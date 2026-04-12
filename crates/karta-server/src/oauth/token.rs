use axum::extract::State;
use axum::http::StatusCode;
use axum::Json;
use base64::Engine;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{Result, ServerError};
use crate::middleware::hash_token;
use crate::state::AppState;

#[derive(Debug, Deserialize)]
pub struct TokenRequest {
    pub grant_type: Option<String>,
    pub code: Option<String>,
    pub redirect_uri: Option<String>,
    pub code_verifier: Option<String>,
    pub client_id: Option<String>,
    pub client_secret: Option<String>,
    pub refresh_token: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct TokenResponse {
    pub access_token: String,
    pub token_type: String,
    pub expires_in: u64,
    pub refresh_token: String,
}

/// `POST /oauth/token` — Token endpoint.
pub async fn token(
    State(state): State<AppState>,
    axum::Form(req): axum::Form<TokenRequest>,
) -> Result<Json<TokenResponse>> {
    let grant_type = req
        .grant_type
        .as_deref()
        .ok_or_else(|| ServerError::oauth(
            "invalid_request",
            "grant_type is required",
            StatusCode::BAD_REQUEST,
        ))?;

    match grant_type {
        "authorization_code" => handle_auth_code_grant(&state, &req).await,
        "refresh_token" => handle_refresh_token_grant(&state, &req).await,
        _ => Err(ServerError::oauth(
            "unsupported_grant_type",
            "Only authorization_code and refresh_token are supported",
            StatusCode::BAD_REQUEST,
        )),
    }
}

async fn handle_auth_code_grant(
    state: &AppState,
    req: &TokenRequest,
) -> Result<Json<TokenResponse>> {
    let code = req
        .code
        .as_deref()
        .ok_or_else(|| ServerError::oauth(
            "invalid_request",
            "code is required",
            StatusCode::BAD_REQUEST,
        ))?;
    let redirect_uri = req
        .redirect_uri
        .as_deref()
        .ok_or_else(|| ServerError::oauth(
            "invalid_request",
            "redirect_uri is required",
            StatusCode::BAD_REQUEST,
        ))?;
    let code_verifier = req
        .code_verifier
        .as_deref()
        .ok_or_else(|| ServerError::oauth(
            "invalid_request",
            "code_verifier is required",
            StatusCode::BAD_REQUEST,
        ))?;

    // Consume the auth code (one-time use)
    let auth_code = state
        .db
        .consume_auth_code(code)?
        .ok_or_else(|| ServerError::oauth(
            "invalid_grant",
            "Invalid, expired, or already used authorization code",
            StatusCode::BAD_REQUEST,
        ))?;

    // Check expiry
    let expires = chrono::NaiveDateTime::parse_from_str(&auth_code.expires_at, "%Y-%m-%d %H:%M:%S")
        .map_err(|e| ServerError::Internal(format!("Bad expiry: {e}")))?;
    if expires.and_utc() < chrono::Utc::now() {
        return Err(ServerError::oauth(
            "invalid_grant",
            "Authorization code has expired",
            StatusCode::BAD_REQUEST,
        ));
    }

    // Verify redirect_uri matches
    if auth_code.redirect_uri != redirect_uri {
        return Err(ServerError::oauth(
            "invalid_grant",
            "redirect_uri does not match",
            StatusCode::BAD_REQUEST,
        ));
    }

    // Verify PKCE: BASE64URL(SHA256(code_verifier)) == code_challenge
    let computed_challenge = {
        let digest = Sha256::digest(code_verifier.as_bytes());
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(digest)
    };

    if computed_challenge != auth_code.code_challenge {
        return Err(ServerError::oauth(
            "invalid_grant",
            "PKCE code_verifier does not match code_challenge",
            StatusCode::BAD_REQUEST,
        ));
    }

    // Validate client authentication if required
    if let Some(client_id) = &req.client_id {
        if *client_id != auth_code.client_id {
            return Err(ServerError::oauth(
                "invalid_client",
                "client_id does not match",
                StatusCode::BAD_REQUEST,
            ));
        }
    }

    // Issue tokens
    issue_tokens(state, &auth_code.client_id, &auth_code.user_id, auth_code.scope.as_deref())
}

async fn handle_refresh_token_grant(
    state: &AppState,
    req: &TokenRequest,
) -> Result<Json<TokenResponse>> {
    let refresh_token = req
        .refresh_token
        .as_deref()
        .ok_or_else(|| ServerError::oauth(
            "invalid_request",
            "refresh_token is required",
            StatusCode::BAD_REQUEST,
        ))?;

    let token_hash = hash_token(refresh_token);

    let (client_id, user_id, scope) = state
        .db
        .consume_refresh_token(&token_hash)?
        .ok_or_else(|| ServerError::oauth(
            "invalid_grant",
            "Invalid, expired, or revoked refresh token",
            StatusCode::BAD_REQUEST,
        ))?;

    // Issue new tokens (rotation)
    issue_tokens(state, &client_id, &user_id, scope.as_deref())
}

fn issue_tokens(
    state: &AppState,
    client_id: &str,
    user_id: &str,
    scope: Option<&str>,
) -> Result<Json<TokenResponse>> {
    // Generate access token (32 random bytes, base64url)
    let access_token_bytes: [u8; 32] = rand::random();
    let access_token = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(access_token_bytes);
    let access_token_hash = hash_token(&access_token);

    // Generate refresh token (32 random bytes, base64url)
    let refresh_token_bytes: [u8; 32] = rand::random();
    let refresh_token = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(refresh_token_bytes);
    let refresh_token_hash = hash_token(&refresh_token);

    let now = chrono::Utc::now();
    let access_expires = (now + chrono::Duration::hours(1))
        .format("%Y-%m-%d %H:%M:%S")
        .to_string();
    let refresh_expires = (now + chrono::Duration::days(30))
        .format("%Y-%m-%d %H:%M:%S")
        .to_string();

    // Store hashed tokens
    state.db.insert_access_token(
        &access_token_hash,
        client_id,
        user_id,
        scope,
        &access_expires,
    )?;
    state.db.insert_refresh_token(
        &refresh_token_hash,
        client_id,
        user_id,
        scope,
        &refresh_expires,
    )?;

    tracing::info!(user_id = %user_id, client_id = %client_id, "Issued new token pair");

    Ok(Json(TokenResponse {
        access_token,
        token_type: "Bearer".to_string(),
        expires_in: 3600,
        refresh_token,
    }))
}
