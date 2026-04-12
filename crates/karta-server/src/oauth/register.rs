use axum::extract::State;
use axum::http::StatusCode;
use axum::Json;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::db::OAuthClient;
use crate::error::{Result, ServerError};
use crate::state::AppState;

#[derive(Debug, Deserialize)]
pub struct RegisterRequest {
    pub redirect_uris: Vec<String>,
    pub client_name: Option<String>,
    pub token_endpoint_auth_method: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct RegisterResponse {
    pub client_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_secret: Option<String>,
    pub redirect_uris: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_name: Option<String>,
    pub token_endpoint_auth_method: String,
}

/// `POST /oauth/register` — Dynamic Client Registration (RFC 7591).
pub async fn register_client(
    State(state): State<AppState>,
    Json(req): Json<RegisterRequest>,
) -> Result<(StatusCode, Json<RegisterResponse>)> {
    if req.redirect_uris.is_empty() {
        return Err(ServerError::BadRequest(
            "redirect_uris must not be empty".to_string(),
        ));
    }

    // Validate all redirect URIs are valid URLs
    for uri in &req.redirect_uris {
        if url::Url::parse(uri).is_err() {
            return Err(ServerError::BadRequest(format!(
                "Invalid redirect_uri: {uri}"
            )));
        }
    }

    let auth_method = req
        .token_endpoint_auth_method
        .as_deref()
        .unwrap_or("none");

    let client_id = uuid::Uuid::new_v4().to_string();

    let (client_secret, client_secret_hash) = if auth_method == "client_secret_post" {
        let secret = generate_random_string();
        let hash = sha256_hex(&secret);
        (Some(secret), Some(hash))
    } else {
        (None, None)
    };

    let client = OAuthClient {
        client_id: client_id.clone(),
        client_secret_hash,
        redirect_uris: req.redirect_uris.clone(),
        client_name: req.client_name.clone(),
    };

    state.db.insert_client(&client)?;

    tracing::info!(client_id = %client_id, "Registered new OAuth client");

    Ok((
        StatusCode::CREATED,
        Json(RegisterResponse {
            client_id,
            client_secret,
            redirect_uris: req.redirect_uris,
            client_name: req.client_name,
            token_endpoint_auth_method: auth_method.to_string(),
        }),
    ))
}

fn generate_random_string() -> String {
    use base64::Engine;
    let bytes: [u8; 32] = rand::random();
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
}

fn sha256_hex(input: &str) -> String {
    let digest = Sha256::digest(input.as_bytes());
    let mut hex = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write;
        write!(hex, "{byte:02x}").unwrap();
    }
    hex
}
