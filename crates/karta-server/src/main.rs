mod config;
mod db;
mod error;
mod middleware;
mod oauth;
mod routes;
mod state;

use axum::Router;
use axum::routing::{get, post};
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

use config::ServerConfig;
use db::AuthDb;
use state::AppState;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load .env if present
    dotenvy::dotenv().ok();

    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "karta_server=info,tower_http=info".into()),
        )
        .init();

    let config = ServerConfig::from_env()?;
    tracing::info!(host = %config.host, port = %config.port, "Starting karta-server");

    // Initialize auth database
    let db = AuthDb::new("karta-auth.db")?;

    // Build application state
    let state = AppState::new(config.clone(), db).await?;

    let app = Router::new()
        // OAuth discovery
        .route(
            "/.well-known/oauth-authorization-server",
            get(oauth::discovery::oauth_metadata),
        )
        // Dynamic client registration
        .route("/oauth/register", post(oauth::register::register_client))
        // Authorization endpoint
        .route("/oauth/authorize", get(oauth::authorize::authorize))
        // Token endpoint
        .route("/oauth/token", post(oauth::token::token))
        // IdP callbacks
        .route("/auth/google/callback", get(oauth::callback::google_callback))
        .route("/auth/github/callback", get(oauth::callback::github_callback))
        // Protected API routes
        .route("/api/health", get(routes::health))
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = format!("{}:{}", config.host, config.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!(addr = %addr, "Listening");

    axum::serve(listener, app).await?;

    Ok(())
}
