use axum::Json;
use serde_json::{Value, json};

use crate::error::Result;
use crate::middleware::AuthenticatedUser;

/// `GET /api/health` — Protected health check.
pub async fn health(user: AuthenticatedUser) -> Result<Json<Value>> {
    Ok(Json(json!({
        "status": "ok",
        "user_id": user.user_id,
    })))
}
