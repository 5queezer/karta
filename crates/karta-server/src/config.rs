use crate::error::{Result, ServerError};

#[derive(Clone, Debug)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub base_url: String,
    pub cookie_secret: Vec<u8>,
    pub google_client_id: String,
    pub google_client_secret: String,
    pub github_client_id: String,
    pub github_client_secret: String,
}

impl ServerConfig {
    pub fn from_env() -> Result<Self> {
        let host = std::env::var("KARTA_HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
        let port = std::env::var("KARTA_PORT")
            .unwrap_or_else(|_| "3000".to_string())
            .parse::<u16>()
            .map_err(|e| ServerError::Config(format!("Invalid KARTA_PORT: {e}")))?;
        let base_url = required_env("KARTA_BASE_URL")?;
        let cookie_secret_b64 = required_env("KARTA_COOKIE_SECRET")?;
        let cookie_secret = base64::Engine::decode(
            &base64::engine::general_purpose::STANDARD,
            &cookie_secret_b64,
        )
        .map_err(|e| ServerError::Config(format!("Invalid KARTA_COOKIE_SECRET base64: {e}")))?;
        if cookie_secret.len() < 32 {
            return Err(ServerError::Config(
                "KARTA_COOKIE_SECRET must decode to at least 32 bytes".to_string(),
            ));
        }

        let google_client_id = required_env("GOOGLE_CLIENT_ID")?;
        let google_client_secret = required_env("GOOGLE_CLIENT_SECRET")?;
        let github_client_id = required_env("GITHUB_CLIENT_ID")?;
        let github_client_secret = required_env("GITHUB_CLIENT_SECRET")?;

        Ok(Self {
            host,
            port,
            base_url,
            cookie_secret,
            google_client_id,
            google_client_secret,
            github_client_id,
            github_client_secret,
        })
    }
}

fn required_env(name: &str) -> Result<String> {
    std::env::var(name).map_err(|_| ServerError::Config(format!("{name} is required")))
}
