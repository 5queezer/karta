use std::sync::Arc;

use rmcp::{
    ErrorData as McpError, ServerHandler,
    handler::server::router::tool::ToolRouter,
    handler::server::wrapper::Parameters,
    model::*,
    schemars, tool, tool_handler, tool_router,
};
use serde::Deserialize;

use karta_core::Karta;

// -- Tool parameter types --

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct AddNoteParams {
    /// The content of the memory note to store
    pub content: String,
    /// Optional session ID for grouping related notes
    #[serde(default)]
    pub session_id: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SearchParams {
    /// The search query
    pub query: String,
    /// Number of results to return (default: 5)
    #[serde(default = "default_top_k")]
    pub top_k: usize,
}

fn default_top_k() -> usize {
    5
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct AskParams {
    /// The question to ask against stored memories
    pub query: String,
    /// Number of context notes to consider (default: 5)
    #[serde(default = "default_top_k")]
    pub top_k: usize,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct GetNoteParams {
    /// The ID of the note to retrieve
    pub id: String,
}

// -- SVG logo as embedded data URI --
const LOGO_SVG: &str = include_str!("logo.svg");

// -- MCP Service --

#[derive(Clone)]
pub struct KartaService {
    karta: Arc<Karta>,
    #[allow(dead_code)]
    tool_router: ToolRouter<KartaService>,
}

#[tool_router]
impl KartaService {
    pub fn new(karta: Arc<Karta>) -> Self {
        Self {
            karta,
            tool_router: Self::tool_router(),
        }
    }

    /// Store a new memory note in the knowledge graph.
    #[tool(description = "Store a new memory note in the knowledge graph. The note will be automatically enriched with LLM-extracted attributes, linked to related notes, and indexed for retrieval.")]
    async fn add_note(
        &self,
        Parameters(params): Parameters<AddNoteParams>,
    ) -> Result<CallToolResult, McpError> {
        let result = if let Some(session_id) = &params.session_id {
            self.karta
                .add_note_with_session(&params.content, session_id)
                .await
        } else {
            self.karta.add_note(&params.content).await
        };

        match result {
            Ok(note) => {
                let response = serde_json::json!({
                    "id": note.id,
                    "content": note.content,
                    "context": note.context,
                    "keywords": note.keywords,
                    "tags": note.tags,
                    "links": note.links,
                    "created_at": note.created_at.to_rfc3339(),
                });
                Ok(CallToolResult::success(vec![Content::text(
                    serde_json::to_string_pretty(&response).unwrap_or_default(),
                )]))
            }
            Err(e) => Err(McpError::new(
                ErrorCode::INTERNAL_ERROR,
                format!("Failed to add note: {e}"),
                None,
            )),
        }
    }

    /// Search stored memories by semantic similarity.
    #[tool(description = "Search stored memories by semantic similarity. Returns the most relevant notes ranked by a combination of vector similarity, recency, and graph connectivity.")]
    async fn search(
        &self,
        Parameters(params): Parameters<SearchParams>,
    ) -> Result<CallToolResult, McpError> {
        match self.karta.search(&params.query, params.top_k).await {
            Ok(results) => {
                let response: Vec<serde_json::Value> = results
                    .iter()
                    .map(|r| {
                        serde_json::json!({
                            "id": r.note.id,
                            "content": r.note.content,
                            "score": r.score,
                            "tags": r.note.tags,
                            "created_at": r.note.created_at.to_rfc3339(),
                        })
                    })
                    .collect();
                Ok(CallToolResult::success(vec![Content::text(
                    serde_json::to_string_pretty(&response).unwrap_or_default(),
                )]))
            }
            Err(e) => Err(McpError::new(
                ErrorCode::INTERNAL_ERROR,
                format!("Search failed: {e}"),
                None,
            )),
        }
    }

    /// Ask a question and get a synthesized answer from stored memories.
    #[tool(description = "Ask a question against stored memories and get a synthesized answer. Uses retrieval-augmented generation: searches for relevant notes, then synthesizes a coherent answer using an LLM.")]
    async fn ask(
        &self,
        Parameters(params): Parameters<AskParams>,
    ) -> Result<CallToolResult, McpError> {
        match self.karta.ask(&params.query, params.top_k).await {
            Ok(result) => {
                let response = serde_json::json!({
                    "answer": result.answer,
                    "query_mode": result.query_mode,
                    "notes_used": result.notes_used,
                    "note_ids": result.note_ids,
                    "has_contradiction": result.has_contradiction,
                });
                Ok(CallToolResult::success(vec![Content::text(
                    serde_json::to_string_pretty(&response).unwrap_or_default(),
                )]))
            }
            Err(e) => Err(McpError::new(
                ErrorCode::INTERNAL_ERROR,
                format!("Ask failed: {e}"),
                None,
            )),
        }
    }

    /// Retrieve a specific note by its ID.
    #[tool(description = "Retrieve a specific memory note by its ID. Returns the full note content and metadata.")]
    async fn get_note(
        &self,
        Parameters(params): Parameters<GetNoteParams>,
    ) -> Result<CallToolResult, McpError> {
        match self.karta.get_note(&params.id).await {
            Ok(Some(note)) => {
                let response = serde_json::json!({
                    "id": note.id,
                    "content": note.content,
                    "context": note.context,
                    "keywords": note.keywords,
                    "tags": note.tags,
                    "links": note.links,
                    "created_at": note.created_at.to_rfc3339(),
                });
                Ok(CallToolResult::success(vec![Content::text(
                    serde_json::to_string_pretty(&response).unwrap_or_default(),
                )]))
            }
            Ok(None) => Err(McpError::resource_not_found(
                format!("Note not found: {}", params.id),
                None,
            )),
            Err(e) => Err(McpError::new(
                ErrorCode::INTERNAL_ERROR,
                format!("Get note failed: {e}"),
                None,
            )),
        }
    }

    /// Get the total count of stored memory notes.
    #[tool(description = "Get the total count of stored memory notes.")]
    async fn note_count(&self) -> Result<CallToolResult, McpError> {
        match self.karta.note_count().await {
            Ok(count) => Ok(CallToolResult::success(vec![Content::text(
                count.to_string(),
            )])),
            Err(e) => Err(McpError::new(
                ErrorCode::INTERNAL_ERROR,
                format!("Count failed: {e}"),
                None,
            )),
        }
    }
}

#[tool_handler]
impl ServerHandler for KartaService {
    fn get_info(&self) -> ServerInfo {
        // Build data URI for the SVG icon
        let icon_data_uri = format!(
            "data:image/svg+xml;base64,{}",
            base64::Engine::encode(&base64::engine::general_purpose::STANDARD, LOGO_SVG)
        );

        ServerInfo::new(
            ServerCapabilities::builder().enable_tools().build(),
        )
        .with_server_info(
            Implementation::new("karta", env!("CARGO_PKG_VERSION"))
                .with_title("Karta Memory Server")
                .with_description("Agentic memory system that thinks, not just stores. Store, search, and reason over knowledge using LLM-enriched notes and graph-based retrieval.")
                .with_icons(vec![
                    Icon::new(icon_data_uri)
                        .with_mime_type("image/svg+xml")
                        .with_sizes(vec!["any".into()]),
                ]),
        )
        .with_instructions("Available tools: add_note (store a memory), search (semantic search), ask (RAG-powered Q&A), get_note (retrieve by ID), note_count (total notes)".to_string())
    }
}
