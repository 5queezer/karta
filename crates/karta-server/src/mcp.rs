use std::{fmt::Display, sync::Arc};

use karta_core::{
    Karta,
    note::{MemoryNote, normalize_scope_id, normalize_scope_type, normalize_source_ref},
};
use rmcp::{
    ErrorData as McpError, ServerHandler,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::*,
    schemars, tool, tool_handler, tool_router,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

const DEFAULT_TOP_K: usize = 5;
const MAX_TOP_K: usize = 100;
const DEFAULT_LIST_LIMIT: usize = 20;
const MAX_LIST_LIMIT: usize = 100;
const DEFAULT_LIST_OFFSET: usize = 0;
const DEFAULT_SCOPE_TYPE: &str = "workspace";
const DEFAULT_SCOPE_ID: &str = "default";

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct AddNoteParams {
    /// Memory note to store.
    pub content: String,
    /// Optional session ID for grouping related notes.
    #[serde(default)]
    pub session_id: Option<String>,
    /// Optional memory scope type (for example: global, repo, workspace).
    #[serde(default)]
    pub scope_type: Option<String>,
    /// Optional memory scope identifier.
    #[serde(default)]
    pub scope_id: Option<String>,
    /// Optional source reference, such as a file path, issue URL, or conversation ID.
    #[serde(default)]
    pub source_ref: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SearchParams {
    /// The search query.
    pub query: String,
    /// Number of results to return (default: 5, max: 100).
    #[serde(default = "default_top_k")]
    pub top_k: usize,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct AskParams {
    /// The question to ask against stored memories.
    pub query: String,
    /// Number of context notes to consider (default: 5, max: 100).
    #[serde(default = "default_top_k")]
    pub top_k: usize,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct GetNoteParams {
    /// The ID of the note to retrieve.
    pub id: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ListNotesParams {
    /// Number of notes to return (default: 20, max: 100).
    #[serde(default = "default_list_limit")]
    pub limit: usize,
    /// Number of notes to skip before collecting results.
    #[serde(default = "default_list_offset")]
    pub offset: usize,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct DreamParams {
    /// Dream scope type (default: "workspace").
    #[serde(default = "default_scope_type")]
    pub scope_type: String,
    /// Scope identifier (default: "default").
    #[serde(default = "default_scope_id")]
    pub scope_id: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct GetLinksParams {
    /// The note ID to get links for.
    pub note_id: String,
}

#[derive(Debug, Serialize)]
struct ListNotesResponse {
    total_count: usize,
    offset: usize,
    limit: usize,
    returned_count: usize,
    has_more: bool,
    notes: Vec<MemoryNote>,
}

fn default_top_k() -> usize {
    DEFAULT_TOP_K
}

fn default_list_limit() -> usize {
    DEFAULT_LIST_LIMIT
}

fn default_list_offset() -> usize {
    DEFAULT_LIST_OFFSET
}

fn default_scope_type() -> String {
    DEFAULT_SCOPE_TYPE.into()
}

fn default_scope_id() -> String {
    DEFAULT_SCOPE_ID.into()
}

fn clamp_top_k(top_k: usize) -> usize {
    top_k.clamp(1, MAX_TOP_K)
}

fn clamp_list_limit(limit: usize) -> usize {
    limit.clamp(1, MAX_LIST_LIMIT)
}

fn tool_result(value: Value) -> CallToolResult {
    CallToolResult::structured(value)
}

fn serialize_tool_result<T: Serialize>(value: T) -> Result<CallToolResult, McpError> {
    serde_json::to_value(value)
        .map(tool_result)
        .map_err(|error| {
            McpError::new(
                ErrorCode::INTERNAL_ERROR,
                format!("Failed to serialize tool result: {error}"),
                None,
            )
        })
}

fn internal_error(action: &str, error: impl Display) -> McpError {
    McpError::new(
        ErrorCode::INTERNAL_ERROR,
        format!("{action} failed: {error}"),
        None,
    )
}

fn paginate_notes(notes: Vec<MemoryNote>, offset: usize, limit: usize) -> ListNotesResponse {
    let total_count = notes.len();
    let page: Vec<MemoryNote> = notes.into_iter().skip(offset).take(limit).collect();
    let returned_count = page.len();
    let has_more = offset.saturating_add(returned_count) < total_count;

    ListNotesResponse {
        total_count,
        offset,
        limit,
        returned_count,
        has_more,
        notes: page,
    }
}

#[derive(Clone)]
pub struct KartaService {
    karta: Arc<Karta>,
    base_url: String,
    #[allow(dead_code)]
    tool_router: ToolRouter<KartaService>,
}

#[tool_router]
impl KartaService {
    pub fn new(karta: Arc<Karta>, base_url: String) -> Self {
        Self {
            karta,
            base_url,
            tool_router: Self::tool_router(),
        }
    }

    /// Store a new memory note in the knowledge graph.
    #[tool(
        description = "Store a durable memory note in Karta. Use for stable facts, preferences, constraints, architecture decisions, and bug findings that should survive beyond the current turn. Avoid secrets, raw logs, transient scratch work, or large code blocks."
    )]
    async fn add_note(
        &self,
        Parameters(params): Parameters<AddNoteParams>,
    ) -> Result<CallToolResult, McpError> {
        let scope_type = normalize_scope_type(params.scope_type.as_deref());
        let scope_id = normalize_scope_id(params.scope_id.as_deref());
        let source_ref = normalize_source_ref(params.source_ref.as_deref());
        let has_scope = params
            .scope_type
            .as_deref()
            .is_some_and(|s| !s.trim().is_empty())
            || params
                .scope_id
                .as_deref()
                .is_some_and(|s| !s.trim().is_empty())
            || source_ref.is_some();

        let result = match (params.session_id.as_deref(), has_scope) {
            (Some(session_id), true) => {
                self.karta
                    .add_note_with_session_scoped(
                        &params.content,
                        session_id,
                        &scope_type,
                        &scope_id,
                        source_ref.as_deref(),
                    )
                    .await
            }
            (Some(session_id), false) => {
                self.karta
                    .add_note_with_session(&params.content, session_id)
                    .await
            }
            (None, true) => {
                self.karta
                    .add_note_scoped(
                        &params.content,
                        &scope_type,
                        &scope_id,
                        source_ref.as_deref(),
                    )
                    .await
            }
            (None, false) => self.karta.add_note(&params.content).await,
        };

        match result {
            Ok(note) => serialize_tool_result(json!({
                "id": note.id,
                "content": note.content,
                "context": note.context,
                "keywords": note.keywords,
                "tags": note.tags,
                "links": note.links,
                "confidence": note.confidence,
                "scope_type": note.scope_type,
                "scope_id": note.scope_id,
                "source_ref": note.source_ref,
                "created_at": note.created_at.to_rfc3339(),
            })),
            Err(error) => Err(internal_error("Add note", error)),
        }
    }

    /// Search stored memories by semantic similarity.
    #[tool(
        description = "Search stored memories by semantic similarity. Prefer this for targeted recall before using broader inspection tools. Keep `top_k` small for focused results; values are clamped to 1-100."
    )]
    async fn search(
        &self,
        Parameters(params): Parameters<SearchParams>,
    ) -> Result<CallToolResult, McpError> {
        let top_k = clamp_top_k(params.top_k);

        match self.karta.search(&params.query, top_k).await {
            Ok(results) => {
                let response = json!({
                    "query": params.query,
                    "top_k": top_k,
                    "returned_count": results.len(),
                    "results": results
                        .iter()
                        .map(|result| json!({
                            "id": result.note.id,
                            "content": result.note.content,
                            "context": result.note.context,
                            "keywords": result.note.keywords,
                            "confidence": result.note.confidence,
                            "score": result.score,
                            "tags": result.note.tags,
                            "scope_type": result.note.scope_type,
                            "scope_id": result.note.scope_id,
                            "source_ref": result.note.source_ref,
                            "created_at": result.note.created_at.to_rfc3339(),
                        }))
                        .collect::<Vec<_>>(),
                });
                Ok(tool_result(response))
            }
            Err(error) => Err(internal_error("Search", error)),
        }
    }

    /// Ask a question and get a synthesized answer from stored memories.
    #[tool(
        description = "Ask a question against stored memories and get a synthesized answer with retrieval metadata. Use after `search` when you want Karta to summarize or reconcile notes. Keep `top_k` focused; values are clamped to 1-100."
    )]
    async fn ask(
        &self,
        Parameters(params): Parameters<AskParams>,
    ) -> Result<CallToolResult, McpError> {
        let top_k = clamp_top_k(params.top_k);

        match self.karta.ask(&params.query, top_k).await {
            Ok(result) => serialize_tool_result(json!({
                "query": params.query,
                "top_k": top_k,
                "answer": result.answer,
                "query_mode": result.query_mode,
                "notes_used": result.notes_used,
                "note_ids": result.note_ids,
                "has_contradiction": result.has_contradiction,
                "contradiction_injected": result.contradiction_injected,
            })),
            Err(error) => Err(internal_error("Ask", error)),
        }
    }

    /// Retrieve a specific note by its ID.
    #[tool(
        description = "Retrieve a specific memory note by ID. Use this after `search`, `ask`, or `list_notes` when you need the full stored record and metadata for one note."
    )]
    async fn get_note(
        &self,
        Parameters(params): Parameters<GetNoteParams>,
    ) -> Result<CallToolResult, McpError> {
        match self.karta.get_note(&params.id).await {
            Ok(Some(note)) => serialize_tool_result(json!({
                "id": note.id,
                "content": note.content,
                "context": note.context,
                "keywords": note.keywords,
                "tags": note.tags,
                "links": note.links,
                "confidence": note.confidence,
                "status": note.status,
                "scope_type": note.scope_type,
                "scope_id": note.scope_id,
                "source_ref": note.source_ref,
                "created_at": note.created_at.to_rfc3339(),
                "updated_at": note.updated_at.to_rfc3339(),
            })),
            Ok(None) => Err(McpError::resource_not_found(
                format!("Note not found: {}", params.id),
                None,
            )),
            Err(error) => Err(internal_error("Get note", error)),
        }
    }

    /// Return stored memory notes with pagination metadata.
    #[tool(
        description = "Inspect stored memory notes in a bounded page. Defaults to 20 notes and clamps `limit` to 100. Prefer `search` for targeted recall; use this for audits or debugging."
    )]
    async fn list_notes(
        &self,
        Parameters(params): Parameters<ListNotesParams>,
    ) -> Result<CallToolResult, McpError> {
        let limit = clamp_list_limit(params.limit);
        let offset = params.offset;
        let total_count = self
            .karta
            .note_count()
            .await
            .map_err(|error| internal_error("List notes", error))?;

        match self.karta.list_notes_page(offset, limit).await {
            Ok(notes) => serialize_tool_result({
                let mut response = paginate_notes(notes, 0, limit);
                response.total_count = total_count;
                response.offset = offset;
                response.has_more = offset.saturating_add(response.returned_count) < total_count;
                response
            }),
            Err(error) => Err(internal_error("List notes", error)),
        }
    }

    /// Get the total count of stored memory notes.
    #[tool(description = "Get the total count of stored memory notes.")]
    async fn note_count(&self) -> Result<CallToolResult, McpError> {
        match self.karta.note_count().await {
            Ok(count) => serialize_tool_result(json!({ "count": count })),
            Err(error) => Err(internal_error("Count", error)),
        }
    }

    /// Check Karta embedded store health and migration status.
    #[tool(
        description = "Check Karta embedded store health and migration status. Read-only diagnostic; does not mutate notes."
    )]
    async fn health_check(&self) -> Result<CallToolResult, McpError> {
        match self.karta.health_check().await {
            Ok(health) => serialize_tool_result(health),
            Err(error) => Err(internal_error("Health check", error)),
        }
    }

    /// Preview what the forgetting engine would do without mutating data.
    #[tool(
        description = "Preview what the forgetting engine would do without mutating data. Read-only dry run for inspection before any forgetting pass."
    )]
    async fn preview_forgetting(&self) -> Result<CallToolResult, McpError> {
        match self.karta.preview_forgetting().await {
            Ok(preview) => serialize_tool_result(preview),
            Err(error) => Err(internal_error("Preview forgetting", error)),
        }
    }

    /// Run background reasoning over the knowledge graph.
    #[tool(
        description = "Run background reasoning over the knowledge graph. This may write inferred notes via deduction, induction, abduction, consolidation, contradiction detection, and episode digests."
    )]
    async fn dream(
        &self,
        Parameters(params): Parameters<DreamParams>,
    ) -> Result<CallToolResult, McpError> {
        match self
            .karta
            .run_dreaming(&params.scope_type, &params.scope_id)
            .await
        {
            Ok(run) => {
                let types: Vec<String> = run
                    .dreams
                    .iter()
                    .filter(|dream| dream.would_write)
                    .map(|dream| dream.dream_type.as_str().to_string())
                    .collect();

                serialize_tool_result(json!({
                    "scope_type": params.scope_type,
                    "scope_id": params.scope_id,
                    "dreams_attempted": run.dreams_attempted,
                    "dreams_written": run.dreams_written,
                    "notes_inspected": run.notes_inspected,
                    "types_produced": types,
                    "total_tokens_used": run.total_tokens_used,
                }))
            }
            Err(error) => Err(internal_error("Dream", error)),
        }
    }

    /// Get all note IDs linked to a given note.
    #[tool(
        description = "Get all note IDs linked to a given note in the knowledge graph. Use this for graph inspection after identifying a note of interest."
    )]
    async fn get_links(
        &self,
        Parameters(params): Parameters<GetLinksParams>,
    ) -> Result<CallToolResult, McpError> {
        match self.karta.get_links(&params.note_id).await {
            Ok(links) => serialize_tool_result(json!({
                "note_id": params.note_id,
                "linked_note_ids": links,
            })),
            Err(error) => Err(internal_error("Get links", error)),
        }
    }
}

#[tool_handler]
impl ServerHandler for KartaService {
    fn get_info(&self) -> ServerInfo {
        let icon_url = format!("{}/icon.svg", self.base_url);

        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(
                Implementation::new("karta", env!("CARGO_PKG_VERSION"))
                    .with_title("Karta Memory Server")
                    .with_description(
                        "Agentic memory system that stores durable notes, retrieves them semantically, and reasons over them with graph-aware workflows.",
                    )
                    .with_icons(vec![
                        Icon::new(icon_url)
                            .with_mime_type("image/svg+xml")
                            .with_sizes(vec!["any".into()]),
                    ]),
            )
            .with_instructions(
                "Prefer search for targeted recall, ask for synthesized answers, get_note for one known ID, and list_notes only for bounded inspection (default limit 20, max 100). health_check and preview_forgetting are read-only diagnostics; dream may write inferred notes."
                    .to_string(),
            )
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_LIST_LIMIT, MAX_LIST_LIMIT, clamp_list_limit, clamp_top_k, paginate_notes,
    };
    use karta_core::note::MemoryNote;

    fn note(id: &str) -> MemoryNote {
        let mut note = MemoryNote::new(format!("note {id}"));
        note.id = id.to_string();
        note.context = format!("context {id}");
        note
    }

    #[test]
    fn clamp_helpers_enforce_bounds() {
        assert_eq!(clamp_top_k(0), 1);
        assert_eq!(clamp_top_k(5), 5);
        assert_eq!(clamp_top_k(999), 100);

        assert_eq!(clamp_list_limit(0), 1);
        assert_eq!(clamp_list_limit(DEFAULT_LIST_LIMIT), DEFAULT_LIST_LIMIT);
        assert_eq!(clamp_list_limit(999), MAX_LIST_LIMIT);
    }

    #[test]
    fn paginate_notes_returns_metadata() {
        let page = paginate_notes(vec![note("a"), note("b"), note("c")], 1, 1);
        assert_eq!(page.total_count, 3);
        assert_eq!(page.offset, 1);
        assert_eq!(page.limit, 1);
        assert_eq!(page.returned_count, 1);
        assert!(page.has_more);
        assert_eq!(page.notes.len(), 1);
        assert_eq!(page.notes[0].id, "b");
    }

    #[test]
    fn paginate_notes_handles_empty_tail_page() {
        let page = paginate_notes(vec![note("a")], 5, 2);
        assert_eq!(page.total_count, 1);
        assert_eq!(page.returned_count, 0);
        assert!(!page.has_more);
        assert!(page.notes.is_empty());
    }
}
