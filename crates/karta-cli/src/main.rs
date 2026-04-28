use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use clap::{Parser, Subcommand};
use karta_core::{Karta, config::KartaConfig, note::MemoryNote};
use serde::Serialize;
use serde_json::json;

#[derive(Debug, Parser)]
#[command(name = "karta")]
#[command(about = "CLI for the Karta agentic memory system")]
#[command(version)]
struct Cli {
    /// Emit machine-readable JSON output.
    #[arg(long, global = true)]
    json: bool,

    /// Embedded storage directory for SQLite graph data and default LanceDB data.
    #[arg(long, global = true, env = "KARTA_DATA_DIR", default_value = ".karta")]
    data_dir: String,

    /// LanceDB URI. Defaults to <data-dir>/lance.
    #[arg(long, global = true, env = "KARTA_LANCE_URI")]
    lance_uri: Option<String>,

    /// Default chat model used by Karta's LLM provider.
    #[arg(long, global = true, env = "KARTA_CHAT_MODEL")]
    model: Option<String>,

    /// OpenAI-compatible base URL (e.g. Ollama/vLLM/Groq). Also honored via OPENAI_API_BASE.
    #[arg(long, global = true, env = "OPENAI_API_BASE")]
    base_url: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Store a new memory note in the knowledge graph.
    AddNote {
        /// Memory content to store.
        #[arg(long)]
        content: String,

        /// Optional session ID for grouping related notes.
        #[arg(long)]
        session_id: Option<String>,

        /// Optional turn index within the source conversation/session.
        #[arg(long)]
        turn_index: Option<u32>,

        /// Optional source timestamp as RFC3339/ISO-8601.
        #[arg(long)]
        source_timestamp: Option<DateTime<Utc>>,
    },

    /// Search stored memories by semantic similarity.
    Search {
        /// Search query.
        #[arg(long)]
        query: String,

        /// Number of results to return.
        #[arg(long, default_value_t = 5)]
        top_k: usize,
    },

    /// Ask a question against stored memories and synthesize an answer.
    Ask {
        /// Question to ask.
        #[arg(long)]
        query: String,

        /// Number of context notes to consider.
        #[arg(long, default_value_t = 5)]
        top_k: usize,
    },

    /// Retrieve a specific note by ID.
    GetNote {
        /// Note ID.
        #[arg(long)]
        id: String,
    },

    /// Return all notes.
    ListNotes,

    /// Get the total count of stored notes.
    NoteCount,

    /// Run background reasoning over the knowledge graph.
    Dream {
        /// Dream scope type.
        #[arg(long, default_value = "workspace")]
        scope_type: String,

        /// Dream scope identifier.
        #[arg(long, default_value = "default")]
        scope_id: String,
    },

    /// Get all note IDs linked to a given note.
    GetLinks {
        /// Note ID to inspect.
        #[arg(long)]
        note_id: String,
    },

    /// Check embedded store health and migration status.
    Health,

    /// Run the forgetting engine.
    Forget,

    /// Preview what the forgetting engine would do without mutating data.
    PreviewForgetting,
}

#[derive(Debug, Serialize)]
struct OkResponse<T> {
    ok: bool,
    #[serde(flatten)]
    data: T,
}

#[derive(Debug, Serialize)]
struct NoteResponse {
    note: MemoryNote,
}

#[derive(Debug, Serialize)]
struct NotesResponse {
    notes: Vec<MemoryNote>,
}

#[derive(Debug, Serialize)]
struct SearchHit {
    note: MemoryNote,
    score: f32,
    linked_notes: Vec<MemoryNote>,
}

#[derive(Debug, Serialize)]
struct SearchResponse {
    query: String,
    top_k: usize,
    results: Vec<SearchHit>,
}

#[derive(Debug, Serialize)]
struct LinksResponse {
    note_id: String,
    linked_note_ids: Vec<String>,
}

#[derive(Debug, Serialize)]
struct CountResponse {
    count: usize,
}

#[tokio::main]
async fn main() {
    dotenvy::dotenv().ok();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "karta_cli=warn,karta_core=warn".into()),
        )
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();
    let json = cli.json;

    if let Err(error) = run(cli).await {
        if json {
            let payload = json!({
                "ok": false,
                "error": error.to_string(),
            });
            eprintln!("{}", serde_json::to_string_pretty(&payload).unwrap());
        } else {
            eprintln!("error: {error:#}");
        }
        std::process::exit(1);
    }
}

async fn run(cli: Cli) -> Result<()> {
    let karta = create_karta(&cli).await?;

    match cli.command {
        Commands::AddNote {
            content,
            session_id,
            turn_index,
            source_timestamp,
        } => {
            let note = match (session_id.as_deref(), turn_index, source_timestamp) {
                (Some(session_id), turn_index, source_timestamp)
                    if turn_index.is_some() || source_timestamp.is_some() =>
                {
                    karta
                        .add_note_with_metadata(&content, session_id, turn_index, source_timestamp)
                        .await?
                }
                (Some(session_id), _, _) => {
                    karta.add_note_with_session(&content, session_id).await?
                }
                (None, None, None) => karta.add_note(&content).await?,
                (None, Some(_), _) | (None, _, Some(_)) => {
                    anyhow::bail!("--turn-index and --source-timestamp require --session-id")
                }
            };
            output(
                cli.json,
                OkResponse {
                    ok: true,
                    data: NoteResponse { note },
                },
                |response| format!("added note {}", response.data.note.id),
            )?;
        }
        Commands::Search { query, top_k } => {
            let top_k = clamp_top_k(top_k);
            let results = karta.search(&query, top_k).await?;
            let response = SearchResponse {
                query,
                top_k,
                results: results
                    .into_iter()
                    .map(|result| SearchHit {
                        note: result.note,
                        score: result.score,
                        linked_notes: result.linked_notes,
                    })
                    .collect(),
            };
            output(
                cli.json,
                OkResponse {
                    ok: true,
                    data: response,
                },
                |response| {
                    response
                        .data
                        .results
                        .iter()
                        .map(|hit| {
                            format!("{:.3}\t{}\t{}", hit.score, hit.note.id, hit.note.content)
                        })
                        .collect::<Vec<_>>()
                        .join("\n")
                },
            )?;
        }
        Commands::Ask { query, top_k } => {
            let result = karta.ask(&query, clamp_top_k(top_k)).await?;
            output(
                cli.json,
                OkResponse {
                    ok: true,
                    data: result,
                },
                |response| response.data.answer.clone(),
            )?;
        }
        Commands::GetNote { id } => {
            let Some(note) = karta.get_note(&id).await? else {
                anyhow::bail!("note not found: {id}");
            };
            output(
                cli.json,
                OkResponse {
                    ok: true,
                    data: NoteResponse { note },
                },
                |response| response.data.note.content.clone(),
            )?;
        }
        Commands::ListNotes => {
            let notes = karta.get_all_notes().await?;
            output(
                cli.json,
                OkResponse {
                    ok: true,
                    data: NotesResponse { notes },
                },
                |response| {
                    response
                        .data
                        .notes
                        .iter()
                        .map(|note| format!("{}\t{}", note.id, note.content))
                        .collect::<Vec<_>>()
                        .join("\n")
                },
            )?;
        }
        Commands::NoteCount => {
            let count = karta.note_count().await?;
            output(
                cli.json,
                OkResponse {
                    ok: true,
                    data: CountResponse { count },
                },
                |response| response.data.count.to_string(),
            )?;
        }
        Commands::Dream {
            scope_type,
            scope_id,
        } => {
            let run = karta.run_dreaming(&scope_type, &scope_id).await?;
            output(
                cli.json,
                OkResponse {
                    ok: true,
                    data: run,
                },
                |response| {
                    format!(
                        "dreams attempted: {}, written: {}, notes inspected: {}",
                        response.data.dreams_attempted,
                        response.data.dreams_written,
                        response.data.notes_inspected
                    )
                },
            )?;
        }
        Commands::GetLinks { note_id } => {
            let linked_note_ids = karta.get_links(&note_id).await?;
            output(
                cli.json,
                OkResponse {
                    ok: true,
                    data: LinksResponse {
                        note_id,
                        linked_note_ids,
                    },
                },
                |response| response.data.linked_note_ids.join("\n"),
            )?;
        }
        Commands::Health => {
            let health = karta.health_check().await?;
            output(
                cli.json,
                OkResponse {
                    ok: true,
                    data: health,
                },
                |response| {
                    let status = if response.data.vector_store_ok && response.data.graph_store_ok {
                        "healthy"
                    } else {
                        "unhealthy"
                    };
                    format!(
                        "{status}\nvector_store_ok: {}\ngraph_store_ok: {}\nschema_version: {}\nwarnings: {}",
                        response.data.vector_store_ok,
                        response.data.graph_store_ok,
                        response
                            .data
                            .schema_version
                            .clone()
                            .unwrap_or_else(|| "unknown".into()),
                        response.data.warnings.join("; ")
                    )
                },
            )?;
        }
        Commands::Forget => {
            let run = karta.run_forgetting().await?;
            output(
                cli.json,
                OkResponse {
                    ok: true,
                    data: run,
                },
                |response| {
                    format!(
                        "notes inspected: {}, archived: {}, deprecated: {}, protected: {}",
                        response.data.notes_inspected,
                        response.data.notes_archived,
                        response.data.notes_deprecated,
                        response.data.notes_protected
                    )
                },
            )?;
        }
        Commands::PreviewForgetting => {
            let preview = karta.preview_forgetting().await?;
            output(
                cli.json,
                OkResponse {
                    ok: true,
                    data: preview,
                },
                |response| {
                    format!(
                        "would archive: {}, deprecate: {}, protected: {}",
                        response.data.total_archived,
                        response.data.total_deprecated,
                        response.data.total_protected
                    )
                },
            )?;
        }
    }

    Ok(())
}

async fn create_karta(cli: &Cli) -> Result<Karta> {
    let mut config = KartaConfig::default();
    config.storage.data_dir = cli.data_dir.clone();
    config.storage.lance_uri = cli.lance_uri.clone();

    if let Some(model) = &cli.model {
        config.llm.default.model = model.clone();
    }
    if let Some(base_url) = &cli.base_url {
        config.llm.default.base_url = Some(base_url.clone());
    }

    Karta::with_defaults(config)
        .await
        .context("failed to initialize Karta")
}

fn clamp_top_k(top_k: usize) -> usize {
    top_k.clamp(1, 100)
}

fn output<T, F>(json: bool, value: T, human: F) -> Result<()>
where
    T: Serialize,
    F: FnOnce(&T) -> String,
{
    if json {
        println!("{}", serde_json::to_string_pretty(&value)?);
    } else {
        println!("{}", human(&value));
    }
    Ok(())
}
