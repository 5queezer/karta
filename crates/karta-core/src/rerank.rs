//! Reranker module — scores query-passage relevance for retrieval quality
//! and abstention decisions.
//!
//! The reranker sits between vector search and synthesis:
//! Query → ANN Search → Top-K → **Reranker** → Filtered & Re-scored → Synthesis
//!
//! Architecture: trait-based, pluggable.
//! - JinaReranker: cross-encoder via Jina AI API (recommended — true relevance scoring)
//! - LlmReranker: uses the configured LLM with a tiny prompt (fallback)
//! - NoopReranker: pass-through, no reranking (for testing/cost saving)

use async_trait::async_trait;
use std::sync::Arc;
#[cfg(feature = "fastembed-reranker")]
use std::sync::Mutex;
use tracing::{debug, warn};

#[cfg(feature = "fastembed-reranker")]
use fastembed::{RerankInitOptions, RerankerModel, TextRerank};
#[cfg(feature = "fastembed-reranker-cuda")]
use ort::ep::CUDA;

use crate::error::Result;
use crate::llm::{ChatMessage, GenConfig, LlmProvider, Role};
use crate::note::MemoryNote;

/// A relevance score for a query-passage pair.
#[derive(Debug, Clone)]
pub struct RerankedResult {
    pub note: MemoryNote,
    /// Original similarity score from vector search.
    pub vector_score: f32,
    /// Reranker relevance score (0.0 = irrelevant, 1.0 = highly relevant).
    pub relevance_score: f32,
}

/// Configuration for the reranker.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RerankerConfig {
    /// Whether reranking is enabled.
    pub enabled: bool,
    /// Score below which the system should abstain.
    pub abstention_threshold: f32,
    /// Maximum number of notes to rerank (to control cost).
    pub max_rerank: usize,
}

impl Default for RerankerConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            abstention_threshold: 0.01, // Jina raw scores: low threshold to avoid false abstention
            max_rerank: 20,
        }
    }
}

/// Trait for reranking query-passage pairs.
#[async_trait]
pub trait Reranker: Send + Sync {
    /// Score the relevance of each note to the query.
    /// Returns notes with relevance scores, sorted by relevance descending.
    async fn rerank(
        &self,
        query: &str,
        notes: Vec<(MemoryNote, f32)>,
    ) -> Result<Vec<RerankedResult>>;
}

/// No-op reranker — passes through vector scores unchanged.
pub struct NoopReranker;

#[async_trait]
impl Reranker for NoopReranker {
    async fn rerank(
        &self,
        _query: &str,
        notes: Vec<(MemoryNote, f32)>,
    ) -> Result<Vec<RerankedResult>> {
        Ok(notes
            .into_iter()
            .map(|(note, score)| RerankedResult {
                note,
                vector_score: score,
                relevance_score: score,
            })
            .collect())
    }
}

/// LLM-based reranker — uses a single batched prompt to score all passages.
/// Cheap: one LLM call with max_tokens ~100 to score up to 10 passages.
pub struct LlmReranker {
    llm: Arc<dyn LlmProvider>,
}

impl LlmReranker {
    pub fn new(llm: Arc<dyn LlmProvider>) -> Self {
        Self { llm }
    }
}

#[async_trait]
impl Reranker for LlmReranker {
    async fn rerank(
        &self,
        query: &str,
        notes: Vec<(MemoryNote, f32)>,
    ) -> Result<Vec<RerankedResult>> {
        if notes.is_empty() {
            return Ok(Vec::new());
        }

        // Build a single prompt that scores all passages at once
        let passages: String = notes
            .iter()
            .enumerate()
            .map(|(i, (note, _))| {
                let content = if note.content.len() > 200 {
                    format!(
                        "{}...",
                        &note.content[..note
                            .content
                            .char_indices()
                            .take(200)
                            .last()
                            .map(|(i, ch)| i + ch.len_utf8())
                            .unwrap_or(note.content.len())]
                    )
                } else {
                    note.content.clone()
                };
                format!("[{}] {}", i + 1, content)
            })
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            "Rate how relevant each passage is to the query. \
             Score each 0 (not relevant) to 10 (directly answers the query).\n\
             A passage is relevant ONLY if it contains information that helps answer the query.\n\
             A passage about a DIFFERENT TOPIC than the query scores 0-2 even if it shares vocabulary.\n\n\
             Query: {}\n\nPassages:\n{}\n\n\
             Respond with JSON only: {{\"scores\": [score1, score2, ...]}}",
            query, passages
        );

        let messages = vec![ChatMessage {
            role: Role::User,
            content: prompt,
        }];

        let config = GenConfig {
            max_tokens: 128,
            temperature: 0.0,
            json_mode: true,
            json_schema: None,
        };

        let response = self.llm.chat(&messages, &config).await?;
        let parsed: serde_json::Value = match serde_json::from_str(&response.content) {
            Ok(value) => value,
            Err(e) => {
                warn!(error = %e, raw_response = %response.content, "LLM reranker returned invalid JSON; using neutral fallback");
                serde_json::Value::Null
            }
        };

        let scores: Vec<f32> = parsed["scores"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .map(|v| v.as_f64().unwrap_or(5.0) as f32 / 10.0)
                    .collect()
            })
            .unwrap_or_else(|| vec![0.5; notes.len()]);

        let mut results: Vec<RerankedResult> = notes
            .into_iter()
            .enumerate()
            .map(|(i, (note, vector_score))| {
                let relevance = scores.get(i).copied().unwrap_or(0.5);
                RerankedResult {
                    note,
                    vector_score,
                    relevance_score: relevance,
                }
            })
            .collect();

        // Sort by relevance score descending
        results.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        debug!(
            scores = ?results.iter().map(|r| format!("{:.2}", r.relevance_score)).collect::<Vec<_>>(),
            "Reranked results"
        );

        Ok(results)
    }
}

/// Jina AI cross-encoder reranker — true relevance scoring via API.
/// This is the recommended reranker for production use.
/// Uses jina-reranker-v3 which provides proper cross-attention scoring.
pub struct JinaReranker {
    api_key: String,
    model: String,
    client: reqwest::Client,
}

impl JinaReranker {
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            model: "jina-reranker-v3".to_string(),
            client: reqwest::Client::new(),
        }
    }

    pub fn with_model(api_key: &str, model: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            model: model.to_string(),
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl Reranker for JinaReranker {
    async fn rerank(
        &self,
        query: &str,
        notes: Vec<(MemoryNote, f32)>,
    ) -> Result<Vec<RerankedResult>> {
        if notes.is_empty() {
            return Ok(Vec::new());
        }

        let documents: Vec<String> = notes
            .iter()
            .map(|(note, _)| {
                let content = &note.content;
                if content.len() > 500 {
                    format!(
                        "{}...",
                        &content[..content
                            .char_indices()
                            .take(500)
                            .last()
                            .map(|(i, ch)| i + ch.len_utf8())
                            .unwrap_or(content.len())]
                    )
                } else {
                    content.clone()
                }
            })
            .collect();

        let body = serde_json::json!({
            "model": self.model,
            "query": query,
            "top_n": notes.len(),
            "documents": documents,
            "return_documents": false,
        });

        let response = self
            .client
            .post("https://api.jina.ai/v1/rerank")
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .await
            .map_err(|e| crate::error::KartaError::Llm(format!("Jina rerank error: {}", e)))?;

        let resp_body: serde_json::Value = response
            .json()
            .await
            .map_err(|e| crate::error::KartaError::Llm(format!("Jina parse error: {}", e)))?;

        // Parse Jina response: results[].index + results[].relevance_score
        let jina_results = resp_body["results"].as_array().ok_or_else(|| {
            crate::error::KartaError::Llm(format!(
                "Jina rerank response missing array field 'results': {}",
                resp_body
            ))
        })?;

        // Build index → score map
        let mut score_map: std::collections::HashMap<usize, f32> = std::collections::HashMap::new();
        for r in jina_results {
            let idx = r["index"].as_u64().unwrap_or(0) as usize;
            let score = r["relevance_score"].as_f64().unwrap_or(0.0) as f32;
            score_map.insert(idx, score);
        }

        // Use raw Jina scores directly — do NOT normalize to 0-1.
        // Jina cross-encoder scores are meaningful as-is:
        //   > 0.5 = highly relevant
        //   0.1 - 0.5 = somewhat relevant
        //   < 0.1 = not relevant (abstention zone)
        //   < 0 = definitely irrelevant
        // Normalizing would destroy the abstention signal (all-low scores
        // would map to 0-1 range, making the "best" look good).
        let mut results: Vec<RerankedResult> = notes
            .into_iter()
            .enumerate()
            .map(|(i, (note, vector_score))| {
                let raw_score = score_map.get(&i).copied().unwrap_or(0.0);
                RerankedResult {
                    note,
                    vector_score,
                    relevance_score: raw_score,
                }
            })
            .collect();

        results.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        debug!(
            scores = ?results.iter().map(|r| format!("{:.3}", r.relevance_score)).collect::<Vec<_>>(),
            "Jina reranked"
        );

        Ok(results)
    }
}

/// Local FastEmbed reranker — cross-encoder scoring via ONNX Runtime.
///
/// fastembed-rs' reranker path uses ONNX Runtime. With the
/// `fastembed-reranker-cuda` feature, `KARTA_RERANKER_DEVICE=auto|cuda` tries
/// CUDA first and falls back to CPU on initialization failure. Without that
/// feature, FastEmbed runs on CPU.
#[cfg(feature = "fastembed-reranker")]
pub struct FastEmbedReranker {
    model: Arc<Mutex<TextRerank>>,
}

#[cfg(feature = "fastembed-reranker")]
impl FastEmbedReranker {
    pub fn new() -> Result<Self> {
        let requested_device = std::env::var("KARTA_RERANKER_DEVICE")
            .unwrap_or_else(|_| "auto".to_string())
            .to_ascii_lowercase();
        let model = match requested_device.as_str() {
            "cuda" | "gpu" | "auto" => match try_new_cuda() {
                Ok(model) => model,
                Err(e) => {
                    warn!(
                        error = %e,
                        requested_device,
                        "FastEmbed CUDA reranker unavailable; falling back to CPU"
                    );
                    try_new_cpu()?
                }
            },
            "cpu" => try_new_cpu()?,
            other => {
                warn!(
                    requested_device = other,
                    "Unknown KARTA_RERANKER_DEVICE; falling back to CPU"
                );
                try_new_cpu()?
            }
        };

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
        })
    }
}

#[cfg(feature = "fastembed-reranker")]
fn try_new_cpu() -> Result<TextRerank> {
    let options =
        RerankInitOptions::new(RerankerModel::BGERerankerBase).with_show_download_progress(false);
    TextRerank::try_new(options)
        .map_err(|e| crate::error::KartaError::Llm(format!("FastEmbed CPU init error: {e}")))
}

#[cfg(all(feature = "fastembed-reranker", feature = "fastembed-reranker-cuda"))]
fn try_new_cuda() -> Result<TextRerank> {
    let options = RerankInitOptions::new(RerankerModel::BGERerankerBase)
        .with_show_download_progress(false)
        .with_execution_providers(vec![CUDA::default().build().error_on_failure()]);
    TextRerank::try_new(options)
        .map_err(|e| crate::error::KartaError::Llm(format!("FastEmbed CUDA init error: {e}")))
}

#[cfg(all(
    feature = "fastembed-reranker",
    not(feature = "fastembed-reranker-cuda")
))]
fn try_new_cuda() -> Result<TextRerank> {
    Err(crate::error::KartaError::Llm(
        "binary was built without the fastembed-reranker-cuda feature".to_string(),
    ))
}

#[cfg(feature = "fastembed-reranker")]
#[async_trait]
impl Reranker for FastEmbedReranker {
    async fn rerank(
        &self,
        query: &str,
        notes: Vec<(MemoryNote, f32)>,
    ) -> Result<Vec<RerankedResult>> {
        if notes.is_empty() {
            return Ok(Vec::new());
        }

        let query = query.to_string();
        let documents: Vec<String> = notes
            .iter()
            .map(|(note, _)| truncate_for_reranker(&note.content, 500))
            .collect();
        let model = Arc::clone(&self.model);

        let fastembed_results = tokio::task::spawn_blocking(move || {
            let mut model = model.lock().unwrap_or_else(|e| {
                warn!("FastEmbed reranker mutex was poisoned; recovering inner model");
                e.into_inner()
            });
            model
                .rerank(query, documents, false, None)
                .map_err(|e| crate::error::KartaError::Llm(format!("FastEmbed rerank error: {e}")))
        })
        .await
        .map_err(|e| crate::error::KartaError::Llm(format!("FastEmbed task join error: {e}")))??;

        let mut score_map: std::collections::HashMap<usize, f32> = std::collections::HashMap::new();
        for r in fastembed_results {
            score_map.insert(r.index, r.score);
        }

        let mut results: Vec<RerankedResult> = notes
            .into_iter()
            .enumerate()
            .map(|(i, (note, vector_score))| RerankedResult {
                note,
                vector_score,
                relevance_score: score_map.get(&i).copied().unwrap_or(0.0),
            })
            .collect();

        results.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        debug!(
            scores = ?results.iter().map(|r| format!("{:.3}", r.relevance_score)).collect::<Vec<_>>(),
            "FastEmbed reranked"
        );

        Ok(results)
    }
}

#[cfg(feature = "fastembed-reranker")]
fn truncate_for_reranker(content: &str, max_chars: usize) -> String {
    if content.chars().count() > max_chars {
        let end = content
            .char_indices()
            .take(max_chars)
            .last()
            .map(|(i, ch)| i + ch.len_utf8())
            .unwrap_or(content.len());
        format!("{}...", &content[..end])
    } else {
        content.to_string()
    }
}
