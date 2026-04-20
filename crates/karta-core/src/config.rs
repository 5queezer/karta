use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KartaConfig {
    pub storage: StorageConfig,
    pub llm: LlmConfig,
    pub read: ReadConfig,
    pub write: WriteConfig,
    pub dream: DreamConfig,
    pub episode: EpisodeConfig,
    pub forget: ForgetConfig,
    pub reranker: crate::rerank::RerankerConfig,
}

impl Default for KartaConfig {
    fn default() -> Self {
        Self {
            storage: StorageConfig::default(),
            llm: LlmConfig::default(),
            read: ReadConfig::default(),
            write: WriteConfig::default(),
            dream: DreamConfig::default(),
            episode: EpisodeConfig::default(),
            forget: ForgetConfig::default(),
            reranker: Default::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeConfig {
    /// Whether episode segmentation is enabled.
    pub enabled: bool,
    /// Time gap in seconds that forces a new episode boundary.
    pub time_gap_threshold_secs: i64,
}

impl Default for EpisodeConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            time_gap_threshold_secs: 1800, // 30 minutes
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Directory for embedded storage (LanceDB + SQLite).
    pub data_dir: String,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            data_dir: ".karta".into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmModelRef {
    pub provider: String,
    pub model: String,
    #[serde(default)]
    pub base_url: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    /// Default model for all operations.
    pub default: LlmModelRef,
    /// Per-operation overrides. Keys: "write.attributes", "write.linking",
    /// "write.evolve", "read.synthesize", "dream.deduction", etc.
    #[serde(default)]
    pub overrides: HashMap<String, LlmModelRef>,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            default: LlmModelRef {
                provider: "openai".into(),
                model: "gpt-4o-mini".into(),
                base_url: None,
            },
            overrides: HashMap::new(),
        }
    }
}

impl LlmConfig {
    /// Get the model ref for a specific operation, falling back to default.
    pub fn model_for(&self, operation: &str) -> &LlmModelRef {
        self.overrides.get(operation).unwrap_or(&self.default)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadConfig {
    /// Weight for temporal recency in scoring (0.0 = pure similarity, 1.0 = heavy recency bias).
    pub recency_weight: f32,
    /// Half-life for temporal decay in days. A note this many days old gets 50% recency score.
    pub recency_half_life_days: f64,
    /// Boost for notes with active foresight signals.
    pub foresight_boost: f32,
    /// Weight for graph connectivity (PageRank-lite). 0.0 = disabled.
    pub graph_weight: f32,
    /// Max graph traversal depth (1 = current behavior, 2-3 = multi-hop).
    pub max_hop_depth: usize,
    /// Decay factor per hop (0.5 = each hop worth half the previous).
    pub hop_decay_factor: f32,
    /// Minimum similarity score for the best result before the system abstains.
    /// If no note scores above this threshold, the system says "no relevant information."
    pub abstention_threshold: f32,
    /// Top-K multiplier for summarization queries (detected by keywords).
    /// Summarization needs broader coverage than factual queries.
    pub summarization_top_k_multiplier: usize,
    /// Whether to use two-level episode retrieval (ANN on episode narratives → drill into notes).
    pub episode_retrieval_enabled: bool,
    /// Max episodes to drill into per query.
    pub max_episode_drilldowns: usize,
    /// Max notes to include per drilled episode.
    pub max_notes_per_episode: usize,
    /// Min ANN score for an episode narrative to trigger drilldown.
    pub episode_drilldown_min_score: f32,
    /// Whether to search atomic facts alongside notes.
    pub fact_retrieval_enabled: bool,
    /// Score boost for notes found via fact match.
    pub fact_match_boost: f32,
    /// ACTIVATE cognitive-retrieval pipeline: ACT-R + Hebbian + PAS + RRF.
    /// When enabled, supersedes the additive scalar scorer in `search_wide()`.
    #[serde(default)]
    pub activate: ActivateConfig,
}

impl Default for ReadConfig {
    fn default() -> Self {
        Self {
            recency_weight: 0.15,
            recency_half_life_days: 30.0,
            foresight_boost: 0.1,
            graph_weight: 0.05,
            max_hop_depth: 2,
            hop_decay_factor: 0.5,
            abstention_threshold: 0.20,
            summarization_top_k_multiplier: 3,
            episode_retrieval_enabled: true,
            max_episode_drilldowns: 3,
            max_notes_per_episode: 10,
            episode_drilldown_min_score: 0.25,
            fact_retrieval_enabled: true,
            fact_match_boost: 0.1,
            activate: ActivateConfig::default(),
        }
    }
}

// ─── ACTIVATE pipeline configuration ────────────────────────────────────────

/// Configuration for the ACTIVATE 6-phase cognitive retrieval pipeline.
///
/// Feature-flagged via `enabled`. Also honours the `KARTA_ACTIVATE_ENABLED`
/// environment variable so benchmarks can toggle without editing TOML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivateConfig {
    /// Master switch. Defaults to `false` so the existing `search_wide()`
    /// scalar scorer remains the production path until this is validated.
    pub enabled: bool,
    /// ACT-R base-level learning decay rate. Default 0.5 (Anderson 2004).
    pub act_r_decay_d: f64,
    /// Drop notes below this base-level activation from the ACT-R channel.
    pub act_r_min_activation: f64,
    /// Hebbian strengthening: per-retrieval weight increment on co-activated semantic links.
    pub hebbian_weight_step: f32,
    /// Upper bound on Hebbian link weight to prevent runaway.
    pub hebbian_max_weight: f32,
    /// Co-activation channel: top-K weight-sorted neighbors to pull per anchor.
    pub hebbian_neighbors_per_anchor: usize,
    /// Reciprocal Rank Fusion constant. 60 is the canonical Cormack 2009 value.
    pub rrf_k: f32,
    /// PAS sequential walk radius (turns in each direction) for Temporal queries.
    pub pas_window: usize,
    /// Fraction of queries that run phase_trace writes. 1.0 = every query.
    pub trace_sample_rate: f32,
    /// Per-QueryMode channel weight overrides. Key = channel name.
    /// Channels: "ann", "keyword", "hebbian", "actr", "integration", "rerank",
    /// "pas", "facts", "foresight", "profile".
    #[serde(default)]
    pub channel_weights: HashMap<String, HashMap<String, f32>>,
}

impl Default for ActivateConfig {
    fn default() -> Self {
        // Channel-weight matrix seeded per QueryMode.  QueryMode variants are
        // stringified via their Debug form so TOML overrides read naturally
        // (e.g. `[read.activate.channel_weights.Temporal] pas = 2.0`).
        fn mk(pairs: &[(&str, f32)]) -> HashMap<String, f32> {
            pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
        }
        let mut m = HashMap::new();
        m.insert("Standard".into(), mk(&[
            ("ann", 1.0), ("keyword", 0.5), ("hebbian", 0.7), ("actr", 0.3),
            ("integration", 0.5), ("rerank", 1.0), ("pas", 0.0),
            ("facts", 0.6), ("foresight", 0.4), ("profile", 1.2),
        ]));
        m.insert("Recency".into(), mk(&[
            ("ann", 0.6), ("keyword", 0.4), ("hebbian", 0.3), ("actr", 1.2),
            ("integration", 0.3), ("rerank", 0.6), ("pas", 0.0),
            ("facts", 0.4), ("foresight", 0.8), ("profile", 0.8),
        ]));
        m.insert("Breadth".into(), mk(&[
            ("ann", 1.0), ("keyword", 0.5), ("hebbian", 1.0), ("actr", 0.3),
            ("integration", 0.8), ("rerank", 0.8), ("pas", 0.0),
            ("facts", 0.5), ("foresight", 0.4), ("profile", 1.0),
        ]));
        m.insert("Computation".into(), mk(&[
            ("ann", 0.8), ("keyword", 0.8), ("hebbian", 0.4), ("actr", 0.3),
            ("integration", 0.6), ("rerank", 1.2), ("pas", 0.0),
            ("facts", 1.0), ("foresight", 0.5), ("profile", 0.8),
        ]));
        m.insert("Temporal".into(), mk(&[
            ("ann", 0.3), ("keyword", 0.3), ("hebbian", 0.0), ("actr", 1.0),
            ("integration", 0.2), ("rerank", 0.0), ("pas", 1.5),
            ("facts", 0.2), ("foresight", 0.3), ("profile", 0.4),
        ]));
        m.insert("Existence".into(), mk(&[
            ("ann", 1.0), ("keyword", 0.8), ("hebbian", 0.5), ("actr", 0.5),
            ("integration", 0.5), ("rerank", 1.2), ("pas", 0.0),
            ("facts", 0.9), ("foresight", 0.5), ("profile", 1.0),
        ]));

        Self {
            enabled: false,
            act_r_decay_d: 0.5,
            act_r_min_activation: -0.5,
            hebbian_weight_step: 0.05,
            hebbian_max_weight: 3.0,
            hebbian_neighbors_per_anchor: 5,
            rrf_k: 60.0,
            pas_window: 6,
            trace_sample_rate: 1.0,
            channel_weights: m,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WriteConfig {
    /// How many similar notes to consider for linking.
    pub top_k_candidates: usize,
    /// Minimum cosine similarity to be a link candidate.
    pub similarity_threshold: f32,
    /// Whether to retroactively update linked notes' context.
    pub evolve_linked_notes: bool,
    /// Max evolutions before a note is flagged for consolidation instead.
    pub max_evolutions_per_note: usize,
    /// Default TTL in days for foresight signals when no explicit expiry is extracted.
    pub foresight_default_ttl_days: i64,
    /// Whether to extract and store atomic facts during note ingestion.
    pub extract_atomic_facts: bool,
    /// Maximum number of atomic facts to extract per note.
    pub max_facts_per_note: usize,
}

impl Default for WriteConfig {
    fn default() -> Self {
        Self {
            top_k_candidates: 5,
            similarity_threshold: 0.3,
            evolve_linked_notes: true,
            max_evolutions_per_note: 5,
            foresight_default_ttl_days: 90,
            extract_atomic_facts: true,
            max_facts_per_note: 5,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamConfig {
    /// Minimum confidence for a dream to be written back as a note.
    pub write_threshold: f32,
    /// Max notes to feed into one dreaming prompt.
    pub max_notes_per_prompt: usize,
    /// Which dream types to run.
    pub enabled_types: Vec<String>,
}

impl Default for DreamConfig {
    fn default() -> Self {
        Self {
            write_threshold: 0.65,
            max_notes_per_prompt: 8,
            enabled_types: vec![
                "deduction".into(),
                "induction".into(),
                "abduction".into(),
                "consolidation".into(),
                "contradiction".into(),
                "episode_digest".into(),
                "cross_episode_digest".into(),
            ],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForgetConfig {
    /// Whether forgetting is enabled.
    pub enabled: bool,
    /// Half-life in days for access-based decay. Notes not accessed in this
    /// many days get 50% decay score.
    pub decay_half_life_days: f64,
    /// Notes with decay score below this threshold get archived.
    pub archive_threshold: f32,
    /// Whether to run forgetting sweep at the end of each dream pass.
    pub sweep_on_dream: bool,
    /// ACTIVATE: archive notes whose ACT-R base-level activation drops below
    /// this floor (and whose age exceeds `decay_half_life_days`).
    #[serde(default = "default_actr_floor")]
    pub actr_decay_floor: f32,
    /// ACTIVATE: multiplicative decay applied to semantic-link weights on
    /// every sweep. Floored at 1.0 so links never fall below their initial weight.
    #[serde(default = "default_link_decay")]
    pub link_weight_decay: f32,
}

fn default_actr_floor() -> f32 { -1.0 }
fn default_link_decay() -> f32 { 0.99 }

impl Default for ForgetConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            decay_half_life_days: 90.0,
            archive_threshold: 0.1,
            sweep_on_dream: true,
            actr_decay_floor: default_actr_floor(),
            link_weight_decay: default_link_decay(),
        }
    }
}
