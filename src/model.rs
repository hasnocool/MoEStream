// src/model.rs

use std::{fmt::Debug, path::PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ModelArchitecture {
    KimiK3,
    QwenMoe,
    DeepSeekMoe,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ExpertId {
    pub layer: usize,
    pub expert: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertLocation {
    pub path: PathBuf,
    pub offset: u64,
    pub length: u64,
}

/// Model-specific behavior is isolated behind this adapter so the storage,
/// caching, scheduling, and API layers can stay model-independent.
pub trait ModelAdapter: Send + Sync + Debug + 'static {
    fn name(&self) -> &str;
    fn architecture(&self) -> ModelArchitecture;
    fn layer_count(&self) -> usize;
    fn expert_count(&self, layer: usize) -> usize;
    fn experts_per_token(&self, layer: usize) -> usize;

    /// Return the authoritative experts selected for this layer.
    /// Implementations must preserve the source model's exact routing semantics.
    fn route_experts(&self, layer: usize, hidden_state: &[f32]) -> anyhow::Result<Vec<ExpertId>>;

    /// Resolve an expert to a byte range in the on-disk expert bank.
    fn expert_location(&self, expert: &ExpertId) -> anyhow::Result<ExpertLocation>;

    /// Optional scheduling-only hint. These predictions may trigger prefetches,
    /// but they must never replace the authoritative router decision.
    fn prefetch_candidates(&self, _layer: usize, _hidden_state: &[f32]) -> Vec<ExpertId> {
        Vec::new()
    }
}
