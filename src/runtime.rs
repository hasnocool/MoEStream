// src/runtime.rs

use std::sync::Arc;

use crate::{
    cache::{ExpertBytes, ExpertCache},
    model::{ExpertId, ModelAdapter},
};

#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub max_cached_experts: usize,
    pub prefetch_concurrency: usize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            max_cached_experts: 32,
            prefetch_concurrency: 4,
        }
    }
}

#[derive(Debug)]
pub struct MoeRuntime<M: ModelAdapter> {
    model: Arc<M>,
    cache: Arc<ExpertCache>,
    config: RuntimeConfig,
}

impl<M: ModelAdapter> MoeRuntime<M> {
    pub fn new(model: M, config: RuntimeConfig) -> Self {
        Self {
            model: Arc::new(model),
            cache: Arc::new(ExpertCache::new(config.max_cached_experts)),
            config,
        }
    }

    pub fn model(&self) -> &Arc<M> {
        &self.model
    }

    pub fn config(&self) -> &RuntimeConfig {
        &self.config
    }

    pub async fn acquire_authoritative_experts(
        &self,
        layer: usize,
        hidden_state: &[f32],
    ) -> anyhow::Result<Vec<(ExpertId, ExpertBytes)>> {
        let routed = self.model.route_experts(layer, hidden_state)?;
        let mut loaded = Vec::with_capacity(routed.len());

        for expert in routed {
            let location = self.model.expert_location(&expert)?;
            let bytes = self.cache.load(expert.clone(), location).await?;
            loaded.push((expert, bytes));
        }

        Ok(loaded)
    }
}
