// src/runtime.rs

use std::sync::Arc;

use crate::{
    cache::{CacheMetrics, ExpertBytes, ExpertCache},
    model::{ExpertId, ModelAdapter},
};

#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub max_cached_experts: usize,
    pub expert_io_concurrency: usize,
    pub prefetch_concurrency: usize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            max_cached_experts: 32,
            expert_io_concurrency: 4,
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
        let cache = Arc::new(ExpertCache::with_io_concurrency(
            config.max_cached_experts,
            config.expert_io_concurrency,
        ));
        Self {
            model: Arc::new(model),
            cache,
            config,
        }
    }

    pub fn model(&self) -> &Arc<M> {
        &self.model
    }

    pub fn config(&self) -> &RuntimeConfig {
        &self.config
    }

    pub fn cache_metrics(&self) -> CacheMetrics {
        self.cache.metrics()
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
