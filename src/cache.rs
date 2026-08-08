// src/cache.rs

use std::{collections::HashMap, sync::Arc};

use tokio::sync::RwLock;

use crate::model::{ExpertId, ExpertLocation};

#[derive(Debug, Clone)]
pub struct ExpertBytes(pub Arc<[u8]>);

#[derive(Debug)]
pub struct ExpertCache {
    entries: RwLock<HashMap<ExpertId, ExpertBytes>>,
    max_entries: usize,
}

impl ExpertCache {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
            max_entries: max_entries.max(1),
        }
    }

    pub async fn get(&self, id: &ExpertId) -> Option<ExpertBytes> {
        self.entries.read().await.get(id).cloned()
    }

    pub async fn load(&self, id: ExpertId, location: ExpertLocation) -> anyhow::Result<ExpertBytes> {
        if let Some(hit) = self.get(&id).await {
            return Ok(hit);
        }

        // Tokio file I/O keeps disk access off the async executor's core threads.
        // Range reads will replace this whole-file prototype in the storage milestone.
        let bytes = tokio::fs::read(&location.path).await?;
        let start = usize::try_from(location.offset)?;
        let length = usize::try_from(location.length)?;
        let end = start
            .checked_add(length)
            .ok_or_else(|| anyhow::anyhow!("expert byte range overflow"))?;
        let slice = bytes
            .get(start..end)
            .ok_or_else(|| anyhow::anyhow!("expert byte range exceeds file size"))?;
        let value = ExpertBytes(Arc::<[u8]>::from(slice));

        let mut entries = self.entries.write().await;
        if entries.len() >= self.max_entries {
            // Deterministic placeholder eviction. Replace with measured LRU/ARC policy.
            if let Some(key) = entries.keys().next().cloned() {
                entries.remove(&key);
            }
        }
        entries.insert(id, value.clone());
        Ok(value)
    }

    pub async fn len(&self) -> usize {
        self.entries.read().await.len()
    }

    pub async fn is_empty(&self) -> bool {
        self.entries.read().await.is_empty()
    }
}
