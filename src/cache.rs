// src/cache.rs

use std::{
    collections::{HashMap, VecDeque},
    io::SeekFrom,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use tokio::{
    fs::File,
    io::{AsyncReadExt, AsyncSeekExt},
    sync::{Mutex, RwLock, Semaphore},
};

use crate::model::{ExpertId, ExpertLocation};

#[derive(Debug, Clone)]
pub struct ExpertBytes(pub Arc<[u8]>);

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheMetrics {
    pub hits: u64,
    pub misses: u64,
    pub coalesced_waits: u64,
    pub evictions: u64,
    pub bytes_read: u64,
}

#[derive(Debug, Default)]
struct Metrics {
    hits: AtomicU64,
    misses: AtomicU64,
    coalesced_waits: AtomicU64,
    evictions: AtomicU64,
    bytes_read: AtomicU64,
}

impl Metrics {
    fn snapshot(&self) -> CacheMetrics {
        CacheMetrics {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            coalesced_waits: self.coalesced_waits.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            bytes_read: self.bytes_read.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Default)]
struct CacheState {
    entries: HashMap<ExpertId, ExpertBytes>,
    lru: VecDeque<ExpertId>,
}

impl CacheState {
    fn touch(&mut self, id: &ExpertId) {
        if let Some(index) = self.lru.iter().position(|candidate| candidate == id) {
            self.lru.remove(index);
        }
        self.lru.push_back(id.clone());
    }
}

#[derive(Debug)]
pub struct ExpertCache {
    state: RwLock<CacheState>,
    inflight: Mutex<HashMap<ExpertId, Arc<Mutex<()>>>>,
    io_slots: Semaphore,
    max_entries: usize,
    metrics: Metrics,
}

impl ExpertCache {
    pub fn new(max_entries: usize) -> Self {
        Self::with_io_concurrency(max_entries, 4)
    }

    pub fn with_io_concurrency(max_entries: usize, io_concurrency: usize) -> Self {
        Self {
            state: RwLock::new(CacheState::default()),
            inflight: Mutex::new(HashMap::new()),
            io_slots: Semaphore::new(io_concurrency.max(1)),
            max_entries: max_entries.max(1),
            metrics: Metrics::default(),
        }
    }

    pub async fn get(&self, id: &ExpertId) -> Option<ExpertBytes> {
        let hit = self.state.read().await.entries.get(id).cloned();
        if hit.is_some() {
            self.metrics.hits.fetch_add(1, Ordering::Relaxed);
            self.state.write().await.touch(id);
        }
        hit
    }

    pub async fn load(&self, id: ExpertId, location: ExpertLocation) -> anyhow::Result<ExpertBytes> {
        if let Some(hit) = self.get(&id).await {
            return Ok(hit);
        }

        self.metrics.misses.fetch_add(1, Ordering::Relaxed);

        let (load_lock, already_inflight) = {
            let mut inflight = self.inflight.lock().await;
            match inflight.get(&id) {
                Some(lock) => (Arc::clone(lock), true),
                None => {
                    let lock = Arc::new(Mutex::new(()));
                    inflight.insert(id.clone(), Arc::clone(&lock));
                    (lock, false)
                }
            }
        };

        if already_inflight {
            self.metrics.coalesced_waits.fetch_add(1, Ordering::Relaxed);
        }

        let _load_guard = load_lock.lock().await;

        if let Some(hit) = self.get(&id).await {
            self.remove_inflight(&id, &load_lock).await;
            return Ok(hit);
        }

        let result = self.read_range(&location).await;
        match result {
            Ok(bytes) => {
                let value = ExpertBytes(Arc::<[u8]>::from(bytes));
                self.insert(id.clone(), value.clone()).await;
                self.remove_inflight(&id, &load_lock).await;
                Ok(value)
            }
            Err(error) => {
                self.remove_inflight(&id, &load_lock).await;
                Err(error)
            }
        }
    }

    async fn read_range(&self, location: &ExpertLocation) -> anyhow::Result<Vec<u8>> {
        let length = usize::try_from(location.length)?;
        let end = location
            .offset
            .checked_add(location.length)
            .ok_or_else(|| anyhow::anyhow!("expert byte range overflow"))?;

        let _permit = self.io_slots.acquire().await?;
        let mut file = File::open(&location.path).await?;
        let file_len = file.metadata().await?.len();
        if end > file_len {
            anyhow::bail!(
                "expert byte range {}..{} exceeds file size {}",
                location.offset,
                end,
                file_len
            );
        }

        file.seek(SeekFrom::Start(location.offset)).await?;
        let mut bytes = vec![0_u8; length];
        file.read_exact(&mut bytes).await?;
        self.metrics
            .bytes_read
            .fetch_add(location.length, Ordering::Relaxed);
        Ok(bytes)
    }

    async fn insert(&self, id: ExpertId, value: ExpertBytes) {
        let mut state = self.state.write().await;

        if !state.entries.contains_key(&id) && state.entries.len() >= self.max_entries {
            if let Some(evicted) = state.lru.pop_front() {
                state.entries.remove(&evicted);
                self.metrics.evictions.fetch_add(1, Ordering::Relaxed);
            }
        }

        state.entries.insert(id.clone(), value);
        state.touch(&id);
    }

    async fn remove_inflight(&self, id: &ExpertId, lock: &Arc<Mutex<()>>) {
        let mut inflight = self.inflight.lock().await;
        if inflight
            .get(id)
            .is_some_and(|candidate| Arc::ptr_eq(candidate, lock))
        {
            inflight.remove(id);
        }
    }

    pub async fn len(&self) -> usize {
        self.state.read().await.entries.len()
    }

    pub async fn is_empty(&self) -> bool {
        self.state.read().await.entries.is_empty()
    }

    pub fn metrics(&self) -> CacheMetrics {
        self.metrics.snapshot()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tempfile::NamedTempFile;
    use tokio::io::AsyncWriteExt;

    use super::ExpertCache;
    use crate::model::{ExpertId, ExpertLocation};

    async fn test_file(bytes: &[u8]) -> NamedTempFile {
        let file = NamedTempFile::new().expect("temporary file");
        let mut async_file = tokio::fs::OpenOptions::new()
            .write(true)
            .open(file.path())
            .await
            .expect("open temporary file");
        async_file.write_all(bytes).await.expect("write test bytes");
        async_file.flush().await.expect("flush test bytes");
        file
    }

    #[tokio::test]
    async fn reads_only_requested_range() {
        let file = test_file(b"0123456789").await;
        let cache = ExpertCache::new(2);
        let id = ExpertId { layer: 0, expert: 1 };
        let location = ExpertLocation {
            path: file.path().to_path_buf(),
            offset: 3,
            length: 4,
        };

        let bytes = cache.load(id, location).await.expect("range read");
        assert_eq!(&*bytes.0, b"3456");
        assert_eq!(cache.metrics().bytes_read, 4);
    }

    #[tokio::test]
    async fn rejects_ranges_past_end_of_file() {
        let file = test_file(b"1234").await;
        let cache = ExpertCache::new(1);
        let location = ExpertLocation {
            path: file.path().to_path_buf(),
            offset: 3,
            length: 2,
        };

        let error = cache
            .load(ExpertId { layer: 0, expert: 0 }, location)
            .await
            .expect_err("invalid range must fail");
        assert!(error.to_string().contains("exceeds file size"));
    }

    #[tokio::test]
    async fn coalesces_concurrent_requests_for_same_expert() {
        let file = test_file(b"abcdefghij").await;
        let cache = Arc::new(ExpertCache::with_io_concurrency(2, 1));
        let id = ExpertId { layer: 1, expert: 7 };
        let location = ExpertLocation {
            path: file.path().to_path_buf(),
            offset: 2,
            length: 5,
        };

        let mut tasks = Vec::new();
        for _ in 0..8 {
            let cache = Arc::clone(&cache);
            let id = id.clone();
            let location = location.clone();
            tasks.push(tokio::spawn(async move { cache.load(id, location).await }));
        }

        for task in tasks {
            let bytes = task.await.expect("task join").expect("expert load");
            assert_eq!(&*bytes.0, b"cdefg");
        }

        assert_eq!(cache.metrics().bytes_read, 5);
        assert!(cache.metrics().coalesced_waits >= 1);
    }

    #[tokio::test]
    async fn evicts_least_recently_used_entry() {
        let file = test_file(b"abcdef").await;
        let cache = ExpertCache::new(2);

        for expert in 0..2 {
            cache
                .load(
                    ExpertId { layer: 0, expert },
                    ExpertLocation {
                        path: file.path().to_path_buf(),
                        offset: expert as u64,
                        length: 1,
                    },
                )
                .await
                .expect("load expert");
        }

        let first = ExpertId { layer: 0, expert: 0 };
        assert!(cache.get(&first).await.is_some());

        cache
            .load(
                ExpertId { layer: 0, expert: 2 },
                ExpertLocation {
                    path: file.path().to_path_buf(),
                    offset: 2,
                    length: 1,
                },
            )
            .await
            .expect("load third expert");

        assert!(cache.get(&first).await.is_some());
        assert!(cache.get(&ExpertId { layer: 0, expert: 1 }).await.is_none());
        assert_eq!(cache.metrics().evictions, 1);
    }
}
