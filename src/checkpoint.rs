// src/checkpoint.rs

use std::{
    collections::{HashMap, HashSet},
    path::{Component, Path, PathBuf},
};

use anyhow::{Context, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::safetensors::{SafetensorsHeader, TensorMetadata, TensorSpan};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SafetensorsIndex {
    #[serde(default)]
    pub metadata: HashMap<String, Value>,
    pub weight_map: HashMap<String, PathBuf>,
}

#[derive(Debug, Clone)]
pub struct CheckpointInventory {
    root: PathBuf,
    index: SafetensorsIndex,
    shards: HashMap<PathBuf, SafetensorsHeader>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexedTensor {
    pub name: String,
    pub shard: PathBuf,
    pub metadata: TensorMetadata,
    pub span: TensorSpan,
}

impl SafetensorsIndex {
    pub fn from_json(json: &str) -> anyhow::Result<Self> {
        let index: Self = serde_json::from_str(json).context("parse safetensors index JSON")?;
        index.validate()?;
        Ok(index)
    }

    pub async fn load(path: &Path) -> anyhow::Result<Self> {
        let json = tokio::fs::read_to_string(path)
            .await
            .with_context(|| format!("read safetensors index {}", path.display()))?;
        Self::from_json(&json)
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        ensure!(
            !self.weight_map.is_empty(),
            "safetensors index weight_map cannot be empty"
        );

        for (tensor, shard) in &self.weight_map {
            ensure!(
                !tensor.trim().is_empty(),
                "safetensors index tensor name cannot be empty"
            );
            validate_shard_path(shard)
                .with_context(|| format!("invalid shard path for tensor {tensor:?}"))?;
        }

        Ok(())
    }

    pub fn shard_for(&self, tensor: &str) -> Option<&Path> {
        self.weight_map.get(tensor).map(PathBuf::as_path)
    }

    pub fn shard_paths(&self) -> HashSet<&Path> {
        self.weight_map.values().map(PathBuf::as_path).collect()
    }

    pub async fn inventory(&self, root: &Path) -> anyhow::Result<CheckpointInventory> {
        self.validate()?;

        let mut shards = HashMap::with_capacity(self.shard_paths().len());
        for shard in self.shard_paths() {
            let path = root.join(shard);
            let header = SafetensorsHeader::load(&path)
                .await
                .with_context(|| format!("inspect checkpoint shard {}", path.display()))?;
            shards.insert(shard.to_path_buf(), header);
        }

        validate_index_against_headers(self, &shards)?;

        Ok(CheckpointInventory {
            root: root.to_path_buf(),
            index: self.clone(),
            shards,
        })
    }
}

impl CheckpointInventory {
    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn tensor_count(&self) -> usize {
        self.index.weight_map.len()
    }

    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    pub fn shard_for(&self, tensor: &str) -> Option<&Path> {
        self.index.shard_for(tensor)
    }

    pub fn tensor(&self, name: &str) -> anyhow::Result<IndexedTensor> {
        let shard = self
            .index
            .weight_map
            .get(name)
            .with_context(|| format!("tensor {name:?} is not present in checkpoint index"))?;
        let header = self
            .shards
            .get(shard)
            .with_context(|| format!("checkpoint shard {:?} is not loaded", shard))?;
        let metadata = header
            .tensor(name)
            .cloned()
            .with_context(|| format!("tensor {name:?} is missing from shard {:?}", shard))?;
        let span = header.absolute_span(name)?;

        Ok(IndexedTensor {
            name: name.to_string(),
            shard: self.root.join(shard),
            metadata,
            span,
        })
    }
}

fn validate_shard_path(path: &Path) -> anyhow::Result<()> {
    ensure!(!path.as_os_str().is_empty(), "shard path cannot be empty");
    ensure!(!path.is_absolute(), "shard path must be relative");
    ensure!(
        path.extension().and_then(|value| value.to_str()) == Some("safetensors"),
        "shard path must end in .safetensors"
    );

    for component in path.components() {
        ensure!(
            matches!(component, Component::Normal(_)),
            "shard path may contain only normal relative components"
        );
    }

    Ok(())
}

fn validate_index_against_headers(
    index: &SafetensorsIndex,
    shards: &HashMap<PathBuf, SafetensorsHeader>,
) -> anyhow::Result<()> {
    let mut mapped_by_shard: HashMap<&Path, HashSet<&str>> = HashMap::new();
    for (tensor, shard) in &index.weight_map {
        let header = shards
            .get(shard)
            .with_context(|| format!("checkpoint shard {:?} was not inspected", shard))?;
        ensure!(
            header.tensor(tensor).is_some(),
            "checkpoint index maps tensor {tensor:?} to shard {:?}, but the shard header does not contain it",
            shard
        );
        mapped_by_shard
            .entry(shard.as_path())
            .or_default()
            .insert(tensor.as_str());
    }

    for (shard, header) in shards {
        let mapped = mapped_by_shard
            .get(shard.as_path())
            .with_context(|| format!("checkpoint shard {:?} has no mapped tensors", shard))?;
        for tensor in header.tensors().keys() {
            ensure!(
                mapped.contains(tensor.as_str()),
                "checkpoint shard {:?} contains tensor {tensor:?} that is absent from the index",
                shard
            );
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use tempfile::tempdir;
    use tokio::io::AsyncWriteExt;

    use super::SafetensorsIndex;

    async fn write_shard(path: &Path, header: &str, payload: &[u8]) {
        let mut file = tokio::fs::File::create(path).await.expect("create shard");
        file.write_all(&(header.len() as u64).to_le_bytes())
            .await
            .expect("write header length");
        file.write_all(header.as_bytes())
            .await
            .expect("write header");
        file.write_all(payload).await.expect("write payload");
        file.flush().await.expect("flush shard");
    }

    #[tokio::test]
    async fn inventories_shards_and_resolves_absolute_tensor_span() {
        let root = tempdir().expect("checkpoint root");
        let shard_a =
            r#"{"layer.0.expert.0.gate":{"dtype":"F16","shape":[2],"data_offsets":[0,4]}}"#;
        let shard_b = r#"{"layer.0.expert.0.up":{"dtype":"F16","shape":[2],"data_offsets":[0,4]}}"#;
        write_shard(
            &root.path().join("model-00001-of-00002.safetensors"),
            shard_a,
            &[0; 4],
        )
        .await;
        write_shard(
            &root.path().join("model-00002-of-00002.safetensors"),
            shard_b,
            &[0; 4],
        )
        .await;

        let json = r#"{
            "metadata":{"total_size":8},
            "weight_map":{
                "layer.0.expert.0.gate":"model-00001-of-00002.safetensors",
                "layer.0.expert.0.up":"model-00002-of-00002.safetensors"
            }
        }"#;
        let index = SafetensorsIndex::from_json(json).expect("valid index");
        let inventory = index.inventory(root.path()).await.expect("inventory");

        assert_eq!(inventory.tensor_count(), 2);
        assert_eq!(inventory.shard_count(), 2);
        let tensor = inventory
            .tensor("layer.0.expert.0.up")
            .expect("indexed tensor");
        assert_eq!(tensor.span.length, 4);
        assert_eq!(tensor.span.offset, 8 + shard_b.len() as u64);
        assert_eq!(
            tensor.shard,
            root.path().join("model-00002-of-00002.safetensors")
        );
    }

    #[test]
    fn rejects_unsafe_shard_paths() {
        let json = r#"{
            "weight_map":{"tensor":"../outside.safetensors"}
        }"#;
        let error = SafetensorsIndex::from_json(json).expect_err("path traversal must fail");
        assert!(error.to_string().contains("invalid shard path"));
    }

    #[tokio::test]
    async fn rejects_tensor_mapped_to_wrong_shard() {
        let root = tempdir().expect("checkpoint root");
        let header = r#"{"actual":{"dtype":"F16","shape":[2],"data_offsets":[0,4]}}"#;
        write_shard(&root.path().join("model.safetensors"), header, &[0; 4]).await;

        let index =
            SafetensorsIndex::from_json(r#"{"weight_map":{"expected":"model.safetensors"}}"#)
                .expect("valid index syntax");
        let error = index
            .inventory(root.path())
            .await
            .expect_err("stale mapping must fail");
        assert!(error.to_string().contains("does not contain it"));
    }

    #[tokio::test]
    async fn rejects_unindexed_tensor_in_referenced_shard() {
        let root = tempdir().expect("checkpoint root");
        let header = r#"{"mapped":{"dtype":"F16","shape":[1],"data_offsets":[0,2]},"extra":{"dtype":"F16","shape":[1],"data_offsets":[2,4]}}"#;
        write_shard(&root.path().join("model.safetensors"), header, &[0; 4]).await;

        let index = SafetensorsIndex::from_json(r#"{"weight_map":{"mapped":"model.safetensors"}}"#)
            .expect("valid index syntax");
        let error = index
            .inventory(root.path())
            .await
            .expect_err("partial index must fail");
        assert!(error.to_string().contains("absent from the index"));
    }
}
