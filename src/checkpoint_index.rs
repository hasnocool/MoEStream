// src/checkpoint_index.rs

use std::{
    collections::{BTreeSet, HashMap},
    path::{Component, Path, PathBuf},
};

use anyhow::{Context, ensure};
use serde::{Deserialize, Serialize};

use crate::safetensors::{SafetensorsHeader, TensorMetadata, TensorSpan};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SafetensorsIndexFile {
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
    pub weight_map: HashMap<String, PathBuf>,
}

#[derive(Debug, Clone)]
pub struct CheckpointTensor {
    pub shard: PathBuf,
    pub metadata: TensorMetadata,
    pub span: TensorSpan,
}

#[derive(Debug, Clone)]
pub struct CheckpointInventory {
    root: PathBuf,
    tensors: HashMap<String, CheckpointTensor>,
    shards: BTreeSet<PathBuf>,
}

impl SafetensorsIndexFile {
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
            validate_relative_path(shard)
                .with_context(|| format!("invalid shard path for tensor {tensor:?}"))?;
        }
        Ok(())
    }

    pub fn shards(&self) -> BTreeSet<PathBuf> {
        self.weight_map.values().cloned().collect()
    }

    pub async fn build_inventory(&self, root: &Path) -> anyhow::Result<CheckpointInventory> {
        self.validate()?;

        let shard_names = self.shards();
        let mut headers = HashMap::with_capacity(shard_names.len());
        for shard in &shard_names {
            let path = root.join(shard);
            let header = SafetensorsHeader::load(&path)
                .await
                .with_context(|| format!("inspect checkpoint shard {}", path.display()))?;
            headers.insert(shard.clone(), header);
        }

        let mut tensors = HashMap::with_capacity(self.weight_map.len());
        for (tensor_name, shard_name) in &self.weight_map {
            let header = headers
                .get(shard_name)
                .with_context(|| format!("missing parsed shard {shard_name:?}"))?;
            let metadata = header
                .tensor(tensor_name)
                .cloned()
                .with_context(|| {
                    format!(
                        "index maps tensor {tensor_name:?} to shard {shard_name:?}, but the shard header does not contain it"
                    )
                })?;
            let span = header.absolute_span(tensor_name)?;
            tensors.insert(
                tensor_name.clone(),
                CheckpointTensor {
                    shard: root.join(shard_name),
                    metadata,
                    span,
                },
            );
        }

        for (shard_name, header) in &headers {
            for tensor_name in header.tensors().keys() {
                let mapped_shard = self.weight_map.get(tensor_name).with_context(|| {
                    format!(
                        "shard {shard_name:?} contains tensor {tensor_name:?} that is absent from the index"
                    )
                })?;
                ensure!(
                    mapped_shard == shard_name,
                    "tensor {tensor_name:?} is physically present in shard {shard_name:?} but index maps it to {mapped_shard:?}"
                );
            }
        }

        Ok(CheckpointInventory {
            root: root.to_path_buf(),
            tensors,
            shards: shard_names,
        })
    }
}

impl CheckpointInventory {
    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub fn shards(&self) -> &BTreeSet<PathBuf> {
        &self.shards
    }

    pub fn tensor(&self, name: &str) -> Option<&CheckpointTensor> {
        self.tensors.get(name)
    }

    pub fn tensors(&self) -> &HashMap<String, CheckpointTensor> {
        &self.tensors
    }
}

fn validate_relative_path(path: &Path) -> anyhow::Result<()> {
    ensure!(!path.as_os_str().is_empty(), "shard path cannot be empty");
    ensure!(
        !path.is_absolute(),
        "shard path must be relative to checkpoint root"
    );
    for component in path.components() {
        ensure!(
            matches!(component, Component::Normal(_)),
            "shard path may contain only normal relative components"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;
    use tokio::io::AsyncWriteExt;

    use super::SafetensorsIndexFile;

    async fn write_safetensors(path: &std::path::Path, tensors: &[(&str, &str, usize)]) {
        let mut offset = 0_u64;
        let mut entries = Vec::new();
        let mut payload = Vec::new();
        for (name, dtype, bytes) in tensors {
            let begin = offset;
            offset += *bytes as u64;
            entries.push(format!(
                "\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{bytes}],\"data_offsets\":[{begin},{offset}]}}"
            ));
            payload.extend(std::iter::repeat_n(0_u8, *bytes));
        }
        let header = format!("{{{}}}", entries.join(","));

        let mut file = tokio::fs::File::create(path).await.expect("create shard");
        file.write_all(&(header.len() as u64).to_le_bytes())
            .await
            .expect("write header length");
        file.write_all(header.as_bytes())
            .await
            .expect("write header");
        file.write_all(&payload).await.expect("write payload");
        file.flush().await.expect("flush shard");
    }

    #[tokio::test]
    async fn builds_inventory_across_multiple_shards() {
        let root = tempdir().expect("checkpoint root");
        write_safetensors(
            &root.path().join("model-00001-of-00002.safetensors"),
            &[("model.embed_tokens.weight", "F16", 8)],
        )
        .await;
        write_safetensors(
            &root.path().join("model-00002-of-00002.safetensors"),
            &[(
                "model.layers.0.mlp.experts.0.gate_proj.weight",
                "F16",
                12,
            )],
        )
        .await;

        let json = r#"{
          "metadata": {"total_size": 20},
          "weight_map": {
            "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
            "model.layers.0.mlp.experts.0.gate_proj.weight": "model-00002-of-00002.safetensors"
          }
        }"#;
        let index = SafetensorsIndexFile::from_json(json).expect("valid index");
        let inventory = index
            .build_inventory(root.path())
            .await
            .expect("inventory");

        assert_eq!(inventory.len(), 2);
        assert_eq!(inventory.shards().len(), 2);
        let tensor = inventory
            .tensor("model.layers.0.mlp.experts.0.gate_proj.weight")
            .expect("expert tensor");
        assert_eq!(tensor.metadata.dtype, "F16");
        assert_eq!(tensor.span.length, 12);
    }

    #[test]
    fn rejects_shard_path_traversal() {
        let json = r#"{"weight_map":{"tensor":"../outside.safetensors"}}"#;
        let error = SafetensorsIndexFile::from_json(json).expect_err("traversal must fail");
        assert!(error.to_string().contains("invalid shard path"));
    }

    #[tokio::test]
    async fn rejects_index_tensor_missing_from_declared_shard() {
        let root = tempdir().expect("checkpoint root");
        write_safetensors(
            &root.path().join("model.safetensors"),
            &[("actual.tensor", "F16", 8)],
        )
        .await;
        let json = r#"{"weight_map":{"missing.tensor":"model.safetensors"}}"#;
        let index = SafetensorsIndexFile::from_json(json).expect("valid syntax");
        let error = index
            .build_inventory(root.path())
            .await
            .expect_err("missing tensor must fail");
        assert!(error.to_string().contains("does not contain it"));
    }

    #[tokio::test]
    async fn rejects_unindexed_tensor_in_shard() {
        let root = tempdir().expect("checkpoint root");
        write_safetensors(
            &root.path().join("model.safetensors"),
            &[
                ("indexed.tensor", "F16", 4),
                ("extra.tensor", "F16", 4),
            ],
        )
        .await;
        let json = r#"{"weight_map":{"indexed.tensor":"model.safetensors"}}"#;
        let index = SafetensorsIndexFile::from_json(json).expect("valid syntax");
        let error = index
            .build_inventory(root.path())
            .await
            .expect_err("unindexed tensor must fail");
        assert!(error.to_string().contains("absent from the index"));
    }
}
