// src/manifest.rs

use std::{
    collections::HashMap,
    fs::File,
    io::Read,
    path::{Component, Path, PathBuf},
};

use anyhow::{Context, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::model::{ExpertId, ExpertLocation};

pub const MANIFEST_SCHEMA_VERSION: u32 = 1;
const HASH_BUFFER_BYTES: usize = 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExpertManifest {
    pub schema_version: u32,
    pub model: String,
    pub files: HashMap<String, ManifestFile>,
    pub experts: Vec<ManifestExpert>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ManifestFile {
    pub path: PathBuf,
    pub size: u64,
    #[serde(default)]
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ManifestExpert {
    pub layer: usize,
    pub expert: usize,
    pub file: String,
    pub offset: u64,
    pub length: u64,
    #[serde(default)]
    pub tensors: Vec<ManifestTensor>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ManifestTensor {
    pub name: String,
    pub offset: u64,
    pub length: u64,
    pub dtype: String,
    pub shape: Vec<u64>,
}

#[derive(Debug, Clone)]
pub struct ExpertIndex {
    model: String,
    entries: HashMap<ExpertId, ExpertLocation>,
    metadata: HashMap<ExpertId, ManifestExpert>,
    files: HashMap<String, IndexedFile>,
}

#[derive(Debug, Clone)]
struct IndexedFile {
    path: PathBuf,
    size: u64,
    sha256: Option<String>,
}

impl ExpertManifest {
    pub fn from_json(json: &str) -> anyhow::Result<Self> {
        let manifest: Self = serde_json::from_str(json).context("parse expert manifest JSON")?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub async fn load(path: &Path) -> anyhow::Result<Self> {
        let json = tokio::fs::read_to_string(path)
            .await
            .with_context(|| format!("read expert manifest {}", path.display()))?;
        Self::from_json(&json)
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        ensure!(
            self.schema_version == MANIFEST_SCHEMA_VERSION,
            "unsupported manifest schema_version {}; expected {}",
            self.schema_version,
            MANIFEST_SCHEMA_VERSION
        );
        ensure!(
            !self.model.trim().is_empty(),
            "manifest model cannot be empty"
        );
        ensure!(!self.files.is_empty(), "manifest files cannot be empty");
        ensure!(!self.experts.is_empty(), "manifest experts cannot be empty");

        for (name, file) in &self.files {
            ensure!(!name.trim().is_empty(), "manifest file key cannot be empty");
            validate_relative_path(&file.path)
                .with_context(|| format!("invalid manifest file {name:?}"))?;
            ensure!(
                file.size > 0,
                "manifest file {name:?} size must be non-zero"
            );
            if let Some(sha256) = &file.sha256 {
                validate_sha256(sha256)
                    .with_context(|| format!("invalid SHA-256 for manifest file {name:?}"))?;
            }
        }

        let mut seen = HashMap::with_capacity(self.experts.len());
        for expert in &self.experts {
            let file = self
                .files
                .get(&expert.file)
                .with_context(|| format!("expert references unknown file {:?}", expert.file))?;
            ensure!(expert.length > 0, "expert length must be non-zero");
            let end = expert
                .offset
                .checked_add(expert.length)
                .context("expert byte range overflow")?;
            ensure!(
                end <= file.size,
                "expert byte range {}..{} exceeds declared file size {}",
                expert.offset,
                end,
                file.size
            );
            validate_tensor_layout(expert)?;

            let id = ExpertId {
                layer: expert.layer,
                expert: expert.expert,
            };
            ensure!(
                seen.insert(id.clone(), ()).is_none(),
                "duplicate expert entry for layer {} expert {}",
                id.layer,
                id.expert
            );
        }

        Ok(())
    }

    pub fn build_index(&self, model_root: &Path) -> anyhow::Result<ExpertIndex> {
        self.validate()?;

        let files = self
            .files
            .iter()
            .map(|(name, file)| {
                (
                    name.clone(),
                    IndexedFile {
                        path: model_root.join(&file.path),
                        size: file.size,
                        sha256: file.sha256.clone(),
                    },
                )
            })
            .collect::<HashMap<_, _>>();

        let mut entries = HashMap::with_capacity(self.experts.len());
        let mut metadata = HashMap::with_capacity(self.experts.len());
        for expert in &self.experts {
            let file = files
                .get(&expert.file)
                .with_context(|| format!("expert references unknown file {:?}", expert.file))?;
            let id = ExpertId {
                layer: expert.layer,
                expert: expert.expert,
            };
            entries.insert(
                id.clone(),
                ExpertLocation {
                    path: file.path.clone(),
                    offset: expert.offset,
                    length: expert.length,
                },
            );
            metadata.insert(id, expert.clone());
        }

        Ok(ExpertIndex {
            model: self.model.clone(),
            entries,
            metadata,
            files,
        })
    }
}

impl ExpertIndex {
    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn location(&self, id: &ExpertId) -> anyhow::Result<ExpertLocation> {
        self.entries
            .get(id)
            .cloned()
            .with_context(|| format!("missing expert layer {} expert {}", id.layer, id.expert))
    }

    pub fn expert_metadata(&self, id: &ExpertId) -> anyhow::Result<&ManifestExpert> {
        self.metadata
            .get(id)
            .with_context(|| format!("missing expert metadata layer {} expert {}", id.layer, id.expert))
    }

    pub async fn verify_file_sizes(&self) -> anyhow::Result<()> {
        for (name, file) in &self.files {
            let metadata = tokio::fs::metadata(&file.path)
                .await
                .with_context(|| format!("stat manifest file {name:?}: {}", file.path.display()))?;
            ensure!(
                metadata.is_file(),
                "manifest source {name:?} is not a regular file"
            );
            ensure!(
                metadata.len() == file.size,
                "manifest source {name:?} size mismatch: expected {}, found {}",
                file.size,
                metadata.len()
            );
        }
        Ok(())
    }

    /// Verify every source that declares a SHA-256 digest.
    ///
    /// Reading and hashing large model files is intentionally isolated on
    /// Tokio's blocking pool so request/runtime worker threads never perform
    /// synchronous disk I/O or sustained CPU hashing work.
    pub async fn verify_declared_hashes(&self) -> anyhow::Result<usize> {
        self.verify_file_sizes().await?;

        let mut verified = 0_usize;
        for (name, file) in &self.files {
            let Some(expected) = file.sha256.clone() else {
                continue;
            };

            let path = file.path.clone();
            let source_name = name.clone();
            let actual = tokio::task::spawn_blocking(move || hash_file_sha256(&path))
                .await
                .with_context(|| {
                    format!("join SHA-256 task for manifest file {source_name:?}")
                })??;

            ensure!(
                actual.eq_ignore_ascii_case(&expected),
                "manifest source {name:?} SHA-256 mismatch: expected {expected}, found {actual}"
            );
            verified += 1;
        }

        Ok(verified)
    }

    pub fn declared_sha256(&self, file_name: &str) -> Option<&str> {
        self.files
            .get(file_name)
            .and_then(|file| file.sha256.as_deref())
    }
}

fn validate_tensor_layout(expert: &ManifestExpert) -> anyhow::Result<()> {
    if expert.tensors.is_empty() {
        return Ok(());
    }

    let mut names = HashMap::with_capacity(expert.tensors.len());
    let mut spans = Vec::with_capacity(expert.tensors.len());
    for tensor in &expert.tensors {
        ensure!(!tensor.name.trim().is_empty(), "tensor name cannot be empty");
        ensure!(!tensor.dtype.trim().is_empty(), "tensor dtype cannot be empty");
        ensure!(tensor.length > 0, "tensor length must be non-zero");
        ensure!(
            names.insert(tensor.name.as_str(), ()).is_none(),
            "duplicate tensor metadata {:?}",
            tensor.name
        );
        let end = tensor
            .offset
            .checked_add(tensor.length)
            .context("tensor byte range overflow")?;
        ensure!(
            end <= expert.length,
            "tensor {:?} range {}..{} exceeds expert length {}",
            tensor.name,
            tensor.offset,
            end,
            expert.length
        );
        spans.push((tensor.offset, end, tensor.name.as_str()));
    }

    spans.sort_unstable_by_key(|(start, end, name)| (*start, *end, *name));
    let mut cursor = 0_u64;
    for (start, end, name) in spans {
        ensure!(
            start == cursor,
            "tensor {name:?} begins at {start}, expected contiguous expert offset {cursor}"
        );
        cursor = end;
    }
    ensure!(
        cursor == expert.length,
        "expert tensor metadata covers {cursor} bytes, expected {}",
        expert.length
    );

    Ok(())
}

fn hash_file_sha256(path: &Path) -> anyhow::Result<String> {
    let mut file =
        File::open(path).with_context(|| format!("open {} for SHA-256", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; HASH_BUFFER_BYTES];

    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("read {} for SHA-256", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }

    Ok(format!("{:x}", hasher.finalize()))
}

fn validate_relative_path(path: &Path) -> anyhow::Result<()> {
    ensure!(!path.as_os_str().is_empty(), "path cannot be empty");
    ensure!(
        !path.is_absolute(),
        "path must be relative to the model root"
    );

    for component in path.components() {
        ensure!(
            matches!(component, Component::Normal(_)),
            "path may contain only normal relative components"
        );
    }
    Ok(())
}

fn validate_sha256(value: &str) -> anyhow::Result<()> {
    ensure!(
        value.len() == 64,
        "SHA-256 must contain 64 hexadecimal characters"
    );
    ensure!(
        value.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "SHA-256 must contain only hexadecimal characters"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use tempfile::tempdir;

    use super::{ExpertManifest, MANIFEST_SCHEMA_VERSION};
    use crate::model::ExpertId;

    const TEST_BYTES: &[u8; 16] = b"0123456789abcdef";
    const TEST_SHA256: &str = "9f9f5111f7b27a781f1f1ddde5ebc2dd2b796bfc7365c9c28b548e564176929f";

    fn valid_manifest_json() -> String {
        format!(
            r#"{{
              "schema_version": {MANIFEST_SCHEMA_VERSION},
              "model": "Qwen/Qwen3-30B-A3B",
              "files": {{
                "experts-0": {{
                  "path": "experts/experts-0.bin",
                  "size": 16,
                  "sha256": "{TEST_SHA256}"
                }}
              }},
              "experts": [
                {{"layer": 0, "expert": 0, "file": "experts-0", "offset": 0, "length": 4}},
                {{"layer": 0, "expert": 1, "file": "experts-0", "offset": 4, "length": 4}}
              ]
            }}"#
        )
    }

    async fn write_test_bank(root: &Path) {
        let experts_dir = root.join("experts");
        tokio::fs::create_dir_all(&experts_dir)
            .await
            .expect("create expert directory");
        tokio::fs::write(experts_dir.join("experts-0.bin"), TEST_BYTES)
            .await
            .expect("write expert bank");
    }

    #[test]
    fn builds_index_with_model_root() {
        let manifest = ExpertManifest::from_json(&valid_manifest_json()).expect("valid manifest");
        let index = manifest
            .build_index(Path::new("/models/qwen3"))
            .expect("build expert index");

        assert_eq!(index.model(), "Qwen/Qwen3-30B-A3B");
        assert_eq!(index.len(), 2);
        let id = ExpertId {
            layer: 0,
            expert: 1,
        };
        let location = index.location(&id).expect("expert location");
        assert_eq!(
            location.path,
            Path::new("/models/qwen3/experts/experts-0.bin")
        );
        assert_eq!(location.offset, 4);
        assert_eq!(location.length, 4);
        assert!(index.expert_metadata(&id).expect("metadata").tensors.is_empty());
    }

    #[test]
    fn validates_contiguous_tensor_subranges() {
        let json = valid_manifest_json().replace(
            "\"offset\": 0, \"length\": 4}",
            "\"offset\": 0, \"length\": 4, \"tensors\":[{\"name\":\"gate_proj\",\"offset\":0,\"length\":2,\"dtype\":\"F16\",\"shape\":[1]},{\"name\":\"up_proj\",\"offset\":2,\"length\":2,\"dtype\":\"F16\",\"shape\":[1]}]}",
        );
        ExpertManifest::from_json(&json).expect("valid tensor subranges");
    }

    #[test]
    fn rejects_tensor_layout_holes() {
        let json = valid_manifest_json().replace(
            "\"offset\": 0, \"length\": 4}",
            "\"offset\": 0, \"length\": 4, \"tensors\":[{\"name\":\"gate_proj\",\"offset\":0,\"length\":2,\"dtype\":\"F16\",\"shape\":[1]},{\"name\":\"up_proj\",\"offset\":3,\"length\":1,\"dtype\":\"F16\",\"shape\":[1]}]}",
        );
        let error = ExpertManifest::from_json(&json).expect_err("tensor hole must fail");
        assert!(error.to_string().contains("expected contiguous expert offset"));
    }

    #[test]
    fn rejects_duplicate_expert_ids() {
        let json = valid_manifest_json().replace(
            "{\"layer\": 0, \"expert\": 1, \"file\": \"experts-0\", \"offset\": 4, \"length\": 4}",
            "{\"layer\": 0, \"expert\": 0, \"file\": \"experts-0\", \"offset\": 4, \"length\": 4}",
        );
        let error = ExpertManifest::from_json(&json).expect_err("duplicate IDs must fail");
        assert!(error.to_string().contains("duplicate expert entry"));
    }

    #[test]
    fn rejects_path_traversal() {
        let json =
            valid_manifest_json().replace("experts/experts-0.bin", "../outside-model-root.bin");
        let error = ExpertManifest::from_json(&json).expect_err("path traversal must fail");
        assert!(error.to_string().contains("invalid manifest file"));
    }

    #[test]
    fn rejects_expert_range_past_declared_file_size() {
        let json = valid_manifest_json().replace(
            "\"offset\": 4, \"length\": 4",
            "\"offset\": 14, \"length\": 4",
        );
        let error = ExpertManifest::from_json(&json).expect_err("out-of-range span must fail");
        assert!(error.to_string().contains("exceeds declared file size"));
    }

    #[tokio::test]
    async fn verifies_declared_file_sizes_without_blocking_executor() {
        let root = tempdir().expect("temporary model root");
        write_test_bank(root.path()).await;

        let manifest = ExpertManifest::from_json(&valid_manifest_json()).expect("valid manifest");
        let index = manifest.build_index(root.path()).expect("build index");
        index
            .verify_file_sizes()
            .await
            .expect("file-size verification");
    }

    #[tokio::test]
    async fn verifies_declared_sha256_off_executor_threads() {
        let root = tempdir().expect("temporary model root");
        write_test_bank(root.path()).await;

        let manifest = ExpertManifest::from_json(&valid_manifest_json()).expect("valid manifest");
        let index = manifest.build_index(root.path()).expect("build index");
        let verified = index
            .verify_declared_hashes()
            .await
            .expect("SHA-256 verification");
        assert_eq!(verified, 1);
    }

    #[tokio::test]
    async fn rejects_corrupted_source_hash() {
        let root = tempdir().expect("temporary model root");
        write_test_bank(root.path()).await;
        tokio::fs::write(
            root.path().join("experts/experts-0.bin"),
            b"fedcba9876543210",
        )
        .await
        .expect("corrupt expert bank");

        let manifest = ExpertManifest::from_json(&valid_manifest_json()).expect("valid manifest");
        let index = manifest.build_index(root.path()).expect("build index");
        let error = index
            .verify_declared_hashes()
            .await
            .expect_err("corrupt source must fail verification");
        assert!(error.to_string().contains("SHA-256 mismatch"));
    }
}
