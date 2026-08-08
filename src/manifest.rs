// src/manifest.rs

use std::{
    collections::HashMap,
    path::{Component, Path, PathBuf},
};

use anyhow::{Context, ensure};
use serde::{Deserialize, Serialize};

use crate::model::{ExpertId, ExpertLocation};

pub const MANIFEST_SCHEMA_VERSION: u32 = 1;

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
}

#[derive(Debug, Clone)]
pub struct ExpertIndex {
    model: String,
    entries: HashMap<ExpertId, ExpertLocation>,
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
        for expert in &self.experts {
            let file = files
                .get(&expert.file)
                .with_context(|| format!("expert references unknown file {:?}", expert.file))?;
            let id = ExpertId {
                layer: expert.layer,
                expert: expert.expert,
            };
            entries.insert(
                id,
                ExpertLocation {
                    path: file.path.clone(),
                    offset: expert.offset,
                    length: expert.length,
                },
            );
        }

        Ok(ExpertIndex {
            model: self.model.clone(),
            entries,
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

    pub fn declared_sha256(&self, file_name: &str) -> Option<&str> {
        self.files
            .get(file_name)
            .and_then(|file| file.sha256.as_deref())
    }
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

    fn valid_manifest_json() -> String {
        format!(
            r#"{{
              "schema_version": {MANIFEST_SCHEMA_VERSION},
              "model": "Qwen/Qwen3-30B-A3B",
              "files": {{
                "experts-0": {{
                  "path": "experts/experts-0.bin",
                  "size": 16,
                  "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                }}
              }},
              "experts": [
                {{"layer": 0, "expert": 0, "file": "experts-0", "offset": 0, "length": 4}},
                {{"layer": 0, "expert": 1, "file": "experts-0", "offset": 4, "length": 4}}
              ]
            }}"#
        )
    }

    #[test]
    fn builds_index_with_model_root() {
        let manifest = ExpertManifest::from_json(&valid_manifest_json()).expect("valid manifest");
        let index = manifest
            .build_index(Path::new("/models/qwen3"))
            .expect("build expert index");

        assert_eq!(index.model(), "Qwen/Qwen3-30B-A3B");
        assert_eq!(index.len(), 2);
        let location = index
            .location(&ExpertId {
                layer: 0,
                expert: 1,
            })
            .expect("expert location");
        assert_eq!(
            location.path,
            Path::new("/models/qwen3/experts/experts-0.bin")
        );
        assert_eq!(location.offset, 4);
        assert_eq!(location.length, 4);
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
        let experts_dir = root.path().join("experts");
        tokio::fs::create_dir_all(&experts_dir)
            .await
            .expect("create expert directory");
        tokio::fs::write(experts_dir.join("experts-0.bin"), [0_u8; 16])
            .await
            .expect("write expert bank");

        let manifest = ExpertManifest::from_json(&valid_manifest_json()).expect("valid manifest");
        let index = manifest.build_index(root.path()).expect("build index");
        index
            .verify_file_sizes()
            .await
            .expect("file-size verification");
    }
}
