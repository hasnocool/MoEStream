// src/adapters/qwen3_prepare.rs

use std::{collections::HashMap, io::ErrorKind, path::Path};

use anyhow::{Context, ensure};
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};

use crate::{
    adapters::qwen3::Qwen3MoeConfig,
    checkpoint::{CheckpointInventory, IndexedTensor},
    manifest::{
        ExpertManifest, MANIFEST_SCHEMA_VERSION, ManifestExpert, ManifestFile, ManifestTensor,
        sha256_file,
    },
    safetensors::TensorSpan,
};

const COPY_BUFFER_BYTES: usize = 1024 * 1024;
const BANK_FILE_KEY: &str = "expert-bank";
const BANK_RELATIVE_PATH: &str = "experts/expert-bank.bin";
const MANIFEST_FILE_NAME: &str = "expert-manifest.json";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen3ExpertTensorNames {
    pub gate_proj: String,
    pub up_proj: String,
    pub down_proj: String,
}

impl Qwen3ExpertTensorNames {
    pub fn canonical(layer: usize, expert: usize) -> Self {
        let prefix = format!("model.layers.{layer}.mlp.experts.{expert}");
        Self {
            gate_proj: format!("{prefix}.gate_proj.weight"),
            up_proj: format!("{prefix}.up_proj.weight"),
            down_proj: format!("{prefix}.down_proj.weight"),
        }
    }
}

/// Repack canonical Hugging Face Qwen3-MoE expert projections into one
/// contiguous bank file and emit a validated MoEStream expert manifest.
///
/// Source tensor reads and destination writes use bounded asynchronous I/O.
/// The final SHA-256 pass is isolated from Tokio workers by `sha256_file`.
pub async fn prepare_qwen3_expert_bank(
    inventory: &CheckpointInventory,
    config: &Qwen3MoeConfig,
    output_root: &Path,
    model_name: &str,
) -> anyhow::Result<ExpertManifest> {
    config.validate()?;
    ensure!(!model_name.trim().is_empty(), "model_name cannot be empty");

    let experts_dir = output_root.join("experts");
    let bank_path = output_root.join(BANK_RELATIVE_PATH);
    let partial_bank_path = experts_dir.join("expert-bank.bin.partial");
    let manifest_path = output_root.join(MANIFEST_FILE_NAME);
    let partial_manifest_path = output_root.join("expert-manifest.json.partial");

    tokio::fs::create_dir_all(&experts_dir)
        .await
        .with_context(|| format!("create output directory {}", experts_dir.display()))?;
    ensure!(
        !tokio::fs::try_exists(&bank_path).await?,
        "expert bank already exists: {}",
        bank_path.display()
    );
    ensure!(
        !tokio::fs::try_exists(&manifest_path).await?,
        "expert manifest already exists: {}",
        manifest_path.display()
    );
    remove_if_exists(&partial_bank_path).await?;
    remove_if_exists(&partial_manifest_path).await?;

    let result = build_bank(
        inventory,
        config,
        model_name,
        &partial_bank_path,
        &partial_manifest_path,
    )
    .await;

    match result {
        Ok(manifest) => {
            tokio::fs::rename(&partial_bank_path, &bank_path)
                .await
                .with_context(|| {
                    format!(
                        "publish expert bank {} -> {}",
                        partial_bank_path.display(),
                        bank_path.display()
                    )
                })?;
            tokio::fs::rename(&partial_manifest_path, &manifest_path)
                .await
                .with_context(|| {
                    format!(
                        "publish expert manifest {} -> {}",
                        partial_manifest_path.display(),
                        manifest_path.display()
                    )
                })?;
            Ok(manifest)
        }
        Err(error) => {
            let _ = remove_if_exists(&partial_bank_path).await;
            let _ = remove_if_exists(&partial_manifest_path).await;
            Err(error)
        }
    }
}

async fn build_bank(
    inventory: &CheckpointInventory,
    config: &Qwen3MoeConfig,
    model_name: &str,
    partial_bank_path: &Path,
    partial_manifest_path: &Path,
) -> anyhow::Result<ExpertManifest> {
    let mut output = tokio::fs::File::create(partial_bank_path)
        .await
        .with_context(|| format!("create expert bank {}", partial_bank_path.display()))?;
    let mut experts = Vec::with_capacity(config.num_hidden_layers * config.num_experts);
    let mut bank_offset = 0_u64;

    for layer in 0..config.num_hidden_layers {
        for expert in 0..config.num_experts {
            let names = Qwen3ExpertTensorNames::canonical(layer, expert);
            let gate = inventory
                .tensor(&names.gate_proj)
                .with_context(|| format!("resolve layer {layer} expert {expert} gate_proj"))?;
            let up = inventory
                .tensor(&names.up_proj)
                .with_context(|| format!("resolve layer {layer} expert {expert} up_proj"))?;
            let down = inventory
                .tensor(&names.down_proj)
                .with_context(|| format!("resolve layer {layer} expert {expert} down_proj"))?;

            validate_projection_shapes(config, layer, expert, &gate, &up, &down)?;

            let expert_offset = bank_offset;
            let mut tensor_offset = 0_u64;
            let mut tensors = Vec::with_capacity(3);

            for (short_name, tensor) in
                [("gate_proj", &gate), ("up_proj", &up), ("down_proj", &down)]
            {
                copy_tensor_span(&tensor.shard, tensor.span, &mut output)
                    .await
                    .with_context(|| format!("copy tensor {:?}", tensor.name))?;
                tensors.push(ManifestTensor {
                    name: short_name.to_string(),
                    offset: tensor_offset,
                    length: tensor.span.length,
                    dtype: tensor.metadata.dtype.clone(),
                    shape: tensor.metadata.shape.clone(),
                });
                tensor_offset = tensor_offset
                    .checked_add(tensor.span.length)
                    .context("expert tensor length overflow")?;
            }

            experts.push(ManifestExpert {
                layer,
                expert,
                file: BANK_FILE_KEY.to_string(),
                offset: expert_offset,
                length: tensor_offset,
                tensors,
            });
            bank_offset = bank_offset
                .checked_add(tensor_offset)
                .context("expert bank size overflow")?;
        }
    }

    output.flush().await.context("flush expert bank")?;
    output.sync_all().await.context("sync expert bank")?;
    drop(output);

    let actual_size = tokio::fs::metadata(partial_bank_path)
        .await
        .with_context(|| format!("stat expert bank {}", partial_bank_path.display()))?
        .len();
    ensure!(
        actual_size == bank_offset,
        "expert bank size mismatch: wrote {bank_offset} bytes, file contains {actual_size}"
    );

    let digest = sha256_file(partial_bank_path).await?;
    let manifest = ExpertManifest {
        schema_version: MANIFEST_SCHEMA_VERSION,
        model: model_name.to_string(),
        files: HashMap::from([(
            BANK_FILE_KEY.to_string(),
            ManifestFile {
                path: BANK_RELATIVE_PATH.into(),
                size: actual_size,
                sha256: Some(digest),
            },
        )]),
        experts,
    };
    manifest.validate()?;

    let manifest_json =
        serde_json::to_vec_pretty(&manifest).context("serialize expert manifest")?;
    tokio::fs::write(partial_manifest_path, manifest_json)
        .await
        .with_context(|| format!("write expert manifest {}", partial_manifest_path.display()))?;

    Ok(manifest)
}

fn validate_projection_shapes(
    config: &Qwen3MoeConfig,
    layer: usize,
    expert: usize,
    gate: &IndexedTensor,
    up: &IndexedTensor,
    down: &IndexedTensor,
) -> anyhow::Result<()> {
    ensure!(
        gate.metadata.dtype == up.metadata.dtype && gate.metadata.dtype == down.metadata.dtype,
        "layer {layer} expert {expert} projection dtypes differ: gate={}, up={}, down={}",
        gate.metadata.dtype,
        up.metadata.dtype,
        down.metadata.dtype
    );

    let hidden = u64::try_from(config.hidden_size).context("hidden_size exceeds u64")?;
    let moe =
        u64::try_from(config.moe_intermediate_size).context("moe_intermediate_size exceeds u64")?;
    let expected_in = vec![moe, hidden];
    let expected_down = vec![hidden, moe];

    ensure!(
        gate.metadata.shape == expected_in,
        "layer {layer} expert {expert} gate_proj shape {:?} != {:?}",
        gate.metadata.shape,
        expected_in
    );
    ensure!(
        up.metadata.shape == expected_in,
        "layer {layer} expert {expert} up_proj shape {:?} != {:?}",
        up.metadata.shape,
        expected_in
    );
    ensure!(
        down.metadata.shape == expected_down,
        "layer {layer} expert {expert} down_proj shape {:?} != {:?}",
        down.metadata.shape,
        expected_down
    );
    Ok(())
}

async fn copy_tensor_span(
    source_path: &Path,
    span: TensorSpan,
    output: &mut tokio::fs::File,
) -> anyhow::Result<()> {
    ensure!(span.length > 0, "tensor span length must be non-zero");
    let mut source = tokio::fs::File::open(source_path)
        .await
        .with_context(|| format!("open tensor source {}", source_path.display()))?;
    source
        .seek(std::io::SeekFrom::Start(span.offset))
        .await
        .with_context(|| format!("seek tensor source {}", source_path.display()))?;

    let mut remaining = span.length;
    let mut buffer = vec![0_u8; COPY_BUFFER_BYTES];
    while remaining > 0 {
        let chunk = usize::try_from(remaining.min(COPY_BUFFER_BYTES as u64))
            .context("copy chunk exceeds usize")?;
        source
            .read_exact(&mut buffer[..chunk])
            .await
            .with_context(|| format!("read tensor source {}", source_path.display()))?;
        output
            .write_all(&buffer[..chunk])
            .await
            .context("write expert bank tensor")?;
        remaining -= chunk as u64;
    }

    Ok(())
}

async fn remove_if_exists(path: &Path) -> anyhow::Result<()> {
    match tokio::fs::remove_file(path).await {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).with_context(|| format!("remove stale file {}", path.display())),
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use tempfile::tempdir;
    use tokio::io::AsyncWriteExt;

    use super::{Qwen3ExpertTensorNames, prepare_qwen3_expert_bank};
    use crate::{adapters::qwen3::Qwen3MoeConfig, checkpoint::SafetensorsIndex};

    fn test_config() -> Qwen3MoeConfig {
        Qwen3MoeConfig {
            model_type: "qwen3_moe".to_string(),
            hidden_size: 2,
            intermediate_size: 4,
            moe_intermediate_size: 1,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            num_experts: 1,
            num_experts_per_tok: 1,
            norm_topk_prob: true,
            vocab_size: 16,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
        }
    }

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

    #[test]
    fn generates_canonical_hugging_face_tensor_names() {
        let names = Qwen3ExpertTensorNames::canonical(12, 34);
        assert_eq!(
            names.gate_proj,
            "model.layers.12.mlp.experts.34.gate_proj.weight"
        );
        assert_eq!(
            names.up_proj,
            "model.layers.12.mlp.experts.34.up_proj.weight"
        );
        assert_eq!(
            names.down_proj,
            "model.layers.12.mlp.experts.34.down_proj.weight"
        );
    }

    #[tokio::test]
    async fn repacks_gate_up_down_and_emits_manifest_subranges() {
        let source_root = tempdir().expect("source root");
        let output_root = tempdir().expect("output root");
        let names = Qwen3ExpertTensorNames::canonical(0, 0);
        let header = format!(
            r#"{{"{}":{{"dtype":"F16","shape":[1,2],"data_offsets":[0,4]}},"{}":{{"dtype":"F16","shape":[1,2],"data_offsets":[4,8]}},"{}":{{"dtype":"F16","shape":[2,1],"data_offsets":[8,12]}}}}"#,
            names.gate_proj, names.up_proj, names.down_proj
        );
        let shard_name = "model-00001-of-00001.safetensors";
        write_shard(
            &source_root.path().join(shard_name),
            &header,
            b"GATEUP__DOWN",
        )
        .await;
        let index_json = format!(
            r#"{{"weight_map":{{"{}":"{shard_name}","{}":"{shard_name}","{}":"{shard_name}"}}}}"#,
            names.gate_proj, names.up_proj, names.down_proj
        );
        let index = SafetensorsIndex::from_json(&index_json).expect("index");
        let inventory = index
            .inventory(source_root.path())
            .await
            .expect("inventory");

        let manifest = prepare_qwen3_expert_bank(
            &inventory,
            &test_config(),
            output_root.path(),
            "Qwen/Qwen3-test",
        )
        .await
        .expect("prepare expert bank");

        let bank = tokio::fs::read(output_root.path().join("experts/expert-bank.bin"))
            .await
            .expect("read bank");
        assert_eq!(bank, b"GATEUP__DOWN");
        assert!(
            tokio::fs::try_exists(output_root.path().join("expert-manifest.json"))
                .await
                .expect("manifest exists")
        );
        assert_eq!(manifest.experts.len(), 1);
        let expert = &manifest.experts[0];
        assert_eq!(expert.offset, 0);
        assert_eq!(expert.length, 12);
        assert_eq!(expert.tensors.len(), 3);
        assert_eq!(expert.tensors[0].name, "gate_proj");
        assert_eq!(expert.tensors[0].offset, 0);
        assert_eq!(expert.tensors[1].name, "up_proj");
        assert_eq!(expert.tensors[1].offset, 4);
        assert_eq!(expert.tensors[2].name, "down_proj");
        assert_eq!(expert.tensors[2].offset, 8);
        assert!(manifest.files["expert-bank"].sha256.is_some());
    }

    #[tokio::test]
    async fn rejects_projection_shape_mismatch_and_cleans_partial_bank() {
        let source_root = tempdir().expect("source root");
        let output_root = tempdir().expect("output root");
        let names = Qwen3ExpertTensorNames::canonical(0, 0);
        let header = format!(
            r#"{{"{}":{{"dtype":"F16","shape":[2,2],"data_offsets":[0,8]}},"{}":{{"dtype":"F16","shape":[1,2],"data_offsets":[8,12]}},"{}":{{"dtype":"F16","shape":[2,1],"data_offsets":[12,16]}}}}"#,
            names.gate_proj, names.up_proj, names.down_proj
        );
        let shard_name = "model.safetensors";
        write_shard(&source_root.path().join(shard_name), &header, &[0_u8; 16]).await;
        let index_json = format!(
            r#"{{"weight_map":{{"{}":"{shard_name}","{}":"{shard_name}","{}":"{shard_name}"}}}}"#,
            names.gate_proj, names.up_proj, names.down_proj
        );
        let index = SafetensorsIndex::from_json(&index_json).expect("index");
        let inventory = index
            .inventory(source_root.path())
            .await
            .expect("inventory");

        let error = prepare_qwen3_expert_bank(
            &inventory,
            &test_config(),
            output_root.path(),
            "Qwen/Qwen3-test",
        )
        .await
        .expect_err("shape mismatch must fail");
        assert!(error.to_string().contains("gate_proj shape"));
        assert!(
            !tokio::fs::try_exists(output_root.path().join("experts/expert-bank.bin.partial"))
                .await
                .expect("partial status")
        );
    }
}
