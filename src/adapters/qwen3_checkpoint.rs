// src/adapters/qwen3_checkpoint.rs

use std::{collections::HashMap, path::Path};

use anyhow::{Context, ensure};
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};

use crate::{
    adapters::qwen3::Qwen3MoeConfig,
    checkpoint::{CheckpointInventory, IndexedTensor},
    manifest::{ExpertManifest, MANIFEST_SCHEMA_VERSION, ManifestExpert, ManifestFile},
    safetensors::TensorSpan,
};

const COPY_BUFFER_BYTES: usize = 1024 * 1024;
const BANK_FILE_NAME: &str = "experts.bin";
const MANIFEST_FILE_NAME: &str = "experts.manifest.json";
const BANK_FILE_KEY: &str = "experts";

#[derive(Debug, Clone)]
pub struct Qwen3ExpertSource {
    pub layer: usize,
    pub expert: usize,
    pub gate_proj: IndexedTensor,
    pub up_proj: IndexedTensor,
    pub down_proj: IndexedTensor,
}

impl Qwen3ExpertSource {
    pub fn packed_length(&self) -> anyhow::Result<u64> {
        self.gate_proj
            .span
            .length
            .checked_add(self.up_proj.span.length)
            .and_then(|value| value.checked_add(self.down_proj.span.length))
            .context("Qwen3 packed expert length overflow")
    }
}

#[derive(Debug, Clone)]
pub struct PackedExpertBank {
    pub bank_path: std::path::PathBuf,
    pub manifest_path: std::path::PathBuf,
    pub manifest: ExpertManifest,
}

pub fn discover_expert_sources(
    inventory: &CheckpointInventory,
    config: &Qwen3MoeConfig,
) -> anyhow::Result<Vec<Qwen3ExpertSource>> {
    config.validate()?;

    let expected_count = config
        .num_hidden_layers
        .checked_mul(config.num_experts)
        .context("Qwen3 expert count overflow")?;
    let mut experts = Vec::with_capacity(expected_count);

    for layer in 0..config.num_hidden_layers {
        for expert in 0..config.num_experts {
            let prefix = format!("model.layers.{layer}.mlp.experts.{expert}");
            let gate_proj = inventory
                .tensor(&format!("{prefix}.gate_proj.weight"))
                .with_context(|| format!("missing Qwen3 layer {layer} expert {expert} gate_proj"))?;
            let up_proj = inventory
                .tensor(&format!("{prefix}.up_proj.weight"))
                .with_context(|| format!("missing Qwen3 layer {layer} expert {expert} up_proj"))?;
            let down_proj = inventory
                .tensor(&format!("{prefix}.down_proj.weight"))
                .with_context(|| format!("missing Qwen3 layer {layer} expert {expert} down_proj"))?;

            validate_expert_tensor_shapes(config, layer, expert, &gate_proj, &up_proj, &down_proj)?;
            experts.push(Qwen3ExpertSource {
                layer,
                expert,
                gate_proj,
                up_proj,
                down_proj,
            });
        }
    }

    Ok(experts)
}

/// Repack Qwen3 expert projection tensors into one contiguous expert bank.
///
/// The output order is deterministic: layer-major, expert-major, then
/// `gate_proj`, `up_proj`, `down_proj`. Source tensor payloads are copied with
/// bounded asynchronous range reads and writes; full shards or experts are
/// never materialized in memory.
pub async fn pack_expert_bank(
    inventory: &CheckpointInventory,
    config: &Qwen3MoeConfig,
    model_id: &str,
    output_root: &Path,
) -> anyhow::Result<PackedExpertBank> {
    ensure!(!model_id.trim().is_empty(), "model_id cannot be empty");
    let experts = discover_expert_sources(inventory, config)?;
    tokio::fs::create_dir_all(output_root)
        .await
        .with_context(|| format!("create expert-bank directory {}", output_root.display()))?;

    let bank_path = output_root.join(BANK_FILE_NAME);
    let bank_tmp = output_root.join(format!("{BANK_FILE_NAME}.tmp"));
    let manifest_path = output_root.join(MANIFEST_FILE_NAME);
    let manifest_tmp = output_root.join(format!("{MANIFEST_FILE_NAME}.tmp"));

    let mut output = tokio::fs::File::create(&bank_tmp)
        .await
        .with_context(|| format!("create temporary expert bank {}", bank_tmp.display()))?;
    let mut manifest_experts = Vec::with_capacity(experts.len());
    let mut bank_offset = 0_u64;

    for source in &experts {
        let expert_offset = bank_offset;
        for tensor in [&source.gate_proj, &source.up_proj, &source.down_proj] {
            copy_tensor_span(&tensor.shard, tensor.span, &mut output).await?;
            bank_offset = bank_offset
                .checked_add(tensor.span.length)
                .context("expert bank size overflow")?;
        }
        let expert_length = bank_offset
            .checked_sub(expert_offset)
            .context("expert bank offset underflow")?;
        ensure!(
            expert_length == source.packed_length()?,
            "packed expert length mismatch for layer {} expert {}",
            source.layer,
            source.expert
        );
        manifest_experts.push(ManifestExpert {
            layer: source.layer,
            expert: source.expert,
            file: BANK_FILE_KEY.to_string(),
            offset: expert_offset,
            length: expert_length,
        });
    }

    output.flush().await.context("flush expert bank")?;
    output.sync_all().await.context("sync expert bank")?;
    drop(output);

    let manifest = ExpertManifest {
        schema_version: MANIFEST_SCHEMA_VERSION,
        model: model_id.to_string(),
        files: HashMap::from([(
            BANK_FILE_KEY.to_string(),
            ManifestFile {
                path: BANK_FILE_NAME.into(),
                size: bank_offset,
                sha256: None,
            },
        )]),
        experts: manifest_experts,
    };
    manifest.validate()?;

    let manifest_json = serde_json::to_vec_pretty(&manifest).context("serialize expert manifest")?;
    tokio::fs::write(&manifest_tmp, manifest_json)
        .await
        .with_context(|| format!("write temporary manifest {}", manifest_tmp.display()))?;

    tokio::fs::rename(&bank_tmp, &bank_path)
        .await
        .with_context(|| format!("publish expert bank {}", bank_path.display()))?;
    tokio::fs::rename(&manifest_tmp, &manifest_path)
        .await
        .with_context(|| format!("publish expert manifest {}", manifest_path.display()))?;

    Ok(PackedExpertBank {
        bank_path,
        manifest_path,
        manifest,
    })
}

async fn copy_tensor_span(
    source_path: &Path,
    span: TensorSpan,
    output: &mut tokio::fs::File,
) -> anyhow::Result<()> {
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
            .context("tensor copy chunk length exceeds usize")?;
        source
            .read_exact(&mut buffer[..chunk])
            .await
            .with_context(|| format!("read tensor source {}", source_path.display()))?;
        output
            .write_all(&buffer[..chunk])
            .await
            .context("write packed expert tensor")?;
        remaining -= chunk as u64;
    }
    Ok(())
}

fn validate_expert_tensor_shapes(
    config: &Qwen3MoeConfig,
    layer: usize,
    expert: usize,
    gate_proj: &IndexedTensor,
    up_proj: &IndexedTensor,
    down_proj: &IndexedTensor,
) -> anyhow::Result<()> {
    ensure!(
        gate_proj.metadata.dtype == up_proj.metadata.dtype
            && gate_proj.metadata.dtype == down_proj.metadata.dtype,
        "Qwen3 layer {layer} expert {expert} projection dtypes differ: gate={}, up={}, down={}",
        gate_proj.metadata.dtype,
        up_proj.metadata.dtype,
        down_proj.metadata.dtype
    );

    let gate_shape = [config.moe_intermediate_size as u64, config.hidden_size as u64];
    let down_shape = [config.hidden_size as u64, config.moe_intermediate_size as u64];
    ensure!(
        gate_proj.metadata.shape.as_slice() == gate_shape,
        "Qwen3 layer {layer} expert {expert} gate_proj shape {:?} does not match {:?}",
        gate_proj.metadata.shape,
        gate_shape
    );
    ensure!(
        up_proj.metadata.shape.as_slice() == gate_shape,
        "Qwen3 layer {layer} expert {expert} up_proj shape {:?} does not match {:?}",
        up_proj.metadata.shape,
        gate_shape
    );
    ensure!(
        down_proj.metadata.shape.as_slice() == down_shape,
        "Qwen3 layer {layer} expert {expert} down_proj shape {:?} does not match {:?}",
        down_proj.metadata.shape,
        down_shape
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;
    use tokio::io::AsyncWriteExt;

    use super::{discover_expert_sources, pack_expert_bank};
    use crate::{adapters::qwen3::Qwen3MoeConfig, checkpoint::SafetensorsIndex};

    fn config() -> Qwen3MoeConfig {
        Qwen3MoeConfig {
            model_type: "qwen3_moe".to_string(),
            hidden_size: 4,
            intermediate_size: 8,
            moe_intermediate_size: 2,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            num_experts: 2,
            num_experts_per_tok: 1,
            norm_topk_prob: true,
            vocab_size: 32,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
        }
    }

    async fn write_fixture(root: &std::path::Path, wrong_down_shape: bool) {
        let mut entries = Vec::new();
        let mut offset = 0_u64;
        let mut payload = Vec::new();
        for expert in 0..2 {
            for (projection_index, projection) in
                ["gate_proj", "up_proj", "down_proj"].into_iter().enumerate()
            {
                let shape = if projection == "down_proj" {
                    if wrong_down_shape && expert == 1 {
                        [3_u64, 2_u64]
                    } else {
                        [4_u64, 2_u64]
                    }
                } else {
                    [2_u64, 4_u64]
                };
                let bytes = shape[0] * shape[1] * 2;
                let begin = offset;
                offset += bytes;
                let name = format!(
                    "model.layers.0.mlp.experts.{expert}.{projection}.weight"
                );
                entries.push(format!(
                    "\"{name}\":{{\"dtype\":\"F16\",\"shape\":[{},{}],\"data_offsets\":[{begin},{offset}]}}",
                    shape[0], shape[1]
                ));
                let value = (expert * 3 + projection_index + 1) as u8;
                payload.resize(offset as usize, value);
            }
        }
        let header = format!("{{{}}}", entries.join(","));
        let shard = root.join("model.safetensors");
        let mut file = tokio::fs::File::create(&shard).await.expect("create shard");
        file.write_all(&(header.len() as u64).to_le_bytes())
            .await
            .expect("header length");
        file.write_all(header.as_bytes()).await.expect("header");
        file.write_all(&payload).await.expect("payload");
        file.flush().await.expect("flush");

        let mut weight_map = serde_json::Map::new();
        for expert in 0..2 {
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                let name = format!(
                    "model.layers.0.mlp.experts.{expert}.{projection}.weight"
                );
                weight_map.insert(
                    name,
                    serde_json::Value::String("model.safetensors".to_string()),
                );
            }
        }
        let index = serde_json::json!({"weight_map": weight_map});
        tokio::fs::write(
            root.join("model.safetensors.index.json"),
            serde_json::to_vec(&index).expect("serialize index"),
        )
        .await
        .expect("write index");
    }

    async fn inventory(root: &std::path::Path) -> crate::checkpoint::CheckpointInventory {
        let index = SafetensorsIndex::load(&root.join("model.safetensors.index.json"))
            .await
            .expect("index");
        index.inventory(root).await.expect("inventory")
    }

    #[tokio::test]
    async fn discovers_complete_qwen3_expert_triplets() {
        let root = tempdir().expect("checkpoint root");
        write_fixture(root.path(), false).await;
        let inventory = inventory(root.path()).await;

        let experts = discover_expert_sources(&inventory, &config()).expect("expert sources");
        assert_eq!(experts.len(), 2);
        assert_eq!(experts[0].layer, 0);
        assert_eq!(experts[0].expert, 0);
        assert_eq!(experts[0].packed_length().expect("packed length"), 48);
    }

    #[tokio::test]
    async fn packs_contiguous_expert_bank_and_manifest() {
        let root = tempdir().expect("checkpoint root");
        let output = tempdir().expect("output root");
        write_fixture(root.path(), false).await;
        let inventory = inventory(root.path()).await;

        let packed = pack_expert_bank(
            &inventory,
            &config(),
            "Qwen/Qwen3-30B-A3B",
            output.path(),
        )
        .await
        .expect("pack expert bank");

        assert_eq!(packed.manifest.experts.len(), 2);
        assert_eq!(packed.manifest.experts[0].offset, 0);
        assert_eq!(packed.manifest.experts[0].length, 48);
        assert_eq!(packed.manifest.experts[1].offset, 48);
        assert_eq!(packed.manifest.experts[1].length, 48);
        assert_eq!(tokio::fs::metadata(&packed.bank_path).await.expect("bank metadata").len(), 96);

        let bytes = tokio::fs::read(&packed.bank_path).await.expect("packed bank");
        assert!(bytes[..16].iter().all(|byte| *byte == 1));
        assert!(bytes[16..32].iter().all(|byte| *byte == 2));
        assert!(bytes[32..48].iter().all(|byte| *byte == 3));
        assert!(bytes[48..64].iter().all(|byte| *byte == 4));
        assert!(bytes[64..80].iter().all(|byte| *byte == 5));
        assert!(bytes[80..96].iter().all(|byte| *byte == 6));

        let manifest_json = tokio::fs::read_to_string(&packed.manifest_path)
            .await
            .expect("manifest file");
        let parsed = crate::manifest::ExpertManifest::from_json(&manifest_json)
            .expect("generated manifest validates");
        assert_eq!(parsed.files["experts"].size, 96);
    }

    #[tokio::test]
    async fn rejects_qwen3_expert_shape_mismatch() {
        let root = tempdir().expect("checkpoint root");
        write_fixture(root.path(), true).await;
        let inventory = inventory(root.path()).await;

        let error = discover_expert_sources(&inventory, &config())
            .expect_err("wrong expert shape must fail");
        assert!(error.to_string().contains("down_proj shape"));
    }
}
