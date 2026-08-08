// src/adapters/qwen3_checkpoint.rs

use anyhow::{Context, ensure};

use crate::{
    adapters::qwen3::Qwen3MoeConfig,
    checkpoint::{CheckpointInventory, IndexedTensor},
};

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

    use super::discover_expert_sources;
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
            for projection in ["gate_proj", "up_proj", "down_proj"] {
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
                payload.resize(offset as usize, 0_u8);
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
                weight_map.insert(name, serde_json::Value::String("model.safetensors".to_string()));
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

    #[tokio::test]
    async fn discovers_complete_qwen3_expert_triplets() {
        let root = tempdir().expect("checkpoint root");
        write_fixture(root.path(), false).await;
        let index = SafetensorsIndex::load(&root.path().join("model.safetensors.index.json"))
            .await
            .expect("index");
        let inventory = index.inventory(root.path()).await.expect("inventory");

        let experts = discover_expert_sources(&inventory, &config()).expect("expert sources");
        assert_eq!(experts.len(), 2);
        assert_eq!(experts[0].layer, 0);
        assert_eq!(experts[0].expert, 0);
        assert_eq!(experts[0].packed_length().expect("packed length"), 48);
    }

    #[tokio::test]
    async fn rejects_qwen3_expert_shape_mismatch() {
        let root = tempdir().expect("checkpoint root");
        write_fixture(root.path(), true).await;
        let index = SafetensorsIndex::load(&root.path().join("model.safetensors.index.json"))
            .await
            .expect("index");
        let inventory = index.inventory(root.path()).await.expect("inventory");

        let error = discover_expert_sources(&inventory, &config())
            .expect_err("wrong expert shape must fail");
        assert!(error.to_string().contains("down_proj shape"));
    }
}
