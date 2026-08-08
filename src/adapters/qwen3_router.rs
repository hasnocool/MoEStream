// src/adapters/qwen3_router.rs

use std::sync::Arc;

use anyhow::{Context, ensure};
use tokio::io::{AsyncReadExt, AsyncSeekExt};

use crate::{
    adapters::qwen3::{Qwen3MoeConfig, RouteSelection, reference_route},
    checkpoint::{CheckpointInventory, IndexedTensor},
};

#[derive(Debug, Clone)]
pub struct Qwen3RouterWeights {
    layer: usize,
    num_experts: usize,
    hidden_size: usize,
    weights: Arc<[f32]>,
}

impl Qwen3RouterWeights {
    pub async fn load(
        inventory: &CheckpointInventory,
        config: &Qwen3MoeConfig,
        layer: usize,
    ) -> anyhow::Result<Self> {
        config.validate()?;
        ensure!(
            layer < config.num_hidden_layers,
            "Qwen3 router layer {layer} is outside configured layer count {}",
            config.num_hidden_layers
        );

        let name = format!("model.layers.{layer}.mlp.gate.weight");
        let tensor = inventory
            .tensor(&name)
            .with_context(|| format!("missing Qwen3 router tensor {name:?}"))?;
        validate_router_tensor(config, layer, &tensor)?;
        let weights = read_router_weights(&tensor).await?;

        Ok(Self {
            layer,
            num_experts: config.num_experts,
            hidden_size: config.hidden_size,
            weights: weights.into(),
        })
    }

    pub fn layer(&self) -> usize {
        self.layer
    }

    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Correctness-oriented f32 matrix-vector reference for router bring-up.
    ///
    /// Production execution should eventually live in an execution provider so
    /// it can reproduce the reference runtime's dtype/accumulation semantics.
    pub fn logits(&self, hidden_state: &[f32]) -> anyhow::Result<Vec<f32>> {
        ensure!(
            hidden_state.len() == self.hidden_size,
            "router hidden state length {} does not match hidden_size {}",
            hidden_state.len(),
            self.hidden_size
        );
        ensure!(
            hidden_state.iter().all(|value| value.is_finite()),
            "router hidden state must contain only finite values"
        );

        let mut logits = vec![0.0_f32; self.num_experts];
        for (expert, output) in logits.iter_mut().enumerate() {
            let row_start = expert
                .checked_mul(self.hidden_size)
                .context("router row offset overflow")?;
            let row_end = row_start
                .checked_add(self.hidden_size)
                .context("router row end overflow")?;
            let row = &self.weights[row_start..row_end];
            *output = row
                .iter()
                .zip(hidden_state)
                .map(|(weight, hidden)| weight * hidden)
                .sum();
        }
        Ok(logits)
    }

    /// Execute router matrix-vector work on Tokio's blocking pool, then apply
    /// the already-defined softmax/top-k reference selection.
    pub async fn route(
        &self,
        hidden_state: &[f32],
        experts_per_token: usize,
        norm_topk_prob: bool,
    ) -> anyhow::Result<Vec<RouteSelection>> {
        ensure!(
            hidden_state.len() == self.hidden_size,
            "router hidden state length {} does not match hidden_size {}",
            hidden_state.len(),
            self.hidden_size
        );
        let router = self.clone();
        let hidden_state = hidden_state.to_vec();
        tokio::task::spawn_blocking(move || {
            let logits = router.logits(&hidden_state)?;
            reference_route(&logits, experts_per_token, norm_topk_prob)
        })
        .await
        .context("join Qwen3 router compute task")?
    }
}

fn validate_router_tensor(
    config: &Qwen3MoeConfig,
    layer: usize,
    tensor: &IndexedTensor,
) -> anyhow::Result<()> {
    let expected_shape = [config.num_experts as u64, config.hidden_size as u64];
    ensure!(
        tensor.metadata.shape.as_slice() == expected_shape,
        "Qwen3 layer {layer} router shape {:?} does not match {:?}",
        tensor.metadata.shape,
        expected_shape
    );
    ensure!(
        matches!(tensor.metadata.dtype.as_str(), "BF16" | "F32"),
        "unsupported Qwen3 router dtype {:?}; expected BF16 or F32",
        tensor.metadata.dtype
    );

    let element_bytes = match tensor.metadata.dtype.as_str() {
        "BF16" => 2_u64,
        "F32" => 4_u64,
        _ => unreachable!("dtype validated above"),
    };
    let expected_elements = (config.num_experts as u64)
        .checked_mul(config.hidden_size as u64)
        .context("router element count overflow")?;
    let expected_bytes = expected_elements
        .checked_mul(element_bytes)
        .context("router byte length overflow")?;
    ensure!(
        tensor.span.length == expected_bytes,
        "Qwen3 layer {layer} router byte length {} does not match expected {expected_bytes}",
        tensor.span.length
    );
    Ok(())
}

async fn read_router_weights(tensor: &IndexedTensor) -> anyhow::Result<Vec<f32>> {
    let byte_len = usize::try_from(tensor.span.length).context("router tensor exceeds usize")?;
    let mut bytes = vec![0_u8; byte_len];
    let mut file = tokio::fs::File::open(&tensor.shard)
        .await
        .with_context(|| format!("open Qwen3 router shard {}", tensor.shard.display()))?;
    file.seek(std::io::SeekFrom::Start(tensor.span.offset))
        .await
        .with_context(|| format!("seek Qwen3 router shard {}", tensor.shard.display()))?;
    file.read_exact(&mut bytes)
        .await
        .with_context(|| format!("read Qwen3 router tensor from {}", tensor.shard.display()))?;

    match tensor.metadata.dtype.as_str() {
        "BF16" => decode_bf16(&bytes),
        "F32" => decode_f32(&bytes),
        dtype => anyhow::bail!("unsupported Qwen3 router dtype {dtype:?}"),
    }
}

fn decode_bf16(bytes: &[u8]) -> anyhow::Result<Vec<f32>> {
    ensure!(bytes.len() % 2 == 0, "BF16 byte length must be divisible by 2");
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            f32::from_bits(u32::from(bits) << 16)
        })
        .collect())
}

fn decode_f32(bytes: &[u8]) -> anyhow::Result<Vec<f32>> {
    ensure!(bytes.len() % 4 == 0, "F32 byte length must be divisible by 4");
    let values = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect::<Vec<_>>();
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "router tensor contains non-finite F32 values"
    );
    Ok(values)
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;
    use tokio::io::AsyncWriteExt;

    use super::Qwen3RouterWeights;
    use crate::{adapters::qwen3::Qwen3MoeConfig, checkpoint::SafetensorsIndex};

    fn config() -> Qwen3MoeConfig {
        Qwen3MoeConfig {
            model_type: "qwen3_moe".to_string(),
            hidden_size: 2,
            intermediate_size: 4,
            moe_intermediate_size: 2,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            num_experts: 3,
            num_experts_per_tok: 2,
            norm_topk_prob: true,
            vocab_size: 16,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
        }
    }

    async fn write_fixture(root: &std::path::Path, dtype: &str, values: &[f32]) {
        let payload = match dtype {
            "F32" => values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
            "BF16" => values
                .iter()
                .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
                .collect::<Vec<_>>(),
            _ => panic!("unsupported test dtype"),
        };
        let name = "model.layers.0.mlp.gate.weight";
        let header = format!(
            "{{\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[3,2],\"data_offsets\":[0,{}]}}}}",
            payload.len()
        );
        let shard = root.join("model.safetensors");
        let mut file = tokio::fs::File::create(&shard).await.expect("create shard");
        file.write_all(&(header.len() as u64).to_le_bytes())
            .await
            .expect("write header length");
        file.write_all(header.as_bytes()).await.expect("write header");
        file.write_all(&payload).await.expect("write payload");
        file.flush().await.expect("flush shard");
        let index = serde_json::json!({"weight_map": {name: "model.safetensors"}});
        tokio::fs::write(
            root.join("model.safetensors.index.json"),
            serde_json::to_vec(&index).expect("serialize index"),
        )
        .await
        .expect("write index");
    }

    async fn load_router(root: &std::path::Path) -> Qwen3RouterWeights {
        let index = SafetensorsIndex::load(&root.join("model.safetensors.index.json"))
            .await
            .expect("load index");
        let inventory = index.inventory(root).await.expect("inventory");
        Qwen3RouterWeights::load(&inventory, &config(), 0)
            .await
            .expect("router")
    }

    #[tokio::test]
    async fn loads_f32_router_and_computes_logits() {
        let root = tempdir().expect("root");
        write_fixture(root.path(), "F32", &[1.0, 0.0, 0.0, 2.0, -1.0, -1.0]).await;
        let router = load_router(root.path()).await;

        let logits = router.logits(&[3.0, 4.0]).expect("logits");
        assert_eq!(logits, vec![3.0, 8.0, -7.0]);
    }

    #[tokio::test]
    async fn decodes_bf16_router_values() {
        let root = tempdir().expect("root");
        write_fixture(root.path(), "BF16", &[1.0, 0.0, 0.0, 2.0, -1.0, -1.0]).await;
        let router = load_router(root.path()).await;

        assert_eq!(router.weights(), &[1.0, 0.0, 0.0, 2.0, -1.0, -1.0]);
    }

    #[tokio::test]
    async fn routes_off_tokio_worker_threads() {
        let root = tempdir().expect("root");
        write_fixture(root.path(), "F32", &[1.0, 0.0, 0.0, 2.0, -1.0, -1.0]).await;
        let router = load_router(root.path()).await;

        let route = router.route(&[3.0, 4.0], 2, true).await.expect("route");
        assert_eq!(route.len(), 2);
        assert_eq!(route[0].expert, 1);
        assert_eq!(route[1].expert, 0);
        let sum: f32 = route.iter().map(|selection| selection.weight).sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
}
