// tests/qwen3_router_parity.rs

use serde::Deserialize;
use tempfile::tempdir;
use tokio::io::AsyncWriteExt;

use moestream::{
    adapters::{qwen3::Qwen3MoeConfig, qwen3_router::Qwen3RouterWeights},
    checkpoint::SafetensorsIndex,
};

const FIXTURE_JSON: &str = include_str!("fixtures/qwen3_router_pytorch_2_10_cpu.json");

#[derive(Debug, Deserialize)]
struct RouterFixture {
    schema_version: u32,
    pytorch_version: String,
    device: String,
    dtype: String,
    shape: [usize; 2],
    weights_bf16_bits: Vec<u16>,
    hidden_bf16_bits: Vec<u16>,
    router_logits_f32: Vec<f32>,
    topk: usize,
    topk_indices: Vec<usize>,
    topk_weights_pre_norm: Vec<f32>,
    topk_weights_norm: Vec<f32>,
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn config(fixture: &RouterFixture) -> Qwen3MoeConfig {
    Qwen3MoeConfig {
        model_type: "qwen3_moe".to_string(),
        hidden_size: fixture.shape[1],
        intermediate_size: fixture.shape[1] * 2,
        moe_intermediate_size: fixture.shape[1],
        num_hidden_layers: 1,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        num_experts: fixture.shape[0],
        num_experts_per_tok: fixture.topk,
        norm_topk_prob: true,
        vocab_size: 32,
        rms_norm_eps: 1e-6,
        rope_theta: 1_000_000.0,
    }
}

async fn load_fixture_router(
    fixture: &RouterFixture,
) -> (tempfile::TempDir, Qwen3RouterWeights) {
    let root = tempdir().expect("temporary checkpoint root");
    let tensor_name = "model.layers.0.mlp.gate.weight";
    let payload = fixture
        .weights_bf16_bits
        .iter()
        .flat_map(|bits| bits.to_le_bytes())
        .collect::<Vec<_>>();
    let header = format!(
        "{{\"{tensor_name}\":{{\"dtype\":\"BF16\",\"shape\":[{},{}],\"data_offsets\":[0,{}]}}}}",
        fixture.shape[0],
        fixture.shape[1],
        payload.len()
    );

    let shard_path = root.path().join("model.safetensors");
    let mut shard = tokio::fs::File::create(&shard_path)
        .await
        .expect("create shard");
    shard
        .write_all(&(header.len() as u64).to_le_bytes())
        .await
        .expect("write safetensors header length");
    shard
        .write_all(header.as_bytes())
        .await
        .expect("write safetensors header");
    shard.write_all(&payload).await.expect("write router payload");
    shard.flush().await.expect("flush shard");

    let index_json = serde_json::json!({
        "weight_map": {tensor_name: "model.safetensors"}
    });
    let index_path = root.path().join("model.safetensors.index.json");
    tokio::fs::write(
        &index_path,
        serde_json::to_vec(&index_json).expect("serialize index"),
    )
    .await
    .expect("write index");

    let index = SafetensorsIndex::load(&index_path)
        .await
        .expect("load index");
    let inventory = index.inventory(root.path()).await.expect("inventory");
    let router = Qwen3RouterWeights::load(&inventory, &config(fixture), 0)
        .await
        .expect("load router");
    (root, router)
}

fn assert_close(actual: f32, expected: f32, tolerance: f32, context: &str) {
    let difference = (actual - expected).abs();
    assert!(
        difference <= tolerance,
        "{context}: actual={actual:?}, expected={expected:?}, difference={difference:?}, tolerance={tolerance:?}"
    );
}

#[tokio::test]
async fn matches_recorded_pytorch_bf16_router_fixture() {
    let fixture: RouterFixture = serde_json::from_str(FIXTURE_JSON).expect("parse fixture");
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.device, "cpu");
    assert_eq!(fixture.dtype, "bfloat16");
    assert!(fixture.pytorch_version.starts_with("2.10."));

    let (_root, router) = load_fixture_router(&fixture).await;
    let hidden = fixture
        .hidden_bf16_bits
        .iter()
        .copied()
        .map(bf16_to_f32)
        .collect::<Vec<_>>();

    let logits = router.logits(&hidden).expect("router logits");
    assert_eq!(logits.len(), fixture.router_logits_f32.len());
    for (index, (actual, expected)) in logits
        .iter()
        .zip(&fixture.router_logits_f32)
        .enumerate()
    {
        assert_close(*actual, *expected, 0.0, &format!("router logit {index}"));
    }

    let pre_norm = router
        .route(&hidden, fixture.topk, false)
        .await
        .expect("pre-normalized route");
    let normalized = router
        .route(&hidden, fixture.topk, true)
        .await
        .expect("normalized route");

    assert_eq!(
        pre_norm.iter().map(|item| item.expert).collect::<Vec<_>>(),
        fixture.topk_indices
    );
    assert_eq!(
        normalized
            .iter()
            .map(|item| item.expert)
            .collect::<Vec<_>>(),
        fixture.topk_indices
    );

    for (index, (actual, expected)) in pre_norm
        .iter()
        .map(|item| item.weight)
        .zip(&fixture.topk_weights_pre_norm)
        .enumerate()
    {
        assert_close(actual, *expected, 1e-6, &format!("pre-norm top-k weight {index}"));
    }
    for (index, (actual, expected)) in normalized
        .iter()
        .map(|item| item.weight)
        .zip(&fixture.topk_weights_norm)
        .enumerate()
    {
        assert_close(actual, *expected, 1e-6, &format!("normalized top-k weight {index}"));
    }
}
