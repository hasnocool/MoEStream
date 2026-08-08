// tests/qwen3_prepare_missing.rs

use std::path::Path;

use moestream::{
    adapters::{
        qwen3::Qwen3MoeConfig,
        qwen3_prepare::{Qwen3ExpertTensorNames, prepare_qwen3_expert_bank},
    },
    checkpoint::SafetensorsIndex,
};
use tempfile::tempdir;
use tokio::io::AsyncWriteExt;

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

#[tokio::test]
async fn rejects_missing_canonical_projection_and_removes_partial_output() {
    let source_root = tempdir().expect("source root");
    let output_root = tempdir().expect("output root");
    let names = Qwen3ExpertTensorNames::canonical(0, 0);
    let header = format!(
        r#"{{"{}":{{"dtype":"F16","shape":[1,2],"data_offsets":[0,4]}},"{}":{{"dtype":"F16","shape":[2,1],"data_offsets":[4,8]}}}}"#,
        names.gate_proj, names.down_proj
    );
    let shard_name = "model.safetensors";
    write_shard(&source_root.path().join(shard_name), &header, b"GATEDOWN").await;

    let index_json = format!(
        r#"{{"weight_map":{{"{}":"{shard_name}","{}":"{shard_name}"}}}}"#,
        names.gate_proj, names.down_proj
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
    .expect_err("missing up_proj must fail");

    assert!(error.to_string().contains("up_proj"));
    assert!(
        !tokio::fs::try_exists(output_root.path().join("experts/expert-bank.bin.partial"))
            .await
            .expect("partial status")
    );
}
