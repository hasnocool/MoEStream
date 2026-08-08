// src/adapters/qwen3_integrity.rs

use std::{fs::File, io::Read, path::Path};

use anyhow::{Context, ensure};
use sha2::{Digest, Sha256};

use crate::{
    adapters::{
        qwen3::Qwen3MoeConfig,
        qwen3_checkpoint::{PackedExpertBank, pack_expert_bank},
    },
    checkpoint::CheckpointInventory,
};

const HASH_BUFFER_BYTES: usize = 1024 * 1024;
const BANK_FILE_NAME: &str = "experts.bin";
const MANIFEST_FILE_NAME: &str = "experts.manifest.json";
const BANK_TEMP_FILE_NAME: &str = "experts.bin.tmp";
const MANIFEST_TEMP_FILE_NAME: &str = "experts.manifest.json.tmp";
const VERIFY_MANIFEST_TEMP_FILE_NAME: &str = "experts.manifest.json.verify.tmp";

/// Pack a Qwen3 expert bank and finalize it with a verified SHA-256 digest.
///
/// The underlying checkpoint conversion remains bounded asynchronous I/O.
/// Full-file hashing uses `spawn_blocking`, keeping synchronous reads and CPU
/// hashing off Tokio runtime worker threads. Any failed conversion/finalization
/// removes temporary output; a failed integrity finalization also removes the
/// just-published bank and manifest so callers never inherit an ambiguous pair.
pub async fn pack_expert_bank_verified(
    inventory: &CheckpointInventory,
    config: &Qwen3MoeConfig,
    model_id: &str,
    output_root: &Path,
) -> anyhow::Result<PackedExpertBank> {
    cleanup_temporary_outputs(output_root).await?;

    let packed = match pack_expert_bank(inventory, config, model_id, output_root).await {
        Ok(packed) => packed,
        Err(error) => {
            let _ = cleanup_temporary_outputs(output_root).await;
            return Err(error);
        }
    };

    match finalize_integrity(packed, output_root).await {
        Ok(packed) => Ok(packed),
        Err(error) => {
            let _ = cleanup_temporary_outputs(output_root).await;
            let _ = remove_if_exists(&output_root.join(BANK_FILE_NAME)).await;
            let _ = remove_if_exists(&output_root.join(MANIFEST_FILE_NAME)).await;
            Err(error)
        }
    }
}

async fn finalize_integrity(
    mut packed: PackedExpertBank,
    output_root: &Path,
) -> anyhow::Result<PackedExpertBank> {
    let actual_size = tokio::fs::metadata(&packed.bank_path)
        .await
        .with_context(|| format!("stat packed bank {}", packed.bank_path.display()))?
        .len();

    let file = packed
        .manifest
        .files
        .get_mut("experts")
        .context("generated Qwen3 manifest is missing experts file entry")?;
    ensure!(
        file.size == actual_size,
        "packed expert bank size mismatch: manifest={}, file={actual_size}",
        file.size
    );

    let digest = sha256_file(&packed.bank_path).await?;
    file.sha256 = Some(digest);
    packed.manifest.validate()?;

    let manifest_tmp = output_root.join(VERIFY_MANIFEST_TEMP_FILE_NAME);
    let json = serde_json::to_vec_pretty(&packed.manifest)
        .context("serialize integrity-finalized expert manifest")?;
    tokio::fs::write(&manifest_tmp, json)
        .await
        .with_context(|| format!("write finalized manifest {}", manifest_tmp.display()))?;
    tokio::fs::rename(&manifest_tmp, &packed.manifest_path)
        .await
        .with_context(|| {
            format!(
                "publish finalized manifest {}",
                packed.manifest_path.display()
            )
        })?;

    Ok(packed)
}

async fn sha256_file(path: &Path) -> anyhow::Result<String> {
    let path = path.to_path_buf();
    tokio::task::spawn_blocking(move || hash_file_sha256(&path))
        .await
        .context("join expert-bank SHA-256 task")?
}

fn hash_file_sha256(path: &Path) -> anyhow::Result<String> {
    let mut file =
        File::open(path).with_context(|| format!("open packed expert bank {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; HASH_BUFFER_BYTES];

    loop {
        let count = file
            .read(&mut buffer)
            .with_context(|| format!("hash packed expert bank {}", path.display()))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }

    Ok(format!("{:x}", hasher.finalize()))
}

async fn cleanup_temporary_outputs(output_root: &Path) -> anyhow::Result<()> {
    for name in [
        BANK_TEMP_FILE_NAME,
        MANIFEST_TEMP_FILE_NAME,
        VERIFY_MANIFEST_TEMP_FILE_NAME,
    ] {
        remove_if_exists(&output_root.join(name)).await?;
    }
    Ok(())
}

async fn remove_if_exists(path: &Path) -> anyhow::Result<()> {
    match tokio::fs::remove_file(path).await {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).with_context(|| format!("remove {}", path.display())),
    }
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;
    use tokio::io::AsyncWriteExt;

    use super::pack_expert_bank_verified;
    use crate::{adapters::qwen3::Qwen3MoeConfig, checkpoint::SafetensorsIndex};

    fn config() -> Qwen3MoeConfig {
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

    async fn inventory(root: &std::path::Path) -> crate::checkpoint::CheckpointInventory {
        let names = ["gate_proj", "up_proj", "down_proj"];
        let mut entries = Vec::new();
        let mut weight_map = serde_json::Map::new();
        let mut offset = 0_u64;
        let mut payload = Vec::new();

        for (index, projection) in names.into_iter().enumerate() {
            let shape = if projection == "down_proj" {
                [2_u64, 1_u64]
            } else {
                [1_u64, 2_u64]
            };
            let length = shape[0] * shape[1] * 2;
            let begin = offset;
            offset += length;
            let name = format!("model.layers.0.mlp.experts.0.{projection}.weight");
            entries.push(format!(
                "\"{name}\":{{\"dtype\":\"F16\",\"shape\":[{},{}],\"data_offsets\":[{begin},{offset}]}}",
                shape[0], shape[1]
            ));
            weight_map.insert(
                name,
                serde_json::Value::String("model.safetensors".to_string()),
            );
            payload.resize(offset as usize, (index + 1) as u8);
        }

        let header = format!("{{{}}}", entries.join(","));
        let mut shard = tokio::fs::File::create(root.join("model.safetensors"))
            .await
            .expect("create shard");
        shard
            .write_all(&(header.len() as u64).to_le_bytes())
            .await
            .expect("write header length");
        shard
            .write_all(header.as_bytes())
            .await
            .expect("write header");
        shard.write_all(&payload).await.expect("write payload");
        shard.flush().await.expect("flush shard");

        let index =
            SafetensorsIndex::from_json(&serde_json::json!({"weight_map": weight_map}).to_string())
                .expect("index");
        index.inventory(root).await.expect("inventory")
    }

    #[tokio::test]
    async fn finalized_manifest_contains_verified_bank_digest() {
        let source = tempdir().expect("source");
        let output = tempdir().expect("output");
        let inventory = inventory(source.path()).await;

        let packed =
            pack_expert_bank_verified(&inventory, &config(), "Qwen/Qwen3-test", output.path())
                .await
                .expect("verified pack");

        let digest = packed.manifest.files["experts"]
            .sha256
            .as_deref()
            .expect("digest");
        assert_eq!(digest.len(), 64);
        assert!(digest.bytes().all(|byte| byte.is_ascii_hexdigit()));

        let manifest_json = tokio::fs::read_to_string(&packed.manifest_path)
            .await
            .expect("manifest");
        let on_disk =
            crate::manifest::ExpertManifest::from_json(&manifest_json).expect("manifest parses");
        assert_eq!(on_disk.files["experts"].sha256.as_deref(), Some(digest));
        assert_eq!(
            on_disk
                .build_index(output.path())
                .expect("index")
                .verify_declared_hashes()
                .await
                .expect("hash verification"),
            1
        );
    }

    #[tokio::test]
    async fn stale_temporary_files_are_removed_before_pack() {
        let source = tempdir().expect("source");
        let output = tempdir().expect("output");
        let inventory = inventory(source.path()).await;
        tokio::fs::write(output.path().join("experts.bin.tmp"), b"stale")
            .await
            .expect("stale bank temp");
        tokio::fs::write(output.path().join("experts.manifest.json.tmp"), b"stale")
            .await
            .expect("stale manifest temp");

        pack_expert_bank_verified(&inventory, &config(), "Qwen/Qwen3-test", output.path())
            .await
            .expect("verified pack");

        assert!(
            !tokio::fs::try_exists(output.path().join("experts.bin.tmp"))
                .await
                .expect("temp status")
        );
        assert!(
            !tokio::fs::try_exists(output.path().join("experts.manifest.json.tmp"))
                .await
                .expect("temp status")
        );
    }
}
