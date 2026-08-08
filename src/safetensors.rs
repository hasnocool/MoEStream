// src/safetensors.rs

use std::{collections::HashMap, path::Path};

use anyhow::{Context, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::io::AsyncReadExt;

const PREFIX_BYTES: u64 = 8;
pub const MAX_HEADER_BYTES: u64 = 128 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TensorMetadata {
    pub dtype: String,
    pub shape: Vec<u64>,
    pub data_offsets: [u64; 2],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorSpan {
    pub offset: u64,
    pub length: u64,
}

#[derive(Debug, Clone)]
pub struct SafetensorsHeader {
    data_start: u64,
    file_size: u64,
    metadata: HashMap<String, String>,
    tensors: HashMap<String, TensorMetadata>,
}

impl SafetensorsHeader {
    /// Read and validate only the safetensors prefix/header.
    ///
    /// This never maps or reads the tensor payload, so it remains practical for
    /// multi-gigabyte model shards. All file operations use Tokio async I/O.
    pub async fn load(path: &Path) -> anyhow::Result<Self> {
        let mut file = tokio::fs::File::open(path)
            .await
            .with_context(|| format!("open safetensors file {}", path.display()))?;
        let file_size = file
            .metadata()
            .await
            .with_context(|| format!("stat safetensors file {}", path.display()))?
            .len();

        ensure!(
            file_size >= PREFIX_BYTES,
            "safetensors file is too small to contain an 8-byte header length"
        );

        let mut prefix = [0_u8; PREFIX_BYTES as usize];
        file.read_exact(&mut prefix)
            .await
            .with_context(|| format!("read safetensors prefix {}", path.display()))?;
        let header_len = u64::from_le_bytes(prefix);

        ensure!(header_len > 0, "safetensors header length must be non-zero");
        ensure!(
            header_len <= MAX_HEADER_BYTES,
            "safetensors header length {header_len} exceeds limit {MAX_HEADER_BYTES}"
        );

        let data_start = PREFIX_BYTES
            .checked_add(header_len)
            .context("safetensors data offset overflow")?;
        ensure!(
            data_start <= file_size,
            "safetensors header extends past end of file"
        );

        let header_len_usize =
            usize::try_from(header_len).context("header length exceeds usize")?;
        let mut header_bytes = vec![0_u8; header_len_usize];
        file.read_exact(&mut header_bytes)
            .await
            .with_context(|| format!("read safetensors header {}", path.display()))?;

        ensure!(
            header_bytes.first() == Some(&b'{'),
            "safetensors header must begin with a JSON object"
        );

        Self::parse_header_bytes(&header_bytes, data_start, file_size)
    }

    fn parse_header_bytes(
        header_bytes: &[u8],
        data_start: u64,
        file_size: u64,
    ) -> anyhow::Result<Self> {
        let root: Value =
            serde_json::from_slice(header_bytes).context("parse safetensors header JSON")?;
        let object = root
            .as_object()
            .context("safetensors header root must be a JSON object")?;

        let metadata = match object.get("__metadata__") {
            Some(value) => serde_json::from_value::<HashMap<String, String>>(value.clone())
                .context("parse safetensors __metadata__")?,
            None => HashMap::new(),
        };

        let mut tensors = HashMap::with_capacity(object.len());
        for (name, value) in object {
            if name == "__metadata__" {
                continue;
            }
            ensure!(!name.is_empty(), "safetensors tensor name cannot be empty");
            let tensor: TensorMetadata = serde_json::from_value(value.clone())
                .with_context(|| format!("parse safetensors tensor metadata {name:?}"))?;
            validate_tensor(name, &tensor)?;
            tensors.insert(name.clone(), tensor);
        }

        ensure!(
            !tensors.is_empty(),
            "safetensors header contains no tensors"
        );

        let data_len = file_size
            .checked_sub(data_start)
            .context("safetensors data section underflow")?;
        validate_data_layout(&tensors, data_len)?;

        Ok(Self {
            data_start,
            file_size,
            metadata,
            tensors,
        })
    }

    pub fn data_start(&self) -> u64 {
        self.data_start
    }

    pub fn file_size(&self) -> u64 {
        self.file_size
    }

    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.metadata
    }

    pub fn tensors(&self) -> &HashMap<String, TensorMetadata> {
        &self.tensors
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorMetadata> {
        self.tensors.get(name)
    }

    /// Resolve the safetensors data-relative offsets into an absolute file span.
    pub fn absolute_span(&self, name: &str) -> anyhow::Result<TensorSpan> {
        let tensor = self
            .tensor(name)
            .with_context(|| format!("unknown safetensors tensor {name:?}"))?;
        let begin = tensor.data_offsets[0];
        let end = tensor.data_offsets[1];
        let offset = self
            .data_start
            .checked_add(begin)
            .context("absolute tensor offset overflow")?;
        let length = end
            .checked_sub(begin)
            .context("tensor offset ordering underflow")?;
        Ok(TensorSpan { offset, length })
    }
}

fn validate_tensor(name: &str, tensor: &TensorMetadata) -> anyhow::Result<()> {
    ensure!(
        !tensor.dtype.trim().is_empty(),
        "tensor {name:?} dtype cannot be empty"
    );
    ensure!(
        tensor.data_offsets[0] <= tensor.data_offsets[1],
        "tensor {name:?} has reversed data offsets"
    );
    Ok(())
}

fn validate_data_layout(
    tensors: &HashMap<String, TensorMetadata>,
    data_len: u64,
) -> anyhow::Result<()> {
    let mut spans = tensors
        .iter()
        .map(|(name, tensor)| {
            (
                name.as_str(),
                tensor.data_offsets[0],
                tensor.data_offsets[1],
            )
        })
        .collect::<Vec<_>>();
    spans.sort_unstable_by(|left, right| {
        left.1
            .cmp(&right.1)
            .then_with(|| left.2.cmp(&right.2))
            .then_with(|| left.0.cmp(right.0))
    });

    let mut cursor = 0_u64;
    for (name, begin, end) in spans {
        ensure!(
            begin == cursor,
            "tensor {name:?} begins at {begin}, expected contiguous offset {cursor}"
        );
        ensure!(
            end <= data_len,
            "tensor {name:?} ends at {end}, beyond data section length {data_len}"
        );
        cursor = end;
    }

    ensure!(
        cursor == data_len,
        "safetensors data section has unindexed trailing bytes: indexed {cursor}, file contains {data_len}"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use tempfile::NamedTempFile;
    use tokio::io::AsyncWriteExt;

    use super::{MAX_HEADER_BYTES, SafetensorsHeader};

    async fn write_fixture(header: &str, payload: &[u8]) -> NamedTempFile {
        let file = NamedTempFile::new().expect("temporary safetensors file");
        let mut async_file = tokio::fs::OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(file.path())
            .await
            .expect("open fixture");
        async_file
            .write_all(&(header.len() as u64).to_le_bytes())
            .await
            .expect("write header length");
        async_file
            .write_all(header.as_bytes())
            .await
            .expect("write header");
        async_file.write_all(payload).await.expect("write payload");
        async_file.flush().await.expect("flush fixture");
        file
    }

    #[tokio::test]
    async fn loads_metadata_without_reading_tensor_payload() {
        let header_json = r#"{"__metadata__":{"format":"pt"},"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F16","shape":[2],"data_offsets":[4,8]}}"#;
        let file = write_fixture(header_json, &[0_u8; 8]).await;

        let header = SafetensorsHeader::load(file.path())
            .await
            .expect("valid safetensors header");
        assert_eq!(header.metadata().get("format"), Some(&"pt".to_string()));
        assert_eq!(header.tensors().len(), 2);
        assert_eq!(header.data_start(), 8 + header_json.len() as u64);

        let span = header.absolute_span("b").expect("absolute tensor span");
        assert_eq!(span.offset, header.data_start() + 4);
        assert_eq!(span.length, 4);
    }

    #[tokio::test]
    async fn rejects_tensor_range_past_file_data() {
        let header_json = r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let file = write_fixture(header_json, &[0_u8; 4]).await;

        let error = SafetensorsHeader::load(file.path())
            .await
            .expect_err("out-of-bounds tensor must fail");
        assert!(error.to_string().contains("beyond data section length"));
    }

    #[tokio::test]
    async fn rejects_holes_between_tensor_spans() {
        let header_json = r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[8,12]}}"#;
        let file = write_fixture(header_json, &[0_u8; 12]).await;

        let error = SafetensorsHeader::load(file.path())
            .await
            .expect_err("layout hole must fail");
        assert!(error.to_string().contains("expected contiguous offset"));
    }

    #[tokio::test]
    async fn rejects_unindexed_trailing_bytes() {
        let header_json = r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let file = write_fixture(header_json, &[0_u8; 8]).await;

        let error = SafetensorsHeader::load(file.path())
            .await
            .expect_err("trailing bytes must fail");
        assert!(error.to_string().contains("unindexed trailing bytes"));
    }

    #[tokio::test]
    async fn rejects_malformed_json_header() {
        let file = write_fixture("{not-json", &[]).await;
        let error = SafetensorsHeader::load(file.path())
            .await
            .expect_err("malformed JSON must fail");
        assert!(error.to_string().contains("parse safetensors header JSON"));
    }

    #[tokio::test]
    async fn rejects_header_larger_than_configured_limit() {
        let file = NamedTempFile::new().expect("temporary safetensors file");
        tokio::fs::write(file.path(), (MAX_HEADER_BYTES + 1).to_le_bytes())
            .await
            .expect("write oversized prefix");

        let error = SafetensorsHeader::load(Path::new(file.path()))
            .await
            .expect_err("oversized header must fail");
        assert!(error.to_string().contains("exceeds limit"));
    }
}
