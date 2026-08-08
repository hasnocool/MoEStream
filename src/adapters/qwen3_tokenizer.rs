// src/adapters/qwen3_tokenizer.rs

use std::{collections::HashMap, path::Path, sync::Arc};

use anyhow::{Context, ensure};
use minijinja::Environment;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokenizers::Tokenizer;

const TOKENIZER_FILE_NAME: &str = "tokenizer.json";
const TOKENIZER_CONFIG_FILE_NAME: &str = "tokenizer_config.json";
const CHAT_TEMPLATE_NAME: &str = "qwen3-chat";

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3TokenizerConfig {
    pub chat_template: String,
    pub eos_token: String,
    pub pad_token: String,
    pub model_max_length: usize,
    pub tokenizer_class: String,
    #[serde(default)]
    pub added_tokens_decoder: HashMap<String, AddedTokenConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AddedTokenConfig {
    pub content: String,
    #[serde(default)]
    pub special: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenIds {
    pub eos: u32,
    pub pad: u32,
    pub im_start: Option<u32>,
    pub im_end: Option<u32>,
}

#[derive(Clone)]
pub struct Qwen3Tokenizer {
    tokenizer: Arc<Tokenizer>,
    environment: Arc<Environment<'static>>,
    config: Arc<Qwen3TokenizerConfig>,
    token_ids: TokenIds,
}

#[derive(Debug, Clone, Serialize)]
struct ChatTemplateContext {
    messages: Vec<Value>,
    tools: Vec<Value>,
    add_generation_prompt: bool,
    enable_thinking: bool,
}

impl Qwen3Tokenizer {
    /// Load tokenizer assets without doing synchronous file I/O or large JSON
    /// parsing on Tokio worker threads.
    pub async fn load(model_root: &Path) -> anyhow::Result<Self> {
        let tokenizer_path = model_root.join(TOKENIZER_FILE_NAME);
        let config_path = model_root.join(TOKENIZER_CONFIG_FILE_NAME);
        let (tokenizer_json, config_json) = tokio::try_join!(
            tokio::fs::read(&tokenizer_path),
            tokio::fs::read(&config_path)
        )
        .with_context(|| format!("read tokenizer assets from {}", model_root.display()))?;

        tokio::task::spawn_blocking(move || Self::from_bytes(tokenizer_json, config_json))
            .await
            .context("join Qwen3 tokenizer parse task")?
    }

    fn from_bytes(tokenizer_json: Vec<u8>, config_json: Vec<u8>) -> anyhow::Result<Self> {
        let config: Qwen3TokenizerConfig =
            serde_json::from_slice(&config_json).context("parse tokenizer_config.json")?;
        validate_config(&config)?;

        let tokenizer = Tokenizer::from_bytes(tokenizer_json)
            .map_err(|error| anyhow::anyhow!("parse tokenizer.json: {error}"))?;
        let token_ids = resolve_token_ids(&tokenizer, &config)?;

        let mut environment = Environment::new();
        minijinja_contrib::add_to_environment(&mut environment);
        environment
            .add_template_owned(CHAT_TEMPLATE_NAME.to_string(), config.chat_template.clone())
            .context("compile Qwen3 chat template")?;

        Ok(Self {
            tokenizer: Arc::new(tokenizer),
            environment: Arc::new(environment),
            config: Arc::new(config),
            token_ids,
        })
    }

    pub fn config(&self) -> &Qwen3TokenizerConfig {
        &self.config
    }

    pub fn token_ids(&self) -> &TokenIds {
        &self.token_ids
    }

    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    /// Render the checkpoint-provided Jinja chat template off the async
    /// executor. Messages and tools are intentionally represented as JSON so
    /// tool calls and future Qwen template fields can pass through losslessly.
    pub async fn render_chat(
        &self,
        messages: &[Value],
        tools: &[Value],
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> anyhow::Result<String> {
        validate_messages(messages)?;
        let environment = Arc::clone(&self.environment);
        let context = ChatTemplateContext {
            messages: messages.to_vec(),
            tools: tools.to_vec(),
            add_generation_prompt,
            enable_thinking,
        };

        tokio::task::spawn_blocking(move || {
            environment
                .get_template(CHAT_TEMPLATE_NAME)
                .context("retrieve compiled Qwen3 chat template")?
                .render(context)
                .context("render Qwen3 chat template")
        })
        .await
        .context("join Qwen3 chat-template render task")?
    }

    /// Tokenize off the async executor because Hugging Face tokenizers may use
    /// CPU-heavy regex/BPE work and Rayon internally.
    pub async fn encode(&self, text: &str, add_special_tokens: bool) -> anyhow::Result<Vec<u32>> {
        let tokenizer = Arc::clone(&self.tokenizer);
        let text = text.to_owned();
        tokio::task::spawn_blocking(move || {
            tokenizer
                .encode(text, add_special_tokens)
                .map(|encoding| encoding.get_ids().to_vec())
                .map_err(|error| anyhow::anyhow!("encode Qwen3 text: {error}"))
        })
        .await
        .context("join Qwen3 encode task")?
    }

    pub async fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> anyhow::Result<String> {
        let tokenizer = Arc::clone(&self.tokenizer);
        let ids = ids.to_vec();
        tokio::task::spawn_blocking(move || {
            tokenizer
                .decode(&ids, skip_special_tokens)
                .map_err(|error| anyhow::anyhow!("decode Qwen3 token IDs: {error}"))
        })
        .await
        .context("join Qwen3 decode task")?
    }

    pub async fn render_and_encode_chat(
        &self,
        messages: &[Value],
        tools: &[Value],
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> anyhow::Result<Vec<u32>> {
        let rendered = self
            .render_chat(
                messages,
                tools,
                add_generation_prompt,
                enable_thinking,
            )
            .await?;
        self.encode(&rendered, false).await
    }
}

fn validate_config(config: &Qwen3TokenizerConfig) -> anyhow::Result<()> {
    ensure!(
        !config.chat_template.trim().is_empty(),
        "Qwen3 tokenizer chat_template cannot be empty"
    );
    ensure!(
        !config.eos_token.is_empty(),
        "Qwen3 tokenizer eos_token cannot be empty"
    );
    ensure!(
        !config.pad_token.is_empty(),
        "Qwen3 tokenizer pad_token cannot be empty"
    );
    ensure!(
        config.model_max_length > 0,
        "Qwen3 tokenizer model_max_length must be non-zero"
    );
    ensure!(
        config.tokenizer_class.starts_with("Qwen2Tokenizer"),
        "unsupported Qwen3 tokenizer class {:?}",
        config.tokenizer_class
    );
    Ok(())
}

fn resolve_token_ids(tokenizer: &Tokenizer, config: &Qwen3TokenizerConfig) -> anyhow::Result<TokenIds> {
    let eos = tokenizer
        .token_to_id(&config.eos_token)
        .with_context(|| format!("eos token {:?} is missing from tokenizer", config.eos_token))?;
    let pad = tokenizer
        .token_to_id(&config.pad_token)
        .with_context(|| format!("pad token {:?} is missing from tokenizer", config.pad_token))?;

    Ok(TokenIds {
        eos,
        pad,
        im_start: tokenizer.token_to_id("<|im_start|>"),
        im_end: tokenizer.token_to_id("<|im_end|>"),
    })
}

fn validate_messages(messages: &[Value]) -> anyhow::Result<()> {
    ensure!(!messages.is_empty(), "chat messages cannot be empty");
    for (index, message) in messages.iter().enumerate() {
        let object = message
            .as_object()
            .with_context(|| format!("chat message {index} must be a JSON object"))?;
        let role = object
            .get("role")
            .and_then(Value::as_str)
            .with_context(|| format!("chat message {index} must contain a string role"))?;
        ensure!(!role.is_empty(), "chat message {index} role cannot be empty");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use serde_json::json;
    use tempfile::tempdir;
    use tokenizers::{
        AddedToken, Tokenizer,
        models::wordlevel::WordLevel,
        pre_tokenizers::whitespace::Whitespace,
    };

    use super::Qwen3Tokenizer;

    async fn write_fixture(root: &std::path::Path) {
        let vocab = HashMap::from([
            ("[UNK]".to_string(), 0_u32),
            ("hello".to_string(), 1_u32),
            ("world".to_string(), 2_u32),
        ]);
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".to_string())
            .build()
            .expect("word-level model");
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Whitespace.into()));
        tokenizer.add_special_tokens(&[
            AddedToken::from("<|endoftext|>".to_string(), true),
            AddedToken::from("<|im_start|>".to_string(), true),
            AddedToken::from("<|im_end|>".to_string(), true),
        ]);
        tokio::fs::write(
            root.join("tokenizer.json"),
            tokenizer.to_string(false).expect("serialize tokenizer"),
        )
        .await
        .expect("write tokenizer");

        // This representative template exercises namespace mutation, reverse
        // slicing and Python-style startswith used by the real Qwen3 template
        // without copying the upstream template into this repository.
        let chat_template = r#"{%- set ns = namespace(last_query_index=messages|length - 1) -%}
{%- for message in messages[::-1] -%}
{%- set index = (messages|length - 1) - loop.index0 -%}
{%- if message.role == 'user' and not message.content.startswith('<tool_response>') -%}
{%- set ns.last_query_index = index -%}
{%- endif -%}
{%- endfor -%}
{%- for message in messages -%}<|im_start|>{{ message.role }}
{{ message.content }}<|im_end|>
{%- endfor -%}
{%- if add_generation_prompt -%}<|im_start|>assistant
{%- if enable_thinking is false -%}<think>

</think>

{%- endif -%}{%- endif -%}"#;
        let config = json!({
            "chat_template": chat_template,
            "eos_token": "<|im_end|>",
            "pad_token": "<|endoftext|>",
            "model_max_length": 131072,
            "tokenizer_class": "Qwen2Tokenizer",
            "added_tokens_decoder": {}
        });
        tokio::fs::write(
            root.join("tokenizer_config.json"),
            serde_json::to_vec(&config).expect("serialize config"),
        )
        .await
        .expect("write tokenizer config");
    }

    #[tokio::test]
    async fn loads_and_tokenizes_off_executor_threads() {
        let root = tempdir().expect("model root");
        write_fixture(root.path()).await;
        let tokenizer = Qwen3Tokenizer::load(root.path()).await.expect("load tokenizer");

        let ids = tokenizer.encode("hello world", false).await.expect("encode");
        assert_eq!(ids, vec![1, 2]);
        assert_eq!(
            tokenizer.decode(&ids, false).await.expect("decode"),
            "hello world"
        );
        assert!(tokenizer.token_ids().im_start.is_some());
        assert!(tokenizer.token_ids().im_end.is_some());
    }

    #[tokio::test]
    async fn renders_checkpoint_chat_template_with_thinking_control() {
        let root = tempdir().expect("model root");
        write_fixture(root.path()).await;
        let tokenizer = Qwen3Tokenizer::load(root.path()).await.expect("load tokenizer");
        let messages = vec![
            json!({"role": "system", "content": "Be concise."}),
            json!({"role": "user", "content": "hello"}),
        ];

        let rendered = tokenizer
            .render_chat(&messages, &[], true, false)
            .await
            .expect("render chat");
        assert!(rendered.contains("<|im_start|>system\nBe concise.<|im_end|>"));
        assert!(rendered.contains("<|im_start|>user\nhello<|im_end|>"));
        assert!(rendered.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    }

    #[tokio::test]
    async fn rejects_invalid_message_shape_before_rendering() {
        let root = tempdir().expect("model root");
        write_fixture(root.path()).await;
        let tokenizer = Qwen3Tokenizer::load(root.path()).await.expect("load tokenizer");
        let error = tokenizer
            .render_chat(&[json!("not-an-object")], &[], true, true)
            .await
            .expect_err("invalid message must fail");
        assert!(error.to_string().contains("must be a JSON object"));
    }
}
