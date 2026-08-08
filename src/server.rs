// src/server.rs

use std::net::SocketAddr;

use axum::{
    Json, Router,
    http::StatusCode,
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Debug, Clone, Serialize)]
struct ModelsResponse {
    object: &'static str,
    data: Vec<ModelCard>,
}

#[derive(Debug, Clone, Serialize)]
struct ModelCard {
    id: &'static str,
    object: &'static str,
    owned_by: &'static str,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<Value>,
    #[serde(default)]
    pub stream: bool,
}

pub async fn serve(address: SocketAddr) -> anyhow::Result<()> {
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions));

    let listener = tokio::net::TcpListener::bind(address).await?;
    tracing::info!(%address, "MoEStream API listening");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn health() -> Json<Value> {
    Json(json!({"status": "ok", "runtime": "moestream"}))
}

async fn models() -> Json<ModelsResponse> {
    Json(ModelsResponse {
        object: "list",
        data: vec![ModelCard {
            id: "moestream-placeholder",
            object: "model",
            owned_by: "local",
        }],
    })
}

async fn chat_completions(Json(request): Json<ChatCompletionRequest>) -> (StatusCode, Json<Value>) {
    let _ = request;
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(json!({
            "error": {
                "message": "Model adapters and tensor execution are the next implementation milestone.",
                "type": "not_implemented"
            }
        })),
    )
}
