// src/adapters/qwen3.rs

use anyhow::{Context, ensure};
use serde::{Deserialize, Serialize};

pub const MODEL_TYPE: &str = "qwen3_moe";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Qwen3MoeConfig {
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub norm_topk_prob: bool,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
}

impl Qwen3MoeConfig {
    pub fn from_json(json: &str) -> anyhow::Result<Self> {
        let config: Self = serde_json::from_str(json).context("parse Qwen3 MoE config JSON")?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        ensure!(
            self.model_type == MODEL_TYPE,
            "unsupported model_type {:?}; expected {MODEL_TYPE:?}",
            self.model_type
        );
        ensure!(self.hidden_size > 0, "hidden_size must be non-zero");
        ensure!(
            self.intermediate_size > 0,
            "intermediate_size must be non-zero"
        );
        ensure!(
            self.moe_intermediate_size > 0,
            "moe_intermediate_size must be non-zero"
        );
        ensure!(
            self.num_hidden_layers > 0,
            "num_hidden_layers must be non-zero"
        );
        ensure!(
            self.num_attention_heads > 0,
            "num_attention_heads must be non-zero"
        );
        ensure!(
            self.num_key_value_heads > 0,
            "num_key_value_heads must be non-zero"
        );
        ensure!(self.num_experts > 0, "num_experts must be non-zero");
        ensure!(
            self.num_experts_per_tok > 0,
            "num_experts_per_tok must be non-zero"
        );
        ensure!(
            self.num_experts_per_tok <= self.num_experts,
            "num_experts_per_tok cannot exceed num_experts"
        );
        ensure!(
            self.hidden_size.is_multiple_of(self.num_attention_heads),
            "hidden_size must be divisible by num_attention_heads"
        );
        ensure!(
            self.num_attention_heads.is_multiple_of(self.num_key_value_heads),
            "num_attention_heads must be divisible by num_key_value_heads"
        );
        ensure!(self.vocab_size > 0, "vocab_size must be non-zero");
        ensure!(
            self.rms_norm_eps.is_finite() && self.rms_norm_eps > 0.0,
            "rms_norm_eps must be finite and positive"
        );
        ensure!(
            self.rope_theta.is_finite() && self.rope_theta > 0.0,
            "rope_theta must be finite and positive"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RouteSelection {
    pub expert: usize,
    pub weight: f32,
}

/// Compute a correctness-oriented reference route from already-computed router
/// logits. This mirrors the documented softmax -> top-k -> optional top-k
/// normalization sequence, but is not yet the tensor-level authoritative router.
///
/// Ties are resolved by lower expert ID to make fixtures deterministic. Exact
/// tie behavior must be checked against the reference runtime before this helper
/// can be used for production token generation.
pub fn reference_route(
    router_logits: &[f32],
    experts_per_token: usize,
    norm_topk_prob: bool,
) -> anyhow::Result<Vec<RouteSelection>> {
    ensure!(!router_logits.is_empty(), "router logits cannot be empty");
    ensure!(
        experts_per_token > 0 && experts_per_token <= router_logits.len(),
        "experts_per_token must be in 1..=router_logits.len()"
    );
    ensure!(
        router_logits.iter().all(|value| value.is_finite()),
        "router logits must be finite"
    );

    let max_logit = router_logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let mut probabilities = Vec::with_capacity(router_logits.len());
    let mut denominator = 0.0_f64;

    for &logit in router_logits {
        let value = f64::from(logit - max_logit).exp();
        probabilities.push(value);
        denominator += value;
    }

    ensure!(
        denominator.is_finite() && denominator > 0.0,
        "router softmax denominator is invalid"
    );

    let mut ranked: Vec<(usize, f64)> = probabilities
        .into_iter()
        .enumerate()
        .map(|(expert, value)| (expert, value / denominator))
        .collect();

    ranked.sort_unstable_by(|(left_id, left_prob), (right_id, right_prob)| {
        right_prob
            .total_cmp(left_prob)
            .then_with(|| left_id.cmp(right_id))
    });
    ranked.truncate(experts_per_token);

    if norm_topk_prob {
        let selected_sum: f64 = ranked.iter().map(|(_, probability)| probability).sum();
        ensure!(
            selected_sum.is_finite() && selected_sum > 0.0,
            "selected router probability sum is invalid"
        );
        for (_, probability) in &mut ranked {
            *probability /= selected_sum;
        }
    }

    Ok(ranked
        .into_iter()
        .map(|(expert, probability)| RouteSelection {
            expert,
            weight: probability as f32,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::{MODEL_TYPE, Qwen3MoeConfig, reference_route};

    const OFFICIAL_CONFIG_SUBSET: &str = r#"
    {
      "model_type": "qwen3_moe",
      "hidden_size": 2048,
      "intermediate_size": 6144,
      "moe_intermediate_size": 768,
      "num_hidden_layers": 48,
      "num_attention_heads": 32,
      "num_key_value_heads": 4,
      "num_experts": 128,
      "num_experts_per_tok": 8,
      "norm_topk_prob": true,
      "vocab_size": 151936,
      "rms_norm_eps": 0.000001,
      "rope_theta": 1000000.0
    }
    "#;

    #[test]
    fn parses_official_qwen3_moe_shape() {
        let config = Qwen3MoeConfig::from_json(OFFICIAL_CONFIG_SUBSET).expect("valid config");
        assert_eq!(config.model_type, MODEL_TYPE);
        assert_eq!(config.num_hidden_layers, 48);
        assert_eq!(config.num_experts, 128);
        assert_eq!(config.num_experts_per_tok, 8);
        assert!(config.norm_topk_prob);
    }

    #[test]
    fn rejects_wrong_model_type() {
        let json = OFFICIAL_CONFIG_SUBSET.replace("qwen3_moe", "dense_model");
        let error = Qwen3MoeConfig::from_json(&json).expect_err("wrong model type must fail");
        assert!(error.to_string().contains("unsupported model_type"));
    }

    #[test]
    fn rejects_more_selected_experts_than_available() {
        let json = OFFICIAL_CONFIG_SUBSET
            .replace("\"num_experts\": 128", "\"num_experts\": 4")
            .replace("\"num_experts_per_tok\": 8", "\"num_experts_per_tok\": 5");
        let error = Qwen3MoeConfig::from_json(&json).expect_err("invalid top-k must fail");
        assert!(
            error
                .to_string()
                .contains("num_experts_per_tok cannot exceed num_experts")
        );
    }

    #[test]
    fn reference_route_selects_highest_logits_and_normalizes() {
        let route = reference_route(&[0.0, 1.0, 3.0, 2.0], 2, true).expect("valid route");
        assert_eq!(route.len(), 2);
        assert_eq!(route[0].expert, 2);
        assert_eq!(route[1].expert, 3);

        let weight_sum: f32 = route.iter().map(|selection| selection.weight).sum();
        assert!((weight_sum - 1.0).abs() < 1e-6);
        assert!(route[0].weight > route[1].weight);
    }

    #[test]
    fn reference_route_preserves_global_softmax_mass_when_not_normalized() {
        let route = reference_route(&[0.0, 1.0, 3.0, 2.0], 2, false).expect("valid route");
        let weight_sum: f32 = route.iter().map(|selection| selection.weight).sum();
        assert!(weight_sum > 0.0 && weight_sum < 1.0);
    }

    #[test]
    fn reference_route_uses_deterministic_expert_id_tie_break() {
        let route = reference_route(&[2.0, 2.0, 1.0], 2, true).expect("valid route");
        assert_eq!(route[0].expert, 0);
        assert_eq!(route[1].expert, 1);
    }
}
