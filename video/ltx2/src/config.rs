//! The DiT's config, from the GGUF's own metadata (the Python engine's `ltx2_config.py`, ported).
//!
//! diffusers' LTX-2 code was written for LTX-2.0; 2.5 differs in seven config flags and one key prefix. The dev and
//! distilled GGUFs carry Lightricks' own config as JSON in their metadata (key `config`, under `transformer`), under
//! their names; this module translates it to diffusers' and checks that what is on disk is the model this engine
//! implements. The engine reads the weights by tensor name, so a 2.0 file would fail deep in the load with a missing
//! tensor; here it fails first, saying which flag is off.
//!
//! The seven flags (Lightricks' name -> diffusers'):
//! ```text
//!   apply_gated_attention          -> gated_attn, audio_gated_attn          true in 2.5 (a per-head gate on attention)
//!   cross_attention_adaln          -> cross_attn_mod, audio_cross_attn_mod  true (the prompt modulation tables)
//!   use_keyframes_abs_pos_embedding-> use_keyframes_abs_pos_embedding       true
//!   qk_norm "rms_norm"             -> "rms_norm_across_heads"               (over the whole inner dimension)
//!   ff_bias                        -> ff_bias                               false (96 weights fewer)
//!   (fixed)                           use_prompt_embeddings                 false (captions are projected in the connector)
//!   (fixed)                           use_prompt_adaln_single               true (prompt_adaln_single.* in the file)
//! ```
//! and the key prefix: diffusers' converter leaves 2.5's prompt modulation under Lightricks' `prompt_adaln_single.` /
//! `audio_prompt_adaln_single.`, which diffusers' model calls `prompt_adaln.` / `audio_prompt_adaln.`.

use nextcuda_core::{Error, Result};
use serde_json::{json, Map, Value};

/// Keys that carry the same name in both configs and are copied as they are
pub const SHARED: [&str; 17] = ["activation_fn", "attention_bias", "attention_head_dim", "audio_attention_head_dim", "audio_cross_attention_dim",
                                "audio_num_attention_heads", "caption_channels", "cross_attention_dim", "in_channels", "norm_elementwise_affine", "norm_eps",
                                "num_attention_heads", "num_layers", "out_channels", "rope_type", "timestep_scale_multiplier", "audio_out_channels"];

/// Lightricks' switch names and diffusers' for them
pub const FLAGS: [(&str, &[&str]); 3] = [("apply_gated_attention", &["gated_attn", "audio_gated_attn"]),
                                         ("cross_attention_adaln", &["cross_attn_mod", "audio_cross_attn_mod"]),
                                         ("use_keyframes_abs_pos_embedding", &["use_keyframes_abs_pos_embedding"])];

/// The `config` of a GGUF's metadata, parsed: the object under `transformer`
pub fn embedded(text: &str) -> Result<Value> {
    let v: Value = serde_json::from_str(text).map_err(|e| Error(format!("the GGUF's config is not JSON: {e}")))?;
    v.get("transformer").filter(|t| t.is_object()).cloned().ok_or_else(|| Error("the GGUF's config has no transformer".into()))
}

/// Lightricks' config -> diffusers' (only the keys diffusers' model takes: the shared ones, the renamed flags, the
/// fixed 2.5 facts and ff_bias)
pub fn derive(src: &Value) -> Map<String, Value> {
    let mut cfg = Map::new();
    for k in SHARED {
        if let Some(v) = src.get(k) {
            cfg.insert(k.into(), v.clone());
        }
    }
    for (theirs, ours) in FLAGS {
        if let Some(v) = src.get(theirs) {
            for name in ours {
                cfg.insert((*name).into(), v.clone());
            }
        }
    }
    // Lightricks' rms_norm normalises over the whole inner dimension, which diffusers calls across_heads
    cfg.insert("qk_norm".into(), json!("rms_norm_across_heads"));
    cfg.insert("use_prompt_embeddings".into(), json!(false));
    cfg.insert("use_prompt_adaln_single".into(), json!(true));
    if let Some(v) = src.get("ff_bias") {
        cfg.insert("ff_bias".into(), v.clone());
    }
    cfg.insert("_class_name".into(), json!("LTX2VideoTransformer3DModel"));
    cfg
}

/// A weight's name in diffusers' model from the file's (2.5's prompt modulation prefix)
pub fn diffusers_name(file_name: &str) -> String {
    file_name.replace("prompt_adaln_single.", "prompt_adaln.")
}

/// What this engine implements, read from the config: LTX-2.5's flags and sizes
pub fn check_ltx25(cfg: &Map<String, Value>) -> Result<()> {
    let want_true = ["gated_attn", "audio_gated_attn", "cross_attn_mod", "audio_cross_attn_mod", "use_keyframes_abs_pos_embedding", "use_prompt_adaln_single"];
    let mut bad: Vec<String> = Vec::new();
    for k in want_true {
        if cfg.get(k).and_then(|v| v.as_bool()) != Some(true) {
            bad.push(format!("{k} is {} (2.5: true)", cfg.get(k).map_or("absent".to_string(), |v| v.to_string())));
        }
    }
    for k in ["ff_bias", "use_prompt_embeddings"] {
        if cfg.get(k).and_then(|v| v.as_bool()) == Some(true) {
            bad.push(format!("{k} is true (2.5: false)"));
        }
    }
    for (k, want) in [("num_layers", 48), ("num_attention_heads", 32), ("attention_head_dim", 128), ("audio_num_attention_heads", 32), ("audio_attention_head_dim", 64),
                      ("cross_attention_dim", 4096), ("audio_cross_attention_dim", 2048), ("caption_channels", 3840), ("in_channels", 128)] {
        if let Some(v) = cfg.get(k).and_then(|v| v.as_u64()) {
            if v != want {
                bad.push(format!("{k} is {v} ({want} here)"));
            }
        }
    }
    if cfg.get("rope_type").and_then(|v| v.as_str()).is_some_and(|r| r != "split") {
        bad.push(format!("rope_type is {} (split here)", cfg["rope_type"]));
    }
    if bad.is_empty() {
        Ok(())
    } else {
        Err(Error(format!("not the LTX-2.5 DiT this engine implements (an LTX-2.0 file?): {}", bad.join("; "))))
    }
}

/// The GGUF's metadata `config` checked, when it has one (a file converted without it is trusted by its tensors)
pub fn check_gguf(f: &nextcuda_gguf::Gguf) -> Result<Option<Map<String, Value>>> {
    let Some(text) = f.meta("config").and_then(|v| v.as_str()) else { return Ok(None) };
    let cfg = derive(&embedded(text)?);
    check_ltx25(&cfg)?;
    Ok(Some(cfg))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `config` the dev GGUF (ltx-2.5-22b-dev-transformer-Q4_K_M) carries in its metadata
    const DEV_GGUF: &str = r#"{"transformer": {"_class_name": "AVTransformer3DModel", "activation_fn": "gelu-approximate", "attention_bias": true, "attention_head_dim": 128, "attention_type": "default", "caption_channels": 3840, "cross_attention_dim": 4096, "double_self_attention": false, "dropout": 0.0, "in_channels": 128, "norm_elementwise_affine": false, "norm_eps": 1e-06, "norm_num_groups": 32, "num_attention_heads": 32, "num_embeds_ada_norm": 1000, "num_layers": 48, "out_channels": 128, "qk_norm": "rms_norm", "standardization_norm": "rms_norm", "positional_embedding_type": "rope", "positional_embedding_theta": 10000.0, "timestep_scale_multiplier": 1000, "av_ca_timestep_scale_multiplier": 1000.0, "audio_num_attention_heads": 32, "audio_attention_head_dim": 64, "use_audio_video_cross_attention": true, "ff_bias": false, "share_ff": false, "audio_out_channels": 128, "audio_cross_attention_dim": 2048, "av_cross_ada_norm": true, "apply_gated_attention": true, "cross_attention_adaln": true, "rope_type": "split", "frequencies_precision": "float64", "use_keyframes_abs_pos_embedding": true}, "scheduler": {"_class_name": "RectifiedFlowScheduler", "sampler": "LinearQuadratic"}}"#;

    /// Lightricks/LTX-2.5-Diffusers transformer/config.json (the official one the Python engine checked its derivation against)
    const OFFICIAL: &str = r#"{"_class_name": "LTX2VideoTransformer3DModel", "activation_fn": "gelu-approximate", "attention_bias": true, "attention_head_dim": 128, "attention_out_bias": true, "audio_attention_head_dim": 64, "audio_cross_attention_dim": 2048, "audio_cross_attn_mod": true, "audio_ff_bias": true, "audio_gated_attn": true, "audio_in_channels": 128, "audio_num_attention_heads": 32, "audio_out_channels": 128, "caption_channels": 3840, "cross_attention_dim": 4096, "cross_attn_mod": true, "ff_bias": false, "gated_attn": true, "in_channels": 128, "norm_elementwise_affine": false, "norm_eps": 1e-06, "num_attention_heads": 32, "num_layers": 48, "out_channels": 128, "qk_norm": "rms_norm_across_heads", "rope_type": "split", "timestep_scale_multiplier": 1000, "use_keyframes_abs_pos_embedding": true, "use_prompt_adaln_single": true, "use_prompt_embeddings": false}"#;

    /// what LTX-2.0's config would say: none of the 2.5 switches, a bias in the feed-forward
    fn v20() -> Value {
        let mut t = embedded(DEV_GGUF).unwrap();
        let o = t.as_object_mut().unwrap();
        for k in ["apply_gated_attention", "cross_attention_adaln", "use_keyframes_abs_pos_embedding"] {
            o.remove(k);
        }
        o.insert("ff_bias".into(), json!(true));
        t
    }

    #[test]
    fn the_embedded_config_is_the_transformer_object() {
        let t = embedded(DEV_GGUF).unwrap();
        assert_eq!(t["num_layers"], 48);
        assert!(embedded("{}").unwrap_err().0.contains("no transformer"));
        assert!(embedded("not json").unwrap_err().0.contains("not JSON"));
    }

    #[test]
    fn the_seven_flags_of_2_5_come_out_as_diffusers_names_them() {
        let c = derive(&embedded(DEV_GGUF).unwrap());
        // 1-3: the three switches Lightricks names, each under diffusers' names (the audio branch's too)
        for k in ["gated_attn", "audio_gated_attn", "cross_attn_mod", "audio_cross_attn_mod", "use_keyframes_abs_pos_embedding"] {
            assert_eq!(c[k], json!(true), "{k}");
        }
        // 4: rms_norm over the whole inner dimension is diffusers' across_heads
        assert_eq!(c["qk_norm"], "rms_norm_across_heads");
        // 5: no bias in the feed-forward (96 weights fewer than 2.0's)
        assert_eq!(c["ff_bias"], json!(false));
        // 6, 7: captions are projected in the connector, and the prompt modulation tables are there
        assert_eq!(c["use_prompt_embeddings"], json!(false));
        assert_eq!(c["use_prompt_adaln_single"], json!(true));
        assert_eq!(c["_class_name"], "LTX2VideoTransformer3DModel");
        // names that are not diffusers' are not carried over
        for k in ["apply_gated_attention", "cross_attention_adaln", "attention_type", "norm_num_groups", "positional_embedding_theta"] {
            assert!(!c.contains_key(k), "{k}");
        }
        assert!(check_ltx25(&c).is_ok());
    }

    #[test]
    fn the_derived_config_agrees_with_lightricks_official_one() {
        // the Python engine's check, 2026-09-15: every key both have is equal
        let official: Map<String, Value> = serde_json::from_str(OFFICIAL).unwrap();
        let mine = derive(&embedded(DEV_GGUF).unwrap());
        let mut compared = 0;
        for (k, v) in &official {
            if k.starts_with('_') {
                continue;
            }
            if let Some(m) = mine.get(k) {
                assert_eq!(m, v, "{k} disagrees with the official config");
                compared += 1;
            }
        }
        assert!(compared >= 20, "{compared} keys compared");
        // the keys the derivation leaves to diffusers' defaults are the ones the file does not carry
        let absent: Vec<&String> = official.keys().filter(|k| !k.starts_with('_') && !mine.contains_key(*k)).collect();
        assert_eq!(absent, ["attention_out_bias", "audio_ff_bias", "audio_in_channels"], "{absent:?}");
        assert!(check_ltx25(&official).is_ok());
    }

    #[test]
    fn a_2_0_file_is_refused_with_the_flags_that_differ() {
        let c = derive(&v20());
        let e = check_ltx25(&c).unwrap_err().0;
        assert!(e.contains("LTX-2.0"), "{e}");
        // gated_attn / cross_attn_mod / the keyframe embedding are absent, ff_bias is on
        for k in ["gated_attn is absent", "audio_gated_attn is absent", "cross_attn_mod is absent", "use_keyframes_abs_pos_embedding is absent", "ff_bias is true"] {
            assert!(e.contains(k), "{k}: {e}");
        }
        // one flag at a time is named
        let mut c = derive(&embedded(DEV_GGUF).unwrap());
        c.insert("cross_attn_mod".into(), json!(false));
        let e = check_ltx25(&c).unwrap_err().0;
        assert!(e.contains("cross_attn_mod is false") && !e.contains("gated_attn"), "{e}");
        // a different size is refused too
        let mut c = derive(&embedded(DEV_GGUF).unwrap());
        c.insert("num_layers".into(), json!(40));
        assert!(check_ltx25(&c).unwrap_err().0.contains("num_layers is 40 (48 here)"));
        let mut c = derive(&embedded(DEV_GGUF).unwrap());
        c.insert("rope_type".into(), json!("interleaved"));
        assert!(check_ltx25(&c).unwrap_err().0.contains("rope_type"));
    }

    #[test]
    fn the_prompt_modulation_key_prefix_is_renamed() {
        assert_eq!(diffusers_name("transformer_blocks.3.prompt_scale_shift_table"), "transformer_blocks.3.prompt_scale_shift_table");
        assert_eq!(diffusers_name("prompt_adaln_single.linear.weight"), "prompt_adaln.linear.weight");
        assert_eq!(diffusers_name("audio_prompt_adaln_single.emb.timestep_embedder.linear_1.bias"), "audio_prompt_adaln.emb.timestep_embedder.linear_1.bias");
        // the other modulation modules keep their names
        assert_eq!(diffusers_name("adaln_single.linear.weight"), "adaln_single.linear.weight");
        assert_eq!(diffusers_name("av_ca_a2v_gate_adaln_single.linear.bias"), "av_ca_a2v_gate_adaln_single.linear.bias");
    }
}
