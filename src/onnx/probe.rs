/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Propose `tests/models/manifest.json` entries for a Hugging Face repository.
//!
//! Lists the repository's `onnx/*.onnx` exports, builds a weight-free skeleton
//! of each (see [`crate::onnx::skeleton`]), reads the graph inputs, and fills
//! the free dimensions from a naming policy (`batch_size` = 1,
//! `sequence_length` = 128, image sizes 224, ...). Merged decoders with a
//! `use_cache_branch` input get one entry per branch. Callers can override
//! any dimension, pin inputs, and convert each proposal to check it.

use std::collections::HashMap;
use std::path::Path;

use prost::Message;

use crate::onnx::convert::{convert_model_proto, sanitize_identifier, ConvertOptions};
use crate::onnx::skeleton::{strip_model, HubSource, KEEP_BYTES};
use crate::protos::onnx::tensor_shape_proto::dimension::Value as DimensionValue;
use crate::protos::onnx::type_proto::Value as TypeProtoValue;
use crate::protos::onnx::{ModelProto, TensorProto_DataType};

/// Weight files above this size run one at a time in the sweep (`"heavy": true`).
pub const HEAVY_BYTES: u64 = 1 << 30;

/// Default sequence and cache lengths used by the dimension policy.
const SEQ_LEN: u32 = 128;
const DECODER_SEQ_LEN: u32 = 16;
const DECODE_PAST_LEN: u32 = 16;

/// An `onnx/*.onnx` export in a Hub repository.
#[derive(Debug, Clone)]
pub struct HubFile {
    /// Path inside the repository, e.g. `onnx/model_q4.onnx`.
    pub path: String,
    pub size: u64,
}

/// List the `.onnx` files under `onnx/` of a Hub repository (`org/repo`).
pub fn list_hub_onnx_files(repo: &str) -> Result<Vec<HubFile>, String> {
    let url = format!("https://huggingface.co/api/models/{repo}/tree/main/onnx");
    let mut request = ureq::get(&url);
    if let Ok(token) = std::env::var("HF_TOKEN") {
        if !token.is_empty() {
            request = request.set("Authorization", &format!("Bearer {token}"));
        }
    }
    let body = request
        .call()
        .map_err(|e| format!("list {url}: {e}"))?
        .into_string()
        .map_err(|e| e.to_string())?;
    let entries: Vec<serde_json::Value> =
        serde_json::from_str(&body).map_err(|e| format!("{url}: unexpected response: {e}"))?;
    let mut files = Vec::new();
    for entry in entries {
        let Some(path) = entry.get("path").and_then(|p| p.as_str()) else {
            continue;
        };
        if !path.ends_with(".onnx") {
            continue;
        }
        let size = entry
            .get("lfs")
            .and_then(|l| l.get("size"))
            .or_else(|| entry.get("size"))
            .and_then(|s| s.as_u64())
            .unwrap_or(0);
        files.push(HubFile {
            path: path.to_string(),
            size,
        });
    }
    files.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(files)
}

/// A graph input with its symbolic dimensions.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphInput {
    pub name: String,
    /// Names of the free dimensions (`DimParam`, or `<input>_dim<axis>` for
    /// unnamed zero dims), in axis order.
    pub free_dims: Vec<String>,
    pub is_bool: bool,
}

/// Graph inputs that are not initializers.
pub fn graph_inputs(model: &ModelProto) -> Vec<GraphInput> {
    let Some(graph) = model.graph.as_ref() else {
        return Vec::new();
    };
    let initializers: std::collections::HashSet<&str> =
        graph.initializer.iter().map(|t| t.name.as_str()).collect();
    graph
        .input
        .iter()
        .filter(|vi| !initializers.contains(vi.name.as_str()))
        .map(|vi| {
            let mut free_dims = Vec::new();
            let mut is_bool = false;
            if let Some(TypeProtoValue::TensorType(tt)) =
                vi.r#type.as_ref().and_then(|t| t.value.as_ref())
            {
                is_bool = tt.elem_type == TensorProto_DataType::Bool as i32;
                if let Some(shape) = tt.shape.as_ref() {
                    for (idx, dim) in shape.dim.iter().enumerate() {
                        match dim.value.as_ref() {
                            Some(DimensionValue::DimParam(p)) => free_dims.push(p.clone()),
                            Some(DimensionValue::DimValue(v)) if *v > 0 => {}
                            _ => free_dims
                                .push(format!("{}_dim{idx}", sanitize_identifier(&vi.name))),
                        }
                    }
                }
            }
            GraphInput {
                name: vi.name.clone(),
                free_dims,
                is_bool,
            }
        })
        .collect()
}

/// Default value for a free dimension by name, or `None` when the policy has
/// no opinion (e.g. dynamo's `s0`, whisper's `encoder_sequence_length / 2`).
pub fn default_dim(name: &str) -> Option<u32> {
    let n = name.to_ascii_lowercase();
    // Expressions such as `encoder_sequence_length / 2` are model specific.
    if n.contains(|c: char| c.is_whitespace() || "/*+-".contains(c)) {
        return None;
    }
    if n.contains("batch") {
        Some(1)
    } else if n.contains("past") {
        Some(0)
    } else if n.contains("decoder_sequence") {
        Some(DECODER_SEQ_LEN)
    } else if n.contains("sequence_length") || n.contains("total_sequence") {
        Some(SEQ_LEN)
    } else if n.contains("num_channels") {
        Some(3)
    } else if n.contains("height") || n.contains("width") {
        Some(224)
    } else if n.contains("num_frames") {
        Some(8)
    } else if n.contains("feature_size") {
        Some(80)
    } else if n.contains("num_samples") {
        Some(16000)
    } else if n.contains("num_choices") {
        Some(2)
    } else {
        None
    }
}

/// One line of `tests/models/manifest.json`.
#[derive(Debug, Clone, PartialEq)]
pub struct ManifestEntry {
    pub file: String,
    pub heavy: bool,
    pub pin_inputs: Vec<(String, i64)>,
    pub override_dims: Vec<(String, u32)>,
}

impl ManifestEntry {
    /// The manifest's one-entry-per-line JSON style.
    pub fn to_json_line(&self) -> String {
        let mut parts = vec![format!(
            "\"file\": {}",
            serde_json::to_string(&self.file).unwrap()
        )];
        if self.heavy {
            parts.push("\"heavy\": true".to_string());
        }
        if !self.pin_inputs.is_empty() {
            let pins: Vec<String> = self
                .pin_inputs
                .iter()
                .map(|(k, v)| format!("{}: {v}", serde_json::to_string(k).unwrap()))
                .collect();
            parts.push(format!("\"pin_inputs\": {{{}}}", pins.join(", ")));
        }
        if !self.override_dims.is_empty() {
            let dims: Vec<String> = self
                .override_dims
                .iter()
                .map(|(k, v)| format!("{}: {v}", serde_json::to_string(k).unwrap()))
                .collect();
            parts.push(format!("\"override_dims\": {{{}}}", dims.join(", ")));
        }
        format!("{{{}}}", parts.join(", "))
    }

    pub fn to_options(&self) -> ConvertOptions {
        ConvertOptions {
            free_dim_overrides: self.override_dims.iter().cloned().collect(),
            optimize: true,
            experimental_dynamic_inputs: false,
            pinned_inputs: self.pin_inputs.iter().cloned().collect(),
            zero_fill_missing_external_data: true,
        }
    }
}

/// Proposed entries plus the free dimensions the policy could not fill.
#[derive(Debug, Default)]
pub struct Proposal {
    pub entries: Vec<ManifestEntry>,
    pub unresolved: Vec<String>,
}

/// Build manifest entries for one export. `overrides` win over the policy;
/// `pins` are applied to every entry. A `use_cache_branch` input yields a
/// prefill entry (branch 0, no cache) and a decode entry (branch 1, one new
/// token, cache of [`DECODE_PAST_LEN`]); `decode_step` adds the same decode
/// entry for cache-carrying models without the gate.
pub fn propose_entries(
    file: &str,
    size: u64,
    inputs: &[GraphInput],
    overrides: &HashMap<String, u32>,
    pins: &[(String, i64)],
    decode_step: bool,
) -> Proposal {
    let mut names: Vec<String> = Vec::new();
    for input in inputs {
        for dim in &input.free_dims {
            if !names.contains(dim) {
                names.push(dim.clone());
            }
        }
    }
    let mut unresolved = Vec::new();
    let resolve = |name: &str, decode: bool| -> Option<u32> {
        if let Some(v) = overrides.get(name) {
            return Some(*v);
        }
        let lower = name.to_ascii_lowercase();
        if decode {
            if lower.contains("past") {
                return Some(DECODE_PAST_LEN);
            }
            if lower.contains("sequence_length")
                && !lower.contains("encoder")
                && !lower.contains("total")
            {
                return Some(1);
            }
        }
        default_dim(name)
    };
    let build = |decode: bool, unresolved: &mut Vec<String>| -> Vec<(String, u32)> {
        let mut dims: Vec<(String, u32)> = Vec::new();
        for name in &names {
            match resolve(name, decode) {
                Some(v) => dims.push((name.clone(), v)),
                None => {
                    if !unresolved.contains(name) {
                        unresolved.push(name.clone());
                    }
                }
            }
        }
        // total_sequence_length = sequence_length + past_sequence_length.
        let seq = dims
            .iter()
            .find(|(n, _)| n == "sequence_length" || n == "decoder_sequence_length")
            .map(|(_, v)| *v);
        let past = dims
            .iter()
            .find(|(n, _)| n.contains("past"))
            .map(|(_, v)| *v);
        if let (Some(seq), Some(past)) = (seq, past) {
            for (n, v) in dims.iter_mut() {
                if n.contains("total_sequence") && !overrides.contains_key(n) {
                    *v = seq + past;
                }
            }
        }
        // encoder_sequence_length_out mirrors encoder_sequence_length.
        let enc = dims
            .iter()
            .find(|(n, _)| n == "encoder_sequence_length")
            .map(|(_, v)| *v);
        if let Some(enc) = enc {
            for (n, v) in dims.iter_mut() {
                if n == "encoder_sequence_length_out" && !overrides.contains_key(n) {
                    *v = enc;
                }
            }
        }
        dims
    };

    let heavy = size >= HEAVY_BYTES;
    let gate = inputs
        .iter()
        .find(|i| i.is_bool && i.name == "use_cache_branch")
        .map(|i| i.name.clone());
    let has_cache = names
        .iter()
        .any(|n| n.to_ascii_lowercase().contains("past"));
    let mut entries = Vec::new();
    let mut entry = |decode: bool, gate_value: Option<i64>, unresolved: &mut Vec<String>| {
        let mut pin_inputs = pins.to_vec();
        if let (Some(gate), Some(value)) = (&gate, gate_value) {
            pin_inputs.push((gate.clone(), value));
        }
        entries.push(ManifestEntry {
            file: file.to_string(),
            heavy,
            pin_inputs,
            override_dims: build(decode, unresolved),
        });
    };
    if gate.is_some() {
        entry(false, Some(0), &mut unresolved);
        entry(true, Some(1), &mut unresolved);
    } else {
        entry(false, None, &mut unresolved);
        if decode_step && has_cache {
            entry(true, None, &mut unresolved);
        }
    }
    Proposal {
        entries,
        unresolved,
    }
}

/// Fetch (or load from `cache_dir`) the skeleton of `file`
/// (`<org>--<repo>/onnx/<name>.onnx`) and decode it.
pub fn load_skeleton(file: &str, cache_dir: &Path) -> Result<ModelProto, String> {
    let cached = cache_dir.join(file);
    let bytes = match std::fs::read(&cached) {
        Ok(bytes) => bytes,
        Err(_) => {
            let (bytes, _) = strip_model(HubSource::open(file)?, KEEP_BYTES)?;
            if let Some(parent) = cached.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let _ = std::fs::write(&cached, &bytes);
            bytes
        }
    };
    ModelProto::decode(&bytes[..]).map_err(|e| format!("{file}: decode skeleton: {e}"))
}

/// Convert a skeleton with the entry's options; `Ok` when the ORT build succeeds.
pub fn try_entry(model: ModelProto, entry: &ManifestEntry) -> Result<(), String> {
    convert_model_proto(model, &entry.to_options())
        .map(|_| ())
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(name: &str, dims: &[&str], is_bool: bool) -> GraphInput {
        GraphInput {
            name: name.to_string(),
            free_dims: dims.iter().map(|d| d.to_string()).collect(),
            is_bool,
        }
    }

    #[test]
    fn merged_decoder_gets_prefill_and_decode_entries() {
        let inputs = vec![
            input("input_ids", &["batch_size", "sequence_length"], false),
            input(
                "attention_mask",
                &["batch_size", "total_sequence_length"],
                false,
            ),
            input(
                "past_key_values.0.key",
                &["batch_size", "past_sequence_length"],
                false,
            ),
            input("use_cache_branch", &[], true),
        ];
        let p = propose_entries(
            "x--y/onnx/decoder_model_merged.onnx",
            2 << 30,
            &inputs,
            &HashMap::new(),
            &[],
            false,
        );
        assert!(p.unresolved.is_empty());
        assert_eq!(p.entries.len(), 2);
        let prefill = &p.entries[0];
        assert!(prefill.heavy);
        assert_eq!(
            prefill.pin_inputs,
            vec![("use_cache_branch".to_string(), 0)]
        );
        assert_eq!(
            prefill.override_dims,
            vec![
                ("batch_size".to_string(), 1),
                ("sequence_length".to_string(), 128),
                ("total_sequence_length".to_string(), 128),
                ("past_sequence_length".to_string(), 0),
            ]
        );
        let decode = &p.entries[1];
        assert_eq!(decode.pin_inputs, vec![("use_cache_branch".to_string(), 1)]);
        assert_eq!(
            decode.override_dims,
            vec![
                ("batch_size".to_string(), 1),
                ("sequence_length".to_string(), 1),
                ("total_sequence_length".to_string(), 17),
                ("past_sequence_length".to_string(), 16),
            ]
        );
        assert_eq!(
            prefill.to_json_line(),
            r#"{"file": "x--y/onnx/decoder_model_merged.onnx", "heavy": true, "pin_inputs": {"use_cache_branch": 0}, "override_dims": {"batch_size": 1, "sequence_length": 128, "total_sequence_length": 128, "past_sequence_length": 0}}"#
        );
    }

    #[test]
    fn unknown_dims_are_reported_and_overrides_win() {
        assert_eq!(default_dim("encoder_sequence_length / 2"), None);
        let inputs = vec![input("pixel_values", &["s0", "s1", "s2"], false)];
        let p = propose_entries(
            "a--b/onnx/vision.onnx",
            10,
            &inputs,
            &HashMap::new(),
            &[],
            false,
        );
        assert_eq!(p.unresolved, vec!["s0", "s1", "s2"]);
        assert!(p.entries[0].override_dims.is_empty());
        let overrides = HashMap::from([
            ("s0".to_string(), 1),
            ("s1".to_string(), 1024),
            ("s2".to_string(), 1024),
        ]);
        let p = propose_entries("a--b/onnx/vision.onnx", 10, &inputs, &overrides, &[], false);
        assert!(p.unresolved.is_empty());
        assert_eq!(p.entries[0].override_dims.len(), 3);
        assert!(!p.entries[0].heavy);
    }
}
