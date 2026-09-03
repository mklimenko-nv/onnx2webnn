/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Whole-model conversion sweep driven by `tests/models/manifest.json`.
//!
//! Each entry names a transformers.js ONNX export plus the dimension overrides
//! and pinned inputs to convert it with. `O2W_MODELS` selects where the models
//! come from:
//!
//! - `hub` (default when `CI` is set): weight-stripped skeletons are built
//!   straight from the Hugging Face Hub with range requests, see
//!   `common::skeleton`, and kept in `target/model-skeletons` (or
//!   `O2W_SKELETON_CACHE`) for the next run. No weights are downloaded.
//! - `dir=<path>`: full downloads laid out as `<org>--<repo>/onnx/<file>.onnx`.
//! - `strip=<path>`: the same local files run through the skeleton scanner
//!   (exercises the scanner offline).
//!
//! Unset outside CI, the sweep is skipped and says so. `O2W_MANIFEST` runs a
//! different manifest file.
//!
//! Skeletons are fetched with `O2W_MODEL_FETCH_JOBS` threads (default 8; the
//! Hub round trips dominate). Light entries then convert on a thread pool of
//! `O2W_MODEL_TEST_JOBS` (default 4; conversions are memory-bound). Entries
//! marked `"heavy": true` (multi-GB weights, >10 GB peak RSS) run one at a
//! time afterwards, or are skipped when `O2W_MODEL_TEST_SKIP_HEAVY` is set.

#[allow(dead_code)]
mod common;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use common::skeleton::{strip_model, FileSource, HubSource, KEEP_BYTES};
use onnx2webnn::protos::onnx::ModelProto;
use onnx2webnn::{convert_model_proto, convert_onnx, ConvertOptions};
use prost::Message;
use serde::Deserialize;

/// Large models recurse deeply in shape inference; the default 2 MB thread
/// stack is not enough.
const WORKER_STACK_BYTES: usize = 256 << 20;

#[derive(Deserialize)]
struct Entry {
    file: String,
    #[serde(default)]
    heavy: bool,
    /// Reason this model cannot build on the CoreML backend (e.g. a hard
    /// backend limit like max tensor rank 5). Skipped when built with
    /// `--features coreml`; still exercised on the ORT backend.
    #[serde(default)]
    coreml_unsupported: Option<String>,
    #[serde(default)]
    override_dims: HashMap<String, u32>,
    #[serde(default)]
    pin_inputs: HashMap<String, i64>,
}

enum Source {
    Hub,
    Dir(PathBuf),
    StripDir(PathBuf),
}

fn source() -> Option<Source> {
    match std::env::var("O2W_MODELS").ok().filter(|v| !v.is_empty()) {
        Some(v) if v == "hub" => Some(Source::Hub),
        Some(v) if v.starts_with("dir=") => Some(Source::Dir(PathBuf::from(&v[4..]))),
        Some(v) if v.starts_with("strip=") => Some(Source::StripDir(PathBuf::from(&v[6..]))),
        Some(v) => panic!("O2W_MODELS={v}: expected hub, dir=<path> or strip=<path>"),
        None => std::env::var_os("CI").map(|_| Source::Hub),
    }
}

/// Where Hub-built skeletons are kept between runs (~39 MB for the manifest).
fn cache_dir() -> PathBuf {
    std::env::var_os("O2W_SKELETON_CACHE")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("target/model-skeletons"))
}

fn env_jobs(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n| n > 0)
        .unwrap_or(default)
}

type Skeleton = Result<(Arc<Vec<u8>>, String), String>;

struct Sweep {
    source: Source,
    /// Stripped models, built once per file even when several entries share it.
    skeletons: Mutex<HashMap<String, Arc<OnceLock<Skeleton>>>>,
    passed: AtomicUsize,
    failures: Mutex<Vec<String>>,
}

impl Sweep {
    fn skeleton(&self, file: &str) -> Skeleton {
        let cell = self
            .skeletons
            .lock()
            .unwrap()
            .entry(file.to_string())
            .or_default()
            .clone();
        cell.get_or_init(|| {
            let started = std::time::Instant::now();
            let cached = matches!(self.source, Source::Hub).then(|| cache_dir().join(file));
            if let Some(bytes) = cached.as_ref().and_then(|path| std::fs::read(path).ok()) {
                let note = format!("skeleton {:.2} MB (cached)", bytes.len() as f64 / 1e6);
                return Ok((Arc::new(bytes), note));
            }
            let (bytes, stats) = match &self.source {
                Source::Hub => strip_model(HubSource::open(file)?, KEEP_BYTES)?,
                Source::StripDir(dir) => strip_model(FileSource::open(&dir.join(file))?, KEEP_BYTES)?,
                Source::Dir(_) => unreachable!("full models are converted from disk"),
            };
            if let Some(path) = cached {
                // Best effort: a failed cache write only costs the next run a rebuild.
                let part = path.with_extension("part");
                let _ = path
                    .parent()
                    .map(std::fs::create_dir_all)
                    .transpose()
                    .and_then(|_| std::fs::write(&part, &bytes))
                    .and_then(|_| std::fs::rename(&part, &path));
            }
            let note = format!(
                "skeleton {:.2} MB: stripped {} tensors ({:.0} MB), read {:.1} MB in {} requests, {:.1} s",
                bytes.len() as f64 / 1e6,
                stats.stripped,
                stats.stripped_bytes as f64 / 1e6,
                stats.fetched as f64 / 1e6,
                stats.requests,
                started.elapsed().as_secs_f32()
            );
            Ok((Arc::new(bytes), note))
        })
        .clone()
    }

    fn convert(&self, idx: usize, entry: &Entry) {
        let label = format!(
            "#{idx} {} dims={:?} pins={:?}",
            entry.file, entry.override_dims, entry.pin_inputs
        );
        let options = ConvertOptions {
            free_dim_overrides: entry.override_dims.clone(),
            optimize: true,
            experimental_dynamic_inputs: false,
            pinned_inputs: entry.pin_inputs.clone(),
            zero_fill_missing_external_data: true,
        };
        let started = std::time::Instant::now();
        let result = match &self.source {
            Source::Dir(dir) => convert_onnx(dir.join(&entry.file), options)
                .map(|_| String::new())
                .map_err(|e| e.to_string()),
            Source::Hub | Source::StripDir(_) => {
                self.skeleton(&entry.file).and_then(|(bytes, note)| {
                    let model = ModelProto::decode(&bytes[..])
                        .map_err(|e| format!("decode skeleton: {e}"))?;
                    convert_model_proto(model, &options)
                        .map(|_| note)
                        .map_err(|e| e.to_string())
                })
            }
        };
        match result {
            Ok(note) => {
                self.passed.fetch_add(1, Ordering::Relaxed);
                eprintln!("ok   {:>6} ms  {label}", started.elapsed().as_millis());
                if !note.is_empty() {
                    eprintln!("     {note}");
                }
            }
            Err(err) => {
                eprintln!(
                    "FAIL {:>6} ms  {label}\n     {err}",
                    started.elapsed().as_millis()
                );
                self.failures
                    .lock()
                    .unwrap()
                    .push(format!("{label}: {err}"));
            }
        }
    }

    /// Build (or load from cache) the skeletons of `entries` with `workers`
    /// threads; failures surface later when the entry converts.
    fn prefetch(&self, entries: &[(usize, &Entry)], workers: usize) {
        if matches!(self.source, Source::Dir(_)) {
            return;
        }
        let mut files: Vec<&str> = entries.iter().map(|(_, e)| e.file.as_str()).collect();
        files.sort_unstable();
        files.dedup();
        let queue = Mutex::new(files);
        std::thread::scope(|scope| {
            for _ in 0..workers {
                scope.spawn(|| loop {
                    let Some(file) = queue.lock().unwrap().pop() else {
                        break;
                    };
                    let _ = self.skeleton(file);
                });
            }
        });
    }

    /// Run `entries` with at most `workers` conversions in flight.
    fn run(&self, entries: Vec<(usize, &Entry)>, workers: usize) {
        let queue = Mutex::new(entries);
        std::thread::scope(|scope| {
            for _ in 0..workers {
                std::thread::Builder::new()
                    .stack_size(WORKER_STACK_BYTES)
                    .spawn_scoped(scope, || loop {
                        let Some((idx, entry)) = queue.lock().unwrap().pop() else {
                            break;
                        };
                        self.convert(idx, entry);
                    })
                    .expect("spawn sweep worker");
            }
        });
    }
}

#[test]
fn manifest_models_convert_and_build() {
    // O2W_MANIFEST points at another manifest (e.g. candidates under evaluation).
    let manifest_path = std::env::var_os("O2W_MANIFEST")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/models/manifest.json")
        });
    let manifest = std::fs::read_to_string(&manifest_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", manifest_path.display()));
    let entries: Vec<Entry> = serde_json::from_str(&manifest).expect("parse manifest");
    let Some(source) = source() else {
        eprintln!("skipping model sweep: set O2W_MODELS=hub, dir=<path> or strip=<path>");
        return;
    };
    let skip_heavy = std::env::var_os("O2W_MODEL_TEST_SKIP_HEAVY").is_some();
    let entries: Vec<Entry> = entries
        .into_iter()
        .filter(|e| {
            if cfg!(feature = "coreml") {
                if let Some(reason) = &e.coreml_unsupported {
                    eprintln!("skipping {} on CoreML: {reason}", e.file);
                    return false;
                }
            }
            true
        })
        .collect();

    let sweep = Sweep {
        source,
        skeletons: Mutex::new(HashMap::new()),
        passed: AtomicUsize::new(0),
        failures: Mutex::new(Vec::new()),
    };
    let (heavy, light): (Vec<_>, Vec<_>) = entries.iter().enumerate().partition(|(_, e)| e.heavy);

    let started = std::time::Instant::now();
    let fetch_jobs = env_jobs("O2W_MODEL_FETCH_JOBS", 8);
    sweep.prefetch(&light, fetch_jobs);
    if !skip_heavy {
        sweep.prefetch(&heavy, fetch_jobs);
    }
    let fetch_secs = started.elapsed().as_secs_f32();
    sweep.run(light, env_jobs("O2W_MODEL_TEST_JOBS", 4));
    let light_secs = started.elapsed().as_secs_f32();
    if skip_heavy {
        eprintln!(
            "skipping {} heavy entries (O2W_MODEL_TEST_SKIP_HEAVY)",
            heavy.len()
        );
    } else {
        sweep.run(heavy, 1);
    }

    let failures = sweep.failures.into_inner().unwrap();
    eprintln!(
        "model sweep: {} passed, {} failed; fetch {:.1} s, light done at {:.1} s, total {:.1} s",
        sweep.passed.load(Ordering::Relaxed),
        failures.len(),
        fetch_secs,
        light_secs,
        started.elapsed().as_secs_f32()
    );
    assert!(
        failures.is_empty(),
        "model conversions failed:\n{}",
        failures.join("\n")
    );
}

/// The scanner must keep the graph intact, keep small initializers inline
/// and turn large ones into external references the converter zero-fills.
#[test]
fn skeleton_scanner_strips_large_initializers_only() {
    use onnx2webnn::test_models::prelude::*;

    let big: Vec<f32> = (0..4096).map(|i| i as f32).collect(); // 16 KB
    let model_proto = model(
        17,
        graph(
            "skel",
            vec![f32_input("x", &[2, 4096])],
            vec![f32_output("y", &[2, 4096])],
            vec![
                node("Add", "add", &["x", "w"], &["t"], &[]),
                node("Mul", "mul", &["t", "s"], &["y"], &[]),
            ],
            vec![f32_init("w", &[4096], &big), f32_init("s", &[], &[2.0])],
        ),
    );
    let bytes = model_proto.encode_to_vec();

    struct Mem(Vec<u8>);
    impl common::skeleton::ByteSource for Mem {
        fn len(&self) -> u64 {
            self.0.len() as u64
        }
        fn read_at(&mut self, start: u64, len: usize) -> Result<Vec<u8>, String> {
            Ok(self.0[start as usize..start as usize + len].to_vec())
        }
    }

    let (stripped, stats) = strip_model(Mem(bytes), KEEP_BYTES).unwrap();
    assert_eq!((stats.stripped, stats.kept), (1, 1));
    let skeleton = ModelProto::decode(&stripped[..]).expect("skeleton decodes");
    let g = skeleton.graph.as_ref().unwrap();
    assert_eq!(g.node.len(), 2);
    let w = g.initializer.iter().find(|t| t.name == "w").unwrap();
    assert_eq!(w.data_location, 1);
    assert!(w.raw_data.is_empty());
    assert_eq!(w.dims, vec![4096]);
    let s = g.initializer.iter().find(|t| t.name == "s").unwrap();
    assert_eq!(s.float_data, vec![2.0]);

    // Zero-filled, the skeleton still converts and builds in ORT.
    let options = ConvertOptions {
        zero_fill_missing_external_data: true,
        ..ConvertOptions::default()
    };
    convert_model_proto(skeleton, &options).expect("skeleton converts");
}
