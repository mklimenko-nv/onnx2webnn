/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Tarek Ziadé <tarek@ziade.org>
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

use clap::{Parser, Subcommand};
use std::collections::HashMap;
use std::path::Path;

use onnx2webnn::convert_onnx;
use onnx2webnn::ConvertOptions;

#[derive(Parser)]
#[command(name = "onnx2webnn")]
#[command(
    about = "Convert ONNX models to WebNN via MLGraphBuilder (ORT validation)",
    long_about = None
)]
struct Cli {
    /// Enable debug output
    #[arg(long, global = true)]
    debug: bool,

    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Lower ONNX to MLGraphBuilder and validate with rustnn ORT (CPU build)
    Convert {
        /// Input ONNX model path
        #[arg(long)]
        input: String,

        /// Override a symbolic dimension, e.g. batch_size=1 (repeatable)
        #[arg(long = "override-dim")]
        override_dims: Vec<String>,

        /// JSON file with dimension overrides (freeDimensionOverrides object)
        #[arg(long = "override-dims-file")]
        override_dims_file: Option<String>,

        /// Freeze a graph input to a constant, e.g. use_cache_branch=false
        /// (repeatable). Constant `If` gates are then inlined.
        #[arg(long = "pin-input")]
        pin_inputs: Vec<String>,

        /// Zero-fill external tensors whose data file is missing (weight-stripped
        /// skeleton models; graph structure only)
        #[arg(long = "allow-missing-external-data")]
        allow_missing_external_data: bool,

        /// Enable constant folding (Shape/Gather/Concat/Reshape pipelines)
        #[arg(long)]
        optimize: bool,

        /// Preserve unresolved symbolic input dims as dynamic metadata (experimental)
        #[arg(long)]
        experimental_dynamic_inputs: bool,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    if cli.debug {
        onnx2webnn::debug::enable();
    }

    match cli.cmd {
        Command::Convert {
            input,
            override_dims,
            override_dims_file,
            pin_inputs,
            allow_missing_external_data,
            optimize,
            experimental_dynamic_inputs,
        } => {
            let mut free_dim_overrides = if let Some(path) = override_dims_file {
                let content = std::fs::read_to_string(&path)?;
                let json: serde_json::Value = serde_json::from_str(&content)?;
                let overrides = json
                    .get("freeDimensionOverrides")
                    .unwrap_or(&json)
                    .as_object()
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "override-dims-file must be a JSON object (optionally nested under freeDimensionOverrides)"
                        )
                    })?;

                let mut map = HashMap::new();
                for (name, value) in overrides {
                    let parsed = value.as_u64().ok_or_else(|| {
                        anyhow::anyhow!(
                            "override value for '{}' must be an integer, got {}",
                            name,
                            value
                        )
                    })?;
                    map.insert(name.to_string(), parsed as u32);
                }
                map
            } else {
                HashMap::new()
            };

            for override_dim in override_dims {
                let parts: Vec<&str> = override_dim.split('=').collect();
                if parts.len() != 2 {
                    return Err(anyhow::anyhow!(
                        "Invalid override-dim format: '{}'. Expected NAME=VALUE",
                        override_dim
                    ));
                }
                let name = parts[0].trim().to_string();
                let value: u32 = parts[1]
                    .trim()
                    .parse()
                    .map_err(|_| anyhow::anyhow!("Invalid dimension value: '{}'", parts[1]))?;
                free_dim_overrides.insert(name, value);
            }

            let input_path = Path::new(&input);

            let mut pinned_inputs = HashMap::new();
            for spec in pin_inputs {
                let (name, value) = onnx2webnn::onnx::convert::parse_pinned_input(&spec)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                pinned_inputs.insert(name, value);
            }

            let options = ConvertOptions {
                free_dim_overrides,
                optimize,
                experimental_dynamic_inputs,
                pinned_inputs,
                zero_fill_missing_external_data: allow_missing_external_data,
            };

            let _validated = convert_onnx(input_path.to_str().unwrap(), options)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            // stdout (not stderr): PowerShell treats native stderr as NativeCommandError
            println!("✓ ORT graph build succeeded for {}", input);
        }
    }

    Ok(())
}
