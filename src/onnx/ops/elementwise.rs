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

// Elementwise binary operators: Add, Sub, Mul, Div, Pow, Mod

use crate::onnx::builder::{map_op_error, OnnxBuilder};
use crate::onnx::convert::{sanitize_identifier, OnnxError};
use crate::onnx::ops::{ConversionContext, ConversionResult, OpHandler};
use crate::protos::onnx::NodeProto;
use rustnn::mlcontext::MLOperand;

pub struct ElementwiseHandler;

impl OpHandler for ElementwiseHandler {
    fn supports(&self, op_type: &str) -> bool {
        matches!(
            op_type,
            "Add" | "Sub" | "Mul" | "Div" | "Pow" | "Min" | "Max" | "Mod"
        )
    }

    fn convert(
        &self,
        node: &NodeProto,
        _context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let op_type = node.op_type.as_str();
        let node_name = if !node.name.is_empty() {
            node.name.as_str().to_string()
        } else {
            "unnamed".to_string()
        };

        let inputs = node.input.as_slice();
        if inputs.is_empty() {
            return Err(OnnxError::InvalidShape(format!(
                "{op_type} expects at least 1 input"
            )));
        }

        let output_name = if node.output.as_slice().is_empty() {
            format!("{}_output", node_name)
        } else {
            sanitize_identifier(&node.output.as_slice()[0].to_string())
        };

        if op_type == "Mod" {
            return convert_mod(node, b, &node_name, &output_name);
        }

        let out = if inputs.len() == 1 {
            let input0 = b.resolve_operand(&inputs[0])?;
            let opts = OnnxBuilder::labeled_options(&output_name);
            b.builder
                .identity_with_options(input0, opts)
                .map_err(map_op_error)?
        } else {
            let mut acc = b.resolve_operand(&inputs[0])?;
            for (step, input_name) in inputs[1..].iter().enumerate() {
                let next = b.resolve_operand(input_name)?;
                let label = if step + 2 == inputs.len() {
                    output_name.clone()
                } else {
                    format!("{output_name}__fold_{step}")
                };
                let opts = OnnxBuilder::labeled_options(&label);
                acc = emit_binary(op_type, b, acc, next, opts, &node_name)?;
            }
            acc
        };

        if let Some(output) = node.output.as_slice().first() {
            b.record_operand(&[output.as_str(), &output_name], out);
        } else {
            b.record_operand(&[&output_name], out);
        }

        Ok(ConversionResult::default())
    }
}

/// ONNX Mod as `A - B * q(A / B)` where `q` is `floor` for `fmod=0`
/// (integer/Python semantics, sign follows the divisor) and truncation
/// (`sign(x) * floor(|x|)`) for `fmod=1` (C fmod, sign follows the dividend).
fn convert_mod(
    node: &NodeProto,
    b: &mut OnnxBuilder<'_, '_, '_>,
    node_name: &str,
    output_name: &str,
) -> Result<ConversionResult, OnnxError> {
    let inputs = node.input.as_slice();
    if inputs.len() < 2 {
        return Err(OnnxError::InvalidShape("Mod expects 2 inputs".to_string()));
    }

    let mut fmod = 0i64;
    for attr in node.attribute.as_slice() {
        if attr.name == "fmod" {
            fmod = attr.i;
        }
    }
    if fmod != 0 && fmod != 1 {
        return Err(OnnxError::unsupported_op(
            format!("Mod (fmod={fmod})"),
            node_name.to_string(),
        ));
    }

    let a = b.resolve_operand(&inputs[0])?;
    let b_in = b.resolve_operand(&inputs[1])?;

    let div_label = format!("{output_name}__div");
    let div_opts = OnnxBuilder::labeled_options(&div_label);
    let quotient = b
        .builder
        .div_with_options(a, b_in, div_opts)
        .map_err(map_op_error)?;

    let floored = if fmod == 0 {
        let floor_label = format!("{output_name}__floor");
        let floor_opts = OnnxBuilder::labeled_options(&floor_label);
        b.builder
            .floor_with_options(quotient, floor_opts)
            .map_err(map_op_error)?
    } else {
        // trunc(q) = sign(q) * floor(|q|)
        let abs = b
            .builder
            .abs_with_options(
                quotient,
                OnnxBuilder::labeled_options(&format!("{output_name}__absq")),
            )
            .map_err(map_op_error)?;
        let abs_floor = b
            .builder
            .floor_with_options(
                abs,
                OnnxBuilder::labeled_options(&format!("{output_name}__floorq")),
            )
            .map_err(map_op_error)?;
        let sign = b
            .builder
            .sign_with_options(
                quotient,
                OnnxBuilder::labeled_options(&format!("{output_name}__signq")),
            )
            .map_err(map_op_error)?;
        b.builder
            .mul_with_options(
                sign,
                abs_floor,
                OnnxBuilder::labeled_options(&format!("{output_name}__truncq")),
            )
            .map_err(map_op_error)?
    };

    let b_operand = b.resolve_operand(&inputs[1])?;
    let mul_label = format!("{output_name}__mul");
    let mul_opts = OnnxBuilder::labeled_options(&mul_label);
    let product = b
        .builder
        .mul_with_options(b_operand, floored, mul_opts)
        .map_err(map_op_error)?;

    let a_operand = b.resolve_operand(&inputs[0])?;
    let sub_opts = OnnxBuilder::labeled_options(output_name);
    let out = b
        .builder
        .sub_with_options(a_operand, product, sub_opts)
        .map_err(map_op_error)?;

    if let Some(output) = node.output.as_slice().first() {
        b.record_operand(&[output.as_str(), output_name], out);
    } else {
        b.record_operand(&[output_name], out);
    }

    Ok(ConversionResult::default())
}

fn emit_binary(
    op_type: &str,
    b: &mut OnnxBuilder<'_, '_, '_>,
    a: MLOperand,
    b_in: MLOperand,
    opts: rustnn::operator_options::MLOperatorOptions,
    node_name: &str,
) -> Result<MLOperand, OnnxError> {
    Ok(match op_type {
        "Add" => b
            .builder
            .add_with_options(a, b_in, opts)
            .map_err(map_op_error)?,
        "Sub" => b
            .builder
            .sub_with_options(a, b_in, opts)
            .map_err(map_op_error)?,
        "Mul" => b
            .builder
            .mul_with_options(a, b_in, opts)
            .map_err(map_op_error)?,
        "Div" => b
            .builder
            .div_with_options(a, b_in, opts)
            .map_err(map_op_error)?,
        "Pow" => b
            .builder
            .pow_with_options(a, b_in, opts)
            .map_err(map_op_error)?,
        "Min" => b
            .builder
            .min_with_options(a, b_in, opts)
            .map_err(map_op_error)?,
        "Max" => b
            .builder
            .max_with_options(a, b_in, opts)
            .map_err(map_op_error)?,
        _ => {
            return Err(OnnxError::unsupported_op(
                op_type.to_string(),
                node_name.to_string(),
            ))
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protos::onnx::NodeProto;

    fn create_test_node(op_type: &str, inputs: Vec<&str>, outputs: Vec<&str>) -> NodeProto {
        NodeProto {
            op_type: op_type.to_string(),
            name: format!("test_{}", op_type.to_lowercase()),
            input: inputs.iter().map(|s| s.to_string()).collect(),
            output: outputs.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        }
    }

    #[test]
    fn test_elementwise_handler_supports() {
        let handler = ElementwiseHandler;
        assert!(handler.supports("Add"));
        assert!(handler.supports("Sub"));
        assert!(handler.supports("Mul"));
        assert!(handler.supports("Div"));
        assert!(handler.supports("Pow"));
        assert!(handler.supports("Min"));
        assert!(handler.supports("Max"));
        assert!(handler.supports("Mod"));
        assert!(!handler.supports("MatMul"));
    }

    #[test]
    fn test_convert_add() {
        let handler = ElementwiseHandler;
        let node = create_test_node("Add", vec!["a", "b"], vec!["c"]);
        crate::onnx::ops::convert_with_test_builder(&handler, &node).unwrap();
    }

    #[test]
    fn test_convert_mul() {
        let handler = ElementwiseHandler;
        let node = create_test_node("Mul", vec!["x", "y"], vec!["z"]);
        crate::onnx::ops::convert_with_test_builder(&handler, &node).unwrap();
    }

    #[test]
    fn test_convert_div() {
        let handler = ElementwiseHandler;
        let node = create_test_node("Div", vec!["a", "b"], vec!["c"]);
        crate::onnx::ops::convert_with_test_builder(&handler, &node).unwrap();
    }

    #[test]
    fn test_convert_min() {
        let handler = ElementwiseHandler;
        let node = create_test_node("Min", vec!["x", "y"], vec!["z"]);
        crate::onnx::ops::convert_with_test_builder(&handler, &node).unwrap();
    }

    #[test]
    fn test_convert_max() {
        let handler = ElementwiseHandler;
        let node = create_test_node("Max", vec!["a", "b"], vec!["c"]);
        crate::onnx::ops::convert_with_test_builder(&handler, &node).unwrap();
    }

    #[test]
    fn test_convert_variadic_min_three_inputs() {
        let handler = ElementwiseHandler;
        let node = create_test_node("Min", vec!["a", "b", "c"], vec!["out"]);
        crate::onnx::ops::convert_with_test_builder(&handler, &node).unwrap();
    }

    #[test]
    fn test_convert_variadic_max_four_inputs() {
        let handler = ElementwiseHandler;
        let node = create_test_node("Max", vec!["a", "b", "c", "d"], vec!["out"]);
        crate::onnx::ops::convert_with_test_builder(&handler, &node).unwrap();
    }
}
