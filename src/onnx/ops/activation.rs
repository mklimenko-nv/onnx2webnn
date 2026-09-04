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

// Activation and unary math operators. Plain unary ops (Relu, Sqrt, Floor, ...) plus parametric
// activations (Elu, LeakyRelu, HardSigmoid), Clip -> clamp, and PRelu (binary).

use crate::onnx::builder::{map_ast_data_type, map_op_error, OnnxBuilder};
use crate::onnx::convert::{sanitize_identifier, OnnxError};
use crate::onnx::ops::{ConversionContext, ConversionResult, OpHandler};
use crate::protos::onnx::{NodeProto, TensorProto, TensorProto_DataType};
use half::f16;
use rustnn::mlcontext::MLOperand;
use rustnn::operator_options::{
    MLClampOptions, MLEluOptions, MLHardSigmoidOptions, MLLeakyReluOptions,
};
use rustnn::DataType;

pub struct ActivationHandler;

impl OpHandler for ActivationHandler {
    fn supports(&self, op_type: &str) -> bool {
        matches!(
            op_type,
            "Relu"
                | "Gelu"
                | "Tanh"
                | "Sigmoid"
                | "Sqrt"
                | "Exp"
                | "Log"
                | "Abs"
                | "Neg"
                | "Erf"
                | "Cos"
                | "Sin"
                | "Identity"
                // Unary math (no attributes)
                | "Floor"
                | "Ceil"
                | "Sign"
                | "Tan"
                | "Reciprocal"
                | "Round"
                | "HardSwish"
                | "Softplus"
                | "Softsign"
                // Parametric activations
                | "Elu"
                | "LeakyRelu"
                | "HardSigmoid"
                | "Clip"
                | "PRelu"
                // Decomposed activations
                | "Swish"
                | "Celu"
                | "Selu"
                | "Mish"
                | "ThresholdedRelu"
                | "Sinh"
                | "Cosh"
                | "Asinh"
                | "Acosh"
                | "Atanh"
                | "Shrink"
        )
    }

    fn convert(
        &self,
        node: &NodeProto,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let op_type = node.op_type.as_str();
        let node_name = if !node.name.is_empty() {
            node.name.as_str().to_string()
        } else {
            "unnamed".to_string()
        };

        match op_type {
            "Clip" => return self.convert_clip(node, &node_name, context, b),
            "PRelu" => return self.convert_prelu(node, &node_name, b),
            "Elu" | "LeakyRelu" | "HardSigmoid" => {
                return self.convert_parametric(node, &node_name, op_type, b)
            }
            "Swish" => return self.convert_swish(node, &node_name, b),
            "Mish" => return self.convert_mish(node, &node_name, b),
            "Celu" => return self.convert_celu(node, &node_name, context, b),
            "Selu" => return self.convert_selu(node, &node_name, context, b),
            "ThresholdedRelu" => {
                return self.convert_thresholded_relu(node, &node_name, context, b)
            }
            "Sinh" => return self.convert_sinh(node, &node_name, context, b),
            "Cosh" => return self.convert_cosh(node, &node_name, context, b),
            "Asinh" => return self.convert_asinh(node, &node_name, context, b),
            "Acosh" => return self.convert_acosh(node, &node_name, context, b),
            "Atanh" => return self.convert_atanh(node, &node_name, context, b),
            "Shrink" => return self.convert_shrink(node, &node_name, context, b),
            _ => {}
        }

        let webnn_op = match op_type {
            "Relu" => "relu",
            "Gelu" => "gelu",
            "Tanh" => "tanh",
            "Sigmoid" => "sigmoid",
            "Sqrt" => "sqrt",
            "Exp" => "exp",
            "Log" => "log",
            "Abs" => "abs",
            "Neg" => "neg",
            "Erf" => "erf",
            "Cos" => "cos",
            "Sin" => "sin",
            "Identity" => "identity",
            "Floor" => "floor",
            "Ceil" => "ceil",
            "Sign" => "sign",
            "Tan" => "tan",
            "Reciprocal" => "reciprocal",
            "Round" => "round_even",
            "HardSwish" => "hard_swish",
            "Softplus" => "softplus",
            "Softsign" => "softsign",
            _ => return Err(OnnxError::unsupported_op(op_type.to_string(), node_name)),
        };

        self.convert_unary(node, &node_name, webnn_op, context, b)
    }
}

impl ActivationHandler {
    fn convert_unary(
        &self,
        node: &NodeProto,
        node_name: &str,
        webnn_op: &str,
        _context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() != 1 {
            return Err(OnnxError::InvalidShape(format!(
                "{} expects 1 input, got {}",
                webnn_op,
                inputs.len()
            )));
        }

        let output_name = output_name_for(node, node_name);
        let input0 = b.resolve_operand(&inputs[0])?;
        let opts = OnnxBuilder::labeled_options(&output_name);
        let out = emit_unary(webnn_op, b, input0, opts, node_name)?;
        record_output(b, node, &output_name, out);
        Ok(ConversionResult::default())
    }

    /// Parametric single-input activations: Elu, LeakyRelu, HardSigmoid.
    fn convert_parametric(
        &self,
        node: &NodeProto,
        node_name: &str,
        op_type: &str,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() != 1 {
            return Err(OnnxError::InvalidShape(format!(
                "{} expects 1 input, got {}",
                op_type,
                inputs.len()
            )));
        }

        let output_name = output_name_for(node, node_name);
        let label = output_name.clone();
        let input0 = b.resolve_operand(&inputs[0])?;

        let out = match op_type {
            "Elu" => {
                let alpha = attr_f64(node, "alpha").unwrap_or(1.0);
                b.builder
                    .elu_with_options(input0, MLEluOptions { label, alpha })
                    .map_err(map_op_error)?
            }
            "LeakyRelu" => {
                let alpha = attr_f64(node, "alpha").unwrap_or(0.01);
                b.builder
                    .leaky_relu_with_options(input0, MLLeakyReluOptions { label, alpha })
                    .map_err(map_op_error)?
            }
            "HardSigmoid" => {
                let alpha = attr_f64(node, "alpha").unwrap_or(0.2);
                let beta = attr_f64(node, "beta").unwrap_or(0.5);
                b.builder
                    .hard_sigmoid_with_options(input0, MLHardSigmoidOptions { label, alpha, beta })
                    .map_err(map_op_error)?
            }
            _ => {
                return Err(OnnxError::unsupported_op(
                    op_type.to_string(),
                    node_name.to_string(),
                ))
            }
        };

        record_output(b, node, &output_name, out);
        Ok(ConversionResult::default())
    }

    /// PRelu is binary: `prelu(input, slope)` with unidirectional slope broadcast.
    fn convert_prelu(
        &self,
        node: &NodeProto,
        node_name: &str,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() != 2 {
            return Err(OnnxError::InvalidShape(format!(
                "PRelu expects 2 inputs (input, slope), got {}",
                inputs.len()
            )));
        }

        let output_name = output_name_for(node, node_name);
        let input0 = b.resolve_operand(&inputs[0])?;
        let slope = b.resolve_operand(&inputs[1])?;
        let opts = OnnxBuilder::labeled_options(&output_name);
        let out = b
            .builder
            .prelu_with_options(input0, slope, opts)
            .map_err(map_op_error)?;

        record_output(b, node, &output_name, out);
        Ok(ConversionResult::default())
    }

    /// Clip -> `clamp`. Bounds come from `min`/`max` attributes (opset 6) or optional constant
    /// inputs (opset 11+). Non-constant bound inputs are rejected as unsupported.
    fn convert_clip(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.is_empty() {
            return Err(OnnxError::InvalidShape(
                "Clip expects at least 1 input".to_string(),
            ));
        }

        let output_name = output_name_for(node, node_name);
        let input0 = b.resolve_operand(&inputs[0])?;

        let mut min_value = attr_f64(node, "min");
        let mut max_value = attr_f64(node, "max");

        // Opset 11+: min/max are optional inputs (an empty name means "not provided").
        if inputs.len() >= 2 && !inputs[1].is_empty() {
            min_value = Some(clip_bound(context, &inputs[1], "min")?);
        }
        if inputs.len() >= 3 && !inputs[2].is_empty() {
            max_value = Some(clip_bound(context, &inputs[2], "max")?);
        }

        let opts = MLClampOptions {
            label: output_name.clone(),
            min_value: min_value.map(|v| serde_json::json!(v)),
            max_value: max_value.map(|v| serde_json::json!(v)),
        };
        let out = b
            .builder
            .clamp_with_options(input0, opts)
            .map_err(map_op_error)?;

        record_output(b, node, &output_name, out);
        Ok(ConversionResult::default())
    }

    /// Swish: `x * sigmoid(x)`.
    fn convert_swish(
        &self,
        node: &NodeProto,
        node_name: &str,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() != 1 {
            return Err(OnnxError::InvalidShape(format!(
                "Swish expects 1 input, got {}",
                inputs.len()
            )));
        }

        let output_name = output_name_for(node, node_name);
        let input0 = b.resolve_operand(&inputs[0])?;
        let sigmoid_label = step_label(&output_name, "sigmoid");
        let sigmoid = b
            .builder
            .sigmoid_with_options(input0, OnnxBuilder::labeled_options(&sigmoid_label))
            .map_err(map_op_error)?;
        let out = b
            .builder
            .mul_with_options(
                b.resolve_operand(&inputs[0])?,
                sigmoid,
                OnnxBuilder::labeled_options(&output_name),
            )
            .map_err(map_op_error)?;

        record_output(b, node, &output_name, out);
        Ok(ConversionResult::default())
    }

    /// Mish: `x * tanh(softplus(x))`.
    fn convert_mish(
        &self,
        node: &NodeProto,
        node_name: &str,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() != 1 {
            return Err(OnnxError::InvalidShape(format!(
                "Mish expects 1 input, got {}",
                inputs.len()
            )));
        }

        let output_name = output_name_for(node, node_name);
        let input0 = b.resolve_operand(&inputs[0])?;
        let softplus_label = step_label(&output_name, "softplus");
        let softplus = b
            .builder
            .softplus_with_options(input0, OnnxBuilder::labeled_options(&softplus_label))
            .map_err(map_op_error)?;
        let tanh_label = step_label(&output_name, "tanh");
        let tanh_out = b
            .builder
            .tanh_with_options(softplus, OnnxBuilder::labeled_options(&tanh_label))
            .map_err(map_op_error)?;
        let out = b
            .builder
            .mul_with_options(
                b.resolve_operand(&inputs[0])?,
                tanh_out,
                OnnxBuilder::labeled_options(&output_name),
            )
            .map_err(map_op_error)?;

        record_output(b, node, &output_name, out);
        Ok(ConversionResult::default())
    }

    /// Celu: `alpha * (exp(x/alpha) - 1)` for `x <= 0`, else `x`.
    fn convert_celu(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() != 1 {
            return Err(OnnxError::InvalidShape(format!(
                "Celu expects 1 input, got {}",
                inputs.len()
            )));
        }

        let output_name = output_name_for(node, node_name);
        let alpha = attr_f64(node, "alpha").unwrap_or(1.0);
        let input0 = b.resolve_operand(&inputs[0])?;
        let input_dtype = input_data_type(context, &inputs[0]);

        let alpha_name = step_label(&output_name, "alpha");
        let alpha_op = register_scalar_like(b, &alpha_name, alpha as f32, input_dtype)?;
        let one_name = step_label(&output_name, "one");
        let one_op = register_scalar_like(b, &one_name, 1.0, input_dtype)?;
        let zero_name = step_label(&output_name, "zero");
        let zero_op = register_scalar_like(b, &zero_name, 0.0, input_dtype)?;

        let div_label = step_label(&output_name, "div");
        let x_over_alpha = b
            .builder
            .div_with_options(input0, alpha_op, OnnxBuilder::labeled_options(&div_label))
            .map_err(map_op_error)?;
        let exp_label = step_label(&output_name, "exp");
        let exp_out = b
            .builder
            .exp_with_options(x_over_alpha, OnnxBuilder::labeled_options(&exp_label))
            .map_err(map_op_error)?;
        let exp_minus_one_label = step_label(&output_name, "exp_minus_one");
        let exp_minus_one = b
            .builder
            .sub_with_options(
                exp_out,
                one_op,
                OnnxBuilder::labeled_options(&exp_minus_one_label),
            )
            .map_err(map_op_error)?;
        let celu_neg_label = step_label(&output_name, "celu_neg");
        let celu_neg = b
            .builder
            .mul_with_options(
                alpha_op,
                exp_minus_one,
                OnnxBuilder::labeled_options(&celu_neg_label),
            )
            .map_err(map_op_error)?;
        // The whole chain is built in input_dtype (scalars via
        // register_scalar_like), so no cast back is needed; an identity
        // f16->f16 cast trips ORT's fp16 transform at optimization level
        // Disable.

        let input0 = b.resolve_operand(&inputs[0])?;
        let gt_label = step_label(&output_name, "gt");
        let cond = b
            .builder
            .greater_with_options(input0, zero_op, OnnxBuilder::labeled_options(&gt_label))
            .map_err(map_op_error)?;
        let out = b
            .builder
            .where_with_options(
                cond,
                b.resolve_operand(&inputs[0])?,
                celu_neg,
                OnnxBuilder::labeled_options(&output_name),
            )
            .map_err(map_op_error)?;

        record_output(b, node, &output_name, out);
        Ok(ConversionResult::default())
    }

    /// Selu: `gamma * elu(x, alpha)` with ONNX default `alpha`/`gamma`.
    fn convert_selu(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() != 1 {
            return Err(OnnxError::InvalidShape(format!(
                "Selu expects 1 input, got {}",
                inputs.len()
            )));
        }

        let output_name = output_name_for(node, node_name);
        let alpha = attr_f64(node, "alpha").unwrap_or(1.6732632423543773);
        let gamma = attr_f64(node, "gamma").unwrap_or(1.0507010298910828);
        let input0 = b.resolve_operand(&inputs[0])?;
        let input_dtype = input_data_type(context, &inputs[0]);
        let elu_label = step_label(&output_name, "elu");
        let elu_out = b
            .builder
            .elu_with_options(
                input0,
                MLEluOptions {
                    label: elu_label,
                    alpha,
                },
            )
            .map_err(map_op_error)?;
        let gamma_name = step_label(&output_name, "gamma");
        let gamma_op = register_scalar_like(b, &gamma_name, gamma as f32, input_dtype)?;
        let out = b
            .builder
            .mul_with_options(
                gamma_op,
                elu_out,
                OnnxBuilder::labeled_options(&output_name),
            )
            .map_err(map_op_error)?;

        record_output(b, node, &output_name, out);
        Ok(ConversionResult::default())
    }

    /// ThresholdedRelu: `x` when `x > alpha`, else `0`.
    fn convert_thresholded_relu(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() != 1 {
            return Err(OnnxError::InvalidShape(format!(
                "ThresholdedRelu expects 1 input, got {}",
                inputs.len()
            )));
        }

        let output_name = output_name_for(node, node_name);
        let threshold = attr_f64(node, "alpha").unwrap_or(1.0);
        let input0 = b.resolve_operand(&inputs[0])?;
        let input_dtype = input_data_type(context, &inputs[0]);
        let threshold_name = step_label(&output_name, "threshold");
        let threshold_op = register_scalar_like(b, &threshold_name, threshold as f32, input_dtype)?;
        let zero_name = step_label(&output_name, "zero");
        let zero_op = register_scalar_like(b, &zero_name, 0.0, input_dtype)?;

        let gt_label = step_label(&output_name, "gt");
        let cond = b
            .builder
            .greater_with_options(
                input0,
                threshold_op,
                OnnxBuilder::labeled_options(&gt_label),
            )
            .map_err(map_op_error)?;
        let out = b
            .builder
            .where_with_options(
                cond,
                b.resolve_operand(&inputs[0])?,
                zero_op,
                OnnxBuilder::labeled_options(&output_name),
            )
            .map_err(map_op_error)?;

        record_output(b, node, &output_name, out);
        Ok(ConversionResult::default())
    }

    /// Sinh: `(exp(x) - exp(-x)) / 2`.
    fn convert_sinh(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() != 1 {
            return Err(OnnxError::InvalidShape(format!(
                "Sinh expects 1 input, got {}",
                inputs.len()
            )));
        }

        let output_name = output_name_for(node, node_name);
        let input0 = b.resolve_operand(&inputs[0])?;
        let input_dtype = input_data_type(context, &inputs[0]);
        let (exp_pos, exp_neg) = exp_and_exp_neg(b, input0, &output_name)?;
        let diff_label = step_label(&output_name, "diff");
        let diff = b
            .builder
            .sub_with_options(exp_pos, exp_neg, OnnxBuilder::labeled_options(&diff_label))
            .map_err(map_op_error)?;
        let half = register_scalar_like(b, &step_label(&output_name, "half"), 0.5, input_dtype)?;
        let out = b
            .builder
            .mul_with_options(diff, half, OnnxBuilder::labeled_options(&output_name))
            .map_err(map_op_error)?;

        record_output(b, node, &output_name, out);
        Ok(ConversionResult::default())
    }

    /// Cosh: `(exp(x) + exp(-x)) / 2`.
    fn convert_cosh(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() != 1 {
            return Err(OnnxError::InvalidShape(format!(
                "Cosh expects 1 input, got {}",
                inputs.len()
            )));
        }

        let output_name = output_name_for(node, node_name);
        let input0 = b.resolve_operand(&inputs[0])?;
        let input_dtype = input_data_type(context, &inputs[0]);
        let (exp_pos, exp_neg) = exp_and_exp_neg(b, input0, &output_name)?;
        let sum_label = step_label(&output_name, "sum");
        let sum = b
            .builder
            .add_with_options(exp_pos, exp_neg, OnnxBuilder::labeled_options(&sum_label))
            .map_err(map_op_error)?;
        let half = register_scalar_like(b, &step_label(&output_name, "half"), 0.5, input_dtype)?;
        let out = b
            .builder
            .mul_with_options(sum, half, OnnxBuilder::labeled_options(&output_name))
            .map_err(map_op_error)?;

        record_output(b, node, &output_name, out);
        Ok(ConversionResult::default())
    }

    /// Asinh: `log(x + sqrt(x^2 + 1))`.
    fn convert_asinh(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() != 1 {
            return Err(OnnxError::InvalidShape(format!(
                "Asinh expects 1 input, got {}",
                inputs.len()
            )));
        }

        let output_name = output_name_for(node, node_name);
        let input0 = b.resolve_operand(&inputs[0])?;
        let input_dtype = input_data_type(context, &inputs[0]);
        let one = register_scalar_like(b, &step_label(&output_name, "one"), 1.0, input_dtype)?;
        let x_sq_label = step_label(&output_name, "x_sq");
        let x_sq = b
            .builder
            .mul_with_options(
                input0,
                b.resolve_operand(&inputs[0])?,
                OnnxBuilder::labeled_options(&x_sq_label),
            )
            .map_err(map_op_error)?;
        let radicand_label = step_label(&output_name, "radicand");
        let radicand = b
            .builder
            .add_with_options(x_sq, one, OnnxBuilder::labeled_options(&radicand_label))
            .map_err(map_op_error)?;
        let sqrt_label = step_label(&output_name, "sqrt");
        let sqrt_term = b
            .builder
            .sqrt_with_options(radicand, OnnxBuilder::labeled_options(&sqrt_label))
            .map_err(map_op_error)?;
        let sum_label = step_label(&output_name, "sum");
        let sum = b
            .builder
            .add_with_options(
                b.resolve_operand(&inputs[0])?,
                sqrt_term,
                OnnxBuilder::labeled_options(&sum_label),
            )
            .map_err(map_op_error)?;
        let out = b
            .builder
            .log_with_options(sum, OnnxBuilder::labeled_options(&output_name))
            .map_err(map_op_error)?;

        record_output(b, node, &output_name, out);
        Ok(ConversionResult::default())
    }

    /// Acosh: `log(x + sqrt(x^2 - 1))` (domain x >= 1).
    fn convert_acosh(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() != 1 {
            return Err(OnnxError::InvalidShape(format!(
                "Acosh expects 1 input, got {}",
                inputs.len()
            )));
        }

        let output_name = output_name_for(node, node_name);
        let input0 = b.resolve_operand(&inputs[0])?;
        let input_dtype = input_data_type(context, &inputs[0]);
        let one = register_scalar_like(b, &step_label(&output_name, "one"), 1.0, input_dtype)?;
        let x_sq_label = step_label(&output_name, "x_sq");
        let x_sq = b
            .builder
            .mul_with_options(
                input0,
                b.resolve_operand(&inputs[0])?,
                OnnxBuilder::labeled_options(&x_sq_label),
            )
            .map_err(map_op_error)?;
        let radicand_label = step_label(&output_name, "radicand");
        let radicand = b
            .builder
            .sub_with_options(x_sq, one, OnnxBuilder::labeled_options(&radicand_label))
            .map_err(map_op_error)?;
        let sqrt_label = step_label(&output_name, "sqrt");
        let sqrt_term = b
            .builder
            .sqrt_with_options(radicand, OnnxBuilder::labeled_options(&sqrt_label))
            .map_err(map_op_error)?;
        let sum_label = step_label(&output_name, "sum");
        let sum = b
            .builder
            .add_with_options(
                b.resolve_operand(&inputs[0])?,
                sqrt_term,
                OnnxBuilder::labeled_options(&sum_label),
            )
            .map_err(map_op_error)?;
        let out = b
            .builder
            .log_with_options(sum, OnnxBuilder::labeled_options(&output_name))
            .map_err(map_op_error)?;

        record_output(b, node, &output_name, out);
        Ok(ConversionResult::default())
    }

    /// Atanh: `0.5 * log((1 + x) / (1 - x))`.
    fn convert_atanh(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() != 1 {
            return Err(OnnxError::InvalidShape(format!(
                "Atanh expects 1 input, got {}",
                inputs.len()
            )));
        }

        let output_name = output_name_for(node, node_name);
        let input0 = b.resolve_operand(&inputs[0])?;
        let input_dtype = input_data_type(context, &inputs[0]);
        let one = register_scalar_like(b, &step_label(&output_name, "one"), 1.0, input_dtype)?;
        let num_label = step_label(&output_name, "num");
        let num = b
            .builder
            .add_with_options(one, input0, OnnxBuilder::labeled_options(&num_label))
            .map_err(map_op_error)?;
        let one_den =
            register_scalar_like(b, &step_label(&output_name, "one_den"), 1.0, input_dtype)?;
        let den_label = step_label(&output_name, "den");
        let den = b
            .builder
            .sub_with_options(
                one_den,
                b.resolve_operand(&inputs[0])?,
                OnnxBuilder::labeled_options(&den_label),
            )
            .map_err(map_op_error)?;
        let quot_label = step_label(&output_name, "quot");
        let quot = b
            .builder
            .div_with_options(num, den, OnnxBuilder::labeled_options(&quot_label))
            .map_err(map_op_error)?;
        let log_label = step_label(&output_name, "log");
        let log_quot = b
            .builder
            .log_with_options(quot, OnnxBuilder::labeled_options(&log_label))
            .map_err(map_op_error)?;
        let half = register_scalar_like(b, &step_label(&output_name, "half"), 0.5, input_dtype)?;
        let out = b
            .builder
            .mul_with_options(half, log_quot, OnnxBuilder::labeled_options(&output_name))
            .map_err(map_op_error)?;

        record_output(b, node, &output_name, out);
        Ok(ConversionResult::default())
    }

    /// Shrink: `x - bias` if `x > lambd`, `x + bias` if `x < -lambd`, else `0`.
    fn convert_shrink(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() != 1 {
            return Err(OnnxError::InvalidShape(format!(
                "Shrink expects 1 input, got {}",
                inputs.len()
            )));
        }

        let output_name = output_name_for(node, node_name);
        let lambd = attr_f64(node, "lambd").unwrap_or(0.5);
        let bias = attr_f64(node, "bias").unwrap_or(0.0);
        let input0 = b.resolve_operand(&inputs[0])?;
        let input_dtype = input_data_type(context, &inputs[0]);
        let lambda_op = register_scalar_like(
            b,
            &step_label(&output_name, "lambd"),
            lambd as f32,
            input_dtype,
        )?;
        let neg_lambda = register_scalar_like(
            b,
            &step_label(&output_name, "neg_lambd"),
            -(lambd as f32),
            input_dtype,
        )?;
        let bias_op = register_scalar_like(
            b,
            &step_label(&output_name, "bias"),
            bias as f32,
            input_dtype,
        )?;
        let zero = register_scalar_like(b, &step_label(&output_name, "zero"), 0.0, input_dtype)?;

        let gt_label = step_label(&output_name, "gt");
        let gt = b
            .builder
            .greater_with_options(input0, lambda_op, OnnxBuilder::labeled_options(&gt_label))
            .map_err(map_op_error)?;
        let lt_label = step_label(&output_name, "lt");
        let lt = b
            .builder
            .lesser_with_options(
                b.resolve_operand(&inputs[0])?,
                neg_lambda,
                OnnxBuilder::labeled_options(&lt_label),
            )
            .map_err(map_op_error)?;

        let high_label = step_label(&output_name, "high");
        let high = b
            .builder
            .sub_with_options(
                b.resolve_operand(&inputs[0])?,
                bias_op,
                OnnxBuilder::labeled_options(&high_label),
            )
            .map_err(map_op_error)?;
        let bias_low = register_scalar_like(
            b,
            &step_label(&output_name, "bias_low"),
            bias as f32,
            input_dtype,
        )?;
        let low_label = step_label(&output_name, "low");
        let low = b
            .builder
            .add_with_options(
                b.resolve_operand(&inputs[0])?,
                bias_low,
                OnnxBuilder::labeled_options(&low_label),
            )
            .map_err(map_op_error)?;

        let mid_label = step_label(&output_name, "mid");
        let mid = b
            .builder
            .where_with_options(lt, low, zero, OnnxBuilder::labeled_options(&mid_label))
            .map_err(map_op_error)?;
        let out = b
            .builder
            .where_with_options(gt, high, mid, OnnxBuilder::labeled_options(&output_name))
            .map_err(map_op_error)?;

        record_output(b, node, &output_name, out);
        Ok(ConversionResult::default())
    }
}

fn step_label(base: &str, step: &str) -> String {
    format!("{base}__{step}")
}

fn exp_and_exp_neg(
    b: &mut OnnxBuilder<'_, '_, '_>,
    input: MLOperand,
    base: &str,
) -> Result<(MLOperand, MLOperand), OnnxError> {
    let exp_pos_label = step_label(base, "exp_pos");
    let exp_pos = b
        .builder
        .exp_with_options(input, OnnxBuilder::labeled_options(&exp_pos_label))
        .map_err(map_op_error)?;
    let neg_label = step_label(base, "neg");
    let neg = b
        .builder
        .neg_with_options(input, OnnxBuilder::labeled_options(&neg_label))
        .map_err(map_op_error)?;
    let exp_neg_label = step_label(base, "exp_neg");
    let exp_neg = b
        .builder
        .exp_with_options(neg, OnnxBuilder::labeled_options(&exp_neg_label))
        .map_err(map_op_error)?;
    Ok((exp_pos, exp_neg))
}

#[allow(dead_code)]
fn cast_to_dtype(
    b: &mut OnnxBuilder<'_, '_, '_>,
    operand: MLOperand,
    data_type: DataType,
    label: &str,
) -> Result<MLOperand, OnnxError> {
    b.builder
        .cast_with_options(
            operand,
            map_ast_data_type(data_type)?,
            OnnxBuilder::labeled_options(label),
        )
        .map_err(map_op_error)
}

fn register_f32_scalar(
    b: &mut OnnxBuilder<'_, '_, '_>,
    name: &str,
    value: f32,
) -> Result<MLOperand, OnnxError> {
    b.register_constant_from_bytes(name, DataType::Float32, &[1], value.to_le_bytes().to_vec())?;
    b.resolve_operand(name)
}

fn register_scalar_like(
    b: &mut OnnxBuilder<'_, '_, '_>,
    name: &str,
    value: f32,
    data_type: DataType,
) -> Result<MLOperand, OnnxError> {
    match data_type {
        DataType::Float16 => {
            let bits = f16::from_f32(value).to_bits();
            b.register_constant_from_bytes(
                name,
                DataType::Float16,
                &[1],
                bits.to_le_bytes().to_vec(),
            )?;
            b.resolve_operand(name)
        }
        _ => register_f32_scalar(b, name, value),
    }
}

fn input_data_type(context: &ConversionContext, input: &str) -> DataType {
    context
        .value_types
        .get(input)
        .or_else(|| context.value_types.get(&sanitize_identifier(input)))
        .copied()
        .unwrap_or(DataType::Float32)
}

fn output_name_for(node: &NodeProto, node_name: &str) -> String {
    if node.output.as_slice().is_empty() {
        format!("{}_output", node_name)
    } else {
        sanitize_identifier(&node.output.as_slice()[0].to_string())
    }
}

fn record_output(
    b: &mut OnnxBuilder<'_, '_, '_>,
    node: &NodeProto,
    output_name: &str,
    out: MLOperand,
) {
    if let Some(output) = node.output.as_slice().first() {
        b.record_operand(&[output.as_str(), output_name], out);
    } else {
        b.record_operand(&[output_name], out);
    }
}

fn attr_f64(node: &NodeProto, name: &str) -> Option<f64> {
    node.attribute
        .as_slice()
        .iter()
        .find(|a| a.name == name)
        .map(|a| a.f as f64)
}

/// Read a scalar Clip bound from a constant initializer. Rejects non-constant bound inputs.
fn clip_bound(context: &ConversionContext, name: &str, which: &str) -> Result<f64, OnnxError> {
    let sanitized = sanitize_identifier(name);
    let tensor = context
        .initializers
        .get(name)
        .or_else(|| context.initializers.get(sanitized.as_str()))
        .ok_or_else(|| {
            OnnxError::unsupported_op(
                format!("Clip with non-constant {which} bound '{name}'"),
                name.to_string(),
            )
        })?;

    scalar_from_tensor(tensor).ok_or_else(|| {
        OnnxError::InvalidShape(format!("Clip {which} bound '{name}' is not a scalar"))
    })
}

/// First element of a (one-element) float, float16, bfloat16 or integer tensor.
fn scalar_from_tensor(tensor: &TensorProto) -> Option<f64> {
    if let Some(v) = tensor.double_data.first() {
        return Some(*v);
    }
    if let Ok(values) = crate::onnx::ops::matmul::decode_float_tensor_as_f32(tensor) {
        if let Some(v) = values.first() {
            return Some(f64::from(*v));
        }
    }
    if tensor.data_type == TensorProto_DataType::Double as i32 {
        return tensor
            .raw_data
            .get(0..8)
            .map(|b| f64::from_le_bytes(b.try_into().unwrap()));
    }
    crate::onnx::shape_inference::read_int_tensor(tensor)
        .first()
        .map(|&v| v as f64)
}

fn emit_unary(
    webnn_op: &str,
    b: &mut OnnxBuilder<'_, '_, '_>,
    input: MLOperand,
    opts: rustnn::operator_options::MLOperatorOptions,
    node_name: &str,
) -> Result<MLOperand, OnnxError> {
    Ok(match webnn_op {
        "relu" => b
            .builder
            .relu_with_options(input, opts)
            .map_err(map_op_error)?,
        "gelu" => b
            .builder
            .gelu_with_options(input, opts)
            .map_err(map_op_error)?,
        "tanh" => b
            .builder
            .tanh_with_options(input, opts)
            .map_err(map_op_error)?,
        "sigmoid" => b
            .builder
            .sigmoid_with_options(input, opts)
            .map_err(map_op_error)?,
        "sqrt" => b
            .builder
            .sqrt_with_options(input, opts)
            .map_err(map_op_error)?,
        "exp" => b
            .builder
            .exp_with_options(input, opts)
            .map_err(map_op_error)?,
        "log" => b
            .builder
            .log_with_options(input, opts)
            .map_err(map_op_error)?,
        "abs" => b
            .builder
            .abs_with_options(input, opts)
            .map_err(map_op_error)?,
        "neg" => b
            .builder
            .neg_with_options(input, opts)
            .map_err(map_op_error)?,
        "erf" => b
            .builder
            .erf_with_options(input, opts)
            .map_err(map_op_error)?,
        "cos" => b
            .builder
            .cos_with_options(input, opts)
            .map_err(map_op_error)?,
        "sin" => b
            .builder
            .sin_with_options(input, opts)
            .map_err(map_op_error)?,
        "identity" => b
            .builder
            .identity_with_options(input, opts)
            .map_err(map_op_error)?,
        "floor" => b
            .builder
            .floor_with_options(input, opts)
            .map_err(map_op_error)?,
        "ceil" => b
            .builder
            .ceil_with_options(input, opts)
            .map_err(map_op_error)?,
        "sign" => b
            .builder
            .sign_with_options(input, opts)
            .map_err(map_op_error)?,
        "tan" => b
            .builder
            .tan_with_options(input, opts)
            .map_err(map_op_error)?,
        "reciprocal" => b
            .builder
            .reciprocal_with_options(input, opts)
            .map_err(map_op_error)?,
        "round_even" => b
            .builder
            .round_even_with_options(input, opts)
            .map_err(map_op_error)?,
        "hard_swish" => b
            .builder
            .hard_swish_with_options(input, opts)
            .map_err(map_op_error)?,
        "softplus" => b
            .builder
            .softplus_with_options(input, opts)
            .map_err(map_op_error)?,
        "softsign" => b
            .builder
            .softsign_with_options(input, opts)
            .map_err(map_op_error)?,
        _ => {
            return Err(OnnxError::unsupported_op(
                webnn_op.to_string(),
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
    fn test_activation_handler_supports() {
        let handler = ActivationHandler;
        assert!(handler.supports("Relu"));
        assert!(handler.supports("Gelu"));
        assert!(handler.supports("Tanh"));
        assert!(handler.supports("Sigmoid"));
        assert!(handler.supports("Sqrt"));
        assert!(handler.supports("Exp"));
        assert!(handler.supports("Log"));
        assert!(handler.supports("Abs"));
        assert!(handler.supports("Neg"));
        assert!(handler.supports("Erf"));
        assert!(handler.supports("Cos"));
        assert!(handler.supports("Sin"));
        // Unary math added as quick wins.
        assert!(handler.supports("Floor"));
        assert!(handler.supports("Ceil"));
        assert!(handler.supports("Sign"));
        assert!(handler.supports("Tan"));
        assert!(handler.supports("Reciprocal"));
        assert!(handler.supports("Round"));
        assert!(handler.supports("HardSwish"));
        assert!(handler.supports("Softplus"));
        assert!(handler.supports("Softsign"));
        // Parametric activations.
        assert!(handler.supports("Elu"));
        assert!(handler.supports("LeakyRelu"));
        assert!(handler.supports("HardSigmoid"));
        assert!(handler.supports("Clip"));
        assert!(handler.supports("PRelu"));
        assert!(handler.supports("Swish"));
        assert!(handler.supports("Celu"));
        assert!(handler.supports("Selu"));
        assert!(handler.supports("Mish"));
        assert!(handler.supports("ThresholdedRelu"));
        assert!(handler.supports("Sinh"));
        assert!(handler.supports("Cosh"));
        assert!(handler.supports("Asinh"));
        assert!(handler.supports("Acosh"));
        assert!(handler.supports("Atanh"));
        assert!(handler.supports("Shrink"));
        assert!(!handler.supports("Add"));
    }

    #[test]
    fn test_convert_floor() {
        let handler = ActivationHandler;
        let node = create_test_node("Floor", vec!["x"], vec!["y"]);
        crate::onnx::ops::convert_with_test_builder(&handler, &node).unwrap();
    }

    #[test]
    fn test_convert_elu() {
        let handler = ActivationHandler;
        let node = create_test_node("Elu", vec!["x"], vec!["y"]);
        crate::onnx::ops::convert_with_test_builder(&handler, &node).unwrap();
    }

    #[test]
    fn test_convert_clip_unbounded() {
        let handler = ActivationHandler;
        let node = create_test_node("Clip", vec!["x"], vec!["y"]);
        crate::onnx::ops::convert_with_test_builder(&handler, &node).unwrap();
    }

    #[test]
    fn test_convert_relu() {
        let handler = ActivationHandler;
        let node = create_test_node("Relu", vec!["x"], vec!["y"]);
        crate::onnx::ops::convert_with_test_builder(&handler, &node).unwrap();
    }

    #[test]
    fn test_convert_sqrt() {
        let handler = ActivationHandler;
        let node = create_test_node("Sqrt", vec!["x"], vec!["y"]);
        crate::onnx::ops::convert_with_test_builder(&handler, &node).unwrap();
    }

    #[test]
    fn test_convert_gelu() {
        let handler = ActivationHandler;
        let node = create_test_node("Gelu", vec!["x"], vec!["y"]);
        crate::onnx::ops::convert_with_test_builder(&handler, &node).unwrap();
    }

    #[test]
    fn test_convert_cos() {
        let handler = ActivationHandler;
        let node = create_test_node("Cos", vec!["x"], vec!["y"]);
        crate::onnx::ops::convert_with_test_builder(&handler, &node).unwrap();
    }

    #[test]
    fn test_convert_sin() {
        let handler = ActivationHandler;
        let node = create_test_node("Sin", vec!["x"], vec!["y"]);
        crate::onnx::ops::convert_with_test_builder(&handler, &node).unwrap();
    }
}
