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

// Static shape/type inference scaffold for ONNX graphs.
// Conservative: records only fully-static shapes and folds small integer constants
// to unblock reshape/axes/starts/ends calculations. Dynamic dims cause errors so
// callers can ask users to run onnx-simplifier or provide overrides.
use crate::onnx::convert::{map_onnx_data_type, sanitize_identifier};
use crate::protos::onnx::{
    tensor_shape_proto::dimension::Value as DimensionValue, type_proto::Value as TypeProtoValue,
    GraphProto, ModelProto, NodeProto, TensorProto, TensorProto_DataType,
};
use rustnn::graph::{Dimension, DynamicDimension};
use rustnn::DataType;
use std::collections::BTreeMap;
use std::collections::{HashMap, HashSet};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ShapeInferenceError {
    #[error("input '{0}' is missing shape information")]
    MissingInputShape(String),
    #[error("input '{input}' has dynamic dimension '{dim}', please provide an override")]
    DynamicDim { input: String, dim: String },
    #[error("unsupported ONNX data type: {0}")]
    UnsupportedDataType(i32),
    #[error("could not infer shape for op '{op}'")]
    CannotInfer { op: String },
}

#[derive(Debug, Default)]
pub struct InferenceResult {
    pub value_shapes: HashMap<String, Vec<i64>>,
    pub value_types: HashMap<String, DataType>,
    pub const_values: HashMap<String, Vec<i64>>,
}

/// Run a lightweight static shape/type inference pass.
/// Returns only fully-known shapes; dynamic dimensions trigger an error.
pub fn infer_static_shapes(
    model: &ModelProto,
    overrides: &HashMap<String, u32>,
) -> Result<InferenceResult, ShapeInferenceError> {
    let mut result = InferenceResult::default();

    if model.graph.is_none() {
        return Ok(result);
    }

    let graph = model.graph.as_ref().unwrap();
    let initializer_names: HashSet<String> = graph
        .initializer
        .as_slice()
        .iter()
        .map(|i| i.name.as_str().to_string())
        .collect();
    let initializers: HashMap<String, &TensorProto> = graph
        .initializer
        .as_slice()
        .iter()
        .map(|init| (init.name.clone(), init))
        .collect();

    seed_inputs(graph, overrides, &initializer_names, &mut result)?;
    seed_initializers(graph, &mut result)?;
    seed_constant_nodes(graph, &mut result)?;

    propagate_node_shapes(graph, &initializers, &mut result)?;
    apply_value_info_shape_hints(graph, &mut result)?;
    // Concrete value_info dimensions can correct earlier best-effort shapes.
    // Refresh shape-derived constants so downstream Resize sizes use the hints.
    for _ in 0..8 {
        if !fold_integer_constants(graph, &mut result) {
            break;
        }
    }

    Ok(result)
}

fn apply_value_info_shape_hints(
    graph: &GraphProto,
    result: &mut InferenceResult,
) -> Result<(), ShapeInferenceError> {
    for value_info in graph.value_info.iter().chain(graph.output.iter()) {
        let Some(type_proto) = value_info.r#type.as_ref() else {
            continue;
        };
        let Some(TypeProtoValue::TensorType(tensor_type)) = type_proto.value.as_ref() else {
            continue;
        };
        if tensor_type.elem_type != 0 {
            let dtype = map_onnx_data_type(tensor_type.elem_type)
                .map_err(|_| ShapeInferenceError::UnsupportedDataType(tensor_type.elem_type))?;
            result
                .value_types
                .entry(value_info.name.clone())
                .or_insert(dtype);
        }
        let Some(shape_proto) = tensor_type.shape.as_ref() else {
            continue;
        };

        let concrete: Vec<Option<i64>> = shape_proto
            .dim
            .iter()
            .map(|dim| match dim.value.as_ref() {
                Some(DimensionValue::DimValue(value)) if *value > 0 => Some(*value),
                _ => None,
            })
            .collect();
        if let Some(existing) = result.value_shapes.get_mut(&value_info.name) {
            if existing.len() == concrete.len() {
                for (dimension, hint) in existing.iter_mut().zip(concrete.iter()) {
                    if let Some(hint) = hint {
                        *dimension = *hint;
                    }
                }
            }
        } else if concrete.iter().all(Option::is_some) {
            result.value_shapes.insert(
                value_info.name.clone(),
                concrete.into_iter().flatten().collect(),
            );
        }
    }
    Ok(())
}

fn seed_inputs(
    graph: &GraphProto,
    overrides: &HashMap<String, u32>,
    initializer_names: &HashSet<String>,
    result: &mut InferenceResult,
) -> Result<(), ShapeInferenceError> {
    for input in graph.input.as_slice() {
        let name = input.name.as_str().to_string();

        if initializer_names.contains(&name) {
            continue;
        }

        let type_proto = input
            .r#type
            .as_ref()
            .ok_or_else(|| ShapeInferenceError::MissingInputShape(name.clone()))?;

        let tensor_type = match &type_proto.value {
            Some(TypeProtoValue::TensorType(tt)) => tt,
            _ => return Err(ShapeInferenceError::MissingInputShape(name.clone())),
        };

        let dtype = if tensor_type.elem_type != 0 {
            map_onnx_data_type(tensor_type.elem_type)
                .map_err(|_| ShapeInferenceError::UnsupportedDataType(tensor_type.elem_type))?
        } else {
            return Err(ShapeInferenceError::UnsupportedDataType(0));
        };

        let shape = tensor_type
            .shape
            .as_ref()
            .ok_or_else(|| ShapeInferenceError::MissingInputShape(name.clone()))?;

        let mut shape_dims = Vec::new();
        for dim in shape.dim.as_slice() {
            if let Some(value) = &dim.value {
                match value {
                    DimensionValue::DimValue(v) if *v > 0 => {
                        shape_dims.push(*v);
                    }
                    DimensionValue::DimValue(_) => {
                        // Unnamed dynamic dim: overridable as `<input>_dim<axis>`.
                        let hint =
                            format!("{}_dim{}", sanitize_identifier(&name), shape_dims.len());
                        if let Some(v) = overrides.get(hint.as_str()) {
                            shape_dims.push(*v as i64);
                        } else {
                            return Err(ShapeInferenceError::DynamicDim {
                                input: name.clone(),
                                dim: hint,
                            });
                        }
                    }
                    DimensionValue::DimParam(key) => {
                        if let Some(v) = overrides.get(key.as_str()) {
                            shape_dims.push(*v as i64);
                        } else {
                            return Err(ShapeInferenceError::DynamicDim {
                                input: name.clone(),
                                dim: key.clone(),
                            });
                        }
                    }
                }
            } else {
                return Err(ShapeInferenceError::MissingInputShape(name.clone()));
            }
        }

        result.value_types.insert(name.clone(), dtype);
        result.value_shapes.insert(name, shape_dims);
    }
    Ok(())
}

fn seed_initializers(
    graph: &GraphProto,
    result: &mut InferenceResult,
) -> Result<(), ShapeInferenceError> {
    for init in graph.initializer.as_slice() {
        let name = init.name.as_str().to_string();

        let dtype = map_onnx_data_type(init.data_type)
            .map_err(|_| ShapeInferenceError::UnsupportedDataType(init.data_type))?;
        let shape: Vec<i64> = init.dims.as_slice().to_vec();
        result.value_types.insert(name.clone(), dtype);
        result.value_shapes.insert(name.clone(), shape);

        if matches!(
            dtype,
            DataType::Int32 | DataType::Int64 | DataType::Uint32 | DataType::Uint64
        ) {
            let values = read_int_tensor(init);
            if !values.is_empty() {
                result.const_values.insert(name, values);
            }
        }
    }
    Ok(())
}

fn seed_constant_nodes(
    graph: &GraphProto,
    result: &mut InferenceResult,
) -> Result<(), ShapeInferenceError> {
    for node in graph.node.as_slice() {
        if node.op_type.as_str() != "Constant" {
            continue;
        }

        if let Some(out) = node.output.as_slice().first() {
            let out_name = out.to_string();

            if let Some(attr) = node
                .attribute
                .as_slice()
                .iter()
                .find(|a| a.name.as_str() == "value" && a.t.is_some())
            {
                let t = attr.t.as_ref().unwrap();
                let dtype = map_onnx_data_type(t.data_type)
                    .map_err(|_| ShapeInferenceError::UnsupportedDataType(t.data_type))?;
                result.value_types.insert(out_name.clone(), dtype);

                let vals = read_int_tensor(t);
                if !vals.is_empty() {
                    result.const_values.insert(out_name.clone(), vals.clone());
                    // Keep the tensor's declared rank (e.g. a [1,1,S,S] bool
                    // mask); only synthesize a flat shape when dims are absent.
                    let shape: Vec<i64> = if !t.dims.is_empty() {
                        t.dims.clone()
                    } else if vals.len() == 1 {
                        Vec::new()
                    } else {
                        vec![vals.len() as i64]
                    };
                    result.value_shapes.insert(out_name.clone(), shape);
                }
            }
        }
    }
    Ok(())
}

fn propagate_node_shapes(
    graph: &GraphProto,
    initializers: &HashMap<String, &TensorProto>,
    result: &mut InferenceResult,
) -> Result<(), ShapeInferenceError> {
    let mut progress = true;
    let max_iters = 8;
    let mut iter = 0;

    while progress && iter < max_iters {
        progress = false;
        iter += 1;

        for node in graph.node.as_slice() {
            let outputs = node.output.as_slice();
            if outputs.is_empty() {
                continue;
            }
            // Shapes may have been guessed before an upstream constant (e.g. a
            // Reshape target) folded; recompute every node and track changes.

            if node.op_type.as_str() == "DynamicQuantizeLinear" {
                if let Some(input_name) = node.input.first() {
                    if let Some(input_shape) = result.value_shapes.get(input_name).cloned() {
                        if let [y, scale, zero_point] = outputs {
                            result.value_shapes.insert(y.clone(), input_shape);
                            result.value_shapes.insert(scale.clone(), Vec::new());
                            result.value_shapes.insert(zero_point.clone(), Vec::new());
                            result.value_types.insert(y.clone(), DataType::Uint8);
                            result.value_types.insert(scale.clone(), DataType::Float32);
                            result
                                .value_types
                                .insert(zero_point.clone(), DataType::Uint8);
                            progress = true;
                            continue;
                        }
                    }
                }
            }

            if node.op_type.as_str() == "GroupQueryAttention" {
                if let Some(shapes) =
                    crate::onnx::ops::attention::gqa_output_shapes(node, &result.value_shapes)
                {
                    let input_type = node
                        .input
                        .first()
                        .and_then(|name| result.value_types.get(name).copied());
                    for (output, shape) in outputs.iter().zip(shapes) {
                        if output.is_empty() {
                            continue;
                        }
                        result.value_shapes.entry(output.clone()).or_insert(shape);
                        if let Some(dtype) = input_type {
                            result.value_types.entry(output.clone()).or_insert(dtype);
                        }
                    }
                    progress = true;
                    continue;
                }
            }

            if node.op_type.as_str() == "SkipSimplifiedLayerNormalization" {
                if let Some(input_name) = node.input.first() {
                    if let Some(input_shape) = result.value_shapes.get(input_name).cloned() {
                        let input_type = result.value_types.get(input_name).copied();
                        for output in outputs.iter().filter(|name| !name.is_empty()) {
                            result
                                .value_shapes
                                .entry(output.clone())
                                .or_insert_with(|| input_shape.clone());
                            if let Some(dtype) = input_type {
                                result.value_types.entry(output.clone()).or_insert(dtype);
                            }
                        }
                        progress = true;
                        continue;
                    }
                }
            }

            if node.op_type.as_str() == "Split" {
                if let Some(shapes) = infer_split_output_shapes(
                    node,
                    &result.value_shapes,
                    initializers,
                    &result.const_values,
                ) {
                    for (output, shape) in outputs.iter().zip(shapes) {
                        result
                            .value_shapes
                            .entry(output.to_string())
                            .or_insert(shape);
                        if let Some(first_in) = node.input.as_slice().first() {
                            if let Some(dtype) = result.value_types.get(first_in).cloned() {
                                result
                                    .value_types
                                    .entry(output.to_string())
                                    .or_insert(dtype);
                            }
                        }
                    }
                    progress = true;
                    continue;
                }
            }

            if let Some(shape) = infer_node_output_shape(
                node,
                &result.value_shapes,
                initializers,
                &result.const_values,
            ) {
                let out_name = outputs[0].to_string();
                let advanced = result.value_shapes.get(&out_name) != Some(&shape);
                result.value_shapes.insert(out_name.clone(), shape);

                // Propagate dtype from first input if available.
                if node.op_type.as_str() == "OneHot" {
                    // Output takes the type of `values`, not of the indices.
                    if let Some(values) = node.input.as_slice().get(2) {
                        let dtype =
                            result
                                .value_types
                                .get(values.as_str())
                                .copied()
                                .or_else(|| {
                                    initializers
                                        .get(values.as_str())
                                        .and_then(|t| map_onnx_data_type(t.data_type).ok())
                                });
                        if let Some(dtype) = dtype {
                            result.value_types.insert(out_name.clone(), dtype);
                        }
                    }
                } else if node.op_type.as_str() == "Cast" {
                    // The output type is the `to` attribute, not the input's.
                    let to = node
                        .attribute
                        .as_slice()
                        .iter()
                        .find(|a| a.name.as_str() == "to")
                        .map(|a| a.i as i32);
                    if let Some(dtype) = to.and_then(|t| map_onnx_data_type(t).ok()) {
                        result.value_types.insert(out_name.clone(), dtype);
                    }
                } else if matches!(
                    node.op_type.as_str(),
                    "Equal"
                        | "Greater"
                        | "GreaterOrEqual"
                        | "Less"
                        | "LessOrEqual"
                        | "And"
                        | "Or"
                        | "Xor"
                        | "Not"
                        | "IsNaN"
                        | "IsInf"
                ) {
                    result.value_types.insert(out_name.clone(), DataType::Uint8);
                } else if matches!(node.op_type.as_str(), "ConvInteger" | "MatMulInteger") {
                    result
                        .value_types
                        .entry(out_name.clone())
                        .or_insert(DataType::Int32);
                } else if node.op_type.as_str() == "GatherBlockQuantized" {
                    // The dequantized output takes the scales' float type, not
                    // the packed uint8 table's.
                    if let Some(dtype) =
                        gather_block_quantized_dtype(node, &result.value_types, initializers)
                    {
                        result.value_types.entry(out_name.clone()).or_insert(dtype);
                    }
                } else if let Some(first_in) = node.input.as_slice().first() {
                    if let Some(dtype) = result.value_types.get(first_in).cloned() {
                        result.value_types.entry(out_name.clone()).or_insert(dtype);
                    }
                }

                if advanced {
                    progress = true;
                }
            }
        }

        // Opportunistic const folding for integer tensors to unlock more shapes.
        progress |= fold_integer_constants(graph, result);
    }

    Ok(())
}

#[allow(dead_code)]
pub fn infer_node_output_shape(
    node: &crate::protos::onnx::NodeProto,
    value_shapes: &HashMap<String, Vec<i64>>,
    initializers: &HashMap<String, &TensorProto>,
    const_values: &HashMap<String, Vec<i64>>,
) -> Option<Vec<i64>> {
    let op = node.op_type.as_str();

    match op {
        // Unary operations that preserve shape
        "Cast"
        | "Relu"
        | "Tanh"
        | "Sigmoid"
        | "Erf"
        | "Softmax"
        | "Gelu"
        | "Exp"
        | "Log"
        | "Abs"
        | "Neg"
        | "Sqrt"
        | "LayerNormalization"
        | "RMSNormalization"
        | "SimplifiedLayerNormalization"
        | "SkipSimplifiedLayerNormalization"
        | "RotaryEmbedding"
        | "MoE"
        | "QMoE"
        | "CumSum"
        | "CumProd"
        | "DequantizeLinear"
        | "QuantizeLinear"
        | "Sin"
        | "Cos"
        | "Tan"
        | "Asin"
        | "Acos"
        | "Atan"
        | "Sinh"
        | "Cosh"
        | "Asinh"
        | "Acosh"
        | "Atanh"
        | "Floor"
        | "Ceil"
        | "Round"
        | "Sign"
        | "Reciprocal"
        | "Identity"
        | "LeakyRelu"
        | "Elu"
        | "Selu"
        | "Celu"
        | "Softplus"
        | "Softsign"
        | "HardSigmoid"
        | "HardSwish"
        | "Mish"
        | "Clip"
        | "Not"
        | "IsNaN"
        | "IsInf"
        | "ScatterND"
        | "ScatterElements"
        | "BatchNormalization"
        | "InstanceNormalization"
        | "Trilu" => {
            let ins = node.input.as_slice();
            if ins.is_empty() {
                return None;
            }
            value_shapes.get(ins[0].as_str()).cloned()
        }

        // Where broadcasts its three inputs together.
        "Where" => {
            let ins = node.input.as_slice();
            if ins.len() < 3 {
                return None;
            }
            let mut out: Vec<i64> = Vec::new();
            for name in ins.iter().take(3) {
                let shape = value_shapes.get(name.as_str())?;
                out = broadcast_shape(&out, shape)?;
            }
            Some(out)
        }

        // Binary operations with NumPy-style broadcasting semantics.
        // (Min/Max are variadic; two inputs covers the common exports.)
        "Add" | "Sub" | "Mul" | "Div" | "Mod" | "Pow" | "Equal" | "Greater" | "GreaterOrEqual"
        | "Less" | "LessOrEqual" | "And" | "Or" | "Xor" | "Min" | "Max" => {
            let ins = node.input.as_slice();
            if ins.len() < 2 {
                return None;
            }

            let shape_a = value_shapes.get(ins[0].as_str());
            let shape_b = value_shapes.get(ins[1].as_str());

            match (shape_a, shape_b) {
                (Some(a), Some(b)) => {
                    let rank = a.len().max(b.len());
                    let mut out_rev = Vec::with_capacity(rank);
                    for i in 0..rank {
                        let da = a.get(a.len().wrapping_sub(1 + i)).copied().unwrap_or(1);
                        let db = b.get(b.len().wrapping_sub(1 + i)).copied().unwrap_or(1);
                        if da == db || da == 1 {
                            out_rev.push(db);
                        } else if db == 1 {
                            out_rev.push(da);
                        } else {
                            return None;
                        }
                    }
                    out_rev.reverse();
                    Some(out_rev)
                }
                (Some(a), None) => Some(a.clone()),
                (None, Some(b)) => Some(b.clone()),
                (None, None) => None,
            }
        }

        // MatMul (2D matrix multiplication)
        "MatMul" | "MatMulInteger" => {
            let ins = node.input.as_slice();
            if ins.len() < 2 {
                return None;
            }

            // Weight operands are often initializers absent from value_shapes.
            let shape_of = |name: &str| {
                value_shapes
                    .get(name)
                    .cloned()
                    .or_else(|| initializers.get(name).map(|t| t.dims.clone()))
            };
            let a_shape = &shape_of(ins[0].as_str())?;
            let b_shape = &shape_of(ins[1].as_str())?;

            // Handle 2D case: [M, K] @ [K, N] -> [M, N]
            if a_shape.len() >= 2 && b_shape.len() >= 2 {
                let m = a_shape[a_shape.len() - 2];
                let n = b_shape[b_shape.len() - 1];

                // For higher-dim inputs, preserve batch dimensions
                if a_shape.len() == 2 && b_shape.len() == 2 {
                    return Some(vec![m, n]);
                } else if a_shape.len() > 2 {
                    let mut result = a_shape[..a_shape.len() - 2].to_vec();
                    result.push(m);
                    result.push(n);
                    return Some(result);
                }
            }
            None
        }

        "Einsum" => {
            let equation = node
                .attribute
                .iter()
                .find(|attr| attr.name == "equation")
                .map(|attr| String::from_utf8_lossy(&attr.s).to_string())?;
            let input_shapes: Vec<Vec<i64>> = node
                .input
                .iter()
                .map(|name| value_shapes.get(name.as_str()).cloned())
                .collect::<Option<_>>()?;
            crate::onnx::ops::einsum::einsum_output_shape(&equation, &input_shapes)
        }

        "MatMulNBits" | "MatMulBnb4" => {
            let ins = node.input.as_slice();
            if ins.is_empty() {
                return None;
            }
            let a_shape = value_shapes.get(ins[0].as_str())?;
            if a_shape.is_empty() {
                return None;
            }
            let n = node
                .attribute
                .iter()
                .find(|attr| attr.name == "N")
                .map(|attr| attr.i)
                .filter(|&n| n > 0)?;
            let mut out = a_shape.clone();
            *out.last_mut()? = n;
            Some(out)
        }

        // Transpose preserves shape with permuted dimensions
        "Transpose" => {
            let ins = node.input.as_slice();
            if ins.is_empty() {
                return None;
            }
            let input_shape = value_shapes.get(ins[0].as_str())?;

            // Get perm attribute
            let perm: Vec<usize> = node
                .attribute
                .as_slice()
                .iter()
                .find(|a| a.name.as_str() == "perm")
                .map(|a| a.ints.iter().map(|&i| i as usize).collect::<Vec<usize>>())
                .unwrap_or_else(|| (0..input_shape.len()).rev().collect());

            // A tracked shape whose rank disagrees with the permutation is a
            // stale guess from partial propagation - refuse rather than panic.
            if perm.len() != input_shape.len() || perm.iter().any(|&i| i >= input_shape.len()) {
                return None;
            }
            Some(perm.iter().map(|&i| input_shape[i]).collect())
        }

        // Reduce operations
        "ReduceMean" | "ReduceSum" | "ReduceMax" | "ReduceMin" => {
            let ins = node.input.as_slice();
            if ins.is_empty() {
                return None;
            }
            let input_shape = value_shapes.get(ins[0].as_str())?;

            // Check keepdims attribute (default is 1/true)
            let keepdims = node
                .attribute
                .as_slice()
                .iter()
                .find(|a| a.name.as_str() == "keepdims")
                .map(|a| a.i != 0)
                .unwrap_or(true);

            // Axes: attribute (opset < 18) or constant second input (opset 18+).
            let mut axes: Vec<i64> = node
                .attribute
                .as_slice()
                .iter()
                .find(|a| a.name.as_str() == "axes")
                .map(|a| a.ints.clone())
                .unwrap_or_default();
            if axes.is_empty() {
                if let Some(axes_name) = ins.get(1).filter(|n| !n.is_empty()) {
                    axes = const_values
                        .get(axes_name.as_str())
                        .cloned()
                        .unwrap_or_default();
                }
            }

            if axes.is_empty() {
                // Reduce all dimensions
                if keepdims {
                    Some(vec![1; input_shape.len()])
                } else {
                    Some(vec![])
                }
            } else {
                // Reduce specific axes
                let mut output_shape = input_shape.clone();
                for &axis in &axes {
                    let idx = if axis < 0 {
                        (input_shape.len() as i64 + axis) as usize
                    } else {
                        axis as usize
                    };
                    if idx < output_shape.len() {
                        if keepdims {
                            output_shape[idx] = 1;
                        } else {
                            output_shape[idx] = -1; // Mark for removal
                        }
                    }
                }
                if !keepdims {
                    output_shape.retain(|&d| d != -1);
                }
                Some(output_shape)
            }
        }

        // Gemm (generalized matrix multiplication)
        "Gemm" => {
            let ins = node.input.as_slice();
            if ins.len() < 2 {
                return None;
            }

            let a_shape = value_shapes.get(ins[0].as_str())?;
            let b_shape = value_shapes.get(ins[1].as_str())?;

            if a_shape.len() != 2 || b_shape.len() != 2 {
                return None;
            }

            // Check transA and transB attributes
            let trans_a = node
                .attribute
                .as_slice()
                .iter()
                .find(|a| a.name.as_str() == "transA")
                .and_then(|a| if a.i != 0 { Some(a.i != 0) } else { None })
                .unwrap_or(false);

            let trans_b = node
                .attribute
                .as_slice()
                .iter()
                .find(|a| a.name.as_str() == "transB")
                .and_then(|a| if a.i != 0 { Some(a.i != 0) } else { None })
                .unwrap_or(false);

            let m = if trans_a { a_shape[1] } else { a_shape[0] };
            let n = if trans_b { b_shape[0] } else { b_shape[1] };

            Some(vec![m, n])
        }

        "Gather" | "GatherBlockQuantized" => {
            let ins = node.input.as_slice();
            if ins.len() < 2 {
                return None;
            }

            let mut data_shape = value_shapes
                .get(ins[0].as_str())
                .cloned()
                .or_else(|| initializers.get(ins[0].as_str()).map(|t| t.dims.clone()))?;
            // The quantized table packs 8/bits values per byte on its last axis.
            if op == "GatherBlockQuantized" {
                let bits = node
                    .attribute
                    .as_slice()
                    .iter()
                    .find(|a| a.name.as_str() == "bits")
                    .map(|a| a.i)
                    .unwrap_or(4);
                if bits != 4 && bits != 8 {
                    return None;
                }
                // INT4/UINT4 tables (data types 22/21) already have logical dims.
                let four_bit_dtype = initializers
                    .get(ins[0].as_str())
                    .is_some_and(|t| t.data_type == 21 || t.data_type == 22);
                if !four_bit_dtype {
                    let last = data_shape.last_mut()?;
                    *last *= 8 / bits;
                }
            }
            let data_shape = &data_shape;
            let indices_shape = value_shapes.get(ins[1].as_str())?;

            let mut axis = node
                .attribute
                .as_slice()
                .iter()
                .find(|a| {
                    a.name.as_str()
                        == if op == "GatherBlockQuantized" {
                            "gather_axis"
                        } else {
                            "axis"
                        }
                })
                .and_then(|a| if a.i != 0 { Some(a.i) } else { None })
                .unwrap_or(0);

            if axis < 0 {
                axis += data_shape.len() as i64;
            }

            let axis_usize = axis as usize;
            if axis_usize > data_shape.len() {
                return None;
            }

            let mut output = Vec::new();
            output.extend_from_slice(&data_shape[..axis_usize]);
            output.extend(indices_shape.iter().cloned());
            if axis_usize < data_shape.len() {
                output.extend_from_slice(&data_shape[axis_usize + 1..]);
            }
            Some(output)
        }

        "Unsqueeze" => {
            let ins = node.input.as_slice();
            if ins.is_empty() {
                return None;
            }

            let input_shape = value_shapes.get(ins[0].as_str())?.clone();
            let mut axes: Vec<i64> = node
                .attribute
                .as_slice()
                .iter()
                .find(|a| a.name.as_str() == "axes")
                .map(|a| a.ints.clone())
                .unwrap_or_default();
            if axes.is_empty() {
                // Opset 13+: axes is the second input.
                let axes_name = ins.get(1)?;
                axes = read_int64_values_from_maps(axes_name.as_str(), initializers, const_values)?;
            }
            if axes.is_empty() {
                return None;
            }

            // Negative axes are relative to the output rank.
            let out_rank = input_shape.len() as i64 + axes.len() as i64;
            let mut normalized: Vec<i64> = axes
                .iter()
                .map(|&a| if a < 0 { a + out_rank } else { a })
                .collect();
            if normalized.iter().any(|&a| a < 0 || a >= out_rank) {
                return None;
            }
            normalized.sort_unstable();
            normalized.dedup();
            if normalized.len() as i64 + input_shape.len() as i64 != out_rank {
                return None;
            }
            let mut output_shape = input_shape;
            for axis in normalized {
                output_shape.insert(axis as usize, 1);
            }
            Some(output_shape)
        }

        "Squeeze" => {
            let ins = node.input.as_slice();
            let input_shape = value_shapes.get(ins.first()?.as_str())?;
            let mut axes = node
                .attribute
                .as_slice()
                .iter()
                .find(|a| a.name.as_str() == "axes")
                .map(|a| a.ints.clone())
                .unwrap_or_default();
            if axes.is_empty() {
                if let Some(axes_name) = ins.get(1) {
                    axes =
                        read_int64_values_from_maps(axes_name.as_str(), initializers, const_values)
                            .unwrap_or_default();
                }
            }

            if axes.is_empty() {
                return Some(
                    input_shape
                        .iter()
                        .copied()
                        .filter(|&dim| dim != 1)
                        .collect(),
                );
            }

            let rank = input_shape.len() as i64;
            let mut normalized = HashSet::new();
            for axis in axes {
                let axis = if axis < 0 { axis + rank } else { axis };
                if axis < 0 || axis >= rank || input_shape[axis as usize] != 1 {
                    return None;
                }
                normalized.insert(axis as usize);
            }
            Some(
                input_shape
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, &dim)| (!normalized.contains(&idx)).then_some(dim))
                    .collect(),
            )
        }

        "Expand" => {
            let ins = node.input.as_slice();
            if ins.len() < 2 {
                return None;
            }
            let input_shape = value_shapes.get(ins[0].as_str())?;
            // An empty shape vector broadcasts to the input itself.
            if const_values
                .get(ins[1].as_str())
                .is_some_and(|v| v.is_empty())
            {
                return Some(input_shape.clone());
            }
            let target = read_int64_values_from_maps(ins[1].as_str(), initializers, const_values)?;
            broadcast_shape(input_shape, &target)
        }

        "Tile" => {
            let ins = node.input.as_slice();
            if ins.len() < 2 {
                return None;
            }
            let input_shape = value_shapes.get(ins[0].as_str())?;
            let repeats = read_int64_values_from_maps(ins[1].as_str(), initializers, const_values)?;
            if repeats.len() != input_shape.len() || repeats.iter().any(|&repeat| repeat < 0) {
                return None;
            }
            input_shape
                .iter()
                .zip(repeats.iter())
                .map(|(&dim, &repeat)| dim.checked_mul(repeat))
                .collect()
        }

        "OneHot" => {
            let ins = node.input.as_slice();
            let indices = value_shapes.get(ins.first()?.as_str())?;
            let depth =
                read_int64_values_from_maps(ins.get(1)?.as_str(), initializers, const_values)?
                    .first()
                    .copied()?;
            let out_rank = indices.len() as i64 + 1;
            let axis = node
                .attribute
                .as_slice()
                .iter()
                .find(|a| a.name.as_str() == "axis")
                .map(|a| a.i)
                .unwrap_or(-1);
            let axis = if axis < 0 { axis + out_rank } else { axis };
            if axis < 0 || axis >= out_rank || depth <= 0 {
                return None;
            }
            let mut out = indices.clone();
            out.insert(axis as usize, depth);
            Some(out)
        }

        "Range" => {
            let ins = node.input.as_slice();
            if ins.len() != 3 {
                return None;
            }
            // ONNX: number_of_elements = max(ceil((limit - start) / delta), 0).
            // Computed in f64 so float ranges (dynamo `arange(0.5, n, 1.0)`) work.
            let scalar = |name: &str| read_range_scalar_f64(name, initializers, const_values);
            let start = scalar(ins[0].as_str())?;
            let limit = scalar(ins[1].as_str())?;
            let delta = scalar(ins[2].as_str())?;
            if delta == 0.0 || !delta.is_finite() {
                return None;
            }
            let len = ((limit - start) / delta).ceil();
            if !len.is_finite() {
                return None;
            }
            Some(vec![if len > 0.0 { len as i64 } else { 0 }])
        }

        "Concat" => {
            let mut shapes = Vec::new();
            for inp in node.input.as_slice() {
                let shape = value_shapes.get(inp.as_str())?;
                shapes.push(shape.clone());
            }

            if shapes.is_empty() {
                return None;
            }

            let mut axis = node
                .attribute
                .as_slice()
                .iter()
                .find(|a| a.name.as_str() == "axis")
                .and_then(|a| if a.i != 0 { Some(a.i) } else { None })
                .unwrap_or(0);

            if axis < 0 {
                axis += shapes[0].len() as i64;
            }
            let axis_usize = axis as usize;

            let mut output = shapes[0].clone();
            for shape in shapes.iter().skip(1) {
                if shape.len() != output.len() || axis_usize >= shape.len() {
                    return None;
                }
                output[axis_usize] += shape[axis_usize];
            }
            Some(output)
        }

        "Pad" => {
            let ins = node.input.as_slice();
            if ins.is_empty() {
                return None;
            }
            let input_shape = value_shapes.get(ins[0].as_str())?;
            let rank = input_shape.len();
            let pads = crate::onnx::ops::pad::read_onnx_pads_from_maps(
                node,
                initializers,
                const_values,
                rank,
            )
            .ok()?;
            crate::onnx::ops::pad::infer_pad_output_shape(input_shape, &pads)
        }

        "Reshape" => {
            let ins = node.input.as_slice();
            if ins.len() < 2 {
                return None;
            }

            // The data shape is only needed to resolve -1 / 0 target entries;
            // a fully positive constant target stands on its own.
            let input_shape_opt = value_shapes.get(ins[0].as_str());
            let shape_input = ins[1].as_str();
            let mut target: Vec<i64> = if let Some(values) = const_values.get(shape_input) {
                values.clone()
            } else if let Some(shape_tensor) = initializers.get(shape_input) {
                if !shape_tensor.raw_data.as_slice().is_empty() {
                    if shape_tensor.data_type == TensorProto_DataType::Int32 as i32 {
                        shape_tensor
                            .raw_data
                            .as_slice()
                            .chunks_exact(4)
                            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64)
                            .collect()
                    } else {
                        shape_tensor
                            .raw_data
                            .as_slice()
                            .chunks_exact(8)
                            .map(|c| {
                                i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
                            })
                            .collect()
                    }
                } else if !shape_tensor.int64_data.as_slice().is_empty() {
                    shape_tensor.int64_data.as_slice().to_vec()
                } else if !shape_tensor.int32_data.as_slice().is_empty() {
                    shape_tensor
                        .int32_data
                        .as_slice()
                        .iter()
                        .map(|&v| v as i64)
                        .collect()
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            };

            if target.is_empty() {
                return None;
            }

            // ONNX: a 0 entry copies the corresponding input dimension.
            if target.contains(&0) {
                let input_shape = input_shape_opt?;
                for (idx, dim) in target.iter_mut().enumerate() {
                    if *dim == 0 {
                        *dim = *input_shape.get(idx)?;
                    }
                }
            }
            if target.contains(&-1) {
                let input_shape = input_shape_opt?;
                let total_input: i64 = input_shape.iter().product();
                let known: i64 = target.iter().filter(|&&d| d != -1).product();
                if known == 0 || total_input % known != 0 {
                    return None;
                }
                if let Some(idx) = target.iter().position(|&d| d == -1) {
                    target[idx] = total_input / known;
                }
            }

            Some(target)
        }

        // Pooling: maxPool / averagePool / global variants.  Only handles fully-static inputs.
        "MaxPool" | "AveragePool" => {
            let ins = node.input.as_slice();
            if ins.is_empty() {
                return None;
            }
            let x_shape = value_shapes.get(ins[0].as_str())?.clone();
            if x_shape.len() < 3 {
                return None;
            }
            let spatial_rank = x_shape.len() - 2;

            let mut auto_pad = String::from("NOTSET");
            let mut strides: Vec<i64> = vec![1; spatial_rank];
            let mut dilations: Vec<i64> = vec![1; spatial_rank];
            let mut pads: Vec<i64> = vec![0; 2 * spatial_rank];
            let mut kernel_shape: Vec<i64> = Vec::new();
            let mut ceil_mode = false;
            for attr in node.attribute.as_slice() {
                match attr.name.as_str() {
                    "auto_pad" => {
                        if let Ok(s) = String::from_utf8(attr.s.clone()) {
                            if !s.is_empty() {
                                auto_pad = s;
                            }
                        }
                    }
                    "kernel_shape" if !attr.ints.is_empty() => kernel_shape = attr.ints.clone(),
                    "strides" if !attr.ints.is_empty() => strides = attr.ints.clone(),
                    "dilations" if !attr.ints.is_empty() => dilations = attr.ints.clone(),
                    "pads" if !attr.ints.is_empty() => pads = attr.ints.clone(),
                    "ceil_mode" => ceil_mode = attr.i != 0,
                    _ => {}
                }
            }
            if kernel_shape.len() != spatial_rank
                || strides.len() != spatial_rank
                || dilations.len() != spatial_rank
                || pads.len() != 2 * spatial_rank
            {
                return None;
            }

            let mut out_spatial = Vec::with_capacity(spatial_rank);
            for i in 0..spatial_rank {
                let in_dim = x_shape[2 + i];
                let k = kernel_shape[i];
                let s = strides[i];
                let d = dilations[i];
                let dilated_k = d * (k - 1) + 1;
                let out_dim = match auto_pad.as_str() {
                    "SAME_UPPER" | "SAME_LOWER" => (in_dim + s - 1) / s,
                    "VALID" => (in_dim - dilated_k) / s + 1,
                    _ => {
                        let pad_begin = pads[i];
                        let pad_end = pads[i + spatial_rank];
                        let numerator = in_dim + pad_begin + pad_end - dilated_k;
                        if ceil_mode {
                            (numerator + s - 1) / s + 1
                        } else {
                            numerator / s + 1
                        }
                    }
                };
                if out_dim < 0 {
                    return None;
                }
                out_spatial.push(out_dim);
            }

            let mut out = vec![x_shape[0], x_shape[1]];
            out.extend(out_spatial);
            Some(out)
        }

        "GlobalMaxPool" | "GlobalAveragePool" => {
            let ins = node.input.as_slice();
            if ins.is_empty() {
                return None;
            }
            let x_shape = value_shapes.get(ins[0].as_str())?.clone();
            if x_shape.len() < 3 {
                return None;
            }
            let mut out = vec![x_shape[0], x_shape[1]];
            out.extend(std::iter::repeat_n(1i64, x_shape.len() - 2));
            Some(out)
        }

        "Flatten" => {
            let ins = node.input.as_slice();
            if ins.is_empty() {
                return None;
            }
            let x_shape = value_shapes.get(ins[0].as_str())?.clone();
            let axis = node
                .attribute
                .as_slice()
                .iter()
                .find(|a| a.name.as_str() == "axis")
                .map(|a| a.i)
                .unwrap_or(1);
            let rank = x_shape.len() as i64;
            let norm = if axis < 0 { axis + rank } else { axis };
            if norm < 0 || norm > rank {
                return None;
            }
            let norm = norm as usize;
            let outer: i64 = if norm == 0 {
                1
            } else {
                x_shape[..norm].iter().product()
            };
            let inner: i64 = if norm == x_shape.len() {
                1
            } else {
                x_shape[norm..].iter().product()
            };
            Some(vec![outer, inner])
        }

        // Convolution / transposed convolution: derive output spatial dims.
        // Only handles fully-static inputs.  Higher-rank cases fall through to None.
        "Conv" | "ConvTranspose" | "ConvInteger" => {
            let ins = node.input.as_slice();
            if ins.len() < 2 {
                return None;
            }
            let x_shape = value_shapes.get(ins[0].as_str())?.clone();
            let w_shape = value_shapes.get(ins[1].as_str()).cloned().or_else(|| {
                initializers
                    .get(ins[1].as_str())
                    .map(|t| t.dims.as_slice().to_vec())
            })?;
            if x_shape.len() < 3 || w_shape.len() < 3 {
                return None;
            }
            let spatial_rank = x_shape.len() - 2;
            if w_shape.len() != x_shape.len() {
                return None;
            }

            // Read attributes.
            let mut auto_pad = String::from("NOTSET");
            let mut strides: Vec<i64> = vec![1; spatial_rank];
            let mut dilations: Vec<i64> = vec![1; spatial_rank];
            let mut pads: Vec<i64> = vec![0; 2 * spatial_rank];
            let mut kernel_shape: Vec<i64> = w_shape[2..].to_vec();
            let mut group: i64 = 1;
            let mut output_padding: Vec<i64> = vec![0; spatial_rank];
            let mut output_shape_attr: Vec<i64> = Vec::new();
            for attr in node.attribute.as_slice() {
                match attr.name.as_str() {
                    "auto_pad" => {
                        if let Ok(s) = String::from_utf8(attr.s.clone()) {
                            if !s.is_empty() {
                                auto_pad = s;
                            }
                        }
                    }
                    "strides" if !attr.ints.is_empty() => strides = attr.ints.clone(),
                    "dilations" if !attr.ints.is_empty() => dilations = attr.ints.clone(),
                    "pads" if !attr.ints.is_empty() => pads = attr.ints.clone(),
                    "kernel_shape" if !attr.ints.is_empty() => kernel_shape = attr.ints.clone(),
                    "group" if attr.i > 0 => group = attr.i,
                    "output_padding" if !attr.ints.is_empty() => output_padding = attr.ints.clone(),
                    "output_shape" if !attr.ints.is_empty() => {
                        output_shape_attr = attr.ints.clone()
                    }
                    _ => {}
                }
            }
            if strides.len() != spatial_rank
                || dilations.len() != spatial_rank
                || kernel_shape.len() != spatial_rank
                || pads.len() != 2 * spatial_rank
                || output_padding.len() != spatial_rank
            {
                return None;
            }
            let _ = group; // not needed for shape inference

            let transpose = op == "ConvTranspose";
            // Output channel count.
            let m = if transpose {
                // Filter layout for ConvTranspose: (C_in, M/group, kSpatial...).
                // M = w_shape[1] * group, but with default group=1 we just use w_shape[1].
                w_shape[1] * group
            } else {
                w_shape[0]
            };

            // If output_shape attr is provided (ConvTranspose), it directly tells us H/W.
            if transpose && !output_shape_attr.is_empty() {
                let sizes = if output_shape_attr.len() == spatial_rank {
                    output_shape_attr.clone()
                } else if output_shape_attr.len() == x_shape.len() {
                    output_shape_attr[2..].to_vec()
                } else {
                    return None;
                };
                let mut out = vec![x_shape[0], m];
                out.extend(sizes);
                return Some(out);
            }

            let mut out_spatial = Vec::with_capacity(spatial_rank);
            for i in 0..spatial_rank {
                let in_dim = x_shape[2 + i];
                let k = kernel_shape[i];
                let s = strides[i];
                let d = dilations[i];
                let dilated_k = d * (k - 1) + 1;

                let out_dim = match auto_pad.as_str() {
                    "SAME_UPPER" | "SAME_LOWER" if !transpose => {
                        // Standard "SAME": out = ceil(in / stride)
                        (in_dim + s - 1) / s
                    }
                    "SAME_UPPER" | "SAME_LOWER" if transpose => {
                        // For transpose: out = in * stride
                        in_dim * s
                    }
                    "VALID" if !transpose => (in_dim - dilated_k) / s + 1,
                    "VALID" if transpose => (in_dim - 1) * s + dilated_k,
                    _ => {
                        // explicit pads (NOTSET) - pads layout: [b1, b2, ..., bk, e1, e2, ..., ek]
                        let pad_begin = pads[i];
                        let pad_end = pads[i + spatial_rank];
                        if transpose {
                            (in_dim - 1) * s - pad_begin - pad_end + dilated_k + output_padding[i]
                        } else {
                            (in_dim + pad_begin + pad_end - dilated_k) / s + 1
                        }
                    }
                };
                if out_dim < 0 {
                    return None;
                }
                out_spatial.push(out_dim);
            }

            let mut out = vec![x_shape[0], m];
            out.extend(out_spatial);
            Some(out)
        }

        "Slice" => {
            let ins = node.input.as_slice();
            if ins.is_empty() {
                return None;
            }

            let input_shape = value_shapes.get(ins[0].as_str())?;

            let read_ints = |name: Option<&String>| -> Option<Vec<i64>> {
                if let Some(n) = name {
                    if let Some(v) = const_values.get(n) {
                        return Some(v.clone());
                    }
                    if let Some(t) = initializers.get(n) {
                        let raw = t.raw_data.as_slice();
                        if !raw.is_empty() {
                            if t.data_type == TensorProto_DataType::Int32 as i32 {
                                return Some(
                                    raw.chunks_exact(4)
                                        .map(|c| {
                                            i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64
                                        })
                                        .collect(),
                                );
                            } else {
                                return Some(
                                    raw.chunks_exact(8)
                                        .map(|c| {
                                            i64::from_le_bytes([
                                                c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7],
                                            ])
                                        })
                                        .collect(),
                                );
                            }
                        } else if !t.int64_data.as_slice().is_empty() {
                            return Some(t.int64_data.as_slice().to_vec());
                        } else if !t.int32_data.as_slice().is_empty() {
                            return Some(
                                t.int32_data.as_slice().iter().map(|&v| v as i64).collect(),
                            );
                        }
                    }
                }
                None
            };

            let starts = read_ints(ins.get(1))?;
            let ends = read_ints(ins.get(2))?;
            let axes =
                read_ints(ins.get(3)).unwrap_or_else(|| (0..input_shape.len() as i64).collect());
            let steps = read_ints(ins.get(4)).unwrap_or_else(|| vec![1; axes.len()]);

            if axes.len() != starts.len() || axes.len() != ends.len() || axes.len() != steps.len() {
                return None;
            }

            let mut output = input_shape.clone();
            for i in 0..axes.len() {
                let axis = if axes[i] < 0 {
                    (input_shape.len() as i64 + axes[i]) as usize
                } else {
                    axes[i] as usize
                };
                if axis >= output.len() {
                    return None;
                }

                let step = steps[i];
                if step <= 0 {
                    return None;
                }

                let dim = input_shape[axis];
                let mut start = starts[i];
                let mut end = ends[i];

                if start < 0 {
                    start += dim;
                }
                if end < 0 {
                    end += dim;
                }

                start = start.max(0);
                end = end.min(dim);

                if end < start {
                    output[axis] = 0;
                } else {
                    // Strided slices (e.g. sin/cos interleave `[..., 0::2]`).
                    output[axis] = (end - start + step - 1) / step;
                }
            }

            Some(output)
        }

        // Resize: when sizes/scales are known constants, output shape is computable.
        // U-Net decoders (e.g. RMBG) chain Resize -> Concat skip -> Conv -> Shape -> Resize.
        "Resize" => infer_resize_output_shape(node, value_shapes, initializers, const_values),

        _ => None,
    }
}

/// Decode a float16/float32/float64 tensor whose every element is integral
/// (e.g. Range start/delta scalars in fp16 exports) into exact i64 values.
fn integral_float_tensor_as_i64(tensor: &TensorProto) -> Option<Vec<i64>> {
    if tensor.dims.as_slice().contains(&0) {
        return None;
    }
    let raw = tensor.raw_data.as_slice();
    let floats: Vec<f64> = if tensor.data_type == TensorProto_DataType::Float as i32 {
        if !raw.is_empty() {
            raw.chunks_exact(4)
                .map(|c| f64::from(f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
                .collect()
        } else {
            tensor.float_data.iter().map(|&v| f64::from(v)).collect()
        }
    } else if tensor.data_type == TensorProto_DataType::Float16 as i32 {
        if !raw.is_empty() {
            raw.chunks_exact(2)
                .map(|c| f64::from(half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()))
                .collect()
        } else {
            tensor
                .int32_data
                .iter()
                .map(|&v| f64::from(half::f16::from_bits(v as u16).to_f32()))
                .collect()
        }
    } else if tensor.data_type == TensorProto_DataType::Double as i32 {
        if !raw.is_empty() {
            raw.chunks_exact(8)
                .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect()
        } else {
            tensor.double_data.clone()
        }
    } else {
        return None;
    };
    if floats.is_empty() || floats.iter().any(|v| v.fract() != 0.0 || !v.is_finite()) {
        return None;
    }
    Some(floats.into_iter().map(|v| v as i64).collect())
}

/// Scalar Range operand as i64: integer values from const/initializer maps,
/// or an integral float initializer (dynamo emits float Range for arange).
fn read_range_scalar(
    name: &str,
    initializers: &HashMap<String, &TensorProto>,
    const_values: &HashMap<String, Vec<i64>>,
) -> Option<i64> {
    if let Some(values) = read_int64_values_from_maps(name, initializers, const_values) {
        return values.first().copied();
    }
    integral_float_tensor_as_i64(initializers.get(name)?).and_then(|v| v.first().copied())
}

/// Scalar Range operand as f64 (any numeric initializer or folded constant),
/// used for output-length inference only.
fn read_range_scalar_f64(
    name: &str,
    initializers: &HashMap<String, &TensorProto>,
    const_values: &HashMap<String, Vec<i64>>,
) -> Option<f64> {
    if let Some(values) = read_int64_values_from_maps(name, initializers, const_values) {
        return values.first().map(|&v| v as f64);
    }
    let tensor = initializers.get(name)?;
    if tensor.dims.as_slice().contains(&0) {
        return None;
    }
    let raw = tensor.raw_data.as_slice();
    let dtype = tensor.data_type;
    if dtype == TensorProto_DataType::Float as i32 {
        if raw.len() >= 4 {
            return Some(f64::from(f32::from_le_bytes([
                raw[0], raw[1], raw[2], raw[3],
            ])));
        }
        return tensor.float_data.first().map(|&v| f64::from(v));
    }
    if dtype == TensorProto_DataType::Float16 as i32 {
        if raw.len() >= 2 {
            return Some(f64::from(
                half::f16::from_bits(u16::from_le_bytes([raw[0], raw[1]])).to_f32(),
            ));
        }
        return tensor
            .int32_data
            .first()
            .map(|&v| f64::from(half::f16::from_bits(v as u16).to_f32()));
    }
    if dtype == TensorProto_DataType::Double as i32 {
        if raw.len() >= 8 {
            return Some(f64::from_le_bytes([
                raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7],
            ]));
        }
        return tensor.double_data.first().copied();
    }
    None
}

/// Element type of a value from the tracked types or its initializer.
fn value_or_initializer_dtype(
    name: &str,
    value_types: &HashMap<String, DataType>,
    initializers: &HashMap<String, &TensorProto>,
) -> Option<DataType> {
    if let Some(dtype) = value_types.get(name) {
        return Some(*dtype);
    }
    let tensor = initializers.get(name)?;
    match tensor.data_type {
        x if x == TensorProto_DataType::Float as i32 => Some(DataType::Float32),
        x if x == TensorProto_DataType::Float16 as i32 => Some(DataType::Float16),
        x if x == TensorProto_DataType::Int32 as i32 => Some(DataType::Int32),
        x if x == TensorProto_DataType::Int64 as i32 => Some(DataType::Int64),
        x if x == TensorProto_DataType::Uint8 as i32 => Some(DataType::Uint8),
        x if x == TensorProto_DataType::Int8 as i32 => Some(DataType::Int8),
        x if x == TensorProto_DataType::Uint32 as i32 => Some(DataType::Uint32),
        x if x == TensorProto_DataType::Bool as i32 => Some(DataType::Uint8),
        _ => None,
    }
}

/// Output element type of a GatherBlockQuantized node: the scales' float type.
fn gather_block_quantized_dtype(
    node: &NodeProto,
    value_types: &HashMap<String, DataType>,
    initializers: &HashMap<String, &TensorProto>,
) -> Option<DataType> {
    let scales = node.input.as_slice().get(2)?;
    if let Some(dtype) = value_types.get(scales.as_str()) {
        return Some(*dtype);
    }
    let tensor = initializers.get(scales.as_str())?;
    if tensor.data_type == TensorProto_DataType::Float16 as i32 {
        Some(DataType::Float16)
    } else if tensor.data_type == TensorProto_DataType::Float as i32 {
        Some(DataType::Float32)
    } else {
        None
    }
}

fn read_int64_values_from_maps(
    name: &str,
    initializers: &HashMap<String, &TensorProto>,
    const_values: &HashMap<String, Vec<i64>>,
) -> Option<Vec<i64>> {
    if let Some(v) = const_values.get(name) {
        if v.is_empty() {
            return None;
        }
        return Some(v.clone());
    }
    let tensor = initializers.get(name)?;
    if tensor.dims.as_slice().contains(&0) {
        return None;
    }
    let raw = tensor.raw_data.as_slice();
    if !raw.is_empty() {
        if tensor.data_type == TensorProto_DataType::Int32 as i32 {
            return Some(
                raw.chunks_exact(4)
                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64)
                    .collect(),
            );
        }
        if tensor.data_type == TensorProto_DataType::Bool as i32
            || tensor.data_type == TensorProto_DataType::Uint8 as i32
            || tensor.data_type == TensorProto_DataType::Int8 as i32
        {
            return Some(raw.iter().map(|&b| i64::from(b)).collect());
        }
        // Only interpret raw bytes as packed i64 for actual int64 tensors;
        // decoding e.g. a bool mask this way poisons const folding.
        if tensor.data_type != TensorProto_DataType::Int64 as i32 {
            return None;
        }
        return Some(
            raw.chunks_exact(8)
                .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect(),
        );
    }
    if !tensor.int64_data.as_slice().is_empty() {
        return Some(tensor.int64_data.as_slice().to_vec());
    }
    if !tensor.int32_data.as_slice().is_empty() {
        return Some(
            tensor
                .int32_data
                .as_slice()
                .iter()
                .map(|&v| v as i64)
                .collect(),
        );
    }
    None
}

fn infer_resize_output_shape(
    node: &NodeProto,
    value_shapes: &HashMap<String, Vec<i64>>,
    initializers: &HashMap<String, &TensorProto>,
    const_values: &HashMap<String, Vec<i64>>,
) -> Option<Vec<i64>> {
    let ins = node.input.as_slice();
    let input_shape = value_shapes.get(ins.first()?.as_str())?;

    if let Some(sizes_name) = ins.get(3).filter(|s| !s.is_empty()) {
        if let Some(sizes) =
            read_int64_values_from_maps(sizes_name.as_str(), initializers, const_values)
        {
            if sizes.len() == input_shape.len() {
                return Some(sizes);
            }
        }
    }

    if let Some(scales_name) = ins.get(2).filter(|s| !s.is_empty()) {
        if let Some(scales) = read_float32_values_from_maps(scales_name.as_str(), initializers) {
            if scales.is_empty() {
                return None;
            }
            // ONNX Resize output size = floor(input_size * scale); nearest_mode only
            // affects which input pixel is sampled, not the output dimension.
            if scales.len() == input_shape.len() {
                let mut out = Vec::with_capacity(input_shape.len());
                for (in_dim, scale) in input_shape.iter().zip(scales.iter()) {
                    let scaled = (*in_dim as f32 * scale).floor() as i64;
                    out.push(scaled.max(1));
                }
                return Some(out);
            }
            if scales.len() == 2 && input_shape.len() == 4 {
                let mut out = input_shape.to_vec();
                for (axis, scale) in [(2usize, scales[0]), (3, scales[1])] {
                    out[axis] = ((out[axis] as f32) * scale).floor() as i64;
                    out[axis] = out[axis].max(1);
                }
                return Some(out);
            }
        }
    }

    None
}

fn infer_split_output_shapes(
    node: &NodeProto,
    value_shapes: &HashMap<String, Vec<i64>>,
    initializers: &HashMap<String, &TensorProto>,
    const_values: &HashMap<String, Vec<i64>>,
) -> Option<Vec<Vec<i64>>> {
    let inputs = node.input.as_slice();
    let input_shape = value_shapes.get(inputs.first()?.as_str())?;
    let output_count = node.output.len();
    if output_count == 0 {
        return None;
    }

    let mut axis = node
        .attribute
        .as_slice()
        .iter()
        .find(|attribute| attribute.name.as_str() == "axis")
        .map(|attribute| attribute.i)
        .unwrap_or(0);
    if axis < 0 {
        axis += input_shape.len() as i64;
    }
    let axis = usize::try_from(axis).ok()?;
    let axis_size = *input_shape.get(axis)?;

    let split_sizes = inputs
        .get(1)
        .filter(|name| !name.is_empty())
        .and_then(|name| read_int64_values_from_maps(name.as_str(), initializers, const_values))
        .or_else(|| {
            node.attribute
                .as_slice()
                .iter()
                .find(|attribute| attribute.name.as_str() == "split")
                .filter(|attribute| !attribute.ints.is_empty())
                .map(|attribute| attribute.ints.clone())
        })
        .unwrap_or_else(|| {
            let count = output_count as i64;
            if axis_size >= 0 && axis_size % count == 0 {
                vec![axis_size / count; output_count]
            } else {
                Vec::new()
            }
        });

    if split_sizes.len() != output_count
        || split_sizes.iter().any(|&size| size < 0)
        || split_sizes.iter().sum::<i64>() != axis_size
    {
        return None;
    }

    Some(
        split_sizes
            .into_iter()
            .map(|size| {
                let mut shape = input_shape.clone();
                shape[axis] = size;
                shape
            })
            .collect(),
    )
}

fn read_float32_values_from_maps(
    name: &str,
    initializers: &HashMap<String, &TensorProto>,
) -> Option<Vec<f32>> {
    let tensor = initializers.get(name)?;
    if tensor.data_type != TensorProto_DataType::Float as i32 {
        return None;
    }
    if !tensor.float_data.is_empty() {
        return Some(tensor.float_data.clone());
    }
    if !tensor.raw_data.is_empty() {
        return Some(
            tensor
                .raw_data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        );
    }
    None
}

fn shape_numel(shape: &[i64]) -> Option<usize> {
    shape.iter().try_fold(1usize, |acc, &d| {
        if d < 0 {
            return None;
        }
        usize::try_from(d).ok().map(|v| acc.saturating_mul(v))
    })
}

fn const_shape_for_folding(
    name: &str,
    values: &[i64],
    value_shapes: &HashMap<String, Vec<i64>>,
) -> Vec<i64> {
    if let Some(shape) = value_shapes.get(name) {
        if shape_numel(shape) == Some(values.len()) {
            return shape.clone();
        }
    }

    if values.len() == 1 {
        Vec::new()
    } else {
        vec![values.len() as i64]
    }
}

fn broadcast_shape(shape_a: &[i64], shape_b: &[i64]) -> Option<Vec<i64>> {
    let rank = shape_a.len().max(shape_b.len());
    let mut out_rev = Vec::with_capacity(rank);
    for i in 0..rank {
        let da = shape_a
            .get(shape_a.len().wrapping_sub(1 + i))
            .copied()
            .unwrap_or(1);
        let db = shape_b
            .get(shape_b.len().wrapping_sub(1 + i))
            .copied()
            .unwrap_or(1);
        if da <= 0 || db <= 0 {
            return None;
        }
        if da == db || da == 1 {
            out_rev.push(db);
        } else if db == 1 {
            out_rev.push(da);
        } else {
            return None;
        }
    }
    out_rev.reverse();
    Some(out_rev)
}

fn linear_index_for_broadcast_operand(
    out_linear_idx: usize,
    out_shape: &[i64],
    in_shape: &[i64],
) -> Option<usize> {
    if in_shape.is_empty() {
        return Some(0);
    }

    let in_rank = in_shape.len();
    let out_rank = out_shape.len();
    if in_rank > out_rank {
        return None;
    }

    let mut in_linear_idx = 0usize;
    let mut in_stride = 1usize;
    let mut rem = out_linear_idx;

    for out_axis_rev in 0..out_rank {
        let out_axis = out_rank - 1 - out_axis_rev;
        let out_dim = usize::try_from(out_shape[out_axis]).ok()?;
        if out_dim == 0 {
            return None;
        }
        let out_coord = rem % out_dim;
        rem /= out_dim;

        if out_axis_rev < in_rank {
            let in_axis = in_rank - 1 - out_axis_rev;
            let in_dim = usize::try_from(in_shape[in_axis]).ok()?;
            if in_dim == 0 {
                return None;
            }
            let in_coord = if in_dim == 1 { 0 } else { out_coord };
            in_linear_idx = in_linear_idx.saturating_add(in_coord.saturating_mul(in_stride));
            in_stride = in_stride.saturating_mul(in_dim);
        }
    }

    Some(in_linear_idx)
}

fn fold_binary_const_i64(
    op_type: &str,
    a_values: &[i64],
    b_values: &[i64],
    a_shape: &[i64],
    b_shape: &[i64],
) -> Option<(Vec<i64>, Vec<i64>)> {
    let out_shape = broadcast_shape(a_shape, b_shape)?;
    let out_numel = shape_numel(&out_shape)?;

    let mut out_values = Vec::with_capacity(out_numel);
    for out_idx in 0..out_numel {
        let a_idx = linear_index_for_broadcast_operand(out_idx, &out_shape, a_shape)?;
        let b_idx = linear_index_for_broadcast_operand(out_idx, &out_shape, b_shape)?;
        let av = *a_values.get(a_idx)?;
        let bv = *b_values.get(b_idx)?;
        let v = match op_type {
            "Add" => av + bv,
            "Sub" => av - bv,
            "Mul" => av * bv,
            "Div" => {
                if bv == 0 {
                    return None;
                }
                av / bv
            }
            "Mod" => {
                if bv == 0 {
                    return None;
                }
                // ONNX Mod fmod=0: result takes the divisor's sign.
                ((av % bv) + bv) % bv
            }
            "Equal" => {
                if av == bv {
                    1
                } else {
                    0
                }
            }
            _ => return None,
        };
        out_values.push(v);
    }

    Some((out_values, out_shape))
}

pub(crate) fn value_shape_dims_for<'a>(
    name: &str,
    value_shape_dims: &'a HashMap<String, Vec<Dimension>>,
) -> Option<&'a [Dimension]> {
    let sanitized = sanitize_identifier(name);
    let trimmed = name.trim_start_matches('/');
    value_shape_dims
        .get(name)
        .or_else(|| value_shape_dims.get(&sanitized))
        .or_else(|| value_shape_dims.get(trimmed))
        .map(Vec::as_slice)
}

fn dims_contain_dynamic(dims: &[Dimension]) -> bool {
    dims.iter().any(|d| matches!(d, Dimension::Dynamic(_)))
}

/// Propagate `value_shape_dims` through shape-preserving ONNX ops so downstream
/// Shape/Gather/Concat/Reshape chains retain dynamic batch/sequence metadata.
fn propagate_dynamic_dims_metadata(
    graph: &GraphProto,
    value_shape_dims: &mut HashMap<String, Vec<Dimension>>,
) {
    const PRESERVE_INPUT_SHAPE: &[&str] = &[
        "Abs",
        "Add",
        "Cast",
        "Div",
        "Equal",
        "Greater",
        "GreaterOrEqual",
        "LayerNormalization",
        "Less",
        "LessOrEqual",
        "Mul",
        "Neg",
        "Not",
        "Relu",
        "Sigmoid",
        "Sin",
        "Cos",
        "Sqrt",
        "Sub",
        "Softmax",
        "Tanh",
        "Where",
    ];

    for _ in 0..graph.node.as_slice().len().max(1) {
        let mut changed = false;
        for node in graph.node.as_slice() {
            let Some(out) = node.output.as_slice().first() else {
                continue;
            };
            if value_shape_dims.contains_key(out.as_str()) {
                continue;
            }

            let op_type = node.op_type.as_str();
            let input_dims = if op_type == "MatMul" || op_type == "Gemm" {
                node.input
                    .as_slice()
                    .first()
                    .and_then(|inp| value_shape_dims_for(inp.as_str(), value_shape_dims))
            } else if PRESERVE_INPUT_SHAPE.contains(&op_type) {
                node.input
                    .as_slice()
                    .first()
                    .and_then(|inp| value_shape_dims_for(inp.as_str(), value_shape_dims))
            } else {
                None
            };

            if let Some(dims) = input_dims {
                if !dims.is_empty() {
                    value_shape_dims.insert(out.to_string(), dims.to_vec());
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
}

pub(crate) fn parse_dynamic_dim_expr(dim_name: &str) -> (String, i64) {
    let s = dim_name.trim();
    if let Some((lhs, rhs)) = s.rsplit_once('+') {
        if let Ok(offset) = rhs.trim().parse::<i64>() {
            return (lhs.trim().to_string(), offset);
        }
    }
    if let Some((lhs, rhs)) = s.rsplit_once('-') {
        if let Ok(offset) = rhs.trim().parse::<i64>() {
            return (lhs.trim().to_string(), -offset);
        }
    }
    (s.to_string(), 0)
}

pub(crate) fn format_dynamic_dim_expr(base: &str, offset: i64) -> String {
    if offset > 0 {
        format!("{base} + {offset}")
    } else if offset < 0 {
        format!("{base} - {}", offset.abs())
    } else {
        base.to_string()
    }
}

fn parse_additive_dynamic_dim_expr(dim_name: &str) -> Option<(BTreeMap<String, i64>, i64)> {
    let expr = dim_name.trim();
    if expr.is_empty() {
        return None;
    }

    let normalized = expr.replace('+', " + ").replace('-', " - ");
    let mut terms = BTreeMap::new();
    let mut constant = 0i64;
    let mut sign = 1i64;
    let mut saw_term = false;

    for token in normalized.split_whitespace() {
        match token {
            "+" => sign = 1,
            "-" => sign = -1,
            _ => {
                saw_term = true;
                if let Ok(value) = token.parse::<i64>() {
                    constant += sign * value;
                } else {
                    *terms.entry(token.to_string()).or_insert(0) += sign;
                }
                sign = 1;
            }
        }
    }

    if !saw_term {
        return None;
    }

    terms.retain(|_, coeff| *coeff != 0);
    Some((terms, constant))
}

fn format_additive_dynamic_dim_expr(
    terms: &BTreeMap<String, i64>,
    constant: i64,
) -> Option<String> {
    if terms.is_empty() && constant == 0 {
        return None;
    }

    let mut out = String::new();
    for (name, coeff) in terms {
        for _ in 0..coeff.abs() {
            if out.is_empty() {
                if *coeff < 0 {
                    out.push_str("- ");
                }
                out.push_str(name);
            } else if *coeff < 0 {
                out.push_str(" - ");
                out.push_str(name);
            } else {
                out.push_str(" + ");
                out.push_str(name);
            }
        }
    }

    if constant != 0 {
        if out.is_empty() {
            out.push_str(&constant.to_string());
        } else if constant < 0 {
            out.push_str(" - ");
            out.push_str(&constant.abs().to_string());
        } else {
            out.push_str(" + ");
            out.push_str(&constant.to_string());
        }
    }

    Some(out)
}

fn is_runtime_resolvable_dynamic_dim_expr(dim_name: &str) -> bool {
    let s = dim_name.trim();
    if s.is_empty() || s.contains('*') || s.contains('/') {
        return false;
    }
    if let Some((lhs, rhs)) = s.rsplit_once('+') {
        return !lhs.trim().is_empty() && rhs.trim().parse::<i64>().is_ok();
    }
    if let Some((lhs, rhs)) = s.rsplit_once('-') {
        return !lhs.trim().is_empty() && rhs.trim().parse::<i64>().is_ok();
    }
    true
}

fn shift_dynamic_dimension(dim: &DynamicDimension, delta: i64) -> Option<DynamicDimension> {
    let (base, offset) = parse_dynamic_dim_expr(&dim.name);
    let name = format_dynamic_dim_expr(&base, offset.checked_add(delta)?);
    let shifted_max = (dim.max_size as i64).checked_add(delta)?.max(0);
    let max_size = u32::try_from(shifted_max).ok()?;
    Some(DynamicDimension { name, max_size })
}

pub(crate) fn dynamic_scalar_dimension_for_value(
    name: &str,
    value_shape_dims: &HashMap<String, Vec<Dimension>>,
) -> Option<DynamicDimension> {
    let dims = value_shape_dims_for(name, value_shape_dims)?;
    if dims.len() != 1 {
        return None;
    }
    match &dims[0] {
        Dimension::Dynamic(dim) => Some(dim.clone()),
        Dimension::Static(_) => None,
    }
}

fn dimension_vector_for_value(
    name: &str,
    const_values: &HashMap<String, Vec<i64>>,
    value_shape_dims: &HashMap<String, Vec<Dimension>>,
) -> Option<Vec<Dimension>> {
    if let Some(dims) = value_shape_dims_for(name, value_shape_dims) {
        return Some(dims.to_vec());
    }
    let values = const_values.get(name)?;
    values
        .iter()
        .map(|&v| u32::try_from(v).ok().map(Dimension::Static))
        .collect()
}

fn is_trivial_static_dimension_vector(dims: &[Dimension]) -> bool {
    !dims.is_empty() && dims.iter().all(|d| matches!(d, Dimension::Static(1)))
}

fn is_all_ones_shape_vector(values: &[i64]) -> bool {
    !values.is_empty() && values.iter().all(|&v| v == 1)
}

fn combine_binary_dimension(
    op_type: &str,
    dynamic: &DynamicDimension,
    static_value: i64,
    dynamic_on_lhs: bool,
) -> Option<Dimension> {
    match op_type {
        "Add" => shift_dynamic_dimension(dynamic, static_value).map(Dimension::Dynamic),
        "Sub" if dynamic_on_lhs => {
            shift_dynamic_dimension(dynamic, -static_value).map(Dimension::Dynamic)
        }
        "Mul" if static_value == 0 => Some(Dimension::Static(0)),
        "Mul" if static_value == 1 => Some(Dimension::Dynamic(dynamic.clone())),
        "Mul" if static_value > 1 => Some(Dimension::Dynamic(DynamicDimension {
            name: if dynamic_on_lhs {
                format!("{} * {}", dynamic.name, static_value)
            } else {
                format!("{} * {}", static_value, dynamic.name)
            },
            max_size: dynamic.max_size.saturating_mul(static_value as u32),
        })),
        "Div" if dynamic_on_lhs && static_value == 1 => Some(Dimension::Dynamic(dynamic.clone())),
        "Div" if dynamic_on_lhs && static_value > 1 => Some(Dimension::Dynamic(DynamicDimension {
            name: format!("{} / {}", dynamic.name, static_value),
            max_size: dynamic.max_size / (static_value as u32),
        })),
        _ => None,
    }
}

fn combine_dynamic_dimensions(
    op_type: &str,
    lhs: &DynamicDimension,
    rhs: &DynamicDimension,
    lhs_value: i64,
    rhs_value: i64,
) -> Option<Dimension> {
    match op_type {
        "Add" | "Sub" => {
            let (mut terms, mut constant) = parse_additive_dynamic_dim_expr(&lhs.name)?;
            let (rhs_terms, rhs_constant) = parse_additive_dynamic_dim_expr(&rhs.name)?;
            let rhs_sign = if op_type == "Add" { 1 } else { -1 };

            for (name, coeff) in rhs_terms {
                *terms.entry(name).or_insert(0) += rhs_sign * coeff;
            }
            constant += rhs_sign * rhs_constant;
            terms.retain(|_, coeff| *coeff != 0);

            let value = if op_type == "Add" {
                lhs_value.checked_add(rhs_value)?
            } else {
                lhs_value.checked_sub(rhs_value)?
            };
            if terms.is_empty() {
                return u32::try_from(value).ok().map(Dimension::Static);
            }

            let name = format_additive_dynamic_dim_expr(&terms, constant)?;
            let max_size = u32::try_from(value).ok()?;
            Some(Dimension::Dynamic(DynamicDimension { name, max_size }))
        }
        _ => None,
    }
}

fn fold_binary_dynamic_dims(
    op_type: &str,
    a_values: &[i64],
    b_values: &[i64],
    a_shape: &[i64],
    b_shape: &[i64],
    a_dims: Option<&[Dimension]>,
    b_dims: Option<&[Dimension]>,
) -> Option<Vec<Dimension>> {
    let out_shape = broadcast_shape(a_shape, b_shape)?;
    let out_numel = shape_numel(&out_shape)?;
    let mut out_dims = Vec::with_capacity(out_numel);
    let mut has_dynamic = false;

    for out_idx in 0..out_numel {
        let a_idx = linear_index_for_broadcast_operand(out_idx, &out_shape, a_shape)?;
        let b_idx = linear_index_for_broadcast_operand(out_idx, &out_shape, b_shape)?;
        let av = *a_values.get(a_idx)?;
        let bv = *b_values.get(b_idx)?;
        let a_dim = a_dims.and_then(|dims| dims.get(a_idx));
        let b_dim = b_dims.and_then(|dims| dims.get(b_idx));

        let out_dim = match (a_dim, b_dim) {
            (Some(Dimension::Dynamic(dynamic)), Some(Dimension::Static(_)))
            | (Some(Dimension::Dynamic(dynamic)), None) => {
                let dim = combine_binary_dimension(op_type, dynamic, bv, true)?;
                has_dynamic |= matches!(dim, Dimension::Dynamic(_));
                dim
            }
            (Some(Dimension::Static(_)), Some(Dimension::Dynamic(dynamic)))
            | (None, Some(Dimension::Dynamic(dynamic))) => {
                let dim = combine_binary_dimension(op_type, dynamic, av, false)?;
                has_dynamic |= matches!(dim, Dimension::Dynamic(_));
                dim
            }
            (Some(Dimension::Dynamic(a_dynamic)), Some(Dimension::Dynamic(b_dynamic))) => {
                let dim = combine_dynamic_dimensions(op_type, a_dynamic, b_dynamic, av, bv)?;
                has_dynamic |= matches!(dim, Dimension::Dynamic(_));
                dim
            }
            _ => {
                let value = match op_type {
                    "Add" => av + bv,
                    "Sub" => av - bv,
                    "Mul" => av * bv,
                    "Div" => {
                        if bv == 0 {
                            return None;
                        }
                        av / bv
                    }
                    _ => return None,
                };
                Dimension::Static(u32::try_from(value).ok()?)
            }
        };

        out_dims.push(out_dim);
    }

    has_dynamic.then_some(out_dims)
}

pub(crate) fn dynamic_range_length_dimension(
    start: i64,
    delta: i64,
    start_dim: Option<&DynamicDimension>,
    limit: &DynamicDimension,
) -> Option<DynamicDimension> {
    if delta != 1 {
        return None;
    }

    let (mut terms, mut constant) = parse_additive_dynamic_dim_expr(&limit.name)?;
    if let Some(start_dim) = start_dim {
        let (start_terms, start_constant) = parse_additive_dynamic_dim_expr(&start_dim.name)?;
        for (name, coeff) in start_terms {
            *terms.entry(name).or_insert(0) -= coeff;
        }
        constant -= start_constant;
    } else {
        constant -= start;
    }
    terms.retain(|_, coeff| *coeff != 0);
    if terms.is_empty() {
        return None;
    }

    let name = format_additive_dynamic_dim_expr(&terms, constant)?;
    if !is_runtime_resolvable_dynamic_dim_expr(&name) {
        return None;
    }

    let max_size = u32::try_from((limit.max_size as i64).checked_sub(start)?).ok()?;
    Some(DynamicDimension { name, max_size })
}
/// Options controlling shape-subgraph constant folding.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FoldShapeConstantsOptions {
    pub require_positive_dims: bool,
    pub experimental_dynamic_inputs: bool,
    pub fold_where_values: bool,
    pub fold_reshape: bool,
    pub fold_unsqueeze_axes: bool,
}

impl FoldShapeConstantsOptions {
    fn early_pass() -> Self {
        Self {
            require_positive_dims: false,
            experimental_dynamic_inputs: false,
            fold_where_values: true,
            fold_reshape: true,
            fold_unsqueeze_axes: true,
        }
    }

    fn from_propagate(opts: &PropagateOptions) -> Self {
        Self {
            require_positive_dims: true,
            experimental_dynamic_inputs: opts.experimental_dynamic_inputs,
            // Where must stay foldable here: the early pass may have folded it
            // from placeholder shapes, and only a refresh with resolved inputs
            // can correct the value (e.g. torch expand_as target vectors).
            fold_where_values: true,
            fold_reshape: false,
            fold_unsqueeze_axes: false,
        }
    }
}

fn producer_of<'a>(graph: &'a GraphProto, output: &str) -> Option<&'a NodeProto> {
    graph
        .node
        .as_slice()
        .iter()
        .find(|n| n.output.as_slice().first().map(|s| s.as_str()) == Some(output))
}

/// Resolve scalar shape values through Unsqueeze wrappers (common in shape-vector Concat).
fn const_values_for_input(
    graph: &GraphProto,
    name: &str,
    const_values: &HashMap<String, Vec<i64>>,
) -> Option<Vec<i64>> {
    if let Some(v) = const_values.get(name) {
        return Some(v.clone());
    }
    if let Some(node) = producer_of(graph, name) {
        if node.op_type.as_str() == "Unsqueeze" {
            if let Some(inp) = node.input.as_slice().first() {
                return const_values.get(inp.as_str()).cloned();
            }
        }
    }
    None
}

fn value_shape_dims_for_input<'a>(
    graph: &GraphProto,
    name: &str,
    value_shape_dims: &'a HashMap<String, Vec<Dimension>>,
) -> Option<&'a [Dimension]> {
    if let Some(dims) = value_shape_dims_for(name, value_shape_dims) {
        return Some(dims);
    }
    if let Some(node) = producer_of(graph, name) {
        if node.op_type.as_str() == "Unsqueeze" {
            if let Some(inp) = node.input.as_slice().first() {
                return value_shape_dims_for(inp, value_shape_dims);
            }
        }
    }
    None
}

/// Parse ConstantOfShape `value` attribute (default: float32 zero).
fn constant_of_shape_fill(node: &NodeProto) -> (DataType, i64) {
    // ONNX default when `value` is absent: float32 zero.
    let mut fill_value: i64 = 0f32.to_bits() as i64;
    let mut dtype = DataType::Float32;
    for attr in node.attribute.as_slice() {
        if attr.name.as_str() != "value" {
            continue;
        }
        let Some(t) = attr.t.as_ref() else {
            continue;
        };
        match t.data_type {
            x if x == TensorProto_DataType::Float as i32 => {
                dtype = DataType::Float32;
                if !t.float_data.as_slice().is_empty() {
                    fill_value = t.float_data.as_slice()[0].to_bits() as i64;
                } else if t.raw_data.as_slice().len() >= 4 {
                    let raw = &t.raw_data.as_slice()[..4];
                    fill_value = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]) as i64;
                } else {
                    fill_value = 0f32.to_bits() as i64;
                }
            }
            x if x == TensorProto_DataType::Int64 as i32 => {
                dtype = DataType::Int64;
                if !t.int64_data.as_slice().is_empty() {
                    fill_value = t.int64_data.as_slice()[0];
                } else if t.raw_data.as_slice().len() >= 8 {
                    let raw = &t.raw_data.as_slice()[..8];
                    fill_value = i64::from_le_bytes([
                        raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7],
                    ]);
                }
            }
            _ => {}
        }
    }
    (dtype, fill_value)
}

/// Fold integer shape subgraphs (Shape/Gather/Concat/Range/Where/...).
fn fold_shape_constants(
    graph: &GraphProto,
    value_shapes: &mut HashMap<String, Vec<i64>>,
    value_types: &mut HashMap<String, DataType>,
    const_values: &mut HashMap<String, Vec<i64>>,
    value_shape_dims: &mut HashMap<String, Vec<Dimension>>,
    options: &FoldShapeConstantsOptions,
) -> bool {
    let consts_before = const_values.len();
    let mut any_folded = false;
    let initializers: HashMap<String, &TensorProto> = graph
        .initializer
        .as_slice()
        .iter()
        .map(|t| (t.name.to_string(), t))
        .collect();
    let initializers = &initializers;

    // Cascade Shape -> Slice -> Concat -> Cast within one propagation pass.
    for _ in 0..16 {
        let pass_before = const_values.len();

        for node in graph.node.as_slice() {
            let op_type = node.op_type.as_str();
            let outputs = node.output.as_slice();
            if outputs.is_empty() {
                continue;
            }
            // Shape-derived constants may have been seeded before later propagation
            // corrected an upstream tensor shape. Recompute the common shape-vector
            // chain instead of keeping stale values (e.g. Resize target sizes).
            let refresh_shape_value = matches!(
                op_type,
                "Shape"
                    | "Gather"
                    | "Unsqueeze"
                    | "Squeeze"
                    | "Slice"
                    | "Concat"
                    | "Cast"
                    | "Transpose"
                    | "Expand"
                    | "Equal"
                    | "Where"
                    | "Range"
                    | "ConstantOfShape"
                    | "Add"
                    | "Sub"
                    | "Mul"
                    | "Div"
                    | "Neg"
                    | "Reshape"
            );
            if const_values.contains_key(outputs[0].as_str()) && !refresh_shape_value {
                continue;
            }

            if options.fold_reshape && op_type == "Reshape" {
                let inputs = node.input.as_slice();
                if inputs.len() >= 2 {
                    if let (Some(data), Some(mut target)) = (
                        const_values.get(inputs[0].as_str()).cloned(),
                        const_values.get(inputs[1].as_str()).cloned(),
                    ) {
                        // An empty const for a non-empty tensor is a stale
                        // placeholder from an earlier pass; wait for the refresh.
                        let input_numel: i64 = value_shapes
                            .get(inputs[0].as_str())
                            .map(|s| s.iter().product())
                            .unwrap_or(0);
                        if data.is_empty() && input_numel > 0 {
                            continue;
                        }
                        if target.contains(&-1) {
                            let total: i64 = if data.is_empty() {
                                1
                            } else {
                                data.len() as i64
                            };
                            let known: i64 = target.iter().filter(|&&d| d != -1).product();
                            if known != 0 {
                                if let Some(idx) = target.iter().position(|&d| d == -1) {
                                    target[idx] = total / known;
                                }
                            }
                        }
                        let out_name = outputs[0].to_string();
                        const_values.insert(out_name.clone(), data);
                        value_shapes.insert(out_name, target);
                    }
                }
                continue;
            }

            if options.fold_unsqueeze_axes && op_type == "Unsqueeze" {
                let inputs = node.input.as_slice();
                if let Some(data) = inputs
                    .first()
                    .and_then(|i| const_values.get(i.as_str()).cloned())
                {
                    let mut axes: Vec<i64> = node
                        .attribute
                        .as_slice()
                        .iter()
                        .find(|a| a.name.as_str() == "axes")
                        .map(|a| a.ints.clone())
                        .unwrap_or_default();
                    if axes.is_empty() && inputs.len() > 1 {
                        axes = const_values
                            .get(inputs[1].as_str())
                            .cloned()
                            .unwrap_or_default();
                    }
                    if !axes.is_empty() {
                        let mut sorted_axes = axes;
                        sorted_axes.sort();
                        let mut out_shape = value_shapes
                            .get(inputs[0].as_str())
                            .cloned()
                            .unwrap_or_else(|| {
                                if data.len() <= 1 {
                                    Vec::new()
                                } else {
                                    vec![data.len() as i64]
                                }
                            });
                        for axis in sorted_axes {
                            let idx = if axis < 0 {
                                (out_shape.len() as i64 + axis + 1) as usize
                            } else {
                                axis as usize
                            };
                            if idx <= out_shape.len() {
                                out_shape.insert(idx, 1);
                            }
                        }
                        let out_name = outputs[0].to_string();
                        const_values.insert(out_name.clone(), data);
                        value_shapes.insert(out_name, out_shape);
                        continue;
                    }
                }
            }

            if op_type == "Where" && options.fold_where_values {
                let inputs = node.input.as_slice();
                if inputs.len() < 3 {
                    continue;
                }

                let cond = const_values.get(inputs[0].as_str()).cloned();
                let a = const_values.get(inputs[1].as_str()).cloned();
                let b = const_values.get(inputs[2].as_str()).cloned();
                let cond_is_const = cond.is_some();

                // Case 1: All inputs are constant - evaluate fully
                if let (Some(cond), Some(a), Some(b)) = (cond, a, b) {
                    if cond.len() != a.len() || a.len() != b.len() {
                        continue;
                    }

                    // Prefer a non-all-ones branch over an all-ones placeholder when
                    // folding shape vectors (e.g. Where(cond, [1,1,1], [1,32,1])).
                    let mut out = if is_all_ones_shape_vector(&a) && !is_all_ones_shape_vector(&b) {
                        b
                    } else if is_all_ones_shape_vector(&b) && !is_all_ones_shape_vector(&a) {
                        a
                    } else {
                        let mut result = Vec::with_capacity(a.len());
                        for i in 0..a.len() {
                            result.push(if cond[i] != 0 { a[i] } else { b[i] });
                        }
                        result
                    };

                    // Resolve reshape placeholders (-1) from a consumer Expand's data shape.
                    if out.contains(&-1) && !outputs.is_empty() {
                        let output_name = outputs[0].as_str();
                        for node in graph.node.as_slice() {
                            if node.op_type.as_str() == "Expand"
                                && node.input.len() >= 2
                                && node.input[1].as_str() == output_name
                            {
                                let data_input = node.input[0].as_str();
                                if let Some(data_shape) = value_shapes.get(data_input) {
                                    if out.len() == data_shape.len() {
                                        for i in 0..out.len() {
                                            if out[i] == -1 {
                                                out[i] = data_shape[i];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    let out_name = outputs[0].to_string();
                    let shape = if out.len() == 1 {
                        Vec::new()
                    } else {
                        vec![out.len() as i64]
                    };
                    const_values.insert(out_name.clone(), out);
                    value_shapes.insert(out_name, shape);
                } else {
                    // Case 2: Mixed constant/dynamic inputs. Prefer a concrete shape over an
                    // all-ones placeholder. When the condition is dynamic, never bake the
                    // placeholder branch into const_values.
                    let a_const = const_values.get(inputs[1].as_str());
                    let b_const = const_values.get(inputs[2].as_str());
                    let a_shape = value_shapes.get(inputs[1].as_str());
                    let b_shape = value_shapes.get(inputs[2].as_str());

                    let preferred_values = if cond_is_const {
                        if let (Some(a_vals), None) = (a_const, b_const) {
                            if is_all_ones_shape_vector(a_vals) && b_shape.is_some() {
                                b_shape.cloned()
                            } else {
                                Some(a_vals.clone())
                            }
                        } else if let (None, Some(b_vals)) = (a_const, b_const) {
                            if is_all_ones_shape_vector(b_vals) && a_shape.is_some() {
                                a_shape.cloned()
                            } else {
                                Some(b_vals.clone())
                            }
                        } else {
                            None
                        }
                    } else if let (Some(a_vals), None) = (a_const, b_const) {
                        if is_all_ones_shape_vector(a_vals) {
                            b_shape.cloned()
                        } else {
                            None
                        }
                    } else if let (None, Some(b_vals)) = (a_const, b_const) {
                        if is_all_ones_shape_vector(b_vals) {
                            a_shape.cloned()
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    if let Some(values) = preferred_values {
                        let out_name = outputs[0].to_string();
                        let shape = if values.len() == 1 {
                            Vec::new()
                        } else {
                            vec![values.len() as i64]
                        };
                        const_values.insert(out_name.clone(), values);
                        value_shapes.insert(out_name, shape);
                    }
                }
                continue;
            }

            if op_type == "Shape" {
                if let (Some(inp), Some(out)) = (
                    node.input.as_slice().first(),
                    node.output.as_slice().first(),
                ) {
                    let out = out.to_string();
                    if let Some(shape) = value_shapes.get(inp).cloned() {
                        if !options.require_positive_dims || shape.iter().all(|d| *d > 0) {
                            // Propagate dynamic dim metadata: Shape output is a 1-D
                            // tensor whose elements correspond to input dimensions.
                            if options.experimental_dynamic_inputs {
                                let inp_s = inp.to_string();
                                if let Some(dims) = value_shape_dims
                                    .get(&inp_s)
                                    .or_else(|| value_shape_dims.get(&sanitize_identifier(&inp_s)))
                                {
                                    // Each element of the Shape output corresponds to one
                                    // input dimension.  Build a 1-D dim vector where
                                    // dynamic input dims become Dynamic elements.
                                    let out_dims: Vec<rustnn::graph::Dimension> = dims
                                        .iter()
                                        .map(|d| match d {
                                            rustnn::graph::Dimension::Dynamic(dd) => {
                                                rustnn::graph::Dimension::Dynamic(dd.clone())
                                            }
                                            rustnn::graph::Dimension::Static(v) => {
                                                rustnn::graph::Dimension::Static(*v)
                                            }
                                        })
                                        .collect();
                                    value_shape_dims.insert(out.clone(), out_dims);
                                }
                            }
                            let (start, end) = shape_attr_range(node, shape.len());
                            let sliced = shape[start..end].to_vec();
                            const_values.insert(out.clone(), sliced.clone());
                            let inferred_shape = vec![sliced.len() as i64];
                            // Force the correct shape - Shape operation computes exact output shape
                            value_shapes.insert(out.clone(), inferred_shape.clone());
                            value_shapes.insert(sanitize_identifier(&out), inferred_shape);
                            value_types.insert(out, DataType::Int64);
                        }
                    }
                }
            } else if op_type == "Gather" {
                if let (Some(data_name), Some(indices_name), Some(out)) = (
                    node.input.as_slice().first(),
                    node.input.as_slice().get(1),
                    node.output.as_slice().first(),
                ) {
                    if let (Some(data), Some(indices)) =
                        (const_values.get(data_name), const_values.get(indices_name))
                    {
                        let axis = node
                            .attribute
                            .as_slice()
                            .iter()
                            .find(|a| a.name.as_str() == "axis" && a.i != 0)
                            .map(|a| a.i)
                            .unwrap_or(0);

                        if axis == 0 {
                            let mut gathered = Vec::new();
                            let mut gathered_dims = Vec::new();
                            let data_dims = if options.experimental_dynamic_inputs {
                                value_shape_dims
                                    .get(data_name)
                                    .or_else(|| {
                                        value_shape_dims.get(&sanitize_identifier(data_name))
                                    })
                                    .cloned()
                            } else {
                                None
                            };
                            for &idx in indices {
                                let i = if idx < 0 {
                                    (data.len() as i64 + idx) as usize
                                } else {
                                    idx as usize
                                };
                                if let Some(v) = data.get(i) {
                                    gathered.push(*v);
                                    if let Some(ref dd) = data_dims {
                                        if let Some(dim) = dd.get(i) {
                                            gathered_dims.push(dim.clone());
                                        }
                                    }
                                }
                            }
                            if !gathered.is_empty() {
                                if options.experimental_dynamic_inputs
                                    && gathered_dims.len() == gathered.len()
                                    && gathered_dims
                                        .iter()
                                        .any(|d| matches!(d, rustnn::graph::Dimension::Dynamic(_)))
                                {
                                    value_shape_dims.insert(out.to_string(), gathered_dims);
                                }
                                const_values.insert(out.to_string(), gathered.clone());
                                let out_shape = if gathered.len() == 1 {
                                    Vec::new()
                                } else {
                                    vec![gathered.len() as i64]
                                };
                                // Force the correct shape - Gather operation computes exact output shape
                                value_shapes.insert(out.to_string(), out_shape.clone());
                                value_shapes.insert(sanitize_identifier(out), out_shape);
                                value_types.insert(out.to_string(), DataType::Int64);
                            }
                        }
                    }
                }
            } else if matches!(op_type, "Add" | "Sub" | "Mul" | "Div" | "Mod") {
                if node.input.as_slice().len() >= 2 {
                    if let (Some(a_name), Some(b_name), Some(out)) = (
                        node.input.as_slice().first(),
                        node.input.as_slice().get(1),
                        node.output.as_slice().first(),
                    ) {
                        let a = const_values.get(a_name);
                        let b = const_values.get(b_name);
                        if let (Some(a), Some(b)) = (a, b) {
                            let a_shape = const_shape_for_folding(a_name, a, value_shapes);
                            let b_shape = const_shape_for_folding(b_name, b, value_shapes);
                            if let Some((result_vals, out_shape)) =
                                fold_binary_const_i64(op_type, a, b, &a_shape, &b_shape)
                            {
                                if options.experimental_dynamic_inputs {
                                    let a_dims = value_shape_dims_for(a_name, value_shape_dims);
                                    let b_dims = value_shape_dims_for(b_name, value_shape_dims);
                                    if let Some(out_dims) = fold_binary_dynamic_dims(
                                        op_type, a, b, &a_shape, &b_shape, a_dims, b_dims,
                                    ) {
                                        value_shape_dims.insert(out.to_string(), out_dims);
                                    }
                                }
                                const_values.insert(out.to_string(), result_vals.clone());
                                // Force the correct shape - Binary operations compute exact output shape
                                value_shapes.insert(out.to_string(), out_shape.clone());
                                value_shapes.insert(sanitize_identifier(out), out_shape);
                                if let Some(dtype) = node
                                    .input
                                    .as_slice()
                                    .iter()
                                    .find_map(|i| value_types.get(i).cloned())
                                {
                                    value_types.insert(out.to_string(), dtype);
                                }
                            }
                        }
                    }
                }
            } else if op_type == "Unsqueeze" || op_type == "Squeeze" {
                if let (Some(inp), Some(out)) = (
                    node.input.as_slice().first(),
                    node.output.as_slice().first(),
                ) {
                    if let Some(vals) = const_values.get(inp).cloned() {
                        // Propagate dynamic dim metadata
                        if options.experimental_dynamic_inputs {
                            if let Some(dims) = value_shape_dims
                                .get(inp)
                                .or_else(|| value_shape_dims.get(&sanitize_identifier(inp)))
                                .cloned()
                            {
                                value_shape_dims.insert(out.to_string(), dims);
                            }
                        }
                        const_values.insert(out.to_string(), vals.clone());
                        // Track the real rank change: consumers like Tile need
                        // the unsqueezed shape, not a flattened vector.
                        let base_shape = value_shapes.get(inp).cloned().unwrap_or_else(|| {
                            if vals.len() == 1 {
                                Vec::new()
                            } else {
                                vec![vals.len() as i64]
                            }
                        });
                        let mut axes: Vec<i64> = node
                            .attribute
                            .as_slice()
                            .iter()
                            .find(|a| a.name.as_str() == "axes")
                            .map(|a| a.ints.clone())
                            .unwrap_or_default();
                        if axes.is_empty() {
                            if let Some(axes_name) = node.input.as_slice().get(1) {
                                axes = const_values
                                    .get(axes_name.as_str())
                                    .cloned()
                                    .unwrap_or_default();
                            }
                        }
                        let out_shape = if op_type == "Unsqueeze" {
                            let out_rank = base_shape.len() as i64 + axes.len() as i64;
                            let mut norm: Vec<i64> = axes
                                .iter()
                                .map(|&a| if a < 0 { a + out_rank } else { a })
                                .collect();
                            norm.sort_unstable();
                            let mut s = base_shape;
                            for a in norm {
                                let idx = a.clamp(0, s.len() as i64) as usize;
                                s.insert(idx, 1);
                            }
                            s
                        } else if axes.is_empty() {
                            base_shape.into_iter().filter(|&d| d != 1).collect()
                        } else {
                            let rank = base_shape.len() as i64;
                            let norm: Vec<usize> = axes
                                .iter()
                                .map(|&a| (if a < 0 { a + rank } else { a }) as usize)
                                .collect();
                            base_shape
                                .into_iter()
                                .enumerate()
                                .filter(|(i, _)| !norm.contains(i))
                                .map(|(_, d)| d)
                                .collect()
                        };
                        value_shapes.insert(out.to_string(), out_shape);
                        if let Some(dtype) = value_types.get(inp).cloned() {
                            value_types.insert(out.to_string(), dtype);
                        }
                    }
                }
            } else if op_type == "Cast" {
                // Integer Cast is common in Resize sizes subgraphs (Gather/Concat -> Cast -> Concat).
                if let (Some(inp), Some(out)) = (
                    node.input.as_slice().first(),
                    node.output.as_slice().first(),
                ) {
                    let to_type = node
                        .attribute
                        .as_slice()
                        .iter()
                        .find(|a| a.name.as_str() == "to")
                        .map(|a| a.i)
                        .unwrap_or(0);
                    let to_int = to_type == TensorProto_DataType::Int64 as i64
                        || to_type == TensorProto_DataType::Int32 as i64
                        || to_type == TensorProto_DataType::Bool as i64
                        || to_type == 0; // missing attr: still allow int const passthrough
                                         // Integer consts survive a cast to float exactly (e.g. a folded
                                         // sequence length feeding a float Range limit).
                    let to_float = to_type == TensorProto_DataType::Float as i64
                        || to_type == TensorProto_DataType::Float16 as i64
                        || to_type == TensorProto_DataType::Double as i64;
                    if to_int || to_float {
                        // fp16 exports route Range scalars through Cast(float16 init);
                        // integral float initializers fold exactly.
                        let folded = const_values.get(inp).cloned().or_else(|| {
                            graph
                                .initializer
                                .iter()
                                .find(|t| t.name == *inp)
                                .and_then(|t| integral_float_tensor_as_i64(t))
                        });
                        if let Some(vals) = folded {
                            if options.experimental_dynamic_inputs {
                                if let Some(dims) = value_shape_dims
                                    .get(inp)
                                    .or_else(|| value_shape_dims.get(&sanitize_identifier(inp)))
                                    .cloned()
                                {
                                    value_shape_dims.insert(out.to_string(), dims);
                                }
                            }
                            const_values.insert(out.to_string(), vals.clone());
                            let out_shape = value_shapes.get(inp).cloned().unwrap_or_else(|| {
                                if vals.len() <= 1 {
                                    Vec::new()
                                } else {
                                    vec![vals.len() as i64]
                                }
                            });
                            value_shapes.insert(out.to_string(), out_shape);
                            let out_type = if to_type == TensorProto_DataType::Float16 as i64 {
                                DataType::Float16
                            } else if to_float {
                                DataType::Float32
                            } else {
                                DataType::Int64
                            };
                            value_types.insert(out.to_string(), out_type);
                        }
                    }
                }
            } else if op_type == "Neg" {
                if let (Some(inp), Some(out)) = (
                    node.input.as_slice().first(),
                    node.output.as_slice().first(),
                ) {
                    if let Some(vals) = const_values.get(inp.as_str()).cloned() {
                        const_values.insert(out.to_string(), vals.iter().map(|v| -v).collect());
                        if let Some(shape) = value_shapes.get(inp.as_str()).cloned() {
                            value_shapes.insert(out.to_string(), shape);
                        }
                        if let Some(dtype) = value_types.get(inp.as_str()).cloned() {
                            value_types.insert(out.to_string(), dtype);
                        }
                    }
                }
            } else if op_type == "Transpose" {
                // Fold Transpose over small 2-D integer consts (pad-spec chains:
                // build [[l,r],...] -> reverse -> transpose -> flatten).
                if let (Some(inp), Some(out)) = (
                    node.input.as_slice().first(),
                    node.output.as_slice().first(),
                ) {
                    if let (Some(data), Some(shape)) = (
                        const_values.get(inp.as_str()).cloned(),
                        value_shapes.get(inp.as_str()).cloned(),
                    ) {
                        let perm: Vec<usize> = node
                            .attribute
                            .as_slice()
                            .iter()
                            .find(|a| a.name.as_str() == "perm")
                            .map(|a| a.ints.iter().map(|&i| i as usize).collect())
                            .unwrap_or_else(|| (0..shape.len()).rev().collect());
                        if shape.len() == 2
                            && perm == [1, 0]
                            && shape.iter().all(|&d| d > 0)
                            && (shape[0] * shape[1]) as usize == data.len()
                        {
                            let (r, c) = (shape[0] as usize, shape[1] as usize);
                            let mut t = Vec::with_capacity(data.len());
                            for j in 0..c {
                                for i in 0..r {
                                    t.push(data[i * c + j]);
                                }
                            }
                            const_values.insert(out.to_string(), t);
                            value_shapes.insert(out.to_string(), vec![shape[1], shape[0]]);
                            value_types.insert(out.to_string(), DataType::Int64);
                        }
                    }
                }
            } else if op_type == "Slice" {
                // Fold Slice over integer shape vectors (common for Resize sizes = Concat(Slice(Shape), HW)).
                let inputs = node.input.as_slice();
                if let (Some(data_name), Some(out)) =
                    (inputs.first(), node.output.as_slice().first())
                {
                    if let Some(data) = const_values.get(data_name.as_str()).cloned() {
                        let read_ints = |idx: usize| -> Option<Vec<i64>> {
                            let name = inputs.get(idx)?;
                            const_values.get(name.as_str()).cloned()
                        };
                        if let (Some(starts), Some(ends)) = (read_ints(1), read_ints(2)) {
                            let axes =
                                read_ints(3).unwrap_or_else(|| (0..starts.len() as i64).collect());
                            let steps = read_ints(4).unwrap_or_else(|| vec![1; starts.len()]);
                            if axes.len() == starts.len()
                                && ends.len() == starts.len()
                                && steps.len() == starts.len()
                            {
                                // Full reverse along axis 0 (torch.flip export):
                                // reorder rows of the (flat) const payload.
                                if axes == [0] && steps == [-1] && starts.len() == 1 {
                                    let dim0 = value_shapes
                                        .get(data_name.as_str())
                                        .and_then(|s| s.first().copied())
                                        .unwrap_or(data.len() as i64);
                                    let covers_all = (starts[0] == -1 || starts[0] >= dim0 - 1)
                                        && (ends[0] <= -dim0 || ends[0] <= i64::MIN + 1);
                                    if covers_all && dim0 > 0 && data.len() as i64 % dim0 == 0 {
                                        let row = data.len() / dim0 as usize;
                                        let mut rev = Vec::with_capacity(data.len());
                                        for chunk in data.chunks(row).rev() {
                                            rev.extend_from_slice(chunk);
                                        }
                                        const_values.insert(out.to_string(), rev);
                                        if let Some(shape) =
                                            value_shapes.get(data_name.as_str()).cloned()
                                        {
                                            value_shapes.insert(out.to_string(), shape);
                                        }
                                        value_types.insert(out.to_string(), DataType::Int64);
                                    }
                                }
                                // Only the common 1-D shape-vector case is folded here.
                                if axes == [0]
                                    && steps == [1]
                                    && starts.len() == 1
                                    && ends.len() == 1
                                {
                                    // Negative indices count from the end (e.g. shape[-1:]).
                                    let start = if starts[0] < 0 {
                                        (data.len() as i64 + starts[0]).max(0) as usize
                                    } else {
                                        (starts[0] as usize).min(data.len())
                                    };
                                    let end = if ends[0] < 0 {
                                        (data.len() as i64 + ends[0]).max(0) as usize
                                    } else {
                                        (ends[0] as usize).min(data.len())
                                    };
                                    if start <= end && end <= data.len() {
                                        let sliced = data[start..end].to_vec();
                                        const_values.insert(out.to_string(), sliced.clone());
                                        value_shapes
                                            .insert(out.to_string(), vec![sliced.len() as i64]);
                                        value_types.insert(out.to_string(), DataType::Int64);
                                    }
                                }
                            }
                        }
                    }
                }
            } else if op_type == "Range" {
                if node.input.as_slice().len() == 3 {
                    if let (Some(start_name), Some(limit_name), Some(delta_name)) = (
                        node.input.as_slice().first(),
                        node.input.as_slice().get(1),
                        node.input.as_slice().get(2),
                    ) {
                        if options.experimental_dynamic_inputs {
                            let start_dim =
                                dynamic_scalar_dimension_for_value(start_name, value_shape_dims);
                            if let Some(limit_dim) =
                                dynamic_scalar_dimension_for_value(limit_name, value_shape_dims)
                            {
                                if let (Some(start_vals), Some(delta_vals), Some(out)) = (
                                    const_values.get(start_name),
                                    const_values.get(delta_name),
                                    node.output.as_slice().first(),
                                ) {
                                    if !start_vals.is_empty() && !delta_vals.is_empty() {
                                        let start = start_vals[0];
                                        let delta = delta_vals[0];
                                        if let Some(range_dim) = dynamic_range_length_dimension(
                                            start,
                                            delta,
                                            start_dim.as_ref(),
                                            &limit_dim,
                                        ) {
                                            let out_shape = vec![range_dim.max_size as i64];
                                            value_shape_dims.insert(
                                                out.to_string(),
                                                vec![Dimension::Dynamic(range_dim.clone())],
                                            );
                                            value_shapes.insert(out.to_string(), out_shape.clone());
                                            value_shapes
                                                .insert(sanitize_identifier(out), out_shape);
                                            value_types.insert(out.to_string(), DataType::Int64);
                                        }
                                    }
                                }
                                continue;
                            }
                        }

                        // Range(start, limit, delta) -> [start, start+delta, start+2*delta, ...]
                        if let (Some(start), Some(limit), Some(delta)) = (
                            read_range_scalar(start_name, initializers, const_values),
                            read_range_scalar(limit_name, initializers, const_values),
                            read_range_scalar(delta_name, initializers, const_values),
                        ) {
                            {
                                let mut range_vals = Vec::new();
                                if delta > 0 {
                                    let mut current = start;
                                    while current < limit {
                                        range_vals.push(current);
                                        current += delta;
                                    }
                                } else if delta < 0 {
                                    let mut current = start;
                                    while current > limit {
                                        range_vals.push(current);
                                        current += delta;
                                    }
                                }

                                if let Some(out) = node.output.as_slice().first() {
                                    let dtype = value_or_initializer_dtype(
                                        start_name,
                                        value_types,
                                        initializers,
                                    )
                                    .unwrap_or(DataType::Int64);
                                    const_values.insert(out.to_string(), range_vals.clone());
                                    let out_shape = vec![range_vals.len() as i64];
                                    // Force the correct shape - Range computes exact output shape
                                    value_shapes.insert(out.to_string(), out_shape.clone());
                                    value_shapes.insert(sanitize_identifier(out), out_shape);
                                    value_types.insert(out.to_string(), dtype);
                                }
                            }
                        }
                    }
                }
            } else if op_type == "Unsqueeze" && options.experimental_dynamic_inputs {
                if let (Some(inp), Some(out)) = (
                    node.input.as_slice().first(),
                    node.output.as_slice().first(),
                ) {
                    if let Some(dims) = value_shape_dims_for_input(graph, inp, value_shape_dims) {
                        if !dims.is_empty() {
                            value_shape_dims.insert(out.to_string(), dims.to_vec());
                        }
                    }
                }
            } else if op_type == "Reshape" && options.experimental_dynamic_inputs {
                let inputs = node.input.as_slice();
                if inputs.len() >= 2 {
                    if let Some(shape_dims) =
                        value_shape_dims_for_input(graph, inputs[1].as_str(), value_shape_dims)
                    {
                        if !shape_dims.is_empty() {
                            value_shape_dims.insert(outputs[0].to_string(), shape_dims.to_vec());
                        }
                    }
                }
            } else if op_type == "Concat" {
                // Concatenate constant inputs (often used to build shape tensors)
                if let Some(out) = node.output.as_slice().first() {
                    let mut per_input: Vec<Vec<i64>> = Vec::new();
                    let mut concatenated: Vec<i64> = Vec::new();
                    let mut all_const = true;
                    for inp in node.input.as_slice() {
                        if let Some(vals) =
                            const_values_for_input(graph, inp.as_str(), const_values)
                        {
                            concatenated.extend_from_slice(&vals);
                            per_input.push(vals);
                        } else {
                            all_const = false;
                            break;
                        }
                    }

                    // N-D constant concat (e.g. folded meshgrid index tensors):
                    // block-copy along the (normalized) axis.
                    let nd_folded = if all_const {
                        (|| {
                            let shapes: Vec<Vec<i64>> = node
                                .input
                                .as_slice()
                                .iter()
                                .map(|i| value_shapes.get(i.as_str()).cloned())
                                .collect::<Option<_>>()?;
                            let rank = shapes.first()?.len();
                            if rank <= 1 || shapes.iter().any(|s| s.len() != rank) {
                                return None;
                            }
                            let mut axis = node
                                .attribute
                                .as_slice()
                                .iter()
                                .find(|a| a.name.as_str() == "axis")
                                .map(|a| a.i)
                                .unwrap_or(0);
                            if axis < 0 {
                                axis += rank as i64;
                            }
                            let ax = usize::try_from(axis).ok().filter(|&a| a < rank)?;
                            let base = &shapes[0];
                            for s in &shapes {
                                for d in 0..rank {
                                    if d != ax && s[d] != base[d] {
                                        return None;
                                    }
                                }
                                if shape_numel(s)?
                                    != per_input[shapes
                                        .iter()
                                        .position(|x| std::ptr::eq(x, s))
                                        .unwrap_or(0)]
                                    .len()
                                {
                                    return None;
                                }
                            }
                            let inner: i64 = base[ax + 1..].iter().product();
                            let outer: i64 = base[..ax].iter().product();
                            if inner <= 0 || outer <= 0 {
                                return None;
                            }
                            let mut out_shape = base.clone();
                            out_shape[ax] = shapes.iter().map(|s| s[ax]).sum();
                            let mut values = Vec::with_capacity(shape_numel(&out_shape)?);
                            for o in 0..outer as usize {
                                for (vals, s) in per_input.iter().zip(&shapes) {
                                    let block = (s[ax] * inner) as usize;
                                    let start = o * block;
                                    values.extend_from_slice(vals.get(start..start + block)?);
                                }
                            }
                            Some((values, out_shape))
                        })()
                    } else {
                        None
                    };
                    if let Some((values, out_shape)) = nd_folded {
                        const_values.insert(out.to_string(), values);
                        value_shapes.insert(out.to_string(), out_shape.clone());
                        value_shapes.insert(sanitize_identifier(out), out_shape);
                        value_types.insert(out.to_string(), DataType::Int64);
                        continue;
                    }

                    // Handle axis=0 or axis=-1 (common for shape building)
                    let axis = node
                        .attribute
                        .as_slice()
                        .iter()
                        .find(|a| a.name.as_str() == "axis" && a.i != 0)
                        .map(|a| a.i)
                        .unwrap_or(0);

                    if all_const && (axis == 0 || axis == -1) {
                        // Propagate dynamic dim metadata through concat
                        if options.experimental_dynamic_inputs {
                            let mut concat_dims: Vec<rustnn::graph::Dimension> = Vec::new();
                            let mut has_dynamic = false;
                            for inp in node.input.as_slice() {
                                if let Some(dims) = value_shape_dims_for_input(
                                    graph,
                                    inp.as_str(),
                                    value_shape_dims,
                                ) {
                                    for d in dims {
                                        if matches!(d, rustnn::graph::Dimension::Dynamic(_)) {
                                            has_dynamic = true;
                                        }
                                        concat_dims.push(d.clone());
                                    }
                                } else if let Some(vals) =
                                    const_values_for_input(graph, inp.as_str(), const_values)
                                {
                                    for v in vals {
                                        concat_dims
                                            .push(rustnn::graph::Dimension::Static(v as u32));
                                    }
                                }
                            }
                            if has_dynamic && concat_dims.len() == concatenated.len() {
                                value_shape_dims.insert(out.to_string(), concat_dims);
                            }
                        }
                        const_values.insert(out.to_string(), concatenated.clone());
                        let out_shape = vec![concatenated.len() as i64];
                        // Force the correct shape - Concat computes exact output shape
                        value_shapes.insert(out.to_string(), out_shape.clone());
                        value_shapes.insert(sanitize_identifier(out), out_shape);
                        value_types.insert(out.to_string(), DataType::Int64);
                    }
                }
            } else if op_type == "Expand" {
                // Materialize Expand over integer consts (Swin/roll meshgrid
                // index construction). Bounded to keep folded payloads small.
                if let (Some(data_name), Some(shape_name), Some(out)) = (
                    node.input.as_slice().first(),
                    node.input.as_slice().get(1),
                    node.output.as_slice().first(),
                ) {
                    if let (Some(data), Some(target)) = (
                        const_values.get(data_name.as_str()).cloned(),
                        const_values.get(shape_name.as_str()).cloned(),
                    ) {
                        let data_shape = value_shapes
                            .get(data_name.as_str())
                            .cloned()
                            .unwrap_or_else(|| {
                                if data.len() == 1 {
                                    Vec::new()
                                } else {
                                    vec![data.len() as i64]
                                }
                            });
                        let folded = (|| {
                            // Bidirectional broadcast (torch -1 keeps input dim).
                            let rank = data_shape.len().max(target.len());
                            let mut out_shape = Vec::with_capacity(rank);
                            for i in 0..rank {
                                let d = if i + data_shape.len() >= rank {
                                    data_shape[i + data_shape.len() - rank]
                                } else {
                                    1
                                };
                                let t = if i + target.len() >= rank {
                                    target[i + target.len() - rank]
                                } else {
                                    1
                                };
                                let t = if t == -1 { d } else { t };
                                if d != t && d != 1 && t != 1 {
                                    return None;
                                }
                                out_shape.push(d.max(t));
                            }
                            let numel = shape_numel(&out_shape)?;
                            if numel > 1_000_000 || data.is_empty() {
                                return None;
                            }
                            let mut values = Vec::with_capacity(numel);
                            for idx in 0..numel {
                                let src = linear_index_for_broadcast_operand(
                                    idx,
                                    &out_shape,
                                    &data_shape,
                                )?;
                                values.push(*data.get(src)?);
                            }
                            Some((values, out_shape))
                        })();
                        if let Some((values, out_shape)) = folded {
                            const_values.insert(out.to_string(), values);
                            value_shapes.insert(out.to_string(), out_shape);
                            if let Some(dtype) = value_types.get(data_name.as_str()).copied() {
                                value_types.insert(out.to_string(), dtype);
                            } else {
                                value_types.insert(out.to_string(), DataType::Int64);
                            }
                        }
                    }
                }
            } else if op_type == "ConstantOfShape" {
                // ConstantOfShape(shape) -> tensor filled with constant value
                if let Some(shape_name) = node.input.as_slice().first() {
                    let dynamic_output_dims = if options.experimental_dynamic_inputs {
                        value_shape_dims_for(shape_name, value_shape_dims)
                            .map(|dims| dims.to_vec())
                            .filter(|dims| dims_contain_dynamic(dims))
                    } else {
                        None
                    };

                    if let (Some(out), Some(dims)) =
                        (node.output.as_slice().first(), dynamic_output_dims.as_ref())
                    {
                        value_shape_dims.insert(out.to_string(), dims.to_vec());
                        const_values.remove(out.as_str());
                    }

                    if let Some(shape_vals) = const_values.get(shape_name).cloned() {
                        let (fill_dtype, fill_value) = constant_of_shape_fill(node);

                        // Calculate number of elements
                        let numel = if shape_vals.is_empty() {
                            1
                        } else {
                            shape_vals.iter().product::<i64>()
                        };

                        if numel > 0 && numel < 1_000_000 {
                            if let Some(out) = node.output.as_slice().first() {
                                let should_keep_const = dynamic_output_dims
                                    .as_ref()
                                    .is_none_or(|dims| !dims_contain_dynamic(dims));
                                // Only fold integer fills into const_values; float tensors must
                                // lower through convert_constant_of_shape (expand + correct dtype).
                                if should_keep_const && fill_dtype == DataType::Int64 {
                                    let filled_tensor = vec![fill_value; numel as usize];
                                    const_values.insert(out.to_string(), filled_tensor);
                                } else {
                                    const_values.remove(out.as_str());
                                }
                                // Force the correct shape - ConstantOfShape creates exact output shape
                                value_shapes.insert(out.to_string(), shape_vals.clone());
                                value_shapes.insert(sanitize_identifier(out), shape_vals.clone());
                                value_types.insert(out.to_string(), fill_dtype);
                            }
                        }
                    }
                }
            } else if op_type == "Equal" {
                // Equal(a, b) -> boolean tensor (represented as i64: 1 for true, 0 for false)
                if node.input.as_slice().len() >= 2 {
                    if let (Some(a_name), Some(b_name), Some(out)) = (
                        node.input.as_slice().first(),
                        node.input.as_slice().get(1),
                        node.output.as_slice().first(),
                    ) {
                        let a = const_values.get(a_name);
                        let b = const_values.get(b_name);
                        if let (Some(a), Some(b)) = (a, b) {
                            let a_shape = const_shape_for_folding(a_name, a, value_shapes);
                            let b_shape = const_shape_for_folding(b_name, b, value_shapes);
                            if let Some((result_vals, out_shape)) =
                                fold_binary_const_i64("Equal", a, b, &a_shape, &b_shape)
                            {
                                const_values.insert(out.to_string(), result_vals.clone());
                                // Force the correct shape - Equal operation computes exact output shape
                                value_shapes.insert(out.to_string(), out_shape.clone());
                                value_shapes.insert(sanitize_identifier(out), out_shape);
                                value_types.insert(out.to_string(), DataType::Uint8);
                            }
                        }
                    }
                }
            } else if matches!(op_type, "Add" | "Sub" | "Mul" | "Div" | "Mod") {
                // Integer arithmetic over folded shape scalars/vectors (e.g.
                // `Mul(Gather(Shape(x), 0), num_heads)` in attention reshape chains).
                if node.input.as_slice().len() >= 2 {
                    if let (Some(a_name), Some(b_name), Some(out)) = (
                        node.input.as_slice().first(),
                        node.input.as_slice().get(1),
                        node.output.as_slice().first(),
                    ) {
                        if let (Some(a), Some(b)) =
                            (const_values.get(a_name), const_values.get(b_name))
                        {
                            let a_shape = const_shape_for_folding(a_name, a, value_shapes);
                            let b_shape = const_shape_for_folding(b_name, b, value_shapes);
                            if let Some((result_vals, out_shape)) =
                                fold_binary_const_i64(op_type, a, b, &a_shape, &b_shape)
                            {
                                const_values.insert(out.to_string(), result_vals);
                                value_shapes.insert(out.to_string(), out_shape.clone());
                                value_shapes.insert(sanitize_identifier(out), out_shape);
                                let dtype = value_types
                                    .get(a_name.as_str())
                                    .copied()
                                    .unwrap_or(DataType::Int64);
                                value_types.insert(out.to_string(), dtype);
                            }
                        }
                    }
                }
            } else if op_type == "Where" {
                if options.experimental_dynamic_inputs && node.input.as_slice().len() >= 3 {
                    if let Some(out) = node.output.as_slice().first() {
                        let cond = const_values.get(node.input.as_slice()[0].as_str());
                        let a_dims = dimension_vector_for_value(
                            node.input.as_slice()[1].as_str(),
                            const_values,
                            value_shape_dims,
                        );
                        let b_dims = dimension_vector_for_value(
                            node.input.as_slice()[2].as_str(),
                            const_values,
                            value_shape_dims,
                        );
                        let out_dims = if let (Some(cond), Some(a_dims), Some(b_dims)) =
                            (cond, a_dims.as_ref(), b_dims.as_ref())
                        {
                            if cond.len() == 1 && a_dims.len() == b_dims.len() {
                                Some(if cond[0] != 0 {
                                    a_dims.clone()
                                } else {
                                    b_dims.clone()
                                })
                            } else if cond.len() == a_dims.len() && cond.len() == b_dims.len() {
                                Some(
                                    cond.iter()
                                        .enumerate()
                                        .map(|(idx, c)| {
                                            if *c != 0 {
                                                a_dims[idx].clone()
                                            } else {
                                                b_dims[idx].clone()
                                            }
                                        })
                                        .collect(),
                                )
                            } else {
                                None
                            }
                        } else if let (Some(a_dims), Some(b_dims)) =
                            (a_dims.as_ref(), b_dims.as_ref())
                        {
                            let a_has_dynamic =
                                a_dims.iter().any(|d| matches!(d, Dimension::Dynamic(_)));
                            let b_has_dynamic =
                                b_dims.iter().any(|d| matches!(d, Dimension::Dynamic(_)));
                            if a_has_dynamic && !b_has_dynamic {
                                Some(a_dims.clone())
                            } else if b_has_dynamic && !a_has_dynamic {
                                Some(b_dims.clone())
                            } else if a_has_dynamic && b_has_dynamic && a_dims.len() == b_dims.len()
                            {
                                Some(
                                    a_dims
                                        .iter()
                                        .zip(b_dims.iter())
                                        .map(|(a_dim, b_dim)| match (a_dim, b_dim) {
                                            (Dimension::Dynamic(dim), _) => {
                                                Dimension::Dynamic(dim.clone())
                                            }
                                            (_, Dimension::Dynamic(dim)) => {
                                                Dimension::Dynamic(dim.clone())
                                            }
                                            (Dimension::Static(v), _) => Dimension::Static(*v),
                                        })
                                        .collect(),
                                )
                            } else {
                                None
                            }
                        } else if let Some(a_dims) = a_dims.as_ref() {
                            if a_dims.iter().any(|d| matches!(d, Dimension::Dynamic(_)))
                                && !is_trivial_static_dimension_vector(a_dims)
                            {
                                Some(a_dims.clone())
                            } else {
                                None
                            }
                        } else if let Some(b_dims) = b_dims.as_ref() {
                            if b_dims.iter().any(|d| matches!(d, Dimension::Dynamic(_)))
                                && !is_trivial_static_dimension_vector(b_dims)
                            {
                                Some(b_dims.clone())
                            } else {
                                None
                            }
                        } else {
                            None
                        };

                        if let Some(out_dims) = out_dims {
                            if out_dims.iter().any(|d| matches!(d, Dimension::Dynamic(_))) {
                                value_shape_dims.insert(out.to_string(), out_dims);
                            }
                        }
                    }
                }
                // Keep Where dynamic to avoid baking shape-driving expressions
                // (e.g., past_sequence_length + 1) into fixed constants.
                continue;
            }
        }

        if const_values.len() == pass_before {
            break;
        }
        any_folded = true;
    }

    any_folded || const_values.len() > consts_before
}

fn fold_integer_constants(graph: &GraphProto, ctx: &mut InferenceResult) -> bool {
    let mut value_shape_dims = HashMap::new();
    fold_shape_constants(
        graph,
        &mut ctx.value_shapes,
        &mut ctx.value_types,
        &mut ctx.const_values,
        &mut value_shape_dims,
        &FoldShapeConstantsOptions::early_pass(),
    )
}

/// Per-element shapes of a SplitToSequence with static input shape and
/// constant split sizes (mirrors the conversion-time lowering).
fn split_to_sequence_element_shapes(
    node: &NodeProto,
    value_shapes: &HashMap<String, Vec<i64>>,
    initializers: &HashMap<String, &TensorProto>,
    const_values: &HashMap<String, Vec<i64>>,
) -> Option<Vec<Vec<i64>>> {
    let ins = node.input.as_slice();
    let input_shape = value_shapes.get(ins.first()?.as_str())?;
    let rank = input_shape.len();
    let mut axis = node
        .attribute
        .as_slice()
        .iter()
        .find(|a| a.name.as_str() == "axis")
        .map(|a| a.i)
        .unwrap_or(0);
    if axis < 0 {
        axis += rank as i64;
    }
    let axis = usize::try_from(axis).ok().filter(|&a| a < rank)?;
    let dim = input_shape[axis];

    let sizes: Vec<i64> = match ins.get(1).filter(|n| !n.is_empty()) {
        Some(name) => {
            let values = const_values
                .get(name.as_str())
                .cloned()
                .or_else(|| initializers.get(name.as_str()).map(|t| read_int_tensor(t)))
                .filter(|v| !v.is_empty())?;
            let is_scalar = value_shapes
                .get(name.as_str())
                .map(|s| s.is_empty())
                .unwrap_or(values.len() == 1);
            if is_scalar && values.len() == 1 {
                let chunk = values[0].max(1);
                let mut sizes = vec![chunk; (dim / chunk) as usize];
                if dim % chunk != 0 {
                    sizes.push(dim % chunk);
                }
                sizes
            } else {
                values
            }
        }
        None => vec![1; usize::try_from(dim).ok()?],
    };
    if sizes.iter().sum::<i64>() != dim || sizes.iter().any(|&s| s <= 0) {
        return None;
    }
    Some(
        sizes
            .iter()
            .map(|&s| {
                let mut shape = input_shape.clone();
                shape[axis] = s;
                shape
            })
            .collect(),
    )
}

/// Apply ONNX `Shape` start/end attributes (dynamo exports slice shapes,
/// e.g. `Shape(x, start=0, end=1)` for the batch dim).
pub(crate) fn shape_attr_range(node: &NodeProto, rank: usize) -> (usize, usize) {
    let rank_i = rank as i64;
    let norm = |v: i64| -> usize { v.clamp(-rank_i, rank_i).rem_euclid(rank_i.max(1)) as usize };
    let mut start = 0usize;
    let mut end = rank;
    for attr in node.attribute.as_slice() {
        match attr.name.as_str() {
            "start" => start = norm(attr.i),
            "end" => end = if attr.i >= rank_i { rank } else { norm(attr.i) },
            _ => {}
        }
    }
    (start.min(rank), end.clamp(start.min(rank), rank))
}

/// ONNX RNN family output shapes: Y [seq, dirs, batch, hidden],
/// Y_h/Y_c [dirs, batch, hidden].
fn rnn_output_shapes(
    node: &NodeProto,
    value_shapes: &HashMap<String, Vec<i64>>,
) -> Option<Vec<Vec<i64>>> {
    let ins = node.input.as_slice();
    let x = value_shapes.get(ins.first()?.as_str())?;
    let w = value_shapes.get(ins.get(1)?.as_str())?;
    if x.len() != 3 || w.len() != 3 {
        return None;
    }
    let (seq, batch) = (x[0], x[1]);
    let dirs = w[0];
    let hidden = node
        .attribute
        .as_slice()
        .iter()
        .find(|a| a.name.as_str() == "hidden_size")
        .map(|a| a.i)
        .filter(|&h| h > 0)?;
    Some(vec![
        vec![seq, dirs, batch, hidden],
        vec![dirs, batch, hidden],
        vec![dirs, batch, hidden],
    ])
}

pub(crate) fn read_int_tensor(tensor: &TensorProto) -> Vec<i64> {
    let raw = tensor.raw_data.as_slice();
    if !raw.is_empty() {
        match tensor.data_type {
            x if x == TensorProto_DataType::Int32 as i32 => raw
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64)
                .collect(),
            x if x == TensorProto_DataType::Bool as i32
                || x == TensorProto_DataType::Uint8 as i32
                || x == TensorProto_DataType::Int8 as i32 =>
            {
                raw.iter().map(|&b| i64::from(b)).collect()
            }
            x if x == TensorProto_DataType::Int64 as i32 => raw
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect(),
            // Never reinterpret non-integer payloads (float weights, bool
            // masks were handled above) as packed i64 - it poisons folding.
            _ => Vec::new(),
        }
    } else if !tensor.int64_data.as_slice().is_empty() {
        tensor.int64_data.as_slice().to_vec()
    } else if !tensor.int32_data.as_slice().is_empty() {
        tensor
            .int32_data
            .as_slice()
            .iter()
            .map(|&v| v as i64)
            .collect()
    } else {
        Vec::new()
    }
}

/// Options for extended shape propagation during ONNX lowering.
#[derive(Debug, Clone, Copy)]
pub struct PropagateOptions {
    pub optimize: bool,
    pub experimental_dynamic_inputs: bool,
}

/// Propagate ONNX value shapes and fold integer shape subgraphs.
pub fn propagate_shapes_and_fold_constants(
    graph: &GraphProto,
    initializers: &HashMap<String, &TensorProto>,
    value_shapes: &mut HashMap<String, Vec<i64>>,
    value_types: &mut HashMap<String, DataType>,
    const_values: &mut HashMap<String, Vec<i64>>,
    value_shape_dims: &mut HashMap<String, Vec<Dimension>>,
    options: &PropagateOptions,
) {
    // Propagate shapes and fold constant shape expressions in a few passes
    for _ in 0..24 {
        let mut shapes_changed = false;
        if options.optimize {
            let max_iterations = 10;
            for iteration in 0..max_iterations {
                let initial_count = value_shapes.len();

                for onnx_node in graph.node.as_slice() {
                    let all_outputs_known = onnx_node
                        .output
                        .as_slice()
                        .iter()
                        .all(|out| value_shapes.contains_key(out.as_str()));
                    // Shapes recorded from partial information (e.g. the
                    // binary-broadcast single-input guess) go stale once more
                    // inputs resolve. Every inference arm is exact given
                    // resolved inputs and returns None otherwise, so always
                    // recomputing is idempotent; multi-output special cases
                    // above keep their or_insert semantics.
                    let _ = all_outputs_known;

                    if onnx_node.op_type.as_str() == "DynamicQuantizeLinear" {
                        if let Some(input_name) = onnx_node.input.first() {
                            if let Some(input_shape) = value_shapes.get(input_name).cloned() {
                                if let [y, scale, zero_point] = onnx_node.output.as_slice() {
                                    value_shapes.insert(y.clone(), input_shape);
                                    value_shapes.insert(scale.clone(), Vec::new());
                                    value_shapes.insert(zero_point.clone(), Vec::new());
                                    value_types.insert(y.clone(), DataType::Uint8);
                                    value_types.insert(scale.clone(), DataType::Float32);
                                    value_types.insert(zero_point.clone(), DataType::Uint8);
                                    continue;
                                }
                            }
                        }
                    }

                    if onnx_node.op_type.as_str() == "GroupQueryAttention" {
                        if let Some(shapes) =
                            crate::onnx::ops::attention::gqa_output_shapes(onnx_node, value_shapes)
                        {
                            for (output, shape) in onnx_node.output.iter().zip(shapes) {
                                if !output.is_empty() {
                                    value_shapes.insert(output.to_string(), shape);
                                }
                            }
                            continue;
                        }
                    }

                    // Sequences: track per-element shapes under `{name}__seq{i}`
                    // so SequenceAt consumers resolve (mirrors the conversion).
                    if onnx_node.op_type.as_str() == "SplitToSequence" {
                        if let Some(elems) = split_to_sequence_element_shapes(
                            onnx_node,
                            value_shapes,
                            initializers,
                            const_values,
                        ) {
                            if let Some(seq_name) = onnx_node.output.first() {
                                let dtype = onnx_node
                                    .input
                                    .first()
                                    .and_then(|n| value_types.get(n.as_str()).copied());
                                for (i, shape) in elems.iter().enumerate() {
                                    let key = format!("{}__seq{i}", sanitize_identifier(seq_name));
                                    value_shapes.insert(key.clone(), shape.clone());
                                    if let Some(dtype) = dtype {
                                        value_types.insert(key, dtype);
                                    }
                                }
                                value_shapes.insert(
                                    format!("{}__seqlen", sanitize_identifier(seq_name)),
                                    vec![elems.len() as i64],
                                );
                            }
                        }
                        continue;
                    }
                    if onnx_node.op_type.as_str() == "SequenceAt" {
                        let resolved = (|| {
                            let ins = onnx_node.input.as_slice();
                            let seq = sanitize_identifier(ins.first()?);
                            let count = value_shapes
                                .get(&format!("{seq}__seqlen"))?
                                .first()
                                .copied()?;
                            let mut idx = const_values
                                .get(ins.get(1)?.as_str())
                                .and_then(|v| v.first().copied())
                                .or_else(|| {
                                    initializers
                                        .get(ins.get(1)?.as_str())
                                        .and_then(|t| read_int_tensor(t).first().copied())
                                })?;
                            if idx < 0 {
                                idx += count;
                            }
                            value_shapes.get(&format!("{seq}__seq{idx}")).cloned()
                        })();
                        if let (Some(shape), Some(out)) = (resolved, onnx_node.output.first()) {
                            value_shapes.insert(out.to_string(), shape);
                            if let Some(dtype) = onnx_node
                                .input
                                .first()
                                .map(|n| format!("{}__seq0", sanitize_identifier(n)))
                                .and_then(|k| value_types.get(&k).copied())
                            {
                                value_types.insert(out.to_string(), dtype);
                            }
                        }
                        continue;
                    }

                    if matches!(onnx_node.op_type.as_str(), "LSTM" | "GRU" | "RNN") {
                        if let Some(shapes) = rnn_output_shapes(onnx_node, value_shapes) {
                            let dtype = onnx_node
                                .input
                                .first()
                                .and_then(|n| value_types.get(n.as_str()).copied());
                            for (output, shape) in onnx_node.output.iter().zip(shapes) {
                                if !output.is_empty() {
                                    value_shapes.insert(output.to_string(), shape);
                                    if let Some(dtype) = dtype {
                                        value_types.insert(output.to_string(), dtype);
                                    }
                                }
                            }
                            continue;
                        }
                    }

                    if onnx_node.op_type.as_str() == "Split" {
                        if let Some(shapes) = infer_split_output_shapes(
                            onnx_node,
                            value_shapes,
                            initializers,
                            const_values,
                        ) {
                            for (output, shape) in onnx_node.output.iter().zip(shapes) {
                                value_shapes.insert(output.to_string(), shape);
                            }
                            continue;
                        }
                    }

                    let inferred_opt = infer_node_output_shape(
                        onnx_node,
                        value_shapes,
                        initializers,
                        const_values,
                    );
                    if std::env::var("O2W_TRACE").is_ok()
                        && onnx_node
                            .output
                            .first()
                            .is_some_and(|o| o.contains("layers.0/self_attn/Reshape_output_0"))
                    {
                        eprintln!(
                            "[opt-loop] {} stored={:?} inferred={:?} in0={:?}",
                            onnx_node.output[0],
                            value_shapes.get(onnx_node.output[0].as_str()),
                            inferred_opt,
                            onnx_node
                                .input
                                .first()
                                .map(|i| value_shapes.get(i.as_str()))
                        );
                    }
                    if let Some(inferred) = inferred_opt {
                        if let Some(output_name) = onnx_node.output.as_slice().first() {
                            // Force the correct shape - shape inference computes exact output shape
                            if value_shapes.get(output_name.as_str()) != Some(&inferred) {
                                shapes_changed = true;
                            }
                            value_shapes.insert(output_name.to_string(), inferred);
                            if matches!(onnx_node.op_type.as_str(), "ConvInteger" | "MatMulInteger")
                            {
                                value_types.insert(output_name.to_string(), DataType::Int32);
                            } else if onnx_node.op_type.as_str() == "GatherBlockQuantized" {
                                if let Some(dtype) = gather_block_quantized_dtype(
                                    onnx_node,
                                    value_types,
                                    initializers,
                                ) {
                                    value_types.insert(output_name.to_string(), dtype);
                                }
                            }
                        }
                    }
                }

                if value_shapes.len() == initial_count {
                    break;
                }

                if iteration == max_iterations - 1 {
                    crate::debug_println!(
                        "Warning: Shape propagation reached max iterations ({}/{})",
                        value_shapes.len(),
                        graph.node.as_slice().len()
                    );
                }
            }
        }

        // If we know the input_ids shape (batch, seq), upgrade any lone hidden-dim
        // tensors (length-1 shapes) to [batch, seq, hidden] to unblock downstream
        // matmul/reshape resolution in decoder graphs that lost batch/seq dims.
        // Tensors whose rank the model declares (value_info/inputs/outputs) or
        // that are initializers keep their genuine 1-D shapes (e.g. a rotary
        // Range output) - only speculatively-shaped values are upgraded.
        if let Some(ids_shape) = value_shapes.get("input_ids") {
            if ids_shape.len() == 2 {
                let declared_rank1: HashSet<&str> = graph
                    .value_info
                    .iter()
                    .chain(graph.input.iter())
                    .chain(graph.output.iter())
                    .filter(|vi| {
                        vi.r#type
                            .as_ref()
                            .and_then(|t| t.value.as_ref())
                            .is_some_and(|v| match v {
                                TypeProtoValue::TensorType(tt) => {
                                    tt.shape.as_ref().is_some_and(|s| s.dim.len() == 1)
                                }
                                _ => false,
                            })
                    })
                    .map(|vi| vi.name.as_str())
                    .collect();
                let (batch, seq) = (ids_shape[0], ids_shape[1]);
                let upgrades: Vec<(String, Vec<i64>)> = value_shapes
                    .iter()
                    .filter_map(|(k, v)| {
                        if v.len() == 1
                            && v[0] > 1
                            && !declared_rank1.contains(k.as_str())
                            && !initializers.contains_key(k)
                        {
                            Some((k.clone(), vec![batch, seq, v[0]]))
                        } else {
                            None
                        }
                    })
                    .collect();
                for (k, v) in upgrades {
                    value_shapes.insert(k, v);
                }
            }
        }

        let consts_before = const_values.len();

        let fold_opts = FoldShapeConstantsOptions::from_propagate(options);
        if options.experimental_dynamic_inputs {
            propagate_dynamic_dims_metadata(graph, value_shape_dims);
        }
        if fold_shape_constants(
            graph,
            value_shapes,
            value_types,
            const_values,
            value_shape_dims,
            &fold_opts,
        ) {
            // at least one node folded this pass
        }

        // In-place shape corrections (stale broadcast guesses healing as more
        // inputs resolve) count as progress even when no new const appeared.
        if const_values.len() == consts_before && !shapes_changed {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dynamic_dim_requires_override() {
        use crate::protos::onnx::{tensor_shape_proto, type_proto};

        let dim = tensor_shape_proto::Dimension {
            value: Some(tensor_shape_proto::dimension::Value::DimParam(
                "batch".to_string(),
            )),
            denotation: String::new(),
        };
        let shape = crate::protos::onnx::TensorShapeProto { dim: vec![dim] };

        let tensor_type = type_proto::Tensor {
            elem_type: crate::protos::onnx::TensorProto_DataType::Float.into(),
            shape: Some(shape),
        };

        let type_proto = crate::protos::onnx::TypeProto {
            value: Some(type_proto::Value::TensorType(tensor_type)),
            denotation: String::new(),
        };

        let vi = crate::protos::onnx::ValueInfoProto {
            name: "input".to_string(),
            r#type: Some(type_proto),
            ..Default::default()
        };

        let graph = crate::protos::onnx::GraphProto {
            input: vec![vi],
            ..Default::default()
        };

        let model = crate::protos::onnx::ModelProto {
            graph: Some(graph),
            ..Default::default()
        };

        let res = infer_static_shapes(&model, &HashMap::new());
        assert!(matches!(
            res,
            Err(ShapeInferenceError::DynamicDim { dim, .. }) if dim == "batch"
        ));
    }

    #[test]
    fn override_allows_static_shape() {
        use crate::protos::onnx::{tensor_shape_proto, type_proto};

        let dim = tensor_shape_proto::Dimension {
            value: Some(tensor_shape_proto::dimension::Value::DimParam(
                "batch".to_string(),
            )),
            denotation: String::new(),
        };
        let shape = crate::protos::onnx::TensorShapeProto { dim: vec![dim] };

        let tensor_type = type_proto::Tensor {
            elem_type: crate::protos::onnx::TensorProto_DataType::Float.into(),
            shape: Some(shape),
        };

        let type_proto = crate::protos::onnx::TypeProto {
            value: Some(type_proto::Value::TensorType(tensor_type)),
            denotation: String::new(),
        };

        let vi = crate::protos::onnx::ValueInfoProto {
            name: "input".to_string(),
            r#type: Some(type_proto),
            ..Default::default()
        };

        let graph = crate::protos::onnx::GraphProto {
            input: vec![vi],
            ..Default::default()
        };

        let model = crate::protos::onnx::ModelProto {
            graph: Some(graph),
            ..Default::default()
        };

        let mut overrides = HashMap::new();
        overrides.insert("batch".to_string(), 1);
        let res = infer_static_shapes(&model, &overrides).unwrap();
        assert_eq!(res.value_shapes.get("input"), Some(&vec![1]));
    }

    fn shape_node(op_type: &str, inputs: &[&str]) -> NodeProto {
        NodeProto {
            op_type: op_type.to_string(),
            input: inputs.iter().map(|input| (*input).to_string()).collect(),
            output: vec!["output".to_string()],
            ..Default::default()
        }
    }

    fn infer_test_node(
        node: &NodeProto,
        input_shape: &[i64],
        constants: &[(&str, Vec<i64>)],
    ) -> Option<Vec<i64>> {
        let value_shapes = HashMap::from([("input".to_string(), input_shape.to_vec())]);
        let const_values = constants
            .iter()
            .map(|(name, values)| ((*name).to_string(), values.clone()))
            .collect();
        infer_node_output_shape(node, &value_shapes, &HashMap::new(), &const_values)
    }

    fn assert_shape_reaches_downstream_shape_node(
        mut node: NodeProto,
        input_shape: &[i64],
        constants: &[(&str, Vec<i64>)],
        expected: &[i64],
    ) {
        node.output = vec!["intermediate".to_string()];
        let shape = NodeProto {
            op_type: "Shape".to_string(),
            input: vec!["intermediate".to_string()],
            output: vec!["shape_output".to_string()],
            ..Default::default()
        };
        let graph = GraphProto {
            node: vec![node, shape],
            ..Default::default()
        };
        let mut value_shapes = HashMap::from([("input".to_string(), input_shape.to_vec())]);
        let mut value_types = HashMap::new();
        let mut const_values: HashMap<String, Vec<i64>> = constants
            .iter()
            .map(|(name, values)| ((*name).to_string(), values.clone()))
            .collect();
        let mut value_shape_dims = HashMap::new();

        propagate_shapes_and_fold_constants(
            &graph,
            &HashMap::new(),
            &mut value_shapes,
            &mut value_types,
            &mut const_values,
            &mut value_shape_dims,
            &PropagateOptions {
                optimize: true,
                experimental_dynamic_inputs: false,
            },
        );

        assert_eq!(
            value_shapes.get("intermediate").map(Vec::as_slice),
            Some(expected)
        );
        assert_eq!(
            const_values.get("shape_output").map(Vec::as_slice),
            Some(expected),
            "downstream Shape must fold after output-shape propagation"
        );
    }

    #[test]
    fn expand_shape_propagates_broadcast_dimensions() {
        let node = shape_node("Expand", &["input", "target"]);

        assert_eq!(
            infer_test_node(&node, &[2, 1, 3], &[("target", vec![1, 4, 3])]),
            Some(vec![2, 4, 3])
        );
        assert_shape_reaches_downstream_shape_node(
            node,
            &[2, 1, 3],
            &[("target", vec![1, 4, 3])],
            &[2, 4, 3],
        );
    }

    #[test]
    fn expand_shape_rejects_incompatible_dimensions() {
        let node = shape_node("Expand", &["input", "target"]);

        assert_eq!(
            infer_test_node(&node, &[2, 2], &[("target", vec![3, 2])]),
            None
        );
    }

    #[test]
    fn tile_shape_multiplies_each_dimension_by_repeats() {
        let node = shape_node("Tile", &["input", "repeats"]);

        assert_eq!(
            infer_test_node(&node, &[2, 3], &[("repeats", vec![1, 4])]),
            Some(vec![2, 12])
        );
        assert_shape_reaches_downstream_shape_node(
            node,
            &[2, 3],
            &[("repeats", vec![1, 4])],
            &[2, 12],
        );
    }

    #[test]
    fn squeeze_shape_supports_constant_and_negative_axes() {
        let node = shape_node("Squeeze", &["input", "axes"]);

        assert_eq!(
            infer_test_node(&node, &[1, 3, 1, 5], &[("axes", vec![0, -2])]),
            Some(vec![3, 5])
        );
        assert_shape_reaches_downstream_shape_node(
            node,
            &[1, 3, 1, 5],
            &[("axes", vec![0, -2])],
            &[3, 5],
        );
    }

    #[test]
    fn squeeze_shape_without_axes_removes_all_unit_dimensions() {
        let node = shape_node("Squeeze", &["input"]);

        assert_eq!(infer_test_node(&node, &[1, 3, 1, 5], &[]), Some(vec![3, 5]));
    }

    #[test]
    fn range_shape_handles_ascending_and_descending_ranges() {
        let ascending = shape_node("Range", &["start", "limit", "delta"]);
        assert_eq!(
            infer_test_node(
                &ascending,
                &[],
                &[("start", vec![1]), ("limit", vec![8]), ("delta", vec![3])]
            ),
            Some(vec![3])
        );
        assert_shape_reaches_downstream_shape_node(
            ascending,
            &[],
            &[("start", vec![1]), ("limit", vec![8]), ("delta", vec![3])],
            &[3],
        );

        let descending = shape_node("Range", &["start", "limit", "delta"]);
        assert_eq!(
            infer_test_node(
                &descending,
                &[],
                &[("start", vec![8]), ("limit", vec![1]), ("delta", vec![-3])]
            ),
            Some(vec![3])
        );
    }

    #[test]
    fn range_shape_handles_float_initializers() {
        // dynamo: arange(0.5, 14.0, 1.0) -> 14 elements
        let f32_scalar = |name: &str, v: f32| TensorProto {
            name: name.to_string(),
            data_type: TensorProto_DataType::Float as i32,
            raw_data: v.to_le_bytes().to_vec(),
            ..Default::default()
        };
        let start = f32_scalar("start", 0.5);
        let delta = f32_scalar("delta", 1.0);
        let initializers =
            HashMap::from([("start".to_string(), &start), ("delta".to_string(), &delta)]);
        let const_values = HashMap::from([("limit".to_string(), vec![14])]);
        let node = shape_node("Range", &["start", "limit", "delta"]);
        assert_eq!(
            infer_node_output_shape(&node, &HashMap::new(), &initializers, &const_values),
            Some(vec![14])
        );
    }

    #[test]
    fn shape_attr_range_slices_shape_vector() {
        use crate::protos::onnx::AttributeProto;
        let mut node = shape_node("Shape", &["input"]);
        assert_eq!(shape_attr_range(&node, 4), (0, 4));
        node.attribute = vec![
            AttributeProto {
                name: "start".to_string(),
                i: 0,
                ..Default::default()
            },
            AttributeProto {
                name: "end".to_string(),
                i: 1,
                ..Default::default()
            },
        ];
        assert_eq!(shape_attr_range(&node, 4), (0, 1));
        node.attribute[0].i = -2;
        node.attribute[1].i = 100;
        assert_eq!(shape_attr_range(&node, 4), (2, 4));

        // Folded Shape output honours the range.
        let graph = GraphProto {
            node: vec![node],
            ..Default::default()
        };
        let mut value_shapes = HashMap::from([("input".to_string(), vec![1, 3, 224, 224])]);
        let mut value_types = HashMap::new();
        let mut const_values = HashMap::new();
        let mut value_shape_dims = HashMap::new();
        propagate_shapes_and_fold_constants(
            &graph,
            &HashMap::new(),
            &mut value_shapes,
            &mut value_types,
            &mut const_values,
            &mut value_shape_dims,
            &PropagateOptions {
                optimize: true,
                experimental_dynamic_inputs: false,
            },
        );
        assert_eq!(const_values.get("output"), Some(&vec![224, 224]));
        assert_eq!(value_shapes.get("output"), Some(&vec![2]));
    }

    #[test]
    fn unsqueeze_shape_reads_axes_input_with_negative_axes() {
        let node = shape_node("Unsqueeze", &["input", "axes"]);
        assert_eq!(
            infer_test_node(&node, &[14, 14], &[("axes", vec![-1])]),
            Some(vec![14, 14, 1])
        );
        assert_eq!(
            infer_test_node(&node, &[14, 14], &[("axes", vec![0, -1])]),
            Some(vec![1, 14, 14, 1])
        );
        assert_eq!(
            infer_test_node(&node, &[14, 14], &[("axes", vec![5])]),
            None
        );
    }

    #[test]
    fn cast_output_type_follows_to_attribute() {
        use crate::onnx::test_models::prelude::*;
        let m = model(
            17,
            graph(
                "cast_types",
                vec![f32_input("x", &[2, 3])],
                vec![f16_output("z", &[2, 3])],
                vec![
                    node("Cast", "c16", &["x"], &["y"], &[attr_int("to", 10)]),
                    node("Abs", "abs", &["y"], &["z"], &[]),
                ],
                vec![],
            ),
        );
        let res = infer_static_shapes(&m, &HashMap::new()).unwrap();
        assert_eq!(res.value_types.get("y"), Some(&DataType::Float16));
        // Downstream unary ops inherit the cast type.
        assert_eq!(res.value_types.get("z"), Some(&DataType::Float16));
    }

    #[test]
    fn unnamed_zero_dims_are_overridable() {
        use crate::onnx::test_models::prelude::*;
        let m = model(
            17,
            graph(
                "zero_dims",
                vec![tensor_input(
                    "x",
                    TensorProto_DataType::Float as i32,
                    &[0, 4],
                )],
                vec![f32_output("y", &[0, 4])],
                vec![node("Abs", "abs", &["x"], &["y"], &[])],
                vec![],
            ),
        );
        let err = infer_static_shapes(&m, &HashMap::new()).unwrap_err();
        assert!(matches!(err, ShapeInferenceError::DynamicDim { dim, .. } if dim == "x_dim0"));
        let overrides = HashMap::from([("x_dim0".to_string(), 3u32)]);
        let res = infer_static_shapes(&m, &overrides).unwrap();
        assert_eq!(res.value_shapes.get("y"), Some(&vec![3, 4]));
    }

    #[test]
    fn comparison_outputs_are_bool() {
        use crate::onnx::test_models::prelude::*;
        let m = model(
            17,
            graph(
                "cmp",
                vec![f32_input("a", &[2]), f32_input("b", &[2])],
                vec![bool_output("eq", &[2])],
                vec![
                    node("Equal", "eq_node", &["a", "b"], &["eq"], &[]),
                    node("Less", "lt_node", &["a", "b"], &["lt"], &[]),
                ],
                vec![],
            ),
        );
        let res = infer_static_shapes(&m, &HashMap::new()).unwrap();
        assert_eq!(res.value_types.get("eq"), Some(&DataType::Uint8));
        assert_eq!(res.value_types.get("lt"), Some(&DataType::Uint8));
    }

    #[test]
    fn neg_folds_shape_constants() {
        let graph = GraphProto {
            node: vec![
                shape_node_named("Shape", &["input"], "s"),
                shape_node_named("Gather", &["s", "idx"], "g"),
                shape_node_named("Neg", &["g"], "n"),
            ],
            ..Default::default()
        };
        let mut value_shapes = HashMap::from([("input".to_string(), vec![2, 3])]);
        let mut value_types = HashMap::new();
        let mut const_values = HashMap::from([("idx".to_string(), vec![1])]);
        let mut value_shape_dims = HashMap::new();
        propagate_shapes_and_fold_constants(
            &graph,
            &HashMap::new(),
            &mut value_shapes,
            &mut value_types,
            &mut const_values,
            &mut value_shape_dims,
            &PropagateOptions {
                optimize: true,
                experimental_dynamic_inputs: false,
            },
        );
        assert_eq!(const_values.get("n"), Some(&vec![-3]));
    }

    fn shape_node_named(op_type: &str, inputs: &[&str], output: &str) -> NodeProto {
        let mut node = shape_node(op_type, inputs);
        node.output = vec![output.to_string()];
        node
    }

    #[test]
    fn range_shape_rejects_zero_delta() {
        let node = shape_node("Range", &["start", "limit", "delta"]);

        assert_eq!(
            infer_test_node(
                &node,
                &[],
                &[("start", vec![0]), ("limit", vec![4]), ("delta", vec![0])]
            ),
            None
        );
    }

    #[test]
    fn normalization_ops_preserve_input_shape() {
        for op_type in ["BatchNormalization", "InstanceNormalization"] {
            let node = shape_node(op_type, &["input"]);
            assert_eq!(
                infer_test_node(&node, &[1, 16, 32, 32], &[]),
                Some(vec![1, 16, 32, 32]),
                "{op_type} should preserve its data input shape"
            );
        }
    }

    #[test]
    fn split_propagates_every_output_shape_to_downstream_shape_nodes() {
        let split = NodeProto {
            op_type: "Split".to_string(),
            input: vec!["input".to_string(), "split_sizes".to_string()],
            output: vec!["left".to_string(), "right".to_string()],
            attribute: vec![crate::protos::onnx::AttributeProto {
                name: "axis".to_string(),
                i: 1,
                ..Default::default()
            }],
            ..Default::default()
        };
        let shape = NodeProto {
            op_type: "Shape".to_string(),
            input: vec!["right".to_string()],
            output: vec!["right_shape".to_string()],
            ..Default::default()
        };
        let graph = GraphProto {
            node: vec![split, shape],
            ..Default::default()
        };
        let mut value_shapes = HashMap::from([("input".to_string(), vec![2, 5])]);
        let mut value_types = HashMap::new();
        let mut const_values = HashMap::from([("split_sizes".to_string(), vec![2, 3])]);
        let mut value_shape_dims = HashMap::new();

        propagate_shapes_and_fold_constants(
            &graph,
            &HashMap::new(),
            &mut value_shapes,
            &mut value_types,
            &mut const_values,
            &mut value_shape_dims,
            &PropagateOptions {
                optimize: true,
                experimental_dynamic_inputs: false,
            },
        );

        assert_eq!(value_shapes.get("left"), Some(&vec![2, 2]));
        assert_eq!(value_shapes.get("right"), Some(&vec![2, 3]));
        assert_eq!(const_values.get("right_shape"), Some(&vec![2, 3]));
    }

    #[test]
    fn split_shape_rejects_sizes_that_do_not_cover_the_axis() {
        let node = NodeProto {
            op_type: "Split".to_string(),
            input: vec!["input".to_string(), "split_sizes".to_string()],
            output: vec!["left".to_string(), "right".to_string()],
            attribute: vec![crate::protos::onnx::AttributeProto {
                name: "axis".to_string(),
                i: 1,
                ..Default::default()
            }],
            ..Default::default()
        };
        let value_shapes = HashMap::from([("input".to_string(), vec![2, 5])]);
        let const_values = HashMap::from([("split_sizes".to_string(), vec![2, 2])]);

        assert_eq!(
            infer_split_output_shapes(&node, &value_shapes, &HashMap::new(), &const_values),
            None
        );
    }

    #[test]
    fn dynamic_quantize_linear_propagates_all_output_shapes_and_types() {
        let node = NodeProto {
            op_type: "DynamicQuantizeLinear".to_string(),
            input: vec!["input".to_string()],
            output: vec![
                "quantized".to_string(),
                "scale".to_string(),
                "zero_point".to_string(),
            ],
            ..Default::default()
        };
        let graph = GraphProto {
            node: vec![node],
            ..Default::default()
        };
        let mut value_shapes = HashMap::from([("input".to_string(), vec![1, 3, 8, 8])]);
        let mut value_types = HashMap::from([("input".to_string(), DataType::Float32)]);
        let mut const_values = HashMap::new();
        let mut value_shape_dims = HashMap::new();

        propagate_shapes_and_fold_constants(
            &graph,
            &HashMap::new(),
            &mut value_shapes,
            &mut value_types,
            &mut const_values,
            &mut value_shape_dims,
            &PropagateOptions {
                optimize: true,
                experimental_dynamic_inputs: false,
            },
        );

        assert_eq!(value_shapes.get("quantized"), Some(&vec![1, 3, 8, 8]));
        assert_eq!(value_shapes.get("scale"), Some(&vec![]));
        assert_eq!(value_shapes.get("zero_point"), Some(&vec![]));
        assert_eq!(value_types.get("quantized"), Some(&DataType::Uint8));
        assert_eq!(value_types.get("scale"), Some(&DataType::Float32));
        assert_eq!(value_types.get("zero_point"), Some(&DataType::Uint8));
    }

    #[test]
    fn conv_integer_uses_conv_spatial_shape() {
        let node = NodeProto {
            op_type: "ConvInteger".to_string(),
            input: vec!["input".to_string(), "weight".to_string()],
            output: vec!["output".to_string()],
            attribute: vec![
                crate::protos::onnx::AttributeProto {
                    name: "strides".to_string(),
                    ints: vec![2, 2],
                    ..Default::default()
                },
                crate::protos::onnx::AttributeProto {
                    name: "pads".to_string(),
                    ints: vec![1, 1, 1, 1],
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let value_shapes = HashMap::from([
            ("input".to_string(), vec![1, 3, 32, 32]),
            ("weight".to_string(), vec![16, 3, 3, 3]),
        ]);

        assert_eq!(
            infer_node_output_shape(&node, &value_shapes, &HashMap::new(), &HashMap::new()),
            Some(vec![1, 16, 16, 16])
        );
    }

    #[test]
    fn shape_folding_refreshes_stale_seeded_values() {
        let graph = GraphProto {
            node: vec![NodeProto {
                op_type: "Shape".to_string(),
                input: vec!["input".to_string()],
                output: vec!["shape".to_string()],
                ..Default::default()
            }],
            ..Default::default()
        };
        let mut value_shapes = HashMap::from([("input".to_string(), vec![1, 3, 32, 32])]);
        let mut value_types = HashMap::new();
        let mut const_values = HashMap::from([("shape".to_string(), vec![1, 3, 16, 16])]);
        let mut value_shape_dims = HashMap::new();

        propagate_shapes_and_fold_constants(
            &graph,
            &HashMap::new(),
            &mut value_shapes,
            &mut value_types,
            &mut const_values,
            &mut value_shape_dims,
            &PropagateOptions {
                optimize: true,
                experimental_dynamic_inputs: false,
            },
        );

        assert_eq!(const_values.get("shape"), Some(&vec![1, 3, 32, 32]));
    }

    #[test]
    fn matmul_nbits_replaces_last_dim_with_n() {
        let node = NodeProto {
            op_type: "MatMulNBits".to_string(),
            input: vec!["input".to_string(), "b".to_string(), "scales".to_string()],
            output: vec!["y".to_string()],
            attribute: vec![crate::protos::onnx::AttributeProto {
                name: "N".to_string(),
                i: 384,
                ..Default::default()
            }],
            ..Default::default()
        };
        assert_eq!(
            infer_test_node(&node, &[2, 8, 384], &[]),
            Some(vec![2, 8, 384])
        );
    }

    #[test]
    fn fused_llm_ops_propagate_output_shapes() {
        for op_type in ["SimplifiedLayerNormalization", "RotaryEmbedding"] {
            let node = shape_node(op_type, &["input"]);
            assert_eq!(
                infer_test_node(&node, &[2, 7, 640], &[]),
                Some(vec![2, 7, 640])
            );
        }

        let graph = GraphProto {
            node: vec![NodeProto {
                op_type: "SkipSimplifiedLayerNormalization".to_string(),
                input: vec!["input".to_string(), "skip".to_string(), "gamma".to_string()],
                output: vec![
                    "normalized".to_string(),
                    String::new(),
                    String::new(),
                    "residual_sum".to_string(),
                ],
                ..Default::default()
            }],
            ..Default::default()
        };
        let mut result = InferenceResult {
            value_shapes: HashMap::from([("input".to_string(), vec![2, 7, 640])]),
            value_types: HashMap::from([("input".to_string(), DataType::Float32)]),
            ..Default::default()
        };
        propagate_node_shapes(&graph, &HashMap::new(), &mut result).unwrap();
        assert_eq!(
            result.value_shapes.get("normalized"),
            Some(&vec![2, 7, 640])
        );
        assert_eq!(
            result.value_shapes.get("residual_sum"),
            Some(&vec![2, 7, 640])
        );
        assert_eq!(
            result.value_types.get("residual_sum"),
            Some(&DataType::Float32)
        );
    }
}
