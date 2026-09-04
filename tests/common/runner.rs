/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Run ONNX Runtime reference inference and compare rustnn dispatch results.

use std::collections::{BTreeMap, HashMap, HashSet};

use half::f16;
use onnx2webnn::onnx::builder::OnnxBuilder;
use onnx2webnn::protos::onnx::{
    type_proto, AttributeProto, ModelProto, NodeProto, TensorProto, TensorProto_DataType,
    ValueInfoProto,
};
use onnx2webnn::{convert_model_proto, ConvertOptions, ValidatedGraph};
use prost::Message;
use rustnn::graph::OperandDescriptor;
use rustnn::mlcontext::{MLContext, MLTensor, MLTensorDescriptor};
use rustnn::operator_enums::MLOperandDataType;
use rustnn::{run_onnx_with_inputs, OnnxInput, TensorData};

/// Expected outcome for operator-level conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpectConvertOp {
    Success,
    UnsupportedOp,
    /// Conversion must fail for a reason other than an unsupported operator - e.g. a data
    /// type the WebNN mapping cannot represent (`bfloat16`). Used by schema-revision tests
    /// that exercise a dtype a newer ONNX schema added but WebNN cannot convert.
    ConversionError,
}

/// Convert (when supported), execute via rustnn dispatch, and compare against ORT.
///
/// Fixtures are built at the opset declared in `model.opset_import`. The converter
/// itself accepts any `ai.onnx` opset in the supported range (1-26).
pub fn assert_op_matches_ort(model: ModelProto, expect: ExpectConvertOp, test_opset: i64) {
    assert_op_matches_ort_with_options(model, expect, test_opset, &ConvertOptions::default());
}

/// Like [`assert_op_matches_ort`] with explicit [`ConvertOptions`]. Pinned
/// inputs are frozen in the ORT reference model too, so both sides see the
/// same constants and the pinned input is not fed at dispatch.
pub fn assert_op_matches_ort_with_options(
    model: ModelProto,
    expect: ExpectConvertOp,
    test_opset: i64,
    options: &ConvertOptions,
) {
    let declared_opset = model
        .opset_import
        .iter()
        .find(|opset| opset.domain.is_empty())
        .map(|opset| opset.version)
        .unwrap_or_default();
    assert_eq!(
        declared_opset, test_opset,
        "fixture opset and test opset should match"
    );
    let result = convert_model_proto(model.clone(), options);
    let mut model = model;
    onnx2webnn::onnx::convert::pin_graph_inputs(&mut model, &options.pinned_inputs)
        .expect("pinned inputs should exist in the fixture");
    match expect {
        ExpectConvertOp::UnsupportedOp => match result {
            Err(err) if err.is_unsupported_op() => {}
            Err(err) => panic!("expected UnsupportedOp, got {err}"),
            Ok(_) => panic!("expected UnsupportedOp, got Ok"),
        },
        ExpectConvertOp::ConversionError => match result {
            Err(_) => {}
            Ok(_) => panic!("expected conversion failure, got Ok"),
        },
        ExpectConvertOp::Success => {
            let mut validated =
                result.unwrap_or_else(|err| panic!("expected conversion success, got {err}"));
            let inputs = build_ort_inputs(&model).expect("failed to build ORT inputs");
            let reference_model = ort_reference_model(&model);
            let reference_bytes = reference_model.encode_to_vec();
            let reference = run_onnx_with_inputs(&reference_bytes, None, clone_ort_inputs(&inputs))
                .unwrap_or_else(|err| panic!("ORT reference run failed: {err}"));
            let actual = dispatch_and_collect(&mut validated, &model, &inputs)
                .unwrap_or_else(|err| panic!("rustnn dispatch failed: {err}"));
            compare_outputs(&model, &reference, &actual);
        }
    }
}

fn graph_proto(model: &ModelProto) -> &onnx2webnn::protos::onnx::GraphProto {
    model.graph.as_ref().expect("model graph")
}

/// ORT may lack kernels for some ONNX ops (e.g. Swish, float16 Range). Decompose for reference only.
fn ort_reference_model(model: &ModelProto) -> ModelProto {
    let mut reference = model.clone();
    let graph = reference.graph.as_mut().expect("model graph");
    let initializers = graph.initializer.clone();
    let input_elem_types: HashMap<String, i32> = graph
        .input
        .iter()
        .filter_map(|value| value_info_elem_type(value).map(|ty| (value.name.clone(), ty)))
        .collect();
    let mut nodes = Vec::with_capacity(graph.node.len() + 4);
    for node in graph.node.drain(..) {
        if node.op_type == "Swish" {
            let input = node.input.first().cloned().unwrap_or_default();
            let output = node.output.first().cloned().unwrap_or_default();
            let sig_out = format!("{}__sig", node.name);
            nodes.push(NodeProto {
                op_type: "Sigmoid".to_string(),
                name: format!("{}__sigmoid", node.name),
                input: vec![input.clone()],
                output: vec![sig_out.clone()],
                ..Default::default()
            });
            nodes.push(NodeProto {
                op_type: "Mul".to_string(),
                name: node.name,
                input: vec![input, sig_out],
                output: vec![output],
                ..Default::default()
            });
        } else if let Some(folded) = try_fold_float_range(&initializers, &node) {
            nodes.push(folded);
        } else if let Some(wrapped) =
            try_wrap_fp16_celu_for_ort(&initializers, &input_elem_types, &node)
        {
            nodes.extend(wrapped);
        } else if let Some(wrapped) =
            try_wrap_fp16_quantize_linear_for_ort(&initializers, &input_elem_types, &node)
        {
            nodes.extend(wrapped);
        } else {
            nodes.push(node);
        }
    }
    graph.node = nodes;
    reference
}

fn initializer_tensor<'a>(initializers: &'a [TensorProto], name: &str) -> Option<&'a TensorProto> {
    initializers.iter().find(|tensor| tensor.name == name)
}

fn read_tensor_scalar_f64(tensor: &TensorProto) -> Option<f64> {
    match tensor.data_type {
        x if x == TensorProto_DataType::Float as i32 => {
            if !tensor.float_data.is_empty() {
                return Some(f64::from(tensor.float_data[0]));
            }
            let raw = tensor.raw_data.as_slice();
            if raw.len() >= 4 {
                let bits = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
                return Some(f64::from(f32::from_bits(bits)));
            }
            None
        }
        x if x == TensorProto_DataType::Float16 as i32 => {
            let raw = tensor.raw_data.as_slice();
            if raw.len() >= 2 {
                let bits = u16::from_le_bytes([raw[0], raw[1]]);
                return Some(f64::from(f16::from_bits(bits).to_f32()));
            }
            None
        }
        x if x == TensorProto_DataType::Double as i32 => {
            if !tensor.double_data.is_empty() {
                return Some(tensor.double_data[0]);
            }
            let raw = tensor.raw_data.as_slice();
            if raw.len() >= 8 {
                let bits = u64::from_le_bytes([
                    raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7],
                ]);
                return Some(f64::from_bits(bits));
            }
            None
        }
        _ => None,
    }
}

fn try_fold_float_range(initializers: &[TensorProto], node: &NodeProto) -> Option<NodeProto> {
    if node.op_type != "Range" || node.input.len() != 3 {
        return None;
    }
    let start_t = initializer_tensor(initializers, &node.input[0])?;
    let limit_t = initializer_tensor(initializers, &node.input[1])?;
    let delta_t = initializer_tensor(initializers, &node.input[2])?;
    if start_t.data_type != TensorProto_DataType::Float16 as i32 {
        return None;
    }
    let start = read_tensor_scalar_f64(start_t)?;
    let limit = read_tensor_scalar_f64(limit_t)?;
    let delta = read_tensor_scalar_f64(delta_t)?;
    if delta == 0.0 {
        return None;
    }

    let count = ((limit - start) / delta).ceil();
    let count = if count.is_finite() && count > 0.0 {
        count as usize
    } else {
        0
    };
    let mut values: Vec<f32> = (0..count)
        .map(|i| (start + (i as f64) * delta) as f32)
        .collect();
    if values.is_empty() {
        values.push(0.0);
    }

    let raw_data: Vec<u8> = values
        .iter()
        .flat_map(|value| f16::from_f32(*value).to_bits().to_le_bytes())
        .collect();
    let tensor = TensorProto {
        data_type: TensorProto_DataType::Float16 as i32,
        dims: vec![values.len() as i64],
        raw_data,
        ..Default::default()
    };
    let value_attr = AttributeProto {
        name: "value".to_string(),
        r#type: onnx2webnn::protos::onnx::attribute_proto::AttributeType::Tensor as i32,
        t: Some(tensor),
        ..Default::default()
    };
    let output = node.output.first()?.clone();
    let mut folded = NodeProto {
        op_type: "Constant".to_string(),
        name: format!("{}__folded_range", node.name),
        output: vec![output],
        ..Default::default()
    };
    folded.attribute.push(value_attr);
    Some(folded)
}

fn tensor_elem_type(
    initializers: &[TensorProto],
    input_elem_types: &HashMap<String, i32>,
    name: &str,
) -> Option<i32> {
    if let Some(tensor) = initializers.iter().find(|tensor| tensor.name == name) {
        return Some(tensor.data_type);
    }
    input_elem_types.get(name).copied()
}

fn cast_node(name: &str, input: &str, output: &str, to: i32) -> NodeProto {
    NodeProto {
        op_type: "Cast".to_string(),
        name: name.to_string(),
        input: vec![input.to_string()],
        output: vec![output.to_string()],
        attribute: vec![AttributeProto {
            name: "to".to_string(),
            r#type: onnx2webnn::protos::onnx::attribute_proto::AttributeType::Int as i32,
            i: to as i64,
            ..Default::default()
        }],
        ..Default::default()
    }
}

fn try_wrap_fp16_celu_for_ort(
    initializers: &[TensorProto],
    input_elem_types: &HashMap<String, i32>,
    node: &NodeProto,
) -> Option<Vec<NodeProto>> {
    if node.op_type != "Celu" {
        return None;
    }
    let input = node.input.first()?;
    if tensor_elem_type(initializers, input_elem_types, input)?
        != TensorProto_DataType::Float16 as i32
    {
        return None;
    }
    let output = node.output.first()?.clone();
    let f32_input = format!("{input}__ort_f32");
    let f32_output = format!("{output}__celu_f32");
    Some(vec![
        cast_node(
            &format!("{}__cast_in", node.name),
            input,
            &f32_input,
            TensorProto_DataType::Float as i32,
        ),
        NodeProto {
            op_type: "Celu".to_string(),
            name: format!("{}__ort_f32", node.name),
            input: vec![f32_input],
            output: vec![f32_output.clone()],
            attribute: node.attribute.clone(),
            ..Default::default()
        },
        cast_node(
            &format!("{}__cast_out", node.name),
            &f32_output,
            &output,
            TensorProto_DataType::Float16 as i32,
        ),
    ])
}

fn try_wrap_fp16_quantize_linear_for_ort(
    initializers: &[TensorProto],
    input_elem_types: &HashMap<String, i32>,
    node: &NodeProto,
) -> Option<Vec<NodeProto>> {
    if node.op_type != "QuantizeLinear" || node.input.is_empty() {
        return None;
    }
    let input = &node.input[0];
    if tensor_elem_type(initializers, input_elem_types, input)?
        != TensorProto_DataType::Float16 as i32
    {
        return None;
    }
    let f32_input = format!("{input}__ort_f32");
    let mut wrapped = vec![cast_node(
        &format!("{}__cast_in", node.name),
        input,
        &f32_input,
        TensorProto_DataType::Float as i32,
    )];
    let mut quant = node.clone();
    quant.input[0] = f32_input;
    quant.name = format!("{}__ort_f32", node.name);
    wrapped.push(quant);
    Some(wrapped)
}

fn initializer_names(model: &ModelProto) -> HashSet<String> {
    graph_proto(model)
        .initializer
        .iter()
        .map(|init| init.name.clone())
        .collect()
}

fn feedable_inputs(model: &ModelProto) -> Vec<&ValueInfoProto> {
    let inits = initializer_names(model);
    graph_proto(model)
        .input
        .iter()
        .filter(|vi| !inits.contains(&vi.name))
        .collect()
}

fn tensor_dims(vi: &ValueInfoProto) -> Option<(i32, Vec<usize>)> {
    let ty = vi.r#type.as_ref()?;
    let tensor = match ty.value.as_ref()? {
        type_proto::Value::TensorType(t) => t,
        _ => return None,
    };
    let elem_type = tensor.elem_type;
    let shape = tensor.shape.as_ref()?;
    let dims = shape
        .dim
        .iter()
        .map(|d| {
            use onnx2webnn::protos::onnx::tensor_shape_proto::dimension::Value as Dim;
            match d.value.as_ref() {
                Some(Dim::DimValue(v)) => (*v).max(1) as usize,
                _ => 1,
            }
        })
        .collect();
    Some((elem_type, dims))
}

fn deterministic_float_data(name: &str, len: usize) -> Vec<f32> {
    let seed = name.bytes().map(u64::from).sum::<u64>() % 7;
    (0..len)
        .map(|i| 0.125 * (i as f32 + 1.0) + (seed as f32) * 0.01)
        .collect()
}

fn deterministic_float16_data(name: &str, len: usize, acosh_graph: bool) -> Vec<u16> {
    let values = if acosh_graph {
        (0..len).map(|i| 1.25 + 0.5 * i as f32).collect()
    } else {
        deterministic_float_data(name, len)
    };
    values
        .into_iter()
        .map(|v| f16::from_f32(v).to_bits())
        .collect()
}

fn deterministic_int_data(name: &str, len: usize) -> Vec<i64> {
    let seed = (name.bytes().map(u64::from).sum::<u64>() % 5) as i64;
    (0..len).map(|i| (i as i64 + seed) % 7).collect()
}

fn deterministic_bool_data(len: usize) -> Vec<u8> {
    (0..len).map(|i| u8::from(i % 2 == 0)).collect()
}

fn build_ort_inputs(model: &ModelProto) -> Result<Vec<OnnxInput>, String> {
    let acosh_graph = graph_proto(model)
        .node
        .iter()
        .any(|node| node.op_type == "Acosh");
    let mut inputs = Vec::new();
    for vi in feedable_inputs(model) {
        let (elem_type, shape) =
            tensor_dims(vi).ok_or_else(|| format!("unsupported input kind for {}", vi.name))?;
        let count = shape.iter().product::<usize>().max(1);
        let data = match elem_type {
            x if x == TensorProto_DataType::Float as i32 => {
                let values = if acosh_graph {
                    (0..count).map(|i| 1.25 + 0.5 * i as f32).collect()
                } else {
                    deterministic_float_data(&vi.name, count)
                };
                TensorData::Float32(values)
            }
            x if x == TensorProto_DataType::Float16 as i32 => {
                TensorData::Float16(deterministic_float16_data(&vi.name, count, acosh_graph))
            }
            x if x == TensorProto_DataType::Int32 as i32 => {
                let vals = deterministic_int_data(&vi.name, count);
                TensorData::Int32(vals.into_iter().map(|v| v as i32).collect())
            }
            x if x == TensorProto_DataType::Int64 as i32 => {
                TensorData::Int64(deterministic_int_data(&vi.name, count))
            }
            x if x == TensorProto_DataType::Bool as i32 => {
                TensorData::Uint8(deterministic_bool_data(count))
            }
            x if x == TensorProto_DataType::Uint8 as i32 => {
                TensorData::Uint8((0..count).map(|i| (i % 255) as u8).collect())
            }
            other => {
                return Err(format!(
                    "unsupported ORT input dtype {other} for {}",
                    vi.name
                ))
            }
        };
        inputs.push(OnnxInput {
            name: vi.name.clone(),
            shape,
            data,
        });
    }
    Ok(inputs)
}

fn clone_tensor_data(data: &TensorData) -> TensorData {
    match data {
        TensorData::Float32(v) => TensorData::Float32(v.clone()),
        TensorData::Float16(v) => TensorData::Float16(v.clone()),
        TensorData::Int8(v) => TensorData::Int8(v.clone()),
        TensorData::Uint8(v) => TensorData::Uint8(v.clone()),
        TensorData::Int32(v) => TensorData::Int32(v.clone()),
        TensorData::Uint32(v) => TensorData::Uint32(v.clone()),
        TensorData::Int64(v) => TensorData::Int64(v.clone()),
        TensorData::Uint64(v) => TensorData::Uint64(v.clone()),
    }
}

fn clone_ort_inputs(inputs: &[OnnxInput]) -> Vec<OnnxInput> {
    inputs
        .iter()
        .map(|input| OnnxInput {
            name: input.name.clone(),
            shape: input.shape.clone(),
            data: clone_tensor_data(&input.data),
        })
        .collect()
}

fn operand_descriptor_to_tensor_desc(desc: &OperandDescriptor) -> MLTensorDescriptor {
    let shape = desc
        .static_or_max_shape()
        .into_iter()
        .map(|d| d as u64)
        .collect();
    let data_type = MLOperandDataType::try_from(desc.data_type)
        .expect("operand descriptor should map to WebNN type");
    let mut tensor_desc = MLTensorDescriptor::new(data_type, shape);
    tensor_desc.set_readable(true);
    tensor_desc.set_writable(true);
    tensor_desc
}

fn write_ort_input(context: &mut MLContext, tensor: &MLTensor, input: &OnnxInput) {
    match &input.data {
        TensorData::Float32(data) => context.write_tensor(tensor, data).unwrap(),
        TensorData::Float16(data) => context.write_tensor(tensor, data).unwrap(),
        TensorData::Int32(data) => context.write_tensor(tensor, data).unwrap(),
        TensorData::Int64(data) => context.write_tensor(tensor, data).unwrap(),
        TensorData::Uint8(data) => context.write_tensor(tensor, data).unwrap(),
        _ => panic!("unsupported tensor write for ORT input"),
    }
}

fn onnx_input_webnn_names(model: &ModelProto) -> HashSet<String> {
    feedable_inputs(model)
        .iter()
        .map(|vi| OnnxBuilder::webnn_id(&vi.name))
        .collect()
}

fn dispatch_and_collect(
    validated: &mut ValidatedGraph,
    model: &ModelProto,
    ort_inputs: &[OnnxInput],
) -> Result<HashMap<String, Vec<f64>>, String> {
    let context = &mut validated.context;
    let graph = &mut validated.graph;
    let input_webnn_names = onnx_input_webnn_names(model);

    let mut owned_inputs: Vec<MLTensor> = Vec::new();
    let mut input_names: Vec<String> = Vec::new();
    for ort_input in ort_inputs {
        let webnn_name = OnnxBuilder::webnn_id(&ort_input.name);
        let desc = graph
            .input_descriptors
            .get(&webnn_name)
            .ok_or_else(|| format!("missing graph input descriptor for {webnn_name}"))?;
        let tensor_desc = operand_descriptor_to_tensor_desc(desc);
        let tensor = context
            .create_tensor(&tensor_desc)
            .map_err(|e| e.to_string())?;
        write_ort_input(context, &tensor, ort_input);
        owned_inputs.push(tensor);
        input_names.push(webnn_name);
    }
    let mut input_bindings: BTreeMap<&str, &MLTensor> = BTreeMap::new();
    for (name, tensor) in input_names.iter().zip(owned_inputs.iter()) {
        input_bindings.insert(name.as_str(), tensor);
    }

    let mut owned_outputs: Vec<MLTensor> = Vec::new();
    let mut output_names: Vec<String> = Vec::new();
    let mut output_keys: HashMap<String, String> = HashMap::new();
    for out in &graph_proto(model).output {
        let webnn_key = OnnxBuilder::output_key_for(&out.name, &input_webnn_names);
        let desc = graph
            .output_descriptors
            .get(&webnn_key)
            .ok_or_else(|| format!("missing graph output descriptor for {webnn_key}"))?;
        let tensor_desc = operand_descriptor_to_tensor_desc(desc);
        let tensor = context
            .create_tensor(&tensor_desc)
            .map_err(|e| e.to_string())?;
        output_names.push(webnn_key.clone());
        owned_outputs.push(tensor);
        output_keys.insert(out.name.clone(), webnn_key);
    }
    let mut output_bindings: BTreeMap<&str, &MLTensor> = BTreeMap::new();
    for (name, tensor) in output_names.iter().zip(owned_outputs.iter()) {
        output_bindings.insert(name.as_str(), tensor);
    }

    context
        .dispatch(graph, &input_bindings, &output_bindings)
        .map_err(|e| e.to_string())?;

    let mut results = HashMap::new();
    for (onnx_name, webnn_key) in output_keys {
        let desc = graph
            .output_descriptors
            .get(&webnn_key)
            .expect("validated above");
        let tensor = output_bindings
            .get(webnn_key.as_str())
            .expect("output binding");
        let values = read_tensor_as_f64(context, tensor, desc)?;
        results.insert(onnx_name, values);
    }
    Ok(results)
}

fn read_tensor_as_f64(
    context: &mut MLContext,
    tensor: &MLTensor,
    desc: &OperandDescriptor,
) -> Result<Vec<f64>, String> {
    let count = desc.element_count().unwrap_or(1).max(1);
    match desc.data_type {
        rustnn::DataType::Float32 => {
            let mut buf = vec![0.0f32; count];
            context
                .read_tensor(tensor, &mut buf)
                .map_err(|e| e.to_string())?;
            Ok(buf.into_iter().map(f64::from).collect())
        }
        rustnn::DataType::Float16 => {
            let mut buf = vec![0u16; count];
            context
                .read_tensor(tensor, &mut buf)
                .map_err(|e| e.to_string())?;
            Ok(buf
                .into_iter()
                .map(|v| f64::from(f16::from_bits(v).to_f32()))
                .collect())
        }
        rustnn::DataType::Int32 => {
            let mut buf = vec![0i32; count];
            context
                .read_tensor(tensor, &mut buf)
                .map_err(|e| e.to_string())?;
            Ok(buf.into_iter().map(|v| v as f64).collect())
        }
        rustnn::DataType::Int64 => {
            let mut buf = vec![0i64; count];
            context
                .read_tensor(tensor, &mut buf)
                .map_err(|e| e.to_string())?;
            Ok(buf.into_iter().map(|v| v as f64).collect())
        }
        rustnn::DataType::Uint8 => {
            let mut buf = vec![0u8; count];
            context
                .read_tensor(tensor, &mut buf)
                .map_err(|e| e.to_string())?;
            Ok(buf.into_iter().map(f64::from).collect())
        }
        other => Err(format!("unsupported output read dtype {other:?}")),
    }
}

fn value_info_elem_type(vi: &ValueInfoProto) -> Option<i32> {
    let ty = vi.r#type.as_ref()?;
    let tensor = match ty.value.as_ref()? {
        type_proto::Value::TensorType(t) => t,
        _ => return None,
    };
    Some(tensor.elem_type)
}

fn compare_outputs(
    model: &ModelProto,
    reference: &[rustnn::OnnxOutputWithData],
    actual: &HashMap<String, Vec<f64>>,
) {
    let outputs = &graph_proto(model).output;
    assert_eq!(
        reference.len(),
        outputs.len(),
        "reference output count mismatch"
    );
    for (idx, out) in outputs.iter().enumerate() {
        let ort = &reference[idx];
        let got = actual
            .get(&out.name)
            .unwrap_or_else(|| panic!("missing rustnn output for {}", out.name));
        if let Some(expected_bool) = ort.bool_data.as_ref() {
            assert_same_bool_values(&out.name, expected_bool, got);
            continue;
        }
        if let Some(expected_int) = ort.int64_data.as_ref() {
            // Integer semantics: any nonzero difference is a real bug; don't let
            // the relative float tolerance absorb off-by-one at large magnitudes
            // (its threshold reaches 1 once |expected| >= 1e6).
            assert_eq!(
                expected_int.len(),
                got.len(),
                "output length mismatch for {}",
                out.name
            );
            for (i, (e, a)) in expected_int.iter().zip(got.iter()).enumerate() {
                assert!(
                    (a - a.round()).abs() < 1e-6,
                    "output {}[{i}] not integral: {a}",
                    out.name
                );
                assert_eq!(
                    *e,
                    a.round() as i64,
                    "output {}[{i}] mismatch: ORT={e}, rustnn={a}",
                    out.name
                );
            }
            continue;
        }
        let expected = ort
            .float32_data
            .as_ref()
            .map(|data| data.iter().map(|&v| f64::from(v)).collect::<Vec<_>>())
            .unwrap_or_else(|| ort.data.clone());
        let is_float16 = value_info_elem_type(out) == Some(TensorProto_DataType::Float16 as i32);
        assert_same_values(&out.name, &expected, got, is_float16);
    }
}

fn assert_same_bool_values(name: &str, expected: &[bool], actual: &[f64]) {
    assert_eq!(
        expected.len(),
        actual.len(),
        "output length mismatch for {name}"
    );
    for (i, (expected, actual)) in expected.iter().zip(actual.iter()).enumerate() {
        let actual_bool = if (*actual - 0.0).abs() <= f64::EPSILON {
            false
        } else if (*actual - 1.0).abs() <= f64::EPSILON {
            true
        } else {
            panic!("output {name}[{i}] expected bool-compatible 0/1, got {actual}");
        };
        assert_eq!(
            *expected, actual_bool,
            "output {name}[{i}] mismatch: ORT={expected}, rustnn={actual_bool}"
        );
    }
}

fn assert_same_values(name: &str, expected: &[f64], actual: &[f64], is_float16: bool) {
    assert_eq!(
        expected.len(),
        actual.len(),
        "output length mismatch for {name}"
    );
    for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        if e.is_nan() && a.is_nan() {
            continue;
        }
        if e.is_infinite() && a.is_infinite() && e.signum() == a.signum() {
            continue;
        }
        let rounded_e = (e * 1_000_000.0).round() / 1_000_000.0;
        let rounded_a = (a * 1_000_000.0).round() / 1_000_000.0;
        // Absolute floor for near-zero values plus a relative term so values of
        // large magnitude tolerate backend rounding differences (CoreML's
        // FMA/accumulation order differs from ORT's by a few float32 ULPs).
        let (abs_tolerance, rel_tolerance) = if is_float16 {
            (1e-2, 1e-3)
        } else {
            (1e-5, 1e-6)
        };
        let tolerance = abs_tolerance + rel_tolerance * e.abs();
        assert!(
            (rounded_e - rounded_a).abs() <= tolerance,
            "output {name}[{i}] mismatch: ORT={rounded_e}, rustnn={rounded_a}"
        );
    }
}
