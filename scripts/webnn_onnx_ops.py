# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ONNX operators supported by onnx2webnn convert (mirrors src/onnx/ops/*.rs)."""

from __future__ import annotations

# ai.onnx ops accepted by OpRegistry in webnn-graph.
WEBNN_SUPPORTED_ONNX_OPS: frozenset[str] = frozenset(
    {
        # matmul.rs
        "MatMul",
        "Gemm",
        "MatMulNBits",
        "MatMulInteger",
        "MatMulBnb4",
        # attention.rs
        "GroupQueryAttention",
        # moe.rs
        "MoE",
        # convert.rs inline_constant_ifs: If with a constant condition is
        # inlined; runtime conditions remain unsupported.
        "If",
        "QMoE",
        # einsum.rs
        "Einsum",
        # conv.rs
        "Conv",
        "ConvTranspose",
        "ConvInteger",
        # pool.rs
        "MaxPool",
        "AveragePool",
        "LpPool",
        "GlobalMaxPool",
        "GlobalAveragePool",
        # elementwise.rs
        "Add",
        "Sub",
        "Mul",
        "Div",
        "Pow",
        "Min",
        "Max",
        "Mod",
        # comparison.rs
        "Greater",
        "Less",
        "Equal",
        "GreaterOrEqual",
        "LessOrEqual",
        "Not",
        "And",
        "Or",
        "Xor",
        "IsNaN",
        "IsInf",
        # conditional.rs
        "Where",
        # normalization.rs
        "BatchNormalization",
        "InstanceNormalization",
        "LayerNormalization",
        "Softmax",
        "GroupNormalization",
        "RMSNormalization",
        "SimplifiedLayerNormalization",
        "SkipSimplifiedLayerNormalization",
        "LogSoftmax",
        "Hardmax",
        # reshape.rs
        "Reshape",
        "Transpose",
        "Concat",
        "Split",
        "Unsqueeze",
        "Squeeze",
        "Tile",
        "Expand",
        "Flatten",
        # conversion.rs
        "Cast",
        "CastLike",
        "Constant",
        "QuantizeLinear",
        "DequantizeLinear",
        "DynamicQuantizeLinear",
        # utility.rs
        "Shape",
        "OneHot",
        "Gather",
        "GatherBlockQuantized",
        "GatherND",
        "GatherElements",
        "ReverseSequence",
        "Slice",
        "ConstantOfShape",
        "Range",
        "Trilu",
        # reduction.rs
        "ReduceMean",
        "ReduceSum",
        "ReduceMax",
        "ReduceMin",
        "ReduceL1",
        "ReduceL2",
        "ReduceLogSum",
        "ReduceLogSumExp",
        "ReduceProd",
        "ReduceSumSquare",
        "ArgMin",
        "ArgMax",
        "CumSum",
        # activation.rs
        "Relu",
        "Gelu",
        "Tanh",
        "Sigmoid",
        "Sqrt",
        "Exp",
        "Log",
        "Abs",
        "Neg",
        "Erf",
        "Cos",
        "Sin",
        "Identity",
        "Floor",
        "Ceil",
        "Sign",
        "Tan",
        "Reciprocal",
        "Round",
        "HardSwish",
        "Softplus",
        "Softsign",
        "Elu",
        "LeakyRelu",
        "HardSigmoid",
        "Clip",
        "PRelu",
        "Swish",
        "Celu",
        "Selu",
        "Mish",
        "ThresholdedRelu",
        "Sinh",
        "Cosh",
        "Asinh",
        "Acosh",
        "Atanh",
        "Shrink",
        # scatter.rs
        "ScatterND",
        "ScatterElements",
        # resize.rs
        "Resize",
        # pad.rs
        "Pad",
        # misc.rs
        "Mean",
        "Sum",
        "CumProd",
        # rnn.rs
        "GRU",
        "LSTM",
        "RotaryEmbedding",
    }
)

# convert.rs MIN/MAX_SUPPORTED_OPSET for domain ai.onnx
WEBNN_MIN_OPSET: int = 9
WEBNN_MAX_OPSET: int = 26


def is_webnn_supported_op(domain: str, op_type: str) -> bool:
    if domain != "ai.onnx":
        return False
    return op_type in WEBNN_SUPPORTED_ONNX_OPS


def is_webnn_supported_opset(domain: str, version: int) -> bool:
    if domain != "ai.onnx":
        return True
    return WEBNN_MIN_OPSET <= version <= WEBNN_MAX_OPSET
