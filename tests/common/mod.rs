/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared helpers for ONNX op conversion integration tests.

mod runner;
pub mod skeleton;

// Not every test crate uses every helper.
#[allow(unused_imports)]
pub use runner::{assert_op_matches_ort, assert_op_matches_ort_with_options, ExpectConvertOp};
