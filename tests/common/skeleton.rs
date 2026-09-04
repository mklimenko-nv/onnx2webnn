/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Weight-stripped ONNX "skeletons" built at the protobuf wire level.
//!
//! [`strip_model`] copies a serialized `ModelProto` field by field, keeps the
//! graph and small initializers, and replaces every large initializer with an
//! external-data reference to a file that does not exist. The converter
//! zero-fills those tensors (`zero_fill_missing_external_data`), which is all
//! shape inference, lowering and the ORT graph build need.
//!
//! The scanner never reads weight bytes: [`ByteSource`] gives it random
//! access, and the [`HubSource`] implementation uses HTTP range requests
//! against the Hugging Face Hub, so a 1.4 GB export costs ~10 MB of traffic.

// Shared test module; only tests/model_skeletons.rs uses it.
#![allow(dead_code)]

use std::io::{Read, Seek, SeekFrom};
use std::time::Duration;

/// Initializers up to this many bytes stay inline (shape vectors, axes,
/// scalars that constant folding needs).
pub const KEEP_BYTES: usize = 4096;

/// Read window right after a skipped tensor (usually a short header follows),
/// doubling while reads stay contiguous.
const MIN_WINDOW: usize = 32 << 10;
const MAX_WINDOW: usize = 8 << 20;

const VARINT: u8 = 0;
const FIXED64: u8 = 1;
const LENGTH_DELIMITED: u8 = 2;
const FIXED32: u8 = 5;

// ONNX field numbers.
const MODEL_GRAPH: u32 = 7;
const GRAPH_NODE: u32 = 1;
const GRAPH_INITIALIZER: u32 = 5;
const NODE_ATTRIBUTE: u32 = 5;
const ATTR_G: u32 = 6;
const ATTR_GRAPHS: u32 = 11;
const TENSOR_DIMS: u32 = 1;
const TENSOR_DATA_TYPE: u32 = 2;
// float_data, int32_data, string_data, int64_data, raw_data, double_data, uint64_data
const TENSOR_DATA_FIELDS: [u32; 7] = [4, 5, 6, 7, 9, 10, 11];
const TENSOR_EXTERNAL_DATA: u32 = 13;
const TENSOR_DATA_LOCATION: u32 = 14;

/// Random-access bytes: a local file or a remote file read by ranges.
pub trait ByteSource {
    fn len(&self) -> u64;
    fn read_at(&mut self, start: u64, len: usize) -> Result<Vec<u8>, String>;
}

pub struct FileSource {
    file: std::fs::File,
    len: u64,
}

impl FileSource {
    pub fn open(path: &std::path::Path) -> Result<Self, String> {
        let file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let len = file.metadata().map_err(|e| e.to_string())?.len();
        Ok(Self { file, len })
    }
}

impl ByteSource for FileSource {
    fn len(&self) -> u64 {
        self.len
    }

    fn read_at(&mut self, start: u64, len: usize) -> Result<Vec<u8>, String> {
        let mut buf = vec![0u8; len];
        self.file
            .seek(SeekFrom::Start(start))
            .and_then(|_| self.file.read_exact(&mut buf))
            .map_err(|e| e.to_string())?;
        Ok(buf)
    }
}

/// A file on the Hugging Face Hub read with range requests over a keep-alive
/// connection. `HF_TOKEN` is sent when set (raises rate limits).
pub struct HubSource {
    agent: ureq::Agent,
    url: String,
    len: u64,
}

impl HubSource {
    /// `<org>--<repo>/onnx/<file>.onnx` -> `https://huggingface.co/<org>/<repo>/resolve/main/onnx/<file>.onnx`.
    pub fn open(file: &str) -> Result<Self, String> {
        let (org_repo, rel) = file
            .split_once('/')
            .ok_or_else(|| format!("{file}: expected <org>--<repo>/<path>"))?;
        let repo = org_repo.replacen("--", "/", 1);
        let url = format!("https://huggingface.co/{repo}/resolve/main/{rel}");
        let agent = ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(120))
            .build();
        // Resolve the (signed CDN) location once; range requests reuse it.
        let mut request = agent.head(&url);
        if let Ok(token) = std::env::var("HF_TOKEN") {
            if !token.is_empty() {
                request = request.set("Authorization", &format!("Bearer {token}"));
            }
        }
        let response = with_retries(|| request.clone().call().map_err(|e| e.to_string()))
            .map_err(|e| format!("resolve {url}: {e}"))?;
        let len = response
            .header("Content-Length")
            .and_then(|v| v.parse().ok())
            .ok_or_else(|| format!("{url}: no Content-Length"))?;
        Ok(Self {
            agent,
            url: response.get_url().to_string(),
            len,
        })
    }
}

impl ByteSource for HubSource {
    fn len(&self) -> u64 {
        self.len
    }

    fn read_at(&mut self, start: u64, len: usize) -> Result<Vec<u8>, String> {
        let end = start + len as u64 - 1;
        with_retries(|| {
            let response = self
                .agent
                .get(&self.url)
                .set("Range", &format!("bytes={start}-{end}"))
                .call()
                .map_err(|e| e.to_string())?;
            let status = response.status();
            let mut body = Vec::with_capacity(len);
            response
                .into_reader()
                .read_to_end(&mut body)
                .map_err(|e| e.to_string())?;
            match status {
                206 if body.len() == len => Ok(body),
                206 => Err(format!("range {start}-{end}: got {} bytes", body.len())),
                // Server ignored the range and sent the whole file.
                200 if body.len() as u64 >= start + len as u64 => {
                    Ok(body[start as usize..start as usize + len].to_vec())
                }
                other => Err(format!("range {start}-{end}: HTTP {other}")),
            }
        })
    }
}

fn with_retries<T>(mut op: impl FnMut() -> Result<T, String>) -> Result<T, String> {
    let mut last = String::new();
    for attempt in 0..5 {
        match op() {
            Ok(value) => return Ok(value),
            Err(err) => {
                last = err;
                std::thread::sleep(Duration::from_millis(1500 * (attempt + 1)));
            }
        }
    }
    Err(last)
}

/// Sequential reader with cheap `skip` over a [`ByteSource`].
struct Reader<S: ByteSource> {
    src: S,
    pos: u64,
    buf: Vec<u8>,
    buf_start: u64,
    window: usize,
    requests: usize,
    fetched: u64,
}

impl<S: ByteSource> Reader<S> {
    fn new(src: S) -> Self {
        Self {
            src,
            pos: 0,
            buf: Vec::new(),
            buf_start: 0,
            window: MIN_WINDOW,
            requests: 0,
            fetched: 0,
        }
    }

    fn at_end(&self) -> bool {
        self.pos >= self.src.len()
    }

    fn fill(&mut self, n: usize) -> Result<(), String> {
        let buf_end = self.buf_start + self.buf.len() as u64;
        if self.pos >= self.buf_start && self.pos + n as u64 <= buf_end {
            return Ok(());
        }
        if !self.buf.is_empty() && self.pos == buf_end {
            self.window = (self.window * 2).min(MAX_WINDOW);
        }
        let available = self.src.len().saturating_sub(self.pos);
        let len = (n.max(self.window) as u64).min(available) as usize;
        if len < n {
            return Err(format!(
                "truncated model: need {n} bytes at offset {}, {available} available",
                self.pos
            ));
        }
        self.buf = self.src.read_at(self.pos, len)?;
        self.buf_start = self.pos;
        self.requests += 1;
        self.fetched += len as u64;
        Ok(())
    }

    fn read(&mut self, n: usize) -> Result<Vec<u8>, String> {
        self.fill(n)?;
        let off = (self.pos - self.buf_start) as usize;
        self.pos += n as u64;
        Ok(self.buf[off..off + n].to_vec())
    }

    fn read_byte(&mut self) -> Result<u8, String> {
        self.fill(1)?;
        let byte = self.buf[(self.pos - self.buf_start) as usize];
        self.pos += 1;
        Ok(byte)
    }

    fn skip(&mut self, n: u64) {
        self.pos += n;
        if self.pos > self.buf_start + self.buf.len() as u64 {
            self.window = MIN_WINDOW;
        }
    }

    fn read_varint(&mut self) -> Result<u64, String> {
        let mut value = 0u64;
        for shift in (0..64).step_by(7) {
            let byte = self.read_byte()?;
            value |= u64::from(byte & 0x7F) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err("varint too long".to_string())
    }

    fn read_tag(&mut self) -> Result<(u32, u8), String> {
        let tag = self.read_varint()?;
        Ok(((tag >> 3) as u32, (tag & 7) as u8))
    }

    /// Raw encoding of a field value (length prefix included for LEN fields).
    fn read_value(&mut self, wire_type: u8) -> Result<Vec<u8>, String> {
        match wire_type {
            VARINT => {
                let mut out = Vec::with_capacity(4);
                loop {
                    let byte = self.read_byte()?;
                    out.push(byte);
                    if byte & 0x80 == 0 {
                        return Ok(out);
                    }
                }
            }
            FIXED64 => self.read(8),
            FIXED32 => self.read(4),
            LENGTH_DELIMITED => {
                let len = self.read_varint()?;
                let mut out = encode_varint(len);
                out.extend(self.read(len as usize)?);
                Ok(out)
            }
            other => Err(format!("unsupported wire type {other}")),
        }
    }
}

fn encode_varint(mut value: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(10);
    loop {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value == 0 {
            out.push(byte);
            return out;
        }
        out.push(byte | 0x80);
    }
}

fn encode_tag(field: u32, wire_type: u8) -> Vec<u8> {
    encode_varint((u64::from(field) << 3) | u64::from(wire_type))
}

fn encode_len_field(field: u32, payload: &[u8]) -> Vec<u8> {
    let mut out = encode_tag(field, LENGTH_DELIMITED);
    out.extend(encode_varint(payload.len() as u64));
    out.extend_from_slice(payload);
    out
}

fn decode_varint(bytes: &[u8]) -> u64 {
    let mut value = 0u64;
    for (i, byte) in bytes.iter().enumerate() {
        value |= u64::from(byte & 0x7F) << (7 * i);
        if byte & 0x80 == 0 {
            break;
        }
    }
    value
}

fn decode_packed_varints(mut bytes: &[u8]) -> Vec<u64> {
    let mut out = Vec::new();
    while !bytes.is_empty() {
        let end = bytes
            .iter()
            .position(|b| b & 0x80 == 0)
            .map_or(bytes.len(), |i| i + 1);
        out.push(decode_varint(&bytes[..end]));
        bytes = &bytes[end..];
    }
    out
}

fn element_size(data_type: u64) -> u64 {
    match data_type {
        1 | 6 | 12 => 4,      // float, int32, uint32
        2 | 3 | 9 => 1,       // uint8, int8, bool
        4 | 5 | 10 | 16 => 2, // uint16, int16, float16, bfloat16
        7 | 11 | 13 => 8,     // int64, double, uint64
        _ => 1,
    }
}

#[derive(Default, Debug)]
pub struct Stats {
    pub kept: usize,
    pub stripped: usize,
    pub stripped_bytes: u64,
    pub external: usize,
    pub requests: usize,
    pub fetched: u64,
}

/// Copy a TensorProto, replacing large data fields with an external reference.
fn strip_tensor<S: ByteSource>(
    r: &mut Reader<S>,
    len: u64,
    keep_bytes: usize,
    stats: &mut Stats,
) -> Result<Vec<u8>, String> {
    let end = r.pos + len;
    let mut kept = Vec::new();
    let mut data_bytes = 0u64;
    let mut saw_data = false;
    let mut dims = Vec::new();
    let mut data_type = 0u64;
    while r.pos < end {
        let (field, wt) = r.read_tag()?;
        if TENSOR_DATA_FIELDS.contains(&field) {
            saw_data = true;
            if wt == LENGTH_DELIMITED {
                let n = r.read_varint()?;
                data_bytes += n;
                if n as usize <= keep_bytes {
                    kept.extend(encode_len_field(field, &r.read(n as usize)?));
                } else {
                    r.skip(n);
                }
            } else {
                // Unpacked scalar entry (rare, small).
                kept.extend(encode_tag(field, wt));
                kept.extend(r.read_value(wt)?);
            }
            continue;
        }
        let raw = r.read_value(wt)?;
        if field == TENSOR_DIMS {
            if wt == LENGTH_DELIMITED {
                let prefix = raw
                    .iter()
                    .position(|b| b & 0x80 == 0)
                    .map_or(raw.len(), |i| i + 1);
                dims.extend(decode_packed_varints(&raw[prefix..]));
            } else {
                dims.push(decode_varint(&raw));
            }
        } else if field == TENSOR_DATA_TYPE && wt == VARINT {
            data_type = decode_varint(&raw);
        }
        kept.extend(encode_tag(field, wt));
        kept.extend(raw);
    }
    if !saw_data {
        // Already external (or empty): a missing data file is zero-filled.
        stats.external += 1;
        return Ok(kept);
    }
    if data_bytes as usize <= keep_bytes {
        stats.kept += 1;
        return Ok(kept);
    }
    stats.stripped += 1;
    stats.stripped_bytes += data_bytes;
    let length = if dims.is_empty() {
        data_bytes
    } else if data_type == 21 || data_type == 22 {
        // UINT4 / INT4: two elements per byte.
        dims.iter().product::<u64>().div_ceil(2)
    } else {
        dims.iter().product::<u64>() * element_size(data_type)
    };
    let mut entry = encode_len_field(1, b"location");
    entry.extend(encode_len_field(2, b"skeleton.bin"));
    kept.extend(encode_len_field(TENSOR_EXTERNAL_DATA, &entry));
    let mut entry = encode_len_field(1, b"length");
    entry.extend(encode_len_field(2, length.to_string().as_bytes()));
    kept.extend(encode_len_field(TENSOR_EXTERNAL_DATA, &entry));
    kept.extend(encode_tag(TENSOR_DATA_LOCATION, VARINT));
    kept.extend(encode_varint(1));
    Ok(kept)
}

/// Copy a NodeProto, recursing into subgraph attributes (If/Loop/Scan bodies).
fn strip_node<S: ByteSource>(
    r: &mut Reader<S>,
    len: u64,
    keep_bytes: usize,
    stats: &mut Stats,
) -> Result<Vec<u8>, String> {
    let end = r.pos + len;
    let mut out = Vec::new();
    while r.pos < end {
        let (field, wt) = r.read_tag()?;
        if field == NODE_ATTRIBUTE && wt == LENGTH_DELIMITED {
            let n = r.read_varint()?;
            out.extend(encode_len_field(
                field,
                &strip_attribute(r, n, keep_bytes, stats)?,
            ));
            continue;
        }
        out.extend(encode_tag(field, wt));
        out.extend(r.read_value(wt)?);
    }
    Ok(out)
}

fn strip_attribute<S: ByteSource>(
    r: &mut Reader<S>,
    len: u64,
    keep_bytes: usize,
    stats: &mut Stats,
) -> Result<Vec<u8>, String> {
    let end = r.pos + len;
    let mut out = Vec::new();
    while r.pos < end {
        let (field, wt) = r.read_tag()?;
        if (field == ATTR_G || field == ATTR_GRAPHS) && wt == LENGTH_DELIMITED {
            let n = r.read_varint()?;
            out.extend(encode_len_field(
                field,
                &strip_graph(r, n, keep_bytes, stats)?,
            ));
            continue;
        }
        out.extend(encode_tag(field, wt));
        out.extend(r.read_value(wt)?);
    }
    Ok(out)
}

fn strip_graph<S: ByteSource>(
    r: &mut Reader<S>,
    len: u64,
    keep_bytes: usize,
    stats: &mut Stats,
) -> Result<Vec<u8>, String> {
    let end = r.pos + len;
    let mut out = Vec::new();
    while r.pos < end {
        let (field, wt) = r.read_tag()?;
        if wt == LENGTH_DELIMITED && (field == GRAPH_INITIALIZER || field == GRAPH_NODE) {
            let n = r.read_varint()?;
            let payload = if field == GRAPH_INITIALIZER {
                strip_tensor(r, n, keep_bytes, stats)?
            } else {
                strip_node(r, n, keep_bytes, stats)?
            };
            out.extend(encode_len_field(field, &payload));
            continue;
        }
        out.extend(encode_tag(field, wt));
        out.extend(r.read_value(wt)?);
    }
    Ok(out)
}

/// Serialized skeleton `ModelProto` for the model behind `src`.
pub fn strip_model<S: ByteSource>(src: S, keep_bytes: usize) -> Result<(Vec<u8>, Stats), String> {
    let mut r = Reader::new(src);
    let mut stats = Stats::default();
    let mut out = Vec::new();
    while !r.at_end() {
        let (field, wt) = r.read_tag()?;
        if field == MODEL_GRAPH && wt == LENGTH_DELIMITED {
            let n = r.read_varint()?;
            out.extend(encode_len_field(
                field,
                &strip_graph(&mut r, n, keep_bytes, &mut stats)?,
            ));
            continue;
        }
        out.extend(encode_tag(field, wt));
        out.extend(r.read_value(wt)?);
    }
    stats.requests = r.requests;
    stats.fetched = r.fetched;
    Ok((out, stats))
}
