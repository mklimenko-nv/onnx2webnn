#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regenerate tests/models/manifest.json from the transformers.js model ranking.

The sweep manifest is defined by two rules, both reproduced here:

1. Which repos: the most-liked Hugging Face model per task among repos tagged
   `transformers.js` (--top N for more per task; the ranking is fetched from
   the Hub and cached under target/, --refresh-ranking re-fetches), minus
   repos and components documented as unconvertible in
   tests/models/UNSUPPORTED_OPS.md. Tasks with nothing to convert
   (`(tokenizer-only)`) are skipped; models without onnx/*.onnx files are
   reported.
2. Which files: what transformers.js itself loads for that repo -- the
   component set its model class constructs (`model`; `encoder_model` +
   `decoder_model_merged`; `embed_tokens` + `vision_encoder` +
   `decoder_model_merged`; Janus' six parts; ...) at the dtype it would pick
   with no caller override: `transformers.js_config.dtype` (string, or a
   per-file object keyed by base file name), else the device default -- q8
   (`_quantized`) on wasm, fp32 on every other device incl. WebNN. Mirrors
   transformers.js `session_config.js` / `session.js` / `utils/dtypes.js`.

Free dimensions are then filled with the same naming policy as onnx2webnn's
src/onnx/probe.rs (batch_size=1, sequence_length=128, image sizes 224, ...);
a `use_cache_branch` input yields a prefill and a decode entry. Values already
in the current manifest for the same file (or the same component at another
dtype) are carried forward, so hand-tuned dims survive regeneration.

Model files are never downloaded. Like the sweep's skeleton scanner
(tests/common/skeleton.rs) the serialized graph is walked over HTTP range
requests, skipping every node and every initializer's data by its length
prefix and decoding only the graph inputs, so a 1.4 GB export costs about
10 MB of traffic. --reuse-dir reads local copies the same way.

Examples:
  # Regenerate, report the differences, write a review copy to target/.
  python scripts/generate_manifest.py
  diff tests/models/manifest.json target/manifest.generated.json

  # WebNN defaults (fp32) instead of wasm's q8; two models per task.
  python scripts/generate_manifest.py --device webnn --top 2

  # Overwrite the real manifest once the report looks right.
  python scripts/generate_manifest.py --apply

Needs only `onnx` (requirements.txt). HF_TOKEN raises the Hub's rate limits.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from typing import Any

import onnx

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_MANIFEST = REPO_ROOT / "tests" / "models" / "manifest.json"
DEFAULT_UNSUPPORTED_FILE = REPO_ROOT / "tests" / "models" / "UNSUPPORTED_OPS.md"
DEFAULT_OVERRIDES = REPO_ROOT / "tests" / "models" / "manifest-overrides.json"
# target/ is gitignored: the ranking snapshot and the review copy live there.
DEFAULT_MODELS_JSON = REPO_ROOT / "target" / "transformers-js-models.json"
DEFAULT_OUT = REPO_ROOT / "target" / "manifest.generated.json"

HUB = "https://huggingface.co"
USER_AGENT = "onnx2webnn/generate_manifest"

HEAVY_BYTES = 1 << 30
SEQ_LEN = 128
DECODER_SEQ_LEN = 16
DECODE_PAST_LEN = 16
CANONICAL_KEYS = ["file", "heavy", "pin_inputs", "override_dims", "coreml_unsupported", "coreml_slow"]

# transformers.js utils/dtypes.js DEFAULT_DTYPE_SUFFIX_MAPPING.
DTYPE_SUFFIX = {
    "fp32": "",
    "fp16": "_fp16",
    "int8": "_int8",
    "uint8": "_uint8",
    "q8": "_quantized",
    "q4": "_q4",
    "q2": "_q2",
    "q1": "_q1",
    "q4f16": "_q4f16",
    "q2f16": "_q2f16",
    "q1f16": "_q1f16",
    "bnb4": "_bnb4",
}
SUFFIX_DTYPE = sorted(((s, d) for d, s in DTYPE_SUFFIX.items() if s), key=lambda x: -len(x[0]))
# When the dtype transformers.js would pick was never exported it would just
# fail to load; here, prefer the quantized variants nearest q8 over an fp32
# decoder that can run to a dozen GB of weights (Voxtral, Qwen2.5-VL).
DTYPE_FALLBACK = ["q8", "q4", "q4f16", "fp16", "fp32", "int8", "uint8", "bnb4", "q2", "q2f16", "q1", "q1f16"]

EXTERNAL_DATA_SUFFIXES = (".onnx_data", ".onnx.data", ".data", ".pb")
EXTERNAL_DATA_CHUNK_RE = re.compile(r"_\d+$")


# ---- Hub HTTP ----


class HubError(RuntimeError):
    def __init__(self, url: str, code: int | None, detail: str):
        hint = " (gated repo: set HF_TOKEN)" if code in (401, 403) else ""
        super().__init__(f"HTTP {code}{hint} for {url}" if code else f"{url}: {detail}")
        self.code = code


def hub_headers(token: str | None) -> dict[str, str]:
    headers = {"User-Agent": USER_AGENT}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def hub_open(url: str, token: str | None, method: str = "GET"):
    """urlopen with retries on transient failures; 401/403/404 raise at once."""
    last: Exception | None = None
    for attempt in range(5):
        req = urllib.request.Request(url, method=method, headers=hub_headers(token))
        try:
            return urllib.request.urlopen(req, timeout=120)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404):
                raise HubError(url, e.code, "") from None
            last = e
        except (urllib.error.URLError, OSError) as e:
            last = e
        time.sleep(1.5 * (attempt + 1))
    raise HubError(url, None, str(last))


def hub_json(url: str, token: str | None) -> Any:
    with hub_open(url, token) as resp:
        return json.loads(resp.read().decode("utf-8"))


def list_onnx_folder(repo: str, revision: str, token: str | None) -> list[tuple[str, int]]:
    """(path, size) for everything under onnx/; [] when the folder does not exist."""
    try:
        entries = hub_json(f"{HUB}/api/models/{repo}/tree/{revision}/onnx", token)
    except HubError as e:
        if e.code == 404:
            return []
        raise
    files = []
    for entry in entries:
        path = entry.get("path")
        if not isinstance(path, str):
            continue
        size = (entry.get("lfs") or {}).get("size") or entry.get("size") or 0
        files.append((path, int(size)))
    return sorted(files)


# ---- ranking: Hub models tagged transformers.js, task inferred like the Hub UI ----

LINK_NEXT_RE = re.compile(r'<([^>]+)>;\s*rel="next"', re.IGNORECASE)
TASK_TAG_PRIORITY = (
    "automatic-speech-recognition",
    "zero-shot-audio-classification",
    "audio-classification",
    "audio-xvector",
    "text-to-speech",
    "text-to-audio",
    "text-generation",
    "text2text-generation",
    "summarization",
    "translation",
    "zero-shot-classification",
    "text-classification",
    "token-classification",
    "question-answering",
    "fill-mask",
    "text-ranking",
    "sentence-similarity",
    "document-question-answering",
    "image-text-to-text",
    "image-to-text",
    "zero-shot-image-classification",
    "image-classification",
    "background-removal",
    "image-segmentation",
    "zero-shot-object-detection",
    "object-detection",
    "pose-estimation",
    "depth-estimation",
    "image-to-image",
    "image-feature-extraction",
    "feature-extraction",
)
TASK_TAG_ALIASES = {
    "audio_xvector": "audio-xvector",
    "feature_extraction": "feature-extraction",
    "image_feature_extraction": "image-feature-extraction",
    "text_to_speech": "text-to-speech",
}
SKIP_TASKS = {"(tokenizer-only)"}


def fetch_all_models(token: str | None) -> list[dict]:
    params = {"filter": "transformers.js", "limit": "1000", "config": "true", "sort": "likes", "direction": "-1"}
    url: str | None = f"{HUB}/api/models?{urllib.parse.urlencode(params)}"
    models: list[dict] = []
    while url:
        with hub_open(url, token) as resp:
            models.extend(json.loads(resp.read().decode("utf-8")))
            link = resp.headers.get("Link") or ""
        print(f"  fetched {len(models)} models...", file=sys.stderr)
        m = LINK_NEXT_RE.search(link)
        url = m.group(1) if m else None
    return models


def model_task(model: dict) -> str:
    tag = model.get("pipeline_tag")
    if isinstance(tag, str) and tag:
        return tag
    tags = {TASK_TAG_ALIASES.get(t, t) for t in model.get("tags") or [] if isinstance(t, str)}
    for task in TASK_TAG_PRIORITY:
        if task in tags:
            return task
    if "tokenizers" in tags:
        return "(tokenizer-only)"
    return "(unknown)"


def normalize(model: dict) -> dict:
    likes = model.get("likes")
    return {"id": model.get("id"), "task": model_task(model), "likes": int(likes) if isinstance(likes, (int, float)) else 0}


def load_ranking(models_json: Path, refresh: bool, token: str | None) -> list[dict]:
    if models_json.is_file() and not refresh:
        return json.loads(models_json.read_text(encoding="utf-8"))["models"]
    print(f"fetching the transformers.js model list from the Hub (cache: {models_json})", file=sys.stderr)
    models = [normalize(m) for m in fetch_all_models(token)]
    models_json.parent.mkdir(parents=True, exist_ok=True)
    models_json.write_text(
        json.dumps({"library": "transformers.js", "count": len(models), "models": models}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return models


def select_repos(models: list[dict], top: int, only: list[str]) -> list[dict]:
    """Top-N most-liked model per task, ties broken by id; `only` names repos
    to take at whatever rank they hold (or outside the ranking altogether)."""
    by_task: dict[str, list[dict]] = OrderedDict()
    for m in models:
        if m.get("id") and m["task"] not in SKIP_TASKS:
            by_task.setdefault(m["task"], []).append(m)
    picked: list[dict] = []
    for task, ms in by_task.items():
        ms.sort(key=lambda m: (-m["likes"], m["id"].lower()))
        for rank, m in enumerate(ms, 1):
            if rank <= top or m["id"] in only:
                picked.append({"repo": m["id"], "task": task, "likes": m["likes"], "rank": rank})
    if only:
        found = {p["repo"] for p in picked}
        picked = [p for p in picked if p["repo"] in only]
        picked += [{"repo": r, "task": "(not in ranking)", "likes": 0, "rank": 0} for r in only if r not in found]
    picked.sort(key=lambda p: (-p["likes"], p["task"], p["rank"]))
    return picked


# ---- UNSUPPORTED_OPS.md: repos and components known not to convert ----

SECTION_RE = re.compile(r"^## \[[^\]]+\]\(https://huggingface\.co/([^)]+)\)\s*$", re.MULTILINE)
ONNX_REF_RE = re.compile(r"`onnx/([A-Za-z0-9_]+?)\*?(?:\.onnx)?`")


def split_variant(name: str) -> tuple[str, str]:
    """`decoder_model_merged_q4f16.onnx` -> (`decoder_model_merged`, `q4f16`)."""
    stem = name[:-5] if name.lower().endswith(".onnx") else name
    for suffix, dtype in SUFFIX_DTYPE:
        if stem.endswith(suffix):
            return stem[: -len(suffix)], dtype
    return stem, "fp32"


def load_exclusions(path: Path) -> dict[str, set[str] | None]:
    """repo -> excluded component stems, or None for the whole repo (a section
    without any `onnx/...` reference, e.g. a gated repo)."""
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    matches = list(SECTION_RE.finditer(text))
    out: dict[str, set[str] | None] = {}
    for i, m in enumerate(matches):
        body = text[m.end() : matches[i + 1].start() if i + 1 < len(matches) else len(text)]
        stems = {split_variant(ref)[0] for ref in ONNX_REF_RE.findall(body)}
        repo = m.group(1)
        existing = out.get(repo, set())
        out[repo] = None if (not stems or existing is None) else existing | stems
    return out


# ---- manifest helpers ----


def safe_model_dir_name(repo_id: str) -> str:
    parts = [re.sub(r'[<>:"/\\|?*]', "_", p).rstrip(". ") for p in repo_id.split("/")]
    if len(parts) != 2 or any(not p or p in {".", ".."} for p in parts):
        raise ValueError(f"invalid repo id: {repo_id!r}")
    return "--".join(parts)


def file_to_repo(file: str) -> tuple[str, str]:
    org_repo, rel = file.split("/", 1)
    return org_repo.replace("--", "/", 1), rel


def companion_total_size(onnx_path: str, siblings: list[tuple[str, int]]) -> int:
    """Own size plus `<stem>.onnx_data[_N]` companions (>2 GB exports must
    split weights out); matched by exact stem so `model_fp16.onnx_data` is
    not attributed to `model.onnx`."""
    pp = PurePosixPath(onnx_path)
    parent, stem = str(pp.parent), pp.stem.lower()
    total = 0
    for path, size in siblings:
        p = PurePosixPath(path)
        if str(p.parent) != parent:
            continue
        if path == onnx_path:
            total += size
            continue
        base = EXTERNAL_DATA_CHUNK_RE.sub("", p.name.lower())
        for suffix in EXTERNAL_DATA_SUFFIXES:
            if base.endswith(suffix) and base[: -len(suffix)] == stem:
                total += size
                break
    return total


def find_reused_onnx(repo_id: str, rel_path: str, reuse_dirs: list[Path]) -> Path | None:
    """A local `<org>--<repo>/onnx/*.onnx` tree (--reuse-dir) is scanned instead of the Hub."""
    dirname = safe_model_dir_name(repo_id)
    for reuse_dir in reuse_dirs:
        candidate = reuse_dir / dirname / rel_path
        if candidate.is_file():
            return candidate
    return None


# ---- graph inputs without downloading: protobuf wire walk over ranges ----

MIN_WINDOW = 32 << 10
MAX_WINDOW = 8 << 20
# Protobuf wire types and the ONNX field numbers the walk cares about.
WIRE_VARINT, WIRE_FIXED64, WIRE_LEN, WIRE_FIXED32 = 0, 1, 2, 5
MODEL_GRAPH, GRAPH_INITIALIZER, GRAPH_INPUT, TENSOR_NAME = 7, 5, 11, 8


class LocalSource:
    def __init__(self, path: Path):
        self.f = open(path, "rb")
        self.size = path.stat().st_size

    def read_at(self, start: int, n: int) -> bytes:
        self.f.seek(start)
        data = self.f.read(n)
        if len(data) != n:
            raise ValueError(f"short read at {start}")
        return data


class HubSource:
    """A file on the Hub read with range requests over one keep-alive
    connection to the (signed CDN) location a HEAD resolves once."""

    def __init__(self, repo: str, rel_path: str, revision: str, token: str | None, size_hint: int):
        url = f"{HUB}/{repo}/resolve/{revision}/{rel_path}"
        with hub_open(url, token, method="HEAD") as resp:
            self.url = resp.geturl()
            self.size = int(resp.headers.get("Content-Length") or size_hint)
        parts = urllib.parse.urlsplit(self.url)
        self.host = parts.netloc
        self.path = parts.path + (f"?{parts.query}" if parts.query else "")
        self.headers = hub_headers(token if parts.netloc == "huggingface.co" else None)
        self.conn: http.client.HTTPSConnection | None = None

    def read_at(self, start: int, n: int) -> bytes:
        last: Exception | None = None
        for attempt in range(5):
            try:
                if self.conn is None:
                    self.conn = http.client.HTTPSConnection(self.host, timeout=120)
                self.conn.request("GET", self.path, headers={**self.headers, "Range": f"bytes={start}-{start + n - 1}"})
                resp = self.conn.getresponse()
                body = resp.read()
                if resp.status == 206 and len(body) == n:
                    return body
                if resp.status == 200 and len(body) >= start + n:  # range ignored, whole file sent
                    return body[start : start + n]
                raise RuntimeError(f"range {start}-{start + n - 1}: HTTP {resp.status}, {len(body)} bytes")
            except Exception as e:
                last = e
                if self.conn is not None:
                    self.conn.close()
                self.conn = None
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"{self.url}: {last}")


class RangeReader:
    """Sequential reader with cheap skips: fetches a window that doubles while
    reads stay contiguous and resets after a skip lands outside it, so a
    skipped weight tensor costs one small request for the header after it."""

    def __init__(self, src: LocalSource | HubSource):
        self.src = src
        self.size = src.size
        self.pos = 0
        self.buf = b""
        self.buf_start = 0
        self.window = MIN_WINDOW
        self.requests = 0
        self.fetched = 0

    def at_end(self) -> bool:
        return self.pos >= self.size

    def _fill(self, n: int) -> None:
        buf_end = self.buf_start + len(self.buf)
        if self.buf_start <= self.pos and self.pos + n <= buf_end:
            return
        if self.buf and self.pos == buf_end:
            self.window = min(self.window * 2, MAX_WINDOW)
        available = self.size - self.pos
        length = min(max(n, self.window), available)
        if length < n:
            raise ValueError(f"truncated model: need {n} bytes at offset {self.pos}, {available} available")
        self.buf = self.src.read_at(self.pos, length)
        self.buf_start = self.pos
        self.requests += 1
        self.fetched += length

    def read(self, n: int) -> bytes:
        if n == 0:
            return b""
        self._fill(n)
        off = self.pos - self.buf_start
        self.pos += n
        return self.buf[off : off + n]

    def skip(self, n: int) -> None:
        self.pos += n
        if self.pos > self.buf_start + len(self.buf):
            self.window = MIN_WINDOW

    def read_varint(self) -> int:
        value = 0
        for shift in range(0, 64, 7):
            byte = self.read(1)[0]
            value |= (byte & 0x7F) << shift
            if not byte & 0x80:
                return value
        raise ValueError("varint too long")

    def read_tag(self) -> tuple[int, int]:
        tag = self.read_varint()
        return tag >> 3, tag & 7

    def skip_value(self, wire: int) -> None:
        if wire == WIRE_VARINT:
            self.read_varint()
        elif wire == WIRE_FIXED64:
            self.skip(8)
        elif wire == WIRE_FIXED32:
            self.skip(4)
        elif wire == WIRE_LEN:
            self.skip(self.read_varint())
        else:
            raise ValueError(f"unsupported wire type {wire}")


def scan_tensor_name(r: RangeReader, length: int) -> str | None:
    """TensorProto.name, skipping the data fields (name precedes raw_data)."""
    end = r.pos + length
    while r.pos < end:
        field, wire = r.read_tag()
        if wire == WIRE_LEN:
            n = r.read_varint()
            if field == TENSOR_NAME:
                name = r.read(n).decode("utf-8", "replace")
                r.skip(end - r.pos)
                return name
            r.skip(n)
        else:
            r.skip_value(wire)
    return None


class GraphInput:
    __slots__ = ("name", "free_dims", "is_bool")

    def __init__(self, name: str, free_dims: list[str], is_bool: bool):
        self.name = name
        self.free_dims = free_dims
        self.is_bool = is_bool


def scan_graph_inputs(src: LocalSource | HubSource) -> tuple[list[GraphInput], RangeReader]:
    """Walk a serialized ModelProto reading only GraphProto.input (decoded with
    onnx's ValueInfoProto) and initializer names; nodes, outputs, value_info
    and all tensor data are skipped by their length prefixes."""
    r = RangeReader(src)
    value_infos: list[onnx.ValueInfoProto] = []
    initializer_names: set[str] = set()
    while not r.at_end():
        field, wire = r.read_tag()
        if field == MODEL_GRAPH and wire == WIRE_LEN:
            n = r.read_varint()
            end = r.pos + n
            while r.pos < end:
                f, w = r.read_tag()
                if w != WIRE_LEN:
                    r.skip_value(w)
                    continue
                n = r.read_varint()
                if f == GRAPH_INPUT:
                    vi = onnx.ValueInfoProto()
                    vi.ParseFromString(r.read(n))
                    value_infos.append(vi)
                elif f == GRAPH_INITIALIZER:
                    name = scan_tensor_name(r, n)
                    if name:
                        initializer_names.add(name)
                else:
                    r.skip(n)
        else:
            r.skip_value(wire)
    return parse_graph_inputs(value_infos, initializer_names), r


def sanitize_identifier(name: str) -> str:
    base = re.sub(r"[^A-Za-z0-9_]", "_", name)
    return f"_{base}" if base[:1].isdigit() else base


def parse_graph_inputs(value_infos: list[onnx.ValueInfoProto], initializer_names: set[str]) -> list[GraphInput]:
    inputs = []
    for vi in value_infos:
        if vi.name in initializer_names:
            continue
        free_dims: list[str] = []
        is_bool = False
        if vi.type.WhichOneof("value") == "tensor_type":
            tt = vi.type.tensor_type
            is_bool = tt.elem_type == onnx.TensorProto.BOOL
            if tt.HasField("shape"):
                for idx, dim in enumerate(tt.shape.dim):
                    which = dim.WhichOneof("value")
                    if which == "dim_param":
                        free_dims.append(dim.dim_param)
                    elif not (which == "dim_value" and dim.dim_value > 0):
                        free_dims.append(f"{sanitize_identifier(vi.name)}_dim{idx}")
        inputs.append(GraphInput(vi.name, free_dims, is_bool))
    return inputs


# ---- transformers.js file selection ----

# A repo's onnx/ inventory determines the model class transformers.js would
# build for it, and therefore the session files (models/session_config.js).
# Only the split legacy decoder (`decoder_model` + `decoder_with_past_model`)
# is ever present alongside what the class loads, so nothing here needs the
# architecture table: the exporter emits exactly the components the class
# constructs.
def transformers_js_components(components: set[str], config: dict) -> list[str]:
    c = components
    if {"text_encoder", "decoder_model_merged", "encodec_decode"} <= c:  # Musicgen
        return ["text_encoder", "decoder_model_merged", "encodec_decode"]
    if "prompt_encoder_mask_decoder" in c:  # MaskGeneration
        return ["vision_encoder", "prompt_encoder_mask_decoder"]
    if {"prepare_inputs_embeds", "language_model"} <= c:  # MultiModality
        return ["prepare_inputs_embeds", "language_model", "lm_head", "gen_head", "gen_img_embeds", "image_decode"]
    if {"prepare_inputs_embeds", "vision_encoder"} <= c:  # Phi3V
        return ["prepare_inputs_embeds", "model", "vision_encoder"]
    if {"embed_tokens", "decoder_model_merged"} <= c:  # (Image|Audio|ImageAudio)TextToText, VoxtralRealtime
        s = ["embed_tokens", "decoder_model_merged"] + [x for x in ("vision_encoder", "audio_encoder") if x in c]
        if config.get("is_encoder_decoder") and "encoder_model" in c:
            s.append("encoder_model")
        return s
    if {"text_encoder", "latent_denoiser"} <= c:  # Supertonic
        return ["text_encoder", "latent_denoiser", "voice_decoder"]
    if {"speech_encoder", "conditional_decoder"} <= c:  # Chatterbox
        return ["embed_tokens", "speech_encoder", "language_model", "conditional_decoder"]
    if {"encoder_model", "decoder_model_merged"} <= c:  # Seq2Seq, Vision2Seq, EncoderDecoder
        return ["encoder_model", "decoder_model_merged"]
    if {"encoder_model", "decoder_model"} <= c:  # AutoEncoder (codecs)
        return ["encoder_model", "decoder_model"]
    split = [x for x in ("text_model", "vision_model", "audio_model") if x in c]
    if split:  # CLIP/SigLIP/CLAP exported as separate towers
        return split
    if "model" in c:  # EncoderOnly, DecoderOnly
        return ["model"]
    return sorted(c)


def select_dtype(dtype_cfg: Any, base: str, device: str) -> str:
    """transformers.js selectDtype() as called by the runtime (session.js):
    per-file objects are keyed by the base file name."""
    resolved = dtype_cfg.get(base) if isinstance(dtype_cfg, dict) else dtype_cfg
    if isinstance(resolved, str) and resolved in DTYPE_SUFFIX:
        return resolved
    return "q8" if device == "wasm" else "fp32"


def select_device(device_cfg: Any, base: str, default: str) -> str:
    if isinstance(device_cfg, str):
        return device_cfg
    if isinstance(device_cfg, dict) and isinstance(device_cfg.get(base), str):
        return device_cfg[base]
    return default


def choose_files(
    onnx_files: list[str], config: dict, device: str, excluded: set[str] | None
) -> tuple[list[tuple[str, str, str]], list[str]]:
    """(rel_path, component, dtype) per session file transformers.js would
    load, plus notes for fallbacks and exclusions."""
    inventory: dict[str, dict[str, str]] = {}
    for path in onnx_files:
        p = PurePosixPath(path)
        if str(p.parent) != "onnx":
            continue
        stem, dtype = split_variant(p.name)
        inventory.setdefault(stem, {})[dtype] = path
    notes: list[str] = []
    chosen: list[tuple[str, str, str]] = []
    custom = config.get("transformers.js_config") or {}
    for component in transformers_js_components(set(inventory), config):
        variants = inventory.get(component)
        if not variants:
            notes.append(f"{component}: transformers.js would load it but the repo has no such export")
            continue
        if excluded is not None and component in excluded:
            notes.append(f"{component}: excluded ({DEFAULT_UNSUPPORTED_FILE.name})")
            continue
        dev = select_device(custom.get("device"), component, device)
        effective = dict(custom)
        effective.update((custom.get("device_config") or {}).get(dev) or {})
        dtype = select_dtype(effective.get("dtype"), component, dev)
        if dtype not in variants:
            fallback = next((d for d in DTYPE_FALLBACK if d in variants), sorted(variants)[0])
            notes.append(f"{component}: transformers.js default {dtype} not exported, using {fallback}")
            dtype = fallback
        chosen.append((variants[dtype], component, dtype))
    return chosen, notes


# ---- dimension policy (mirrors src/onnx/probe.rs) ----


def default_dim(name: str) -> int | None:
    n = name.lower()
    if any(c.isspace() or c in "/*+-" for c in n):
        return None
    if "batch" in n:
        return 1
    if "past" in n:
        return 0
    if "decoder_sequence" in n:
        return DECODER_SEQ_LEN
    if "sequence_length" in n or "total_sequence" in n:
        return SEQ_LEN
    if "num_channels" in n:
        return 3
    if "height" in n or "width" in n:
        return 224
    if "num_frames" in n:
        return 8
    if "feature_size" in n:
        return 80
    if "num_samples" in n:
        return 16000
    if "num_choices" in n:
        return 2
    return None


def free_dim_names(inputs: list[GraphInput]) -> list[str]:
    names: list[str] = []
    for inp in inputs:
        for d in inp.free_dims:
            if d not in names:
                names.append(d)
    return names


def find_gate(inputs: list[GraphInput]) -> str | None:
    return next((i.name for i in inputs if i.is_bool and i.name == "use_cache_branch"), None)


def resolve_dims(names: list[str], decode: bool, overrides: dict[str, int]) -> tuple[dict[str, int], list[str]]:
    dims: dict[str, int] = OrderedDict()
    unresolved: list[str] = []
    for name in names:
        v = overrides.get(name)
        if v is None and decode:
            lower = name.lower()
            if "past" in lower:
                v = DECODE_PAST_LEN
            elif "sequence_length" in lower and "encoder" not in lower and "total" not in lower:
                v = 1
        if v is None:
            v = default_dim(name)
        if v is None:
            unresolved.append(name)
        else:
            dims[name] = v
    seq = dims.get("sequence_length", dims.get("decoder_sequence_length"))
    past = next((v for k, v in dims.items() if "past" in k), None)
    if seq is not None and past is not None:
        for k in list(dims):
            if "total_sequence" in k and k not in overrides:
                dims[k] = seq + past
    enc = dims.get("encoder_sequence_length")
    if enc is not None and "encoder_sequence_length_out" in dims and "encoder_sequence_length_out" not in overrides:
        dims["encoder_sequence_length_out"] = enc
    return dims, unresolved


def entry_to_dict(file: str, heavy: bool, pins: dict[str, int], dims: dict[str, int]) -> dict:
    d: dict[str, Any] = {"file": file}
    if heavy:
        d["heavy"] = True
    if pins:
        d["pin_inputs"] = dict(pins)
    if dims:
        d["override_dims"] = dict(dims)
    return d


def entry_key(e: dict) -> tuple[str, tuple]:
    return (e["file"], tuple(sorted((e.get("pin_inputs") or {}).items())))


def reorder_entry(e: dict) -> dict:
    out = {k: e[k] for k in CANONICAL_KEYS if k in e}
    out.update({k: v for k, v in e.items() if k not in out})
    return out


def dump_manifest(entries: list[dict]) -> str:
    return "[\n  " + ",\n  ".join(json.dumps(reorder_entry(e), ensure_ascii=False) for e in entries) + "\n]\n"


# ---- per-repo work ----


class RepoResult:
    def __init__(self, pick: dict):
        self.pick = pick
        self.repo = pick["repo"]
        self.entries: list[dict] = []
        self.files: list[tuple[str, str, str]] = []
        self.notes: list[str] = []
        self.error: str | None = None
        self.skipped_reason: str | None = None
        self.fetched = 0
        self.requests = 0


class Baselines:
    """Dims already in the manifest: exact file+pins first, then the same
    component of the same repo at another dtype (identical graph inputs).

    An ungated decoder can appear twice with the same key (a hand-added
    prefill/decode pair, e.g. FastVLM); a nonzero past length tells them apart.
    """

    def __init__(self, existing: list[dict]):
        self.exact: dict[tuple, list[dict]] = {}
        self.by_component: dict[tuple[str, str, tuple], list[dict]] = {}
        self.repo_dims: dict[str, dict[str, int]] = {}
        self.manual_fields: dict[tuple, dict] = {}
        for e in existing:
            key = entry_key(e)
            self.exact.setdefault(key, []).append(e)
            repo, rel = file_to_repo(e["file"])
            stem, _ = split_variant(PurePosixPath(rel).name)
            self.by_component.setdefault((repo, stem, key[1]), []).append(e)
            for name, value in (e.get("override_dims") or {}).items():
                if self._branch_invariant(name):
                    self.repo_dims.setdefault(repo, {}).setdefault(name, value)
            extra = {k: v for k, v in e.items() if k not in ("file", "heavy", "pin_inputs", "override_dims")}
            if extra:
                self.manual_fields[key] = extra

    @staticmethod
    def _branch_invariant(name: str) -> bool:
        """Dims that mean the same thing in every file of a repo and in both
        cache branches -- image sizes, dynamo's s0.., the encoder output length
        a decoder attends over (distilvit 197, donut 4800) -- unlike the
        prefill/decode-dependent sequence, past and total lengths."""
        n = name.lower()
        if "past" in n or "total_sequence" in n:
            return False
        return "sequence_length" not in n or "encoder" in n

    @staticmethod
    def _is_decode(e: dict) -> bool:
        return any("past" in k.lower() and v > 0 for k, v in (e.get("override_dims") or {}).items())

    def _pick(self, candidates: list[dict], decode: bool) -> dict | None:
        return next((e for e in candidates if self._is_decode(e) == decode), candidates[0] if candidates else None)

    def dims_for(self, file: str, pins: dict[str, int], decode: bool) -> dict[str, int]:
        key = (file, tuple(sorted(pins.items())))
        repo, rel = file_to_repo(file)
        stem, _ = split_variant(PurePosixPath(rel).name)
        e = self._pick(self.exact.get(key, []), decode)
        if e is None:
            e = self._pick(self.by_component.get((repo, stem, key[1]), []), decode)
        dims = dict(self.repo_dims.get(repo, {}))
        dims.update((e or {}).get("override_dims") or {})
        return dims


def process_repo(
    pick: dict,
    args: argparse.Namespace,
    token: str | None,
    exclusions: dict[str, set[str] | None],
    baselines: Baselines,
    overrides_by_file: dict[str, dict[str, int]],
) -> RepoResult:
    r = RepoResult(pick)
    repo = r.repo
    excluded = exclusions.get(repo, set())
    if excluded is None:
        r.skipped_reason = f"whole repo excluded by {DEFAULT_UNSUPPORTED_FILE.name}"
        return r
    try:
        siblings = list_onnx_folder(repo, args.revision, token)
        config = hub_json(f"{HUB}/{repo}/resolve/{args.revision}/config.json", token)
    except Exception as e:
        r.error = str(e).splitlines()[0]
        return r
    onnx_files = [p for p, _ in siblings if p.lower().endswith(".onnx")]
    if not onnx_files:
        r.skipped_reason = "no onnx/*.onnx files"
        return r
    r.files, notes = choose_files(onnx_files, config, args.device, excluded)
    r.notes.extend(notes)
    if not r.files:
        r.skipped_reason = "every component excluded or missing"
        return r

    dirname = safe_model_dir_name(repo)
    for rel_path, component, dtype in r.files:
        file_key = f"{dirname}/{rel_path}"
        own_size = next((sz for p, sz in siblings if p == rel_path), 0)
        try:
            local = find_reused_onnx(repo, rel_path, args.reuse_dir)
            source = LocalSource(local) if local else HubSource(repo, rel_path, args.revision, token, own_size)
            inputs, reader = scan_graph_inputs(source)
        except Exception as e:
            r.notes.append(f"skipped {rel_path}: {str(e).splitlines()[0]}")
            continue
        if not local:
            r.fetched += reader.fetched
            r.requests += reader.requests
        names = free_dim_names(inputs)
        gate = find_gate(inputs)
        heavy = companion_total_size(rel_path, siblings) >= HEAVY_BYTES
        # A use_cache_branch gate means two real execution paths, so both are
        # always emitted. Whether an ungated cache-carrying decoder also gets
        # a decode-step entry is a coverage/cost call (the current manifest
        # makes it for FastVLM and SmolLM2 q4 but not q4f16), hence opt-in.
        manual = overrides_by_file.get(file_key, {})
        if {"dims", "pins", "decode_step"} & set(manual):
            manual_dims, manual_pins = manual.get("dims", {}), manual.get("pins", {})
            decode_step = bool(manual.get("decode_step"))
        else:  # flat form: just dims
            manual_dims, manual_pins, decode_step = manual, {}, False
        decode_step = decode_step or file_key in args.decode_step or component in args.decode_step
        if gate:
            branches = [(False, {gate: 0}), (True, {gate: 1})]
        elif decode_step and any("past" in n.lower() for n in names):
            branches = [(False, {}), (True, {})]
        else:
            branches = [(False, {})]
        entries, unresolved_all = [], []
        for decode, gate_pins in branches:
            pins = {**manual_pins, **gate_pins}
            overrides = {**baselines.dims_for(file_key, pins, decode), **manual_dims}
            dims, unresolved = resolve_dims(names, decode, overrides)
            unresolved_all += [u for u in unresolved if u not in unresolved_all]
            entry = entry_to_dict(file_key, heavy, pins, dims)
            entry.update(baselines.manual_fields.get(entry_key(entry), {}))
            entries.append(entry)
        if unresolved_all:
            r.notes.append(f"skipped {rel_path}: unresolved dims {unresolved_all} -- add to --overrides")
            continue
        r.entries.extend(entries)
    return r


# ---- report ----


def report(results: list[RepoResult], existing: list[dict], generated: list[dict], partial: bool) -> None:
    existing_repos: dict[str, list[str]] = OrderedDict()
    for e in existing:
        repo, rel = file_to_repo(e["file"])
        existing_repos.setdefault(repo, [])
        if rel not in existing_repos[repo]:
            existing_repos[repo].append(rel)
    generated_repos = {r.repo for r in results if r.entries}
    existing_keys = {entry_key(e) for e in existing}
    generated_keys = {entry_key(e) for e in generated}

    print("\n== repos ==", file=sys.stderr)
    for r in results:
        p = r.pick
        tag = "SELECTED" if r.entries else ("ERROR" if r.error else "SKIPPED")
        status = r.error or r.skipped_reason or ""
        in_manifest = "" if r.repo in existing_repos else "  (not in current manifest)"
        print(f"  [{tag:8}] {r.repo:<55} {p['task']:<30} #{p['rank']} {p['likes']:>5} likes{in_manifest}", file=sys.stderr)
        if status:
            print(f"             {status}", file=sys.stderr)
        for note in r.notes:
            print(f"             {note}", file=sys.stderr)
    dropped = [repo for repo in existing_repos if repo not in generated_repos]
    if dropped and not partial:
        print("\n  in the current manifest but not selected now:", file=sys.stderr)
        for repo in dropped:
            print(f"    {repo}", file=sys.stderr)

    print("\n== files (per selected repo) ==", file=sys.stderr)
    for r in results:
        if not r.entries:
            continue
        old = existing_repos.get(r.repo, [])
        new = [rel for rel, _, _ in r.files]
        same, added, removed = sorted(set(old) & set(new)), sorted(set(new) - set(old)), sorted(set(old) - set(new))
        if not added and not removed:
            print(f"  {r.repo}: unchanged ({len(same)} files)", file=sys.stderr)
            continue
        print(f"  {r.repo}:", file=sys.stderr)
        for rel in same:
            print(f"      = {rel}", file=sys.stderr)
        for rel in added:
            print(f"      + {rel}", file=sys.stderr)
        for rel in removed:
            print(f"      - {rel}", file=sys.stderr)

    existing_by_key: dict[tuple, list[dict]] = {}
    for e in existing:
        existing_by_key.setdefault(entry_key(e), []).append(e)  # ungated prefill/decode pairs share a key
    unchanged = sum(1 for e in generated if e in existing_by_key.get(entry_key(e), []))
    print("\n== summary ==", file=sys.stderr)
    print(f"  {sum(1 for r in results if r.entries)} repos selected, {sum(1 for r in results if r.skipped_reason)} skipped, {sum(1 for r in results if r.error)} errored", file=sys.stderr)
    print(f"  {len(generated)} entries generated vs {len(existing)} in the current manifest", file=sys.stderr)
    line = f"    {unchanged} identical, {len(generated_keys - existing_keys)} new"
    if not partial:
        line += f", {len(existing_keys - generated_keys)} no longer generated"
    print(line + ("  (--only: comparison limited to the selected repos)" if partial else ""), file=sys.stderr)
    fetched = sum(r.fetched for r in results)
    requests = sum(r.requests for r in results)
    print(f"  read {fetched / 1e6:.1f} MB from the Hub in {requests} range requests", file=sys.stderr)


# ---- CLI ----


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Current manifest (dims baseline + comparison)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Where to write the generated manifest")
    p.add_argument("--apply", action="store_true", help="Write to --manifest instead of --out")
    p.add_argument("--top", type=int, default=1, metavar="N", help="Most-liked models per task (default 1)")
    p.add_argument("--device", default="wasm", help="transformers.js device whose default dtype applies: wasm=q8, anything else (webgpu, webnn, cpu)=fp32")
    p.add_argument("--only", action="append", default=[], metavar="REPO", help="Process just these repo IDs, whatever their rank (repeatable)")
    p.add_argument("--models-json", type=Path, default=DEFAULT_MODELS_JSON, help="Ranking snapshot; fetched from the Hub into this path when missing")
    p.add_argument("--refresh-ranking", action="store_true", help="Re-fetch the ranking even if --models-json exists")
    p.add_argument("--unsupported-file", type=Path, default=DEFAULT_UNSUPPORTED_FILE, help="Repos/components known not to convert, one `## [org/repo](url)` section each")
    p.add_argument(
        "--overrides",
        type=Path,
        default=DEFAULT_OVERRIDES,
        help='JSON keyed by <org>--<repo>/onnx/<file>.onnx: either {"dim": value, ...} or '
        '{"dims": {...}, "pins": {"<input>": value}, "decode_step": true} for dims the policy cannot fill, '
        f"inputs that must be pinned, and decode-step opt-ins (default: {DEFAULT_OVERRIDES.relative_to(REPO_ROOT)} when present)",
    )
    p.add_argument(
        "--decode-step",
        action="append",
        default=[],
        metavar="FILE|COMPONENT",
        help="Also emit a decode-step entry (one new token, 16-token cache) for this ungated cache-carrying "
        "decoder, given as org--repo/onnx/<file>.onnx or a component name like decoder_model_merged (repeatable)",
    )
    p.add_argument("--reuse-dir", action="append", type=Path, default=[], help="Scan org--repo/onnx/*.onnx from this local tree instead of the Hub (repeatable)")
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--token", default=None, help="HF token (default: HF_TOKEN / HUGGING_FACE_HUB_TOKEN)")
    p.add_argument("--revision", default="main")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    existing = json.loads(args.manifest.read_text(encoding="utf-8")) if args.manifest.is_file() else []
    overrides_by_file = json.loads(args.overrides.read_text(encoding="utf-8")) if args.overrides and args.overrides.is_file() else {}
    exclusions = load_exclusions(args.unsupported_file)
    baselines = Baselines(existing)

    picks = select_repos(load_ranking(args.models_json, args.refresh_ranking, token), args.top, args.only)
    print(f"{len(picks)} candidate repos (top {args.top} per task), transformers.js defaults for device={args.device}", file=sys.stderr)

    results: dict[str, RepoResult] = {}
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(process_repo, pick, args, token, exclusions, baselines, overrides_by_file): pick["repo"] for pick in picks
        }
        for fut in as_completed(futures):
            r = fut.result()
            results[r.repo] = r
            state = "FAIL" if r.error else ("skip" if r.skipped_reason else "ok")
            detail = f" -- {r.error or r.skipped_reason}" if state != "ok" else f" ({r.fetched / 1e6:.1f} MB / {r.requests} requests)"
            print(f"  [{state:4}] {r.repo}: {len(r.entries)} entries{detail}", file=sys.stderr)
    ordered = [results[p["repo"]] for p in picks]
    generated = [e for r in ordered for e in r.entries]

    out_path = args.manifest if args.apply else args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(dump_manifest(generated), encoding="utf-8", newline="\n")
    report(ordered, existing, generated, partial=bool(args.only))
    print(f"\nwrote {len(generated)} entries to {out_path}", file=sys.stderr)
    if not args.apply:
        print(f"review with: diff {args.manifest} {out_path}", file=sys.stderr)
    return 1 if any(r.error for r in ordered) else 0


if __name__ == "__main__":
    raise SystemExit(main())
