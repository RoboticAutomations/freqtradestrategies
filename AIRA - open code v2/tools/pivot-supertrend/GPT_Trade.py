#!/usr/bin/env python3
"""
DarkReaper
GPT Trade

Key properties:
- stdlib only
- explicit forward/backward kernels (no generic scalar autograd)
- flat float32 parameter buffers using array('f')
- fused QKV projection
- causal self-attention with optional ALiBi
- optional QK-Norm
- tied input/output embeddings by default
- ReLU^2 MLP by default
- AdamW optimizer
- train + sample + save/load in one file

"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import sys
import tempfile
import time
import urllib.request
from array import array
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# -----------------------------
# Utilities
# -----------------------------


def f32(value: float) -> float:
    return float(value)


def prod(shape: Sequence[int]) -> int:
    out = 1
    for dim in shape:
        out *= dim
    return out


def atomic_write_bytes(path: str, payload: bytes) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", dir=directory)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


@dataclass
class TensorSpec:
    name: str
    shape: Tuple[int, ...]
    offset: int
    size: int


@dataclass
class Config:
    seed: int = 1337
    tokenizer: str = "byte"   # byte | char
    context: int = 128
    n_layer: int = 4
    d_model: int = 128
    n_head: int = 4
    d_ff: int = 512
    position_mode: str = "alibi"  # alibi | learned
    qk_norm: bool = True
    relu2: bool = True
    tie_embeddings: bool = True
    final_norm: bool = True
    residual_gate: bool = True
    init_std: float = 0.02
    lr: float = 2e-3
    min_lr: float = 2e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    adam_eps: float = 1e-8
    grad_clip: float = 1.0
    batch_docs: int = 1
    micro_batch_tokens: int = 0
    steps: int = 200
    warmup_steps: int = 20
    save_every: int = 50
    eval_every: int = 25
    train_ratio: float = 0.9
    temperature: float = 0.9
    top_k: int = 0
    sample_tokens: int = 128

    def validate(self) -> None:
        if self.n_head <= 0:
            raise ValueError("n_head must be positive")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.d_model % self.n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        if self.context < 2:
            raise ValueError("context must be >= 2")
        if self.position_mode not in {"alibi", "learned"}:
            raise ValueError("position_mode must be 'alibi' or 'learned'")
        if self.tokenizer not in {"byte", "char"}:
            raise ValueError("tokenizer must be 'byte' or 'char'")
        if not (0.0 < self.train_ratio < 1.0):
            raise ValueError("train_ratio must be in (0, 1)")


# -----------------------------
# Tokenizers
# -----------------------------


class ByteTokenizer:
    bos_id = 256
    vocab_size = 257
    meta = {"type": "byte", "bos_id": 256, "vocab_size": 257}

    def encode(self, text: str) -> List[int]:
        return list(text.encode("utf-8", errors="replace"))

    def decode(self, tokens: Sequence[int]) -> str:
        data = bytes(t for t in tokens if 0 <= t <= 255)
        return data.decode("utf-8", errors="replace")

    def bos(self) -> int:
        return self.bos_id

    def to_meta(self) -> Dict[str, object]:
        return dict(self.meta)

    @staticmethod
    def from_meta(meta: Dict[str, object]) -> "ByteTokenizer":
        _ = meta
        return ByteTokenizer()


class CharTokenizer:
    def __init__(self, chars: Sequence[str]):
        self.chars = list(chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.bos_id = len(self.chars)
        self.vocab_size = len(self.chars) + 1

    def encode(self, text: str) -> List[int]:
        out = []
        for ch in text:
            if ch not in self.stoi:
                raise ValueError(f"character {ch!r} not present in tokenizer vocabulary")
            out.append(self.stoi[ch])
        return out

    def decode(self, tokens: Sequence[int]) -> str:
        pieces = []
        for token in tokens:
            if token == self.bos_id:
                continue
            pieces.append(self.itos[token])
        return "".join(pieces)

    def bos(self) -> int:
        return self.bos_id

    def to_meta(self) -> Dict[str, object]:
        return {
            "type": "char",
            "chars": self.chars,
            "bos_id": self.bos_id,
            "vocab_size": self.vocab_size,
        }

    @staticmethod
    def from_meta(meta: Dict[str, object]) -> "CharTokenizer":
        chars = meta.get("chars")
        if not isinstance(chars, list):
            raise ValueError("invalid char tokenizer metadata")
        return CharTokenizer(chars)


def build_tokenizer(tokenizer_mode: str, docs: Sequence[str]):
    if tokenizer_mode == "byte":
        return ByteTokenizer()
    chars = sorted(set("".join(docs)))
    if not chars:
        raise ValueError("cannot build char tokenizer from an empty dataset")
    return CharTokenizer(chars)


def tokenizer_from_meta(meta: Dict[str, object]):
    tokenizer_type = meta.get("type")
    if tokenizer_type == "byte":
        return ByteTokenizer.from_meta(meta)
    if tokenizer_type == "char":
        return CharTokenizer.from_meta(meta)
    raise ValueError(f"unsupported tokenizer metadata type: {tokenizer_type!r}")


# -----------------------------
# Dataset
# -----------------------------


DEFAULT_NAMES_URL = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"
DEFAULT_FALLBACK_DOCS = [
    "emma", "olivia", "ava", "isabella", "sophia",
    "mia", "charlotte", "amelia", "harper", "evelyn",
    "liam", "noah", "oliver", "elijah", "james",
    "william", "benjamin", "lucas", "henry", "theodore",
]


def ensure_input_file(path: str) -> str:
    if os.path.exists(path):
        return path
    if os.path.basename(path) == "input.txt":
        print("input.txt not found; downloading the original tiny names dataset...", file=sys.stderr)
        try:
            urllib.request.urlretrieve(DEFAULT_NAMES_URL, path)
        except Exception as exc:  # pragma: no cover - network availability is environment-specific
            print(f"download failed ({exc}); writing embedded fallback dataset instead", file=sys.stderr)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("\n".join(DEFAULT_FALLBACK_DOCS) + "\n")
        return path
    raise FileNotFoundError(path)


def load_docs(path: str) -> List[str]:
    ensure_input_file(path)
    with open(path, "r", encoding="utf-8") as handle:
        docs = [line.strip("\n\r") for line in handle]
    docs = [doc for doc in docs if doc]
    if not docs:
        raise ValueError("dataset is empty after stripping blank lines")
    return docs


class DocumentDataset:
    def __init__(self, docs: Sequence[str], tokenizer, context: int, seed: int = 1337):
        if context < 2:
            raise ValueError("context must be >= 2")
        self.docs = list(docs)
        self.tokenizer = tokenizer
        self.context = context
        self.rng = random.Random(seed)
        self.encoded_docs: List[List[int]] = []
        bos = tokenizer.bos()
        for doc in self.docs:
            toks = [bos] + tokenizer.encode(doc) + [bos]
            if len(toks) >= 2:
                self.encoded_docs.append(toks)
        if not self.encoded_docs:
            raise ValueError("no valid documents after tokenization")

    def sample(self, num_docs: int = 1) -> Tuple[List[int], List[int], int]:
        if num_docs <= 0:
            raise ValueError("num_docs must be positive")
        inputs: List[int] = []
        targets: List[int] = []
        tokens_used = 0
        for _ in range(num_docs):
            seq = self.rng.choice(self.encoded_docs)
            if len(seq) <= self.context + 1:
                chunk = seq
            else:
                start = self.rng.randint(0, len(seq) - (self.context + 1))
                chunk = seq[start:start + self.context + 1]
            chunk_inputs = chunk[:-1]
            chunk_targets = chunk[1:]
            if len(chunk_inputs) < 1:
                continue
            inputs.extend(chunk_inputs)
            targets.extend(chunk_targets)
            tokens_used += len(chunk_inputs)
        if not inputs or not targets:
            raise RuntimeError("failed to sample a non-empty training sequence")
        return inputs, targets, tokens_used

    def split(self, train_ratio: float) -> Tuple["DocumentDataset", "DocumentDataset"]:
        n = len(self.docs)
        pivot = max(1, min(n - 1, int(round(n * train_ratio))))
        train_docs = self.docs[:pivot]
        val_docs = self.docs[pivot:]
        return (
            DocumentDataset(train_docs, self.tokenizer, self.context, seed=self.rng.randint(0, 2**31 - 1)),
            DocumentDataset(val_docs, self.tokenizer, self.context, seed=self.rng.randint(0, 2**31 - 1)),
        )


# -----------------------------
# Parameter store
# -----------------------------


class ParameterStore:
    def __init__(self) -> None:
        self.specs: Dict[str, TensorSpec] = {}
        self.param = array("f")
        self.grad = array("f")
        self.m1 = array("f")
        self.m2 = array("f")

    def register(self, name: str, shape: Sequence[int], init: str, rng: random.Random, std: float, value: float = 0.0) -> TensorSpec:
        shape = tuple(int(x) for x in shape)
        size = prod(shape)
        offset = len(self.param)
        if name in self.specs:
            raise ValueError(f"duplicate parameter name: {name}")
        if init == "normal":
            data = [rng.gauss(0.0, std) for _ in range(size)]
        elif init == "zeros":
            data = [0.0 for _ in range(size)]
        elif init == "constant":
            data = [float(value) for _ in range(size)]
        else:
            raise ValueError(f"unknown init mode: {init}")
        self.param.extend(data)
        self.grad.extend([0.0] * size)
        self.m1.extend([0.0] * size)
        self.m2.extend([0.0] * size)
        spec = TensorSpec(name=name, shape=shape, offset=offset, size=size)
        self.specs[name] = spec
        return spec

    def spec(self, name: str) -> TensorSpec:
        try:
            return self.specs[name]
        except KeyError as exc:
            raise KeyError(f"parameter not found: {name}") from exc

    def zero_grad(self) -> None:
        for i in range(len(self.grad)):
            self.grad[i] = 0.0

    def state_dict(self) -> Dict[str, object]:
        return {
            "specs": {name: asdict(spec) for name, spec in self.specs.items()},
            "param_bytes": self.param.tobytes(),
            "m1_bytes": self.m1.tobytes(),
            "m2_bytes": self.m2.tobytes(),
        }

    @staticmethod
    def from_state(state: Dict[str, object]) -> "ParameterStore":
        store = ParameterStore()
        specs_raw = state["specs"]
        if not isinstance(specs_raw, dict):
            raise ValueError("invalid checkpoint: missing specs")
        store.specs = {name: TensorSpec(**spec) for name, spec in specs_raw.items()}
        store.param = array("f")
        store.param.frombytes(state["param_bytes"])
        store.grad = array("f", [0.0] * len(store.param))
        store.m1 = array("f")
        store.m1.frombytes(state["m1_bytes"])
        store.m2 = array("f")
        store.m2.frombytes(state["m2_bytes"])
        return store


# -----------------------------
# Model
# -----------------------------


class MicroGPT2:
    def __init__(self, config: Config, vocab_size: int):
        config.validate()
        self.cfg = config
        self.vocab_size = int(vocab_size)
        self.d_head = self.cfg.d_model // self.cfg.n_head
        self.scale = 1.0 / math.sqrt(self.d_head)
        self.rng = random.Random(self.cfg.seed)
        self.store = ParameterStore()
        self._build_parameters()
        self.alibi_slopes = self._make_alibi_slopes(self.cfg.n_head)

    def _build_parameters(self) -> None:
        c = self.cfg
        s = self.store
        s.register("wte", (self.vocab_size, c.d_model), "normal", self.rng, c.init_std)
        if not c.tie_embeddings:
            s.register("lm_head", (self.vocab_size, c.d_model), "normal", self.rng, c.init_std)
        if c.position_mode == "learned":
            s.register("wpe", (c.context, c.d_model), "normal", self.rng, c.init_std)
        for li in range(c.n_layer):
            s.register(f"layer{li}.w_qkv", (3 * c.d_model, c.d_model), "normal", self.rng, c.init_std)
            s.register(f"layer{li}.w_o", (c.d_model, c.d_model), "normal", self.rng, c.init_std)
            s.register(f"layer{li}.w1", (c.d_ff, c.d_model), "normal", self.rng, c.init_std)
            s.register(f"layer{li}.w2", (c.d_model, c.d_ff), "normal", self.rng, c.init_std)
            if c.qk_norm:
                s.register(f"layer{li}.attn_alpha", (c.n_head,), "constant", self.rng, c.init_std, value=1.0)
            if c.residual_gate:
                gate_init = 1.0 / math.sqrt(2.0 * max(1, c.n_layer))
                s.register(f"layer{li}.gamma_attn", (1,), "constant", self.rng, c.init_std, value=gate_init)
                s.register(f"layer{li}.gamma_mlp", (1,), "constant", self.rng, c.init_std, value=gate_init)

    @staticmethod
    def _make_alibi_slopes(n_head: int) -> List[float]:
        if n_head <= 0:
            raise ValueError("n_head must be positive")
        def get_slopes_power_of_2(power: int) -> List[float]:
            start = 2.0 ** (-(2.0 ** -(math.log2(power) - 3.0)))
            ratio = start
            return [start * (ratio ** i) for i in range(power)]
        if (n_head & (n_head - 1)) == 0:
            return get_slopes_power_of_2(n_head)
        lower = 2 ** int(math.floor(math.log2(n_head)))
        slopes = get_slopes_power_of_2(lower)
        extra = get_slopes_power_of_2(2 * lower)[0::2]
        slopes.extend(extra[: n_head - lower])
        return slopes

    # ---------- low-level kernels ----------

    def _linear_fwd(self, x: List[float], w_spec: TensorSpec, rows: int, in_dim: int, out_dim: int) -> List[float]:
        p = self.store.param
        y = [0.0] * (rows * out_dim)
        woff = w_spec.offset
        for r in range(rows):
            xoff = r * in_dim
            yoff = r * out_dim
            for o in range(out_dim):
                base = woff + o * in_dim
                acc = 0.0
                for i in range(in_dim):
                    acc += x[xoff + i] * p[base + i]
                y[yoff + o] = acc
        return y

    def _linear_bwd(self, dout: List[float], x: List[float], w_spec: TensorSpec, rows: int, in_dim: int, out_dim: int) -> List[float]:
        p = self.store.param
        g = self.store.grad
        dx = [0.0] * (rows * in_dim)
        woff = w_spec.offset
        for r in range(rows):
            xoff = r * in_dim
            yoff = r * out_dim
            for o in range(out_dim):
                go = dout[yoff + o]
                if go == 0.0:
                    continue
                base = woff + o * in_dim
                for i in range(in_dim):
                    dx[xoff + i] += go * p[base + i]
                    g[base + i] += go * x[xoff + i]
        return dx

    def _rmsnorm_fwd(self, x: List[float], rows: int, dim: int, eps: float = 1e-5) -> Tuple[List[float], List[float]]:
        y = [0.0] * (rows * dim)
        inv = [0.0] * rows
        for r in range(rows):
            off = r * dim
            ms = 0.0
            for i in range(dim):
                xi = x[off + i]
                ms += xi * xi
            ms /= dim
            rinv = 1.0 / math.sqrt(ms + eps)
            inv[r] = rinv
            for i in range(dim):
                y[off + i] = x[off + i] * rinv
        return y, inv

    def _rmsnorm_bwd(self, dout: List[float], x: List[float], inv: List[float], rows: int, dim: int) -> List[float]:
        dx = [0.0] * (rows * dim)
        for r in range(rows):
            off = r * dim
            rinv = inv[r]
            dot = 0.0
            for i in range(dim):
                dot += dout[off + i] * x[off + i]
            coeff = (rinv ** 3) * dot / dim
            for i in range(dim):
                dx[off + i] = dout[off + i] * rinv - x[off + i] * coeff
        return dx

    def _relu2_fwd(self, x: List[float]) -> Tuple[List[float], List[float]]:
        y = [0.0] * len(x)
        positive = [0.0] * len(x)
        for i, xv in enumerate(x):
            p = xv if xv > 0.0 else 0.0
            positive[i] = p
            y[i] = p * p
        return y, positive

    def _relu2_bwd(self, dout: List[float], positive: List[float]) -> List[float]:
        dx = [0.0] * len(dout)
        for i, go in enumerate(dout):
            p = positive[i]
            dx[i] = go * (2.0 * p)
        return dx

    def _embedding_fwd(self, tokens: Sequence[int]) -> List[float]:
        c = self.cfg
        if c.position_mode == "learned" and len(tokens) > c.context:
            raise ValueError("learned position mode cannot process a sequence longer than configured context")
        spec = self.store.spec("wte")
        p = self.store.param
        out = [0.0] * (len(tokens) * c.d_model)
        for t, token in enumerate(tokens):
            if token < 0 or token >= self.vocab_size:
                raise ValueError(f"token {token} out of vocabulary range 0..{self.vocab_size - 1}")
            src = spec.offset + token * c.d_model
            dst = t * c.d_model
            for i in range(c.d_model):
                out[dst + i] = p[src + i]
        if c.position_mode == "learned":
            wpe = self.store.spec("wpe")
            for t in range(len(tokens)):
                src = wpe.offset + t * c.d_model
                dst = t * c.d_model
                for i in range(c.d_model):
                    out[dst + i] += p[src + i]
        return out

    def _embedding_bwd(self, tokens: Sequence[int], d_embed: List[float]) -> None:
        c = self.cfg
        g = self.store.grad
        wte = self.store.spec("wte")
        for t, token in enumerate(tokens):
            dst = wte.offset + token * c.d_model
            src = t * c.d_model
            for i in range(c.d_model):
                g[dst + i] += d_embed[src + i]
        if c.position_mode == "learned":
            wpe = self.store.spec("wpe")
            for t in range(len(tokens)):
                dst = wpe.offset + t * c.d_model
                src = t * c.d_model
                for i in range(c.d_model):
                    g[dst + i] += d_embed[src + i]

    def _qk_norm_fwd(self, x: List[float], rows: int, n_head: int, d_head: int, eps: float = 1e-5) -> Tuple[List[float], List[float]]:
        y = [0.0] * len(x)
        inv = [0.0] * (rows * n_head)
        for r in range(rows):
            for h in range(n_head):
                base = (r * n_head + h) * d_head
                ms = 0.0
                for i in range(d_head):
                    xv = x[base + i]
                    ms += xv * xv
                ms /= d_head
                rinv = 1.0 / math.sqrt(ms + eps)
                inv[r * n_head + h] = rinv
                for i in range(d_head):
                    y[base + i] = x[base + i] * rinv
        return y, inv

    def _qk_norm_bwd(self, dout: List[float], x: List[float], inv: List[float], rows: int, n_head: int, d_head: int) -> List[float]:
        dx = [0.0] * len(dout)
        for r in range(rows):
            for h in range(n_head):
                idx = r * n_head + h
                base = idx * d_head
                rinv = inv[idx]
                dot = 0.0
                for i in range(d_head):
                    dot += dout[base + i] * x[base + i]
                coeff = (rinv ** 3) * dot / d_head
                for i in range(d_head):
                    dx[base + i] = dout[base + i] * rinv - x[base + i] * coeff
        return dx

    def _split_qkv(self, qkv: List[float], rows: int) -> Tuple[List[float], List[float], List[float]]:
        d = self.cfg.d_model
        q = [0.0] * (rows * d)
        k = [0.0] * (rows * d)
        v = [0.0] * (rows * d)
        for r in range(rows):
            base_in = r * (3 * d)
            base_out = r * d
            for i in range(d):
                q[base_out + i] = qkv[base_in + i]
                k[base_out + i] = qkv[base_in + d + i]
                v[base_out + i] = qkv[base_in + 2 * d + i]
        return q, k, v

    def _merge_qkv_grads(self, dq: List[float], dk: List[float], dv: List[float], rows: int) -> List[float]:
        d = self.cfg.d_model
        out = [0.0] * (rows * 3 * d)
        for r in range(rows):
            base_out = r * 3 * d
            base_in = r * d
            for i in range(d):
                out[base_out + i] = dq[base_in + i]
                out[base_out + d + i] = dk[base_in + i]
                out[base_out + 2 * d + i] = dv[base_in + i]
        return out

    def _attention_fwd(self, q: List[float], k: List[float], v: List[float], rows: int, layer_idx: int) -> Tuple[List[float], List[float]]:
        c = self.cfg
        out = [0.0] * (rows * c.d_model)
        probs = [0.0] * (c.n_head * rows * rows)
        alpha = None
        if c.qk_norm:
            alpha = self.store.spec(f"layer{layer_idx}.attn_alpha")
        p = self.store.param
        for h in range(c.n_head):
            slope = self.alibi_slopes[h] if c.position_mode == "alibi" else 0.0
            alpha_h = p[alpha.offset + h] if alpha is not None else self.scale
            for t in range(rows):
                logits: List[float] = []
                max_logit = -1e30
                qbase = (t * c.n_head + h) * self.d_head
                for tau in range(t + 1):
                    kbase = (tau * c.n_head + h) * self.d_head
                    dot = 0.0
                    for i in range(self.d_head):
                        dot += q[qbase + i] * k[kbase + i]
                    logit = alpha_h * dot
                    if not c.qk_norm:
                        logit *= self.scale
                    if c.position_mode == "alibi":
                        logit += slope * (tau - t)
                    logits.append(logit)
                    if logit > max_logit:
                        max_logit = logit
                exps = [math.exp(val - max_logit) for val in logits]
                denom = sum(exps)
                if denom <= 0.0:
                    raise FloatingPointError("attention softmax denominator is non-positive")
                obase = t * c.d_model + h * self.d_head
                pbase = h * rows * rows + t * rows
                for tau, ex in enumerate(exps):
                    prob = ex / denom
                    probs[pbase + tau] = prob
                    vbase = (tau * c.n_head + h) * self.d_head
                    for i in range(self.d_head):
                        out[obase + i] += prob * v[vbase + i]
        return out, probs

    def _attention_bwd(self, dout: List[float], probs: List[float], q: List[float], k: List[float], v: List[float], rows: int, layer_idx: int) -> Tuple[List[float], List[float], List[float]]:
        c = self.cfg
        dq = [0.0] * (rows * c.d_model)
        dk = [0.0] * (rows * c.d_model)
        dv = [0.0] * (rows * c.d_model)
        alpha_spec = self.store.spec(f"layer{layer_idx}.attn_alpha") if c.qk_norm else None
        p = self.store.param
        g = self.store.grad
        for h in range(c.n_head):
            alpha_h = p[alpha_spec.offset + h] if alpha_spec is not None else self.scale
            dprob = [0.0] * (rows * rows)
            for t in range(rows):
                obase = t * c.d_model + h * self.d_head
                for tau in range(t + 1):
                    pidx = h * rows * rows + t * rows + tau
                    prob = probs[pidx]
                    vbase = (tau * c.n_head + h) * self.d_head
                    dot = 0.0
                    for i in range(self.d_head):
                        go = dout[obase + i]
                        dot += go * v[vbase + i]
                        dv[vbase + i] += prob * go
                    dprob[t * rows + tau] = dot
            ds = [0.0] * (rows * rows)
            for t in range(rows):
                row_dot = 0.0
                for tau in range(t + 1):
                    pidx = h * rows * rows + t * rows + tau
                    prob = probs[pidx]
                    row_dot += prob * dprob[t * rows + tau]
                for tau in range(t + 1):
                    pidx = h * rows * rows + t * rows + tau
                    prob = probs[pidx]
                    ds_val = prob * (dprob[t * rows + tau] - row_dot)
                    ds[t * rows + tau] = ds_val
            for t in range(rows):
                qbase = (t * c.n_head + h) * self.d_head
                for tau in range(t + 1):
                    kval = ds[t * rows + tau]
                    if kval == 0.0:
                        continue
                    kbase = (tau * c.n_head + h) * self.d_head
                    dot_qk = 0.0
                    for i in range(self.d_head):
                        qv = q[qbase + i]
                        kv = k[kbase + i]
                        dq[qbase + i] += kval * alpha_h * kv
                        dk[kbase + i] += kval * alpha_h * qv
                        dot_qk += qv * kv
                    if alpha_spec is not None:
                        g[alpha_spec.offset + h] += kval * dot_qk
        return dq, dk, dv

    def _lm_head_spec(self) -> TensorSpec:
        if self.cfg.tie_embeddings:
            return self.store.spec("wte")
        return self.store.spec("lm_head")

    def _lm_head_fwd(self, x: List[float], rows: int) -> List[float]:
        head = self._lm_head_spec()
        p = self.store.param
        d = self.cfg.d_model
        logits = [0.0] * (rows * self.vocab_size)
        for r in range(rows):
            xoff = r * d
            loff = r * self.vocab_size
            for token in range(self.vocab_size):
                woff = head.offset + token * d
                acc = 0.0
                for i in range(d):
                    acc += x[xoff + i] * p[woff + i]
                logits[loff + token] = acc
        return logits

    def _lm_head_bwd(self, dlogits: List[float], x: List[float], rows: int) -> List[float]:
        d = self.cfg.d_model
        head = self._lm_head_spec()
        p = self.store.param
        g = self.store.grad
        dx = [0.0] * (rows * d)
        for r in range(rows):
            xoff = r * d
            loff = r * self.vocab_size
            for token in range(self.vocab_size):
                go = dlogits[loff + token]
                if go == 0.0:
                    continue
                woff = head.offset + token * d
                for i in range(d):
                    dx[xoff + i] += go * p[woff + i]
                    g[woff + i] += go * x[xoff + i]
        return dx

    def _cross_entropy(self, logits: List[float], targets: Sequence[int], rows: int) -> Tuple[float, List[float]]:
        if len(targets) != rows:
            raise ValueError("target length must match number of rows")
        dlogits = [0.0] * len(logits)
        loss = 0.0
        inv_rows = 1.0 / rows
        for r in range(rows):
            base = r * self.vocab_size
            max_logit = max(logits[base:base + self.vocab_size])
            denom = 0.0
            for i in range(self.vocab_size):
                denom += math.exp(logits[base + i] - max_logit)
            log_denom = math.log(denom)
            target = targets[r]
            loss += -(logits[base + target] - max_logit - log_denom)
            for i in range(self.vocab_size):
                prob = math.exp(logits[base + i] - max_logit - log_denom)
                dlogits[base + i] = prob * inv_rows
            dlogits[base + target] -= inv_rows
        return loss * inv_rows, dlogits

    # ---------- forward/backward ----------

    def forward_backward(self, tokens: Sequence[int], targets: Sequence[int]) -> float:
        if len(tokens) != len(targets):
            raise ValueError("tokens and targets must have identical lengths")
        if not tokens:
            raise ValueError("tokens must be non-empty")
        T = len(tokens)
        c = self.cfg
        store = self.store
        store.zero_grad()

        h = self._embedding_fwd(tokens)
        caches: List[Dict[str, object]] = []

        for li in range(c.n_layer):
            layer_cache: Dict[str, object] = {"x_in": h[:]}

            x_norm1, inv1 = self._rmsnorm_fwd(h, T, c.d_model)
            layer_cache["x_norm1"] = x_norm1
            layer_cache["inv1"] = inv1
            qkv = self._linear_fwd(x_norm1, store.spec(f"layer{li}.w_qkv"), T, c.d_model, 3 * c.d_model)
            q, k, v = self._split_qkv(qkv, T)
            layer_cache["q_raw"] = q
            layer_cache["k_raw"] = k
            layer_cache["v_raw"] = v

            if c.qk_norm:
                qn, q_inv = self._qk_norm_fwd(q, T, c.n_head, self.d_head)
                kn, k_inv = self._qk_norm_fwd(k, T, c.n_head, self.d_head)
            else:
                qn, kn = q, k
                q_inv, k_inv = [], []
            layer_cache["q"] = qn
            layer_cache["k"] = kn
            layer_cache["q_inv"] = q_inv
            layer_cache["k_inv"] = k_inv

            attn_cat, probs = self._attention_fwd(qn, kn, v, T, li)
            layer_cache["attn_cat"] = attn_cat
            layer_cache["probs"] = probs
            attn_proj = self._linear_fwd(attn_cat, store.spec(f"layer{li}.w_o"), T, c.d_model, c.d_model)
            layer_cache["attn_proj"] = attn_proj
            gamma_attn = store.param[store.spec(f"layer{li}.gamma_attn").offset] if c.residual_gate else 1.0
            h_attn = [h[i] + gamma_attn * attn_proj[i] for i in range(len(h))]
            layer_cache["h_attn"] = h_attn
            layer_cache["gamma_attn"] = gamma_attn

            x_norm2, inv2 = self._rmsnorm_fwd(h_attn, T, c.d_model)
            layer_cache["x_norm2"] = x_norm2
            layer_cache["inv2"] = inv2
            mlp_pre = self._linear_fwd(x_norm2, store.spec(f"layer{li}.w1"), T, c.d_model, c.d_ff)
            layer_cache["mlp_pre"] = mlp_pre
            if c.relu2:
                mlp_act, positive = self._relu2_fwd(mlp_pre)
                layer_cache["mlp_pos"] = positive
            else:
                raise NotImplementedError("only relu2=True is implemented")
            layer_cache["mlp_act"] = mlp_act
            mlp_proj = self._linear_fwd(mlp_act, store.spec(f"layer{li}.w2"), T, c.d_ff, c.d_model)
            layer_cache["mlp_proj"] = mlp_proj
            gamma_mlp = store.param[store.spec(f"layer{li}.gamma_mlp").offset] if c.residual_gate else 1.0
            h = [h_attn[i] + gamma_mlp * mlp_proj[i] for i in range(len(h_attn))]
            layer_cache["gamma_mlp"] = gamma_mlp
            layer_cache["x_out"] = h[:]
            caches.append(layer_cache)

        if c.final_norm:
            hf, final_inv = self._rmsnorm_fwd(h, T, c.d_model)
        else:
            hf, final_inv = h[:], []
        logits = self._lm_head_fwd(hf, T)
        loss, dlogits = self._cross_entropy(logits, targets, T)

        dh = self._lm_head_bwd(dlogits, hf, T)
        if c.final_norm:
            dh = self._rmsnorm_bwd(dh, h, final_inv, T, c.d_model)

        for li in reversed(range(c.n_layer)):
            cache = caches[li]
            gamma_mlp = cache["gamma_mlp"]
            gamma_attn = cache["gamma_attn"]

            d_h_attn = dh[:]
            d_mlp_proj = [val * gamma_mlp for val in dh]
            if c.residual_gate:
                gamma_spec = store.spec(f"layer{li}.gamma_mlp")
                gate_grad = 0.0
                mlp_proj = cache["mlp_proj"]
                for i in range(len(dh)):
                    gate_grad += dh[i] * mlp_proj[i]
                store.grad[gamma_spec.offset] += gate_grad
            d_mlp_act = self._linear_bwd(d_mlp_proj, cache["mlp_act"], store.spec(f"layer{li}.w2"), T, c.d_ff, c.d_model)
            d_mlp_pre = self._relu2_bwd(d_mlp_act, cache["mlp_pos"])
            d_x_norm2 = self._linear_bwd(d_mlp_pre, cache["x_norm2"], store.spec(f"layer{li}.w1"), T, c.d_model, c.d_ff)
            d_h_attn_norm = self._rmsnorm_bwd(d_x_norm2, cache["h_attn"], cache["inv2"], T, c.d_model)
            d_h_attn = [d_h_attn[i] + d_h_attn_norm[i] for i in range(len(d_h_attn))]

            d_h_in = d_h_attn[:]
            d_attn_proj = [val * gamma_attn for val in d_h_attn]
            if c.residual_gate:
                gamma_spec = store.spec(f"layer{li}.gamma_attn")
                gate_grad = 0.0
                attn_proj = cache["attn_proj"]
                for i in range(len(d_h_attn)):
                    gate_grad += d_h_attn[i] * attn_proj[i]
                store.grad[gamma_spec.offset] += gate_grad
            d_attn_cat = self._linear_bwd(d_attn_proj, cache["attn_cat"], store.spec(f"layer{li}.w_o"), T, c.d_model, c.d_model)
            dq, dk, dv = self._attention_bwd(d_attn_cat, cache["probs"], cache["q"], cache["k"], cache["v_raw"], T, li)
            if c.qk_norm:
                dq = self._qk_norm_bwd(dq, cache["q_raw"], cache["q_inv"], T, c.n_head, self.d_head)
                dk = self._qk_norm_bwd(dk, cache["k_raw"], cache["k_inv"], T, c.n_head, self.d_head)
            dqkv = self._merge_qkv_grads(dq, dk, dv, T)
            d_x_norm1 = self._linear_bwd(dqkv, cache["x_norm1"], store.spec(f"layer{li}.w_qkv"), T, c.d_model, 3 * c.d_model)
            d_h_norm = self._rmsnorm_bwd(d_x_norm1, cache["x_in"], cache["inv1"], T, c.d_model)
            dh = [d_h_in[i] + d_h_norm[i] for i in range(len(d_h_in))]

        self._embedding_bwd(tokens, dh)
        return loss

    # ---------- optimizer / training ----------

    def clip_grads(self, clip: float) -> float:
        if clip <= 0.0:
            return 0.0
        total = 0.0
        for val in self.store.grad:
            total += val * val
        total = math.sqrt(total)
        if total > clip:
            scale = clip / (total + 1e-12)
            for i in range(len(self.store.grad)):
                self.store.grad[i] *= scale
        return total

    def adamw_step(self, step_idx: int) -> float:
        c = self.cfg
        if step_idx < c.warmup_steps:
            lr = c.lr * (step_idx + 1) / max(1, c.warmup_steps)
        else:
            denom = max(1, c.steps - c.warmup_steps)
            progress = min(1.0, max(0.0, (step_idx - c.warmup_steps) / denom))
            lr = c.lr + (c.min_lr - c.lr) * progress
        p = self.store.param
        g = self.store.grad
        m1 = self.store.m1
        m2 = self.store.m2
        beta1 = c.beta1
        beta2 = c.beta2
        eps = c.adam_eps
        t = step_idx + 1
        bc1 = 1.0 - beta1 ** t
        bc2 = 1.0 - beta2 ** t
        inv_bc1 = 1.0 / bc1
        inv_bc2 = 1.0 / bc2
        for i in range(len(p)):
            gi = g[i]
            m1[i] = beta1 * m1[i] + (1.0 - beta1) * gi
            m2[i] = beta2 * m2[i] + (1.0 - beta2) * gi * gi
            mhat = m1[i] * inv_bc1
            vhat = m2[i] * inv_bc2
            update = mhat / (math.sqrt(vhat) + eps)
            if c.weight_decay != 0.0:
                update += c.weight_decay * p[i]
            p[i] -= lr * update
            g[i] = 0.0
        return lr

    def evaluate(self, dataset: DocumentDataset, batches: int = 4) -> float:
        losses = []
        for _ in range(batches):
            tokens, targets, _ = dataset.sample(num_docs=max(1, self.cfg.batch_docs))
            loss = self.forward_backward(tokens, targets)
            losses.append(loss)
            self.store.zero_grad()
        return sum(losses) / len(losses)

    # ---------- inference ----------

    def _embed_token_step(self, token: int, pos: int) -> List[float]:
        if token < 0 or token >= self.vocab_size:
            raise ValueError(f"token {token} out of vocabulary range 0..{self.vocab_size - 1}")
        if self.cfg.position_mode == "learned" and pos >= self.cfg.context:
            raise ValueError("learned position mode cannot decode beyond configured context length")
        c = self.cfg
        p = self.store.param
        wte = self.store.spec("wte")
        out = [0.0] * c.d_model
        src = wte.offset + token * c.d_model
        for i in range(c.d_model):
            out[i] = p[src + i]
        if c.position_mode == "learned":
            wpe = self.store.spec("wpe")
            src = wpe.offset + pos * c.d_model
            for i in range(c.d_model):
                out[i] += p[src + i]
        return out

    def init_kv_cache(self) -> Dict[str, object]:
        return {
            "keys": [[] for _ in range(self.cfg.n_layer)],
            "values": [[] for _ in range(self.cfg.n_layer)],
            "positions": [],
            "next_pos": 0,
        }

    def decode_step(self, token: int, cache: Dict[str, object]) -> List[float]:
        c = self.cfg
        pos = int(cache["next_pos"])
        if len(cache["positions"]) >= c.context:
            cache["positions"].pop(0)
            for li in range(c.n_layer):
                cache["keys"][li].pop(0)
                cache["values"][li].pop(0)
        current_positions = list(cache["positions"]) + [pos]
        x = self._embed_token_step(token, pos)
        for li in range(c.n_layer):
            x_norm1, _ = self._rmsnorm_fwd(x, 1, c.d_model)
            qkv = self._linear_fwd(x_norm1, self.store.spec(f"layer{li}.w_qkv"), 1, c.d_model, 3 * c.d_model)
            q, k, v = self._split_qkv(qkv, 1)
            if c.qk_norm:
                q, _ = self._qk_norm_fwd(q, 1, c.n_head, self.d_head)
                k, _ = self._qk_norm_fwd(k, 1, c.n_head, self.d_head)
            cache["keys"][li].append(k[:])
            cache["values"][li].append(v[:])
            attn_cat = [0.0] * c.d_model
            alpha_spec = self.store.spec(f"layer{li}.attn_alpha") if c.qk_norm else None
            for h in range(c.n_head):
                slope = self.alibi_slopes[h] if c.position_mode == "alibi" else 0.0
                alpha_h = self.store.param[alpha_spec.offset + h] if alpha_spec is not None else self.scale
                qbase = h * self.d_head
                logits = []
                max_logit = -1e30
                for tau_idx, (k_tok, _v_tok, tau_pos) in enumerate(zip(cache["keys"][li], cache["values"][li], current_positions)):
                    kbase = h * self.d_head
                    dot = 0.0
                    for i in range(self.d_head):
                        dot += q[qbase + i] * k_tok[kbase + i]
                    logit = alpha_h * dot
                    if not c.qk_norm:
                        logit *= self.scale
                    if c.position_mode == "alibi":
                        logit += slope * (tau_pos - pos)
                    logits.append((tau_idx, logit))
                    if logit > max_logit:
                        max_logit = logit
                exps = [math.exp(logit - max_logit) for _idx, logit in logits]
                denom = sum(exps)
                if denom <= 0.0:
                    raise FloatingPointError("attention denominator became non-positive during decode")
                for ex, (tau_idx, _logit) in zip(exps, logits):
                    prob = ex / denom
                    v_tok = cache["values"][li][tau_idx]
                    obase = h * self.d_head
                    vbase = h * self.d_head
                    for i in range(self.d_head):
                        attn_cat[obase + i] += prob * v_tok[vbase + i]
            attn_proj = self._linear_fwd(attn_cat, self.store.spec(f"layer{li}.w_o"), 1, c.d_model, c.d_model)
            gamma_attn = self.store.param[self.store.spec(f"layer{li}.gamma_attn").offset] if c.residual_gate else 1.0
            x_attn = [x[i] + gamma_attn * attn_proj[i] for i in range(c.d_model)]
            x_norm2, _ = self._rmsnorm_fwd(x_attn, 1, c.d_model)
            mlp_pre = self._linear_fwd(x_norm2, self.store.spec(f"layer{li}.w1"), 1, c.d_model, c.d_ff)
            mlp_act, _ = self._relu2_fwd(mlp_pre)
            mlp_proj = self._linear_fwd(mlp_act, self.store.spec(f"layer{li}.w2"), 1, c.d_ff, c.d_model)
            gamma_mlp = self.store.param[self.store.spec(f"layer{li}.gamma_mlp").offset] if c.residual_gate else 1.0
            x = [x_attn[i] + gamma_mlp * mlp_proj[i] for i in range(c.d_model)]
        if c.final_norm:
            x, _ = self._rmsnorm_fwd(x, 1, c.d_model)
        logits = self._lm_head_fwd(x, 1)
        cache["positions"].append(pos)
        cache["next_pos"] = pos + 1
        return logits

    def prefill_cache(self, tokens: Sequence[int], cache: Optional[Dict[str, object]] = None) -> Tuple[Dict[str, object], List[float]]:
        if cache is None:
            cache = self.init_kv_cache()
        last_logits: Optional[List[float]] = None
        for token in tokens:
            last_logits = self.decode_step(token, cache)
        if last_logits is None:
            raise ValueError("prefill requires at least one token")
        return cache, last_logits

    def _forward_inference(self, tokens: Sequence[int]) -> List[float]:
        cache, logits = self.prefill_cache(tokens, self.init_kv_cache())
        _ = cache
        return logits

    def sample(self, tokenizer, prompt: str, max_new_tokens: int, temperature: float = 1.0, top_k: int = 0) -> str:
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        bos = tokenizer.bos()
        prompt_tokens = [bos] + tokenizer.encode(prompt)
        if len(prompt_tokens) > self.cfg.context:
            prompt_tokens = prompt_tokens[-self.cfg.context:]
        rng = random.Random(self.cfg.seed)
        cache, last_logits = self.prefill_cache(prompt_tokens, self.init_kv_cache())
        tokens = list(prompt_tokens)
        for _ in range(max_new_tokens):
            next_token = sample_from_logits(last_logits, rng=rng, temperature=temperature, top_k=top_k)
            tokens.append(next_token)
            last_logits = self.decode_step(next_token, cache)
        return tokenizer.decode(tokens[1:])

    # ---------- checkpointing ----------

    def save(self, path: str, tokenizer_meta: Dict[str, object], extra: Optional[Dict[str, object]] = None) -> None:
        payload = {
            "config": asdict(self.cfg),
            "vocab_size": self.vocab_size,
            "tokenizer": tokenizer_meta,
            "store": self.store.state_dict(),
            "extra": extra or {},
        }
        atomic_write_bytes(path, pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))

    @staticmethod
    def load(path: str) -> Tuple["MicroGPT2", object, Dict[str, object]]:
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        cfg = Config(**payload["config"])
        model = MicroGPT2(cfg, int(payload["vocab_size"]))
        model.store = ParameterStore.from_state(payload["store"])
        tokenizer = tokenizer_from_meta(payload["tokenizer"])
        extra = payload.get("extra", {})
        return model, tokenizer, extra


# -----------------------------
# Sampling helper
# -----------------------------


def sample_from_logits(logits: Sequence[float], rng: Optional[random.Random] = None, temperature: float = 1.0, top_k: int = 0) -> int:
    if rng is None:
        rng = random
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    scaled = [x / temperature for x in logits]
    if top_k > 0:
        top_k = max(1, min(top_k, len(scaled)))
        top_idx = sorted(range(len(scaled)), key=lambda i: scaled[i], reverse=True)[:top_k]
        mask = set(top_idx)
        floor = min(scaled)
        scaled = [val if i in mask else floor - 1e9 for i, val in enumerate(scaled)]
    m = max(scaled)
    exps = [math.exp(v - m) for v in scaled]
    total = sum(exps)
    if total <= 0.0:
        return int(max(range(len(logits)), key=lambda i: logits[i]))
    threshold = rng.random() * total
    running = 0.0
    for i, ex in enumerate(exps):
        running += ex
        if running >= threshold:
            return i
    return len(logits) - 1


# -----------------------------
# CLI helpers
# -----------------------------


def build_model_and_data(args) -> Tuple[MicroGPT2, object, DocumentDataset, DocumentDataset]:
    docs = load_docs(args.input)
    rng = random.Random(args.seed)
    rng.shuffle(docs)
    cfg = Config(
        seed=args.seed,
        tokenizer=args.tokenizer,
        context=args.context,
        n_layer=args.n_layer,
        d_model=args.d_model,
        n_head=args.n_head,
        d_ff=args.d_ff,
        position_mode=args.position_mode,
        qk_norm=bool(args.qk_norm),
        relu2=True,
        tie_embeddings=True,
        final_norm=bool(args.final_norm),
        residual_gate=bool(args.residual_gate),
        init_std=args.init_std,
        lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        adam_eps=args.adam_eps,
        grad_clip=args.grad_clip,
        batch_docs=args.batch_docs,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        save_every=args.save_every,
        eval_every=args.eval_every,
        train_ratio=args.train_ratio,
        temperature=args.temperature,
        top_k=args.top_k,
        sample_tokens=args.sample_tokens,
    )
    tokenizer = build_tokenizer(cfg.tokenizer, docs)
    dataset = DocumentDataset(docs, tokenizer, cfg.context, seed=cfg.seed)
    train_dataset, val_dataset = dataset.split(cfg.train_ratio)
    model = MicroGPT2(cfg, tokenizer.vocab_size)
    return model, tokenizer, train_dataset, val_dataset


def train_main(args) -> int:
    model, tokenizer, train_dataset, val_dataset = build_model_and_data(args)
    print(f"train docs: {len(train_dataset.docs)} | val docs: {len(val_dataset.docs)}")
    print(f"vocab size: {tokenizer.vocab_size} | params: {len(model.store.param)}")
    best_val = float("inf")
    history: List[Dict[str, float]] = []
    started = time.time()
    for step in range(model.cfg.steps):
        tokens, targets, tokens_used = train_dataset.sample(num_docs=max(1, model.cfg.batch_docs))
        loss = model.forward_backward(tokens, targets)
        grad_norm = model.clip_grads(model.cfg.grad_clip)
        lr = model.adamw_step(step)

        row = {
            "step": float(step + 1),
            "loss": float(loss),
            "grad_norm": float(grad_norm),
            "lr": float(lr),
            "tokens": float(tokens_used),
        }
        history.append(row)

        if (step + 1) % max(1, model.cfg.eval_every) == 0 or step == 0 or step + 1 == model.cfg.steps:
            val_loss = model.evaluate(val_dataset, batches=min(4, max(1, len(val_dataset.docs))))
            row["val_loss"] = float(val_loss)
            if val_loss < best_val:
                best_val = val_loss
                if args.out:
                    model.save(args.out, tokenizer.to_meta(), extra={"history": history, "best_val": best_val})
            elapsed = time.time() - started
            print(
                f"step {step + 1:5d}/{model.cfg.steps:5d} | train {loss:.4f} | val {val_loss:.4f} | "
                f"gnorm {grad_norm:.4f} | lr {lr:.6f} | tok {tokens_used} | {elapsed:.1f}s"
            )
        else:
            print(
                f"step {step + 1:5d}/{model.cfg.steps:5d} | train {loss:.4f} | "
                f"gnorm {grad_norm:.4f} | lr {lr:.6f} | tok {tokens_used}",
                end="\r",
                flush=True,
            )

        if args.out and ((step + 1) % max(1, model.cfg.save_every) == 0):
            model.save(args.out + ".latest", tokenizer.to_meta(), extra={"history": history, "best_val": best_val})

    print()
    if args.out:
        model.save(args.out, tokenizer.to_meta(), extra={"history": history, "best_val": best_val})
        with open(args.out + ".history.json", "w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)
    return 0


def sample_main(args) -> int:
    model, tokenizer, extra = MicroGPT2.load(args.checkpoint)
    _ = extra
    text = model.sample(
        tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(text)
    return 0


def inspect_main(args) -> int:
    model, tokenizer, extra = MicroGPT2.load(args.checkpoint)
    summary = {
        "config": asdict(model.cfg),
        "vocab_size": model.vocab_size,
        "param_count": len(model.store.param),
        "tokenizer": tokenizer.to_meta(),
        "extra_keys": sorted(extra.keys()),
    }
    print(json.dumps(summary, indent=2))
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dependency-free GPT-style language model")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common_train_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--input", type=str, default="input.txt", help="line-delimited training text file")
        p.add_argument("--out", type=str, default="microgpt2.ckpt", help="checkpoint output path")
        p.add_argument("--seed", type=int, default=1337)
        p.add_argument("--tokenizer", type=str, default="byte", choices=["byte", "char"])
        p.add_argument("--context", type=int, default=128)
        p.add_argument("--n-layer", dest="n_layer", type=int, default=4)
        p.add_argument("--d-model", dest="d_model", type=int, default=128)
        p.add_argument("--n-head", dest="n_head", type=int, default=4)
        p.add_argument("--d-ff", dest="d_ff", type=int, default=512)
        p.add_argument("--position-mode", type=str, default="alibi", choices=["alibi", "learned"])
        p.add_argument("--qk-norm", type=int, default=1, choices=[0, 1])
        p.add_argument("--final-norm", type=int, default=1, choices=[0, 1])
        p.add_argument("--residual-gate", type=int, default=1, choices=[0, 1])
        p.add_argument("--init-std", dest="init_std", type=float, default=0.02)
        p.add_argument("--lr", type=float, default=2e-3)
        p.add_argument("--min-lr", dest="min_lr", type=float, default=2e-4)
        p.add_argument("--weight-decay", dest="weight_decay", type=float, default=0.01)
        p.add_argument("--beta1", type=float, default=0.9)
        p.add_argument("--beta2", type=float, default=0.95)
        p.add_argument("--adam-eps", dest="adam_eps", type=float, default=1e-8)
        p.add_argument("--grad-clip", dest="grad_clip", type=float, default=1.0)
        p.add_argument("--batch-docs", dest="batch_docs", type=int, default=1)
        p.add_argument("--steps", type=int, default=200)
        p.add_argument("--warmup-steps", dest="warmup_steps", type=int, default=20)
        p.add_argument("--save-every", dest="save_every", type=int, default=50)
        p.add_argument("--eval-every", dest="eval_every", type=int, default=25)
        p.add_argument("--train-ratio", dest="train_ratio", type=float, default=0.9)
        p.add_argument("--temperature", type=float, default=0.9)
        p.add_argument("--top-k", dest="top_k", type=int, default=0)
        p.add_argument("--sample-tokens", dest="sample_tokens", type=int, default=128)

    p_train = sub.add_parser("train", help="train a model")
    add_common_train_flags(p_train)

    p_sample = sub.add_parser("sample", help="sample from a checkpoint")
    p_sample.add_argument("--checkpoint", type=str, required=True)
    p_sample.add_argument("--prompt", type=str, default="")
    p_sample.add_argument("--tokens", type=int, default=128)
    p_sample.add_argument("--temperature", type=float, default=0.9)
    p_sample.add_argument("--top-k", type=int, default=0)

    p_inspect = sub.add_parser("inspect", help="inspect checkpoint metadata")
    p_inspect.add_argument("--checkpoint", type=str, required=True)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.command == "train":
        return train_main(args)
    if args.command == "sample":
        return sample_main(args)
    if args.command == "inspect":
        return inspect_main(args)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())