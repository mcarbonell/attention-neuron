"""
v330b — Transferencia espectral controlada en lenguaje real (extensión de convergencia)
===========================================================

Pregunta primaria:
    ¿FWHT/DCT-II aportan una ventaja específica sobre una rotación ortogonal
    aleatoria congelada, cuando la mezcla causal, el corpus, los tokens de
    entrenamiento, la profundidad y el presupuesto paramétrico se mantienen
    controlados?

El experimento usa Tiny Shakespeare a nivel de caracteres con split temporal
train/valid/test. Compara únicamente el FFN de un mismo backbone causal:

    1. lerp_fwht_dct     Candidato: mezcla global aprendible FWHT + DCT-II.
    2. dense_ffn         Control denso con número de parámetros emparejado.
    3. random_orthogonal Control crítico: matriz ortogonal aleatoria fija.
    4. fwht              Base Walsh-Hadamard fija.
    5. dct               Base DCT-II fija.

No declara ganadores ni escribe findings: sólo conserva resultados crudos y
una entrada de ledger [SEÑAL]. El documento de findings debe redactarse tras
inspeccionar el JSON completo.

Uso recomendado:
    python scratch/prototype_v330_spectral_transfer_control.py --mode pilot
    python scratch/prototype_v330_spectral_transfer_control.py --mode level2

El modo pilot (una semilla) es un filtro de harness. El modo level2 usa cinco
semillas, 30 épocas y al menos 1,024 secuencias en validación y test por
configuración. El progreso se imprime una vez por época; el JSON conserva el
historial de cada época completo.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "tinyshakespeare.txt"
RAW_RESULTS_DIR = ROOT / "results" / "raw"
LEDGER_PATH = ROOT / "results" / "master_ledger.jsonl"
TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    mode: str
    rigor_level: int
    seeds: tuple[int, ...]
    model_names: tuple[str, ...]
    device: str
    deterministic: bool
    d_model: int
    n_heads: int
    n_layers: int
    seq_len: int
    batch_size: int
    epochs: int
    steps_per_epoch: int
    lr: float
    weight_decay: float
    grad_clip: float
    warmup_pct: float
    valid_batches: int
    test_batches: int
    log_every: int
    train_split: float
    valid_split: float
    random_basis_seed: int


MODE_CONFIGS: dict[str, dict[str, Any]] = {
    "pilot": {
        "rigor_level": 1,
        "seeds": (42,),
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 2,
        "seq_len": 128,
        "batch_size": 16,
        "epochs": 10,
        "steps_per_epoch": 80,
        "lr": 3e-3,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "warmup_pct": 0.05,
        "valid_batches": 16,
        "test_batches": 16,
    },
    "level2": {
        "rigor_level": 2,
        "seeds": (10, 20, 30, 42, 100),
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 2,
        "seq_len": 128,
        "batch_size": 16,
        "epochs": 30,
        "steps_per_epoch": 150,
        "lr": 3e-3,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "warmup_pct": 0.05,
        "valid_batches": 64,
        "test_batches": 64,
    },
}

# El candidato debe ejecutarse primero, conforme a GEMINI.md.
DEFAULT_MODEL_ORDER = (
    "lerp_fwht_dct",
    "dense_ffn",
    "random_orthogonal",
    "fwht",
    "dct",
)


class TimestampLogger:
    """Imprime únicamente líneas con timestamp relativo y conserva el texto."""

    def __init__(self) -> None:
        self.start = time.perf_counter()

    def elapsed(self) -> str:
        elapsed = time.perf_counter() - self.start
        hours, rem = divmod(elapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        return f"+{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}"

    def log(self, message: str) -> None:
        for line in str(message).splitlines() or [""]:
            print(f"[{self.elapsed()}] {line}", flush=True)

    def log_json(self, label: str, value: Any) -> None:
        self.log(f"{label}:")
        self.log(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v330 spectral transfer control")
    parser.add_argument("--mode", choices=tuple(MODE_CONFIGS), default="pilot")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, etc.")
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODEL_ORDER),
        help="Lista separada por comas; el candidato lerp_fwht_dct debe ir primero.",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Sobrescribe las semillas del modo, por ejemplo: 10,20,30,42,100",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=0,
        help="0 imprime sólo el resumen detallado por época; N>0 añade cada N pasos.",
    )
    parser.add_argument(
        "--run-id",
        default="v330b_spectral_transfer_extended_30ep",
        help="Identificador único para JSON y ledger; evita mezclar corridas con v330.",
    )
    parser.add_argument("--no-deterministic", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    base = dict(MODE_CONFIGS[args.mode])
    model_names = tuple(name.strip() for name in args.models.split(",") if name.strip())
    unknown = sorted(set(model_names) - set(DEFAULT_MODEL_ORDER))
    if unknown:
        raise ValueError(f"Modelos no reconocidos: {unknown}")
    if not model_names:
        raise ValueError("Debe seleccionarse al menos un modelo.")
    if model_names[0] != "lerp_fwht_dct":
        raise ValueError("La Regla de Oro exige ejecutar lerp_fwht_dct primero.")

    if args.seeds:
        base["seeds"] = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
    if args.epochs is not None:
        base["epochs"] = args.epochs
    if args.steps_per_epoch is not None:
        base["steps_per_epoch"] = args.steps_per_epoch
    if args.log_every < 0:
        raise ValueError("--log-every debe ser >= 0.")

    return ExperimentConfig(
        experiment_id=args.run_id,
        mode=args.mode,
        model_names=model_names,
        device=args.device,
        deterministic=not args.no_deterministic,
        log_every=args.log_every,
        train_split=0.70,
        valid_split=0.15,
        random_basis_seed=330,
        **base,
    )


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def hardware_metadata(device: torch.device) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cpu": platform.processor() or "unavailable",
        "torch_threads": torch.get_num_threads(),
        "requested_device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "commit_hash": git_commit_hash(),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        metadata["gpu"] = torch.cuda.get_device_name(device)
        metadata["gpu_memory_bytes"] = torch.cuda.get_device_properties(device).total_memory
    return metadata


def configure_determinism(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def load_tiny_shakespeare(logger: TimestampLogger) -> str:
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        logger.log(f"DATA | download_start | url={TINY_SHAKESPEARE_URL} | path={DATA_PATH}")
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, DATA_PATH)
        logger.log("DATA | download_complete")
    text = DATA_PATH.read_text(encoding="utf-8")
    if len(text) < 10_000:
        raise ValueError(f"Corpus inesperadamente pequeño: {len(text)} caracteres.")
    logger.log(f"DATA | corpus=tiny_shakespeare | characters={len(text):,} | path={DATA_PATH}")
    return text


def encode_corpus(text: str) -> tuple[torch.Tensor, dict[str, int]]:
    chars = sorted(set(text))
    stoi = {char: index for index, char in enumerate(chars)}
    tokens = torch.tensor([stoi[char] for char in text], dtype=torch.long)
    return tokens, stoi


def split_tokens(tokens: torch.Tensor, config: ExperimentConfig) -> dict[str, torch.Tensor]:
    if not 0.0 < config.train_split < 1.0 or not 0.0 < config.valid_split < 1.0:
        raise ValueError("Los splits deben ser fracciones estrictamente positivas.")
    if config.train_split + config.valid_split >= 1.0:
        raise ValueError("train_split + valid_split debe dejar espacio para test.")
    train_end = int(len(tokens) * config.train_split)
    valid_end = train_end + int(len(tokens) * config.valid_split)
    splits = {
        "train": tokens[:train_end].contiguous(),
        "valid": tokens[train_end:valid_end].contiguous(),
        "test": tokens[valid_end:].contiguous(),
    }
    required = config.seq_len + 2
    for name, split in splits.items():
        if len(split) < required:
            raise ValueError(f"Split {name} demasiado corto para seq_len={config.seq_len}.")
    return splits


def sample_batch(
    tokens: torch.Tensor,
    batch_size: int,
    seq_len: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(tokens) - seq_len - 1
    starts = torch.randint(0, max_start, (batch_size,), generator=generator)
    offsets = torch.arange(seq_len + 1)
    windows = tokens[starts[:, None] + offsets[None, :]]
    return windows[:, :-1].to(device), windows[:, 1:].to(device)


def build_fixed_eval_batches(
    tokens: torch.Tensor,
    batch_size: int,
    seq_len: int,
    num_batches: int,
    seed: int,
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Construye ventanas retenidas, fijas y no solapadas por secuencia."""
    num_sequences = batch_size * num_batches
    max_start = len(tokens) - seq_len - 1
    stride = max_start // num_sequences
    if stride < seq_len + 1:
        raise ValueError(
            f"Split insuficiente para {num_sequences} ventanas independientes de longitud {seq_len}."
        )
    slack = stride - (seq_len + 1)
    offset = seed % (slack + 1)
    starts = offset + torch.arange(num_sequences) * stride
    offsets = torch.arange(seq_len + 1)
    windows = tokens[starts[:, None] + offsets[None, :]]
    return [
        (windows[index : index + batch_size, :-1].to(device), windows[index : index + batch_size, 1:].to(device))
        for index in range(0, num_sequences, batch_size)
    ]


def create_hadamard_matrix(dim: int) -> torch.Tensor:
    if dim <= 0 or dim & (dim - 1):
        raise ValueError(f"FWHT requiere una dimensión potencia de 2, recibido {dim}.")
    matrix = torch.ones((1, 1), dtype=torch.float32)
    while matrix.shape[0] < dim:
        matrix = torch.cat(
            (torch.cat((matrix, matrix), dim=1), torch.cat((matrix, -matrix), dim=1)), dim=0
        )
    return matrix / math.sqrt(dim)


def create_dct2_matrix(dim: int) -> torch.Tensor:
    rows = torch.arange(dim, dtype=torch.float32).unsqueeze(1)
    cols = torch.arange(dim, dtype=torch.float32).unsqueeze(0)
    matrix = math.sqrt(2.0 / dim) * torch.cos(math.pi * rows * (2.0 * cols + 1.0) / (2.0 * dim))
    matrix[0, :] = 1.0 / math.sqrt(dim)
    return matrix


def create_random_orthogonal_matrix(dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    raw = torch.randn((dim, dim), generator=generator)
    q, r = torch.linalg.qr(raw)
    signs = torch.sign(torch.diagonal(r))
    signs[signs == 0] = 1.0
    return q * signs.unsqueeze(0)


class SinCosPosition(nn.Module):
    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10_000.0) / d_model)
        )
        pe = torch.zeros((max_len, d_model), dtype=torch.float32)
        pe[:, 0::2] = torch.sin(positions * frequencies)
        pe[:, 1::2] = torch.cos(positions * frequencies)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1]]


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_len: int) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model debe ser divisible por n_heads.")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.register_buffer(
            "causal_mask", torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1), persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, channels = x.shape
        qkv = self.qkv(x).view(batch, steps, 3, self.n_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.causal_mask[:steps, :steps], float("-inf"))
        weights = F.softmax(scores, dim=-1)
        mixed = (weights @ value).transpose(1, 2).contiguous().view(batch, steps, channels)
        return self.out(mixed)


class DenseFFN(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(d_model, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(F.silu(self.in_proj(x)))


class SingleBasisSpectralFFN(nn.Module):
    """Dos bancos trigonométricos en una base ortogonal congelada, sin loops en forward."""

    def __init__(self, basis: torch.Tensor) -> None:
        super().__init__()
        d_model = basis.shape[0]
        self.register_buffer("basis", basis, persistent=False)
        self.phase_cos = nn.Parameter(torch.zeros(2, d_model))
        self.phase_sin = nn.Parameter(torch.zeros(2, d_model))
        self.amplitude_cos = nn.Parameter(torch.ones(2, d_model))
        self.amplitude_sin = nn.Parameter(torch.ones(2, d_model))
        self.combine = nn.Linear(2 * d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frequency = F.linear(x, self.basis)
        modulated = (
            torch.cos(frequency.unsqueeze(-2) + self.phase_cos) * self.amplitude_cos
            + torch.sin(frequency.unsqueeze(-2) + self.phase_sin) * self.amplitude_sin
        )
        combined = self.combine(modulated.flatten(start_dim=-2))
        return F.linear(combined, self.basis.t())


class LerpFwhtDctFFN(nn.Module):
    """Dos bases, un banco por rama y dos logits globales aprendibles por capa."""

    def __init__(self, fwht: torch.Tensor, dct: torch.Tensor) -> None:
        super().__init__()
        d_model = fwht.shape[0]
        self.register_buffer("fwht", fwht, persistent=False)
        self.register_buffer("dct", dct, persistent=False)
        self.substrate_logits = nn.Parameter(torch.zeros(2))
        self.phase_cos = nn.Parameter(torch.zeros(2, d_model))
        self.phase_sin = nn.Parameter(torch.zeros(2, d_model))
        self.amplitude_cos = nn.Parameter(torch.ones(2, d_model))
        self.amplitude_sin = nn.Parameter(torch.ones(2, d_model))
        self.combine = nn.Linear(2 * d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.substrate_logits, dim=0)
        fwht_frequency = F.linear(x, self.fwht)
        dct_frequency = F.linear(x, self.dct)
        fwht_branch = (
            torch.cos(fwht_frequency + self.phase_cos[0]) * self.amplitude_cos[0]
            + torch.sin(fwht_frequency + self.phase_sin[0]) * self.amplitude_sin[0]
        )
        dct_branch = (
            torch.cos(dct_frequency + self.phase_cos[1]) * self.amplitude_cos[1]
            + torch.sin(dct_frequency + self.phase_sin[1]) * self.amplitude_sin[1]
        )
        fwht_output = F.linear(fwht_branch, self.fwht.t())
        dct_output = F.linear(dct_branch, self.dct.t())
        return self.combine(torch.cat((weights[0] * fwht_output, weights[1] * dct_output), dim=-1))


def matched_dense_hidden_dim(d_model: int) -> int:
    spectral_params = 2 * d_model * d_model + 8 * d_model
    return max(1, round((spectral_params - d_model) / (2 * d_model + 1)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_len: int, ffn: nn.Module) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_len)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = ffn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x))
        return x + self.ffn(self.norm_ffn(x))


class SpectralTransferModel(nn.Module):
    def __init__(self, vocab_size: int, config: ExperimentConfig, model_name: str) -> None:
        super().__init__()
        self.model_name = model_name
        self.dense_hidden_dim = matched_dense_hidden_dim(config.d_model)
        self.embedding = nn.Embedding(vocab_size, config.d_model)
        self.position = SinCosPosition(config.d_model, config.seq_len)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.d_model,
                    config.n_heads,
                    config.seq_len,
                    self._build_ffn(config, model_name, layer_index),
                )
                for layer_index in range(config.n_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, vocab_size)

    def _build_ffn(self, config: ExperimentConfig, model_name: str, layer_index: int) -> nn.Module:
        d_model = config.d_model
        if model_name == "dense_ffn":
            return DenseFFN(d_model, self.dense_hidden_dim)
        if model_name == "fwht":
            return SingleBasisSpectralFFN(create_hadamard_matrix(d_model))
        if model_name == "dct":
            return SingleBasisSpectralFFN(create_dct2_matrix(d_model))
        if model_name == "random_orthogonal":
            return SingleBasisSpectralFFN(create_random_orthogonal_matrix(d_model, config.random_basis_seed + layer_index))
        if model_name == "lerp_fwht_dct":
            return LerpFwhtDctFFN(create_hadamard_matrix(d_model), create_dct2_matrix(d_model))
        raise ValueError(f"Modelo desconocido: {model_name}")

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.position(self.embedding(tokens))
        for block in self.blocks:
            hidden = block(hidden)
        return self.head(self.norm_out(hidden))


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def describe_architecture(model: SpectralTransferModel, config: ExperimentConfig) -> list[dict[str, Any]]:
    layers: list[dict[str, Any]] = [
        {"layer": "embedding", "type": "Embedding", "shape": ["vocab", config.d_model], "params": trainable_parameter_count(model.embedding)},
        {"layer": "position", "type": "fixed_sincos", "shape": [config.seq_len, config.d_model], "params": 0},
    ]
    for index, block in enumerate(model.blocks, start=1):
        layers.extend(
            [
                {"layer": f"block_{index}.norm_attn", "type": "LayerNorm", "params": trainable_parameter_count(block.norm_attn)},
                {
                    "layer": f"block_{index}.attention",
                    "type": "causal_QK_softmax_MHA",
                    "heads": config.n_heads,
                    "head_dim": config.d_model // config.n_heads,
                    "params": trainable_parameter_count(block.attn),
                },
                {"layer": f"block_{index}.norm_ffn", "type": "LayerNorm", "params": trainable_parameter_count(block.norm_ffn)},
                {
                    "layer": f"block_{index}.ffn",
                    "type": type(block.ffn).__name__,
                    "params": trainable_parameter_count(block.ffn),
                    "detail": model.model_name,
                },
            ]
        )
    layers.extend(
        [
            {"layer": "norm_out", "type": "LayerNorm", "params": trainable_parameter_count(model.norm_out)},
            {"layer": "head", "type": "Linear", "shape": [config.d_model, "vocab"], "params": trainable_parameter_count(model.head)},
        ]
    )
    return layers


def sequence_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    batch, steps, vocab = logits.shape
    losses = F.cross_entropy(logits.reshape(batch * steps, vocab), targets.reshape(batch * steps), reduction="none")
    return losses.view(batch, steps).mean(dim=1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, float]:
    model.eval()
    per_sequence: list[torch.Tensor] = []
    correct = 0
    total = 0
    function_evaluation_seconds = 0.0
    evaluation_start = time.perf_counter()
    for tokens, targets in batches:
        forward_start = time.perf_counter()
        logits = model(tokens)
        function_evaluation_seconds += time.perf_counter() - forward_start
        per_sequence.append(sequence_loss(logits, targets).cpu())
        predictions = logits.argmax(dim=-1)
        correct += (predictions == targets).sum().item()
        total += targets.numel()
    losses = torch.cat(per_sequence)
    return {
        "loss_mean": losses.mean().item(),
        "loss_std_sequence": losses.std(unbiased=True).item(),
        "loss_se_sequence": (losses.std(unbiased=True) / math.sqrt(len(losses))).item(),
        "ppl": math.exp(min(losses.mean().item(), 20.0)),
        "accuracy": correct / total,
        "n_sequences": int(len(losses)),
        "n_tokens": int(total),
        "wall_clock_seconds": time.perf_counter() - evaluation_start,
        "function_evaluation_seconds": function_evaluation_seconds,
        "internal_overhead_seconds": (time.perf_counter() - evaluation_start) - function_evaluation_seconds,
    }


def run_one_seed(
    model_name: str,
    seed: int,
    vocab_size: int,
    splits: dict[str, torch.Tensor],
    valid_batches: list[tuple[torch.Tensor, torch.Tensor]],
    test_batches: list[tuple[torch.Tensor, torch.Tensor]],
    config: ExperimentConfig,
    logger: TimestampLogger,
    device: torch.device,
) -> dict[str, Any]:
    configure_determinism(seed, config.deterministic)
    model = SpectralTransferModel(vocab_size, config, model_name).to(device)
    architecture = describe_architecture(model, config)
    total_params = trainable_parameter_count(model)
    logger.log(f"RUN_START | model={model_name} | seed={seed} | params={total_params:,}")
    for layer in architecture:
        logger.log(f"ARCH_LAYER | model={model_name} | seed={seed} | {json.dumps(layer, ensure_ascii=False, sort_keys=True)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    total_steps = config.epochs * config.steps_per_epoch
    warmup_steps = max(1, int(total_steps * config.warmup_pct))
    train_generator = torch.Generator(device="cpu").manual_seed(10_000 + seed)
    history: list[dict[str, Any]] = []
    best_checkpoint: dict[str, Any] | None = None
    run_start = time.perf_counter()
    global_step = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss_total = 0.0
        epoch_tokens = 0
        epoch_start = time.perf_counter()
        epoch_forward_seconds = 0.0
        epoch_step_seconds = 0.0
        epoch_grad_norm_total = 0.0
        last_grad_norm = 0.0
        for step in range(1, config.steps_per_epoch + 1):
            global_step += 1
            lr_scale = min(1.0, global_step / warmup_steps)
            current_lr = config.lr * lr_scale
            for group in optimizer.param_groups:
                group["lr"] = current_lr

            step_start = time.perf_counter()
            tokens, targets = sample_batch(
                splits["train"], config.batch_size, config.seq_len, train_generator, device
            )
            optimizer.zero_grad(set_to_none=True)
            forward_start = time.perf_counter()
            logits = model(tokens)
            forward_seconds = time.perf_counter() - forward_start
            epoch_forward_seconds += forward_seconds
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip))
            epoch_grad_norm_total += grad_norm
            last_grad_norm = grad_norm
            optimizer.step()
            step_seconds = time.perf_counter() - step_start
            epoch_step_seconds += step_seconds

            token_count = targets.numel()
            epoch_loss_total += loss.item() * token_count
            epoch_tokens += token_count
            if config.log_every > 0 and step % config.log_every == 0:
                logger.log(
                    "TRAIN_STEP | "
                    f"model={model_name} | seed={seed} | epoch={epoch}/{config.epochs} | "
                    f"step={step}/{config.steps_per_epoch} | global_step={global_step}/{total_steps} | "
                    f"loss={loss.item():.6f} | epoch_loss_so_far={epoch_loss_total / epoch_tokens:.6f} | "
                    f"lr={current_lr:.8f} | grad_norm={grad_norm:.6f} | tokens={token_count} | "
                    f"forward_seconds={forward_seconds:.4f} | step_seconds={step_seconds:.4f}"
                )

        valid_metrics = evaluate(model, valid_batches)
        train_loss = epoch_loss_total / epoch_tokens
        epoch_wall_clock_seconds = time.perf_counter() - epoch_start
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_tokens": epoch_tokens,
            "valid": valid_metrics,
            "epoch_wall_clock_seconds": epoch_wall_clock_seconds,
            "function_evaluation_seconds": epoch_forward_seconds + valid_metrics["function_evaluation_seconds"],
            "training_step_seconds": epoch_step_seconds,
            "internal_overhead_seconds": epoch_wall_clock_seconds - epoch_forward_seconds - valid_metrics["function_evaluation_seconds"],
            "mean_grad_norm": epoch_grad_norm_total / config.steps_per_epoch,
            "last_grad_norm": last_grad_norm,
            "final_lr": current_lr,
        }
        history.append(epoch_record)
        logger.log(
            "EPOCH_END | "
            f"model={model_name} | seed={seed} | epoch={epoch}/{config.epochs} | "
            f"train_loss={train_loss:.6f} | valid_loss={valid_metrics['loss_mean']:.6f} | "
            f"valid_ppl={valid_metrics['ppl']:.4f} | valid_acc={valid_metrics['accuracy']:.6f} | "
            f"valid_sequences={valid_metrics['n_sequences']} | valid_loss_se_sequence={valid_metrics['loss_se_sequence']:.6f} | "
            f"epoch_seconds={epoch_record['epoch_wall_clock_seconds']:.2f} | "
            f"function_evaluation_seconds={epoch_record['function_evaluation_seconds']:.2f} | "
            f"internal_overhead_seconds={epoch_record['internal_overhead_seconds']:.2f} | "
            f"mean_grad_norm={epoch_record['mean_grad_norm']:.6f} | "
            f"last_grad_norm={epoch_record['last_grad_norm']:.6f} | final_lr={epoch_record['final_lr']:.8f}"
        )
        if best_checkpoint is None or valid_metrics["loss_mean"] < best_checkpoint["valid"]["loss_mean"]:
            best_checkpoint = {
                "epoch": epoch,
                "valid": valid_metrics,
                "state_dict": copy.deepcopy(model.state_dict()),
            }
            logger.log(
                f"CHECKPOINT | model={model_name} | seed={seed} | selected_epoch={epoch} | "
                f"criterion=minimum_valid_loss | valid_loss={valid_metrics['loss_mean']:.6f}"
            )

    assert best_checkpoint is not None
    model.load_state_dict(best_checkpoint.pop("state_dict"))
    test_metrics = evaluate(model, test_batches)
    elapsed = time.perf_counter() - run_start
    logger.log(
        "TEST_RESULT | "
        f"model={model_name} | seed={seed} | selected_epoch={best_checkpoint['epoch']} | "
        f"test_loss={test_metrics['loss_mean']:.6f} | test_ppl={test_metrics['ppl']:.4f} | "
        f"test_acc={test_metrics['accuracy']:.6f} | test_sequences={test_metrics['n_sequences']} | "
        f"test_loss_se_sequence={test_metrics['loss_se_sequence']:.6f} | run_seconds={elapsed:.2f}"
    )
    return {
        "model": model_name,
        "seed": seed,
        "params": total_params,
        "architecture": architecture,
        "history": history,
        "selected_checkpoint": best_checkpoint,
        "test": test_metrics,
        "wall_clock_seconds": elapsed,
        "function_evaluation_seconds": sum(item["function_evaluation_seconds"] for item in history),
        "internal_overhead_seconds": sum(item["internal_overhead_seconds"] for item in history),
    }


def mean_and_se(values: list[float]) -> tuple[float, float, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    if len(tensor) == 1:
        return tensor.item(), 0.0, 0.0
    std = tensor.std(unbiased=True).item()
    return tensor.mean().item(), std, std / math.sqrt(len(tensor))


def summarize_by_model(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in seed_results:
        grouped.setdefault(result["model"], []).append(result)
    summary: dict[str, Any] = {}
    for model_name, runs in grouped.items():
        mean_loss, std_loss, se_loss = mean_and_se([run["test"]["loss_mean"] for run in runs])
        mean_ppl, std_ppl, se_ppl = mean_and_se([run["test"]["ppl"] for run in runs])
        mean_seconds, _, _ = mean_and_se([run["wall_clock_seconds"] for run in runs])
        mean_function_seconds, _, _ = mean_and_se([run["function_evaluation_seconds"] for run in runs])
        mean_overhead_seconds, _, _ = mean_and_se([run["internal_overhead_seconds"] for run in runs])
        summary[model_name] = {
            "n_seeds": len(runs),
            "params": runs[0]["params"],
            "test_loss_mean_across_seeds": mean_loss,
            "test_loss_std_across_seeds": std_loss,
            "test_loss_se_across_seeds": se_loss,
            "test_ppl_mean_across_seeds": mean_ppl,
            "test_ppl_std_across_seeds": std_ppl,
            "test_ppl_se_across_seeds": se_ppl,
            "mean_wall_clock_seconds": mean_seconds,
            "mean_function_evaluation_seconds": mean_function_seconds,
            "mean_internal_overhead_seconds": mean_overhead_seconds,
            "selected_epochs": [run["selected_checkpoint"]["epoch"] for run in runs],
        }
    return summary


def write_results(payload: dict[str, Any], config: ExperimentConfig, logger: TimestampLogger) -> Path:
    RAW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = RAW_RESULTS_DIR / f"{config.experiment_id}_{stamp}.json"
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    logger.log(f"ARTIFACT | raw_results={destination}")
    return destination


def append_ledger(summary: dict[str, Any], config: ExperimentConfig, logger: TimestampLogger) -> None:
    candidate = summary["lerp_fwht_dct"]
    ledger_entry = {
        "experiment_id": config.experiment_id,
        "fecha": datetime.now(timezone.utc).date().isoformat(),
        "familia": "spectral_transfer_control",
        "dataset": "Tiny Shakespeare char; temporal split 70/15/15",
        "n_eval": config.test_batches * config.batch_size,
        "metric_name": "test_loss_best_valid_checkpoint",
        "value": round(candidate["test_loss_mean_across_seeds"], 6),
        "SE": round(candidate["test_loss_se_across_seeds"], 6),
        "params": candidate["params"],
        "nivel_rigor": config.rigor_level,
        "etiqueta": "SEÑAL",
    }
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(ledger_entry, ensure_ascii=False) + "\n")
    logger.log_json("LEDGER_ENTRY", ledger_entry)


def main() -> int:
    logger = TimestampLogger()
    try:
        started_utc = datetime.now(timezone.utc).isoformat()
        args = parse_args()
        config = build_config(args)
        device = torch.device(config.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Se solicitó CUDA pero torch.cuda.is_available() es False.")

        logger.log("=" * 96)
        logger.log(f"EXPERIMENT_START | {config.experiment_id} — Transferencia espectral controlada en lenguaje real")
        logger.log("QUESTION | ¿FWHT/DCT-II superan a una base ortogonal aleatoria fija bajo controles iguales?")
        logger.log("CLAIM_BOUNDARY | Nivel 1/2 según modo; el script no declara ganadores ni transferencia a LLMs.")
        logger.log_json("METADATA", {"started_utc": started_utc, "argv": sys.argv, **hardware_metadata(device)})
        logger.log_json("FULL_CONFIG", asdict(config))
        logger.log("PROTOCOL | split temporal 70/15/15; checkpoint por validación; test una sola vez por checkpoint.")
        logger.log("PROTOCOL | candidate=lerp_fwht_dct se ejecuta primero; random_orthogonal es el control crítico.")
        logger.log("=" * 96)

        corpus = load_tiny_shakespeare(logger)
        corpus_sha256 = hashlib.sha256(corpus.encode("utf-8")).hexdigest()
        tokens, stoi = encode_corpus(corpus)
        splits = split_tokens(tokens, config)
        logger.log(
            f"DATA_SPLIT | vocab_size={len(stoi)} | train_tokens={len(splits['train']):,} | "
            f"valid_tokens={len(splits['valid']):,} | test_tokens={len(splits['test']):,} | "
            f"corpus_sha256={corpus_sha256}"
        )
        valid_batches = build_fixed_eval_batches(
            splits["valid"], config.batch_size, config.seq_len, config.valid_batches, 330_001, device
        )
        test_batches = build_fixed_eval_batches(
            splits["test"], config.batch_size, config.seq_len, config.test_batches, 330_002, device
        )
        logger.log(
            f"EVAL_PLAN | valid_sequences={config.valid_batches * config.batch_size} | "
            f"test_sequences={config.test_batches * config.batch_size} | non_overlapping_fixed_windows=true"
        )

        seed_results: list[dict[str, Any]] = []
        for model_name in config.model_names:
            for seed in config.seeds:
                seed_results.append(
                    run_one_seed(
                        model_name,
                        seed,
                        len(stoi),
                        splits,
                        valid_batches,
                        test_batches,
                        config,
                        logger,
                        device,
                    )
                )

        summary = summarize_by_model(seed_results)
        logger.log_json("SUMMARY_BY_MODEL", summary)
        payload = {
            "experiment_id": config.experiment_id,
            "started_utc": started_utc,
            "metadata": {"argv": sys.argv, **hardware_metadata(device)},
            "config": asdict(config),
            "dataset": {
                "name": "Tiny Shakespeare (character-level)",
                "source_url": TINY_SHAKESPEARE_URL,
                "local_path": str(DATA_PATH),
                "sha256": corpus_sha256,
                "vocab_size": len(stoi),
                "split_tokens": {name: len(split) for name, split in splits.items()},
            },
            "question": "¿FWHT/DCT-II superan a una base ortogonal aleatoria fija bajo controles iguales?",
            "seed_results": seed_results,
            "summary_by_model": summary,
        }
        write_results(payload, config, logger)
        append_ledger(summary, config, logger)
        logger.log("EXPERIMENT_COMPLETE | status=success | next_step=inspeccionar_JSON_antes_de_redactar_findings")
        return 0
    except Exception as error:
        logger.log(f"EXPERIMENT_ERROR | type={type(error).__name__} | message={error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
