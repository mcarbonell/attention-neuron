"""v331 — Ablación causal: dos ramas frente a bases espectrales.

Pregunta: a igual FFN de dos ramas, ¿la pareja FWHT+DCT-II mejora a dos
bases ortogonales aleatorias independientes? El candidato se ejecuta primero.
Este archivo reutiliza funciones de datos/evaluación de v330b, pero no modifica
el experimento previo. Cada salida lleva timestamp relativo y el JSON conserva
configuración, arquitectura, semillas/huellas de bases e historia por época.

Uso:
    python scratch/prototype_v331_two_branch_basis_ablation.py --mode pilot
    python scratch/prototype_v331_two_branch_basis_ablation.py --mode level2
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# v330b es un harness de datos/entrenamiento ya comprobado. Se importa sin editarlo.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import prototype_v330_spectral_transfer_control as base


EXPERIMENT_ID = "v331_two_branch_basis_ablation"
MODEL_ORDER = (
    "lerp_fwht_dct",
    "lerp_random_pair",
    "lerp_random_tied",
    "lerp_fwht_random",
    "lerp_dct_random",
    "dense_ffn",
)
_ACTIVE_TRAINING_SEED: int | None = None


class TwoBranchSpectralFFN(nn.Module):
    """Misma capacidad Lerp para todos los pares; sólo cambian buffers congelados."""

    def __init__(self, basis0: torch.Tensor, basis1: torch.Tensor) -> None:
        super().__init__()
        if basis0.shape != basis1.shape or basis0.ndim != 2 or basis0.shape[0] != basis0.shape[1]:
            raise ValueError("Las dos bases deben ser matrices cuadradas de igual dimensión.")
        d_model = basis0.shape[0]
        self.register_buffer("basis0", basis0, persistent=False)
        self.register_buffer("basis1", basis1, persistent=False)
        self.substrate_logits = nn.Parameter(torch.zeros(2))
        self.phase_cos = nn.Parameter(torch.zeros(2, d_model))
        self.phase_sin = nn.Parameter(torch.zeros(2, d_model))
        self.amplitude_cos = nn.Parameter(torch.ones(2, d_model))
        self.amplitude_sin = nn.Parameter(torch.ones(2, d_model))
        self.combine = nn.Linear(2 * d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.substrate_logits, dim=0)
        frequency0 = F.linear(x, self.basis0)
        frequency1 = F.linear(x, self.basis1)
        branch0 = (
            torch.cos(frequency0 + self.phase_cos[0]) * self.amplitude_cos[0]
            + torch.sin(frequency0 + self.phase_sin[0]) * self.amplitude_sin[0]
        )
        branch1 = (
            torch.cos(frequency1 + self.phase_cos[1]) * self.amplitude_cos[1]
            + torch.sin(frequency1 + self.phase_sin[1]) * self.amplitude_sin[1]
        )
        output0 = F.linear(branch0, self.basis0.t())
        output1 = F.linear(branch1, self.basis1.t())
        return self.combine(torch.cat((weights[0] * output0, weights[1] * output1), dim=-1))


def random_basis_seed(config: base.ExperimentConfig, training_seed: int, layer_index: int, branch_index: int) -> int:
    """Comparte R0/R1 entre condiciones de una semilla; los cambia entre semillas."""
    return config.random_basis_seed + training_seed * 10_000 + layer_index * 10 + branch_index


def matrix_sha256(matrix: torch.Tensor) -> str:
    return hashlib.sha256(matrix.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def make_basis_pair(
    model_name: str, config: base.ExperimentConfig, training_seed: int, layer_index: int
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    d_model = config.d_model
    fwht = base.create_hadamard_matrix(d_model)
    dct = base.create_dct2_matrix(d_model)
    r0_seed = random_basis_seed(config, training_seed, layer_index, 0)
    r1_seed = random_basis_seed(config, training_seed, layer_index, 1)
    r0 = base.create_random_orthogonal_matrix(d_model, r0_seed)
    r1 = base.create_random_orthogonal_matrix(d_model, r1_seed)

    if model_name == "lerp_fwht_dct":
        pair, labels, seeds = (fwht, dct), ("fwht", "dct2"), (None, None)
    elif model_name == "lerp_random_pair":
        pair, labels, seeds = (r0, r1), ("random_r0", "random_r1"), (r0_seed, r1_seed)
    elif model_name == "lerp_random_tied":
        pair, labels, seeds = (r0, r0), ("random_r0", "random_r0_tied"), (r0_seed, r0_seed)
    elif model_name == "lerp_fwht_random":
        pair, labels, seeds = (fwht, r0), ("fwht", "random_r0"), (None, r0_seed)
    elif model_name == "lerp_dct_random":
        pair, labels, seeds = (dct, r0), ("dct2", "random_r0"), (None, r0_seed)
    else:
        raise ValueError(f"No hay par de bases para {model_name}.")

    metadata = [
        {"branch": index, "label": label, "seed": seed, "sha256": matrix_sha256(matrix)}
        for index, (matrix, label, seed) in enumerate(zip(pair, labels, seeds))
    ]
    return pair[0], pair[1], metadata


def basis_metadata(model_name: str, config: base.ExperimentConfig, training_seed: int) -> list[dict[str, Any]]:
    if model_name == "dense_ffn":
        return []
    return [
        {"layer_index": layer_index, "branches": make_basis_pair(model_name, config, training_seed, layer_index)[2]}
        for layer_index in range(config.n_layers)
    ]


class V331Model(base.SpectralTransferModel):
    def _build_ffn(self, config: base.ExperimentConfig, model_name: str, layer_index: int) -> nn.Module:
        if model_name == "dense_ffn":
            return base.DenseFFN(config.d_model, self.dense_hidden_dim)
        if _ACTIVE_TRAINING_SEED is None:
            raise RuntimeError("La semilla activa debe fijarse antes de construir el modelo.")
        basis0, basis1, _ = make_basis_pair(model_name, config, _ACTIVE_TRAINING_SEED, layer_index)
        return TwoBranchSpectralFFN(basis0, basis1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v331 two-branch spectral basis ablation")
    parser.add_argument("--mode", choices=("pilot", "level2"), default="pilot")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, etc.")
    parser.add_argument("--models", default=",".join(MODEL_ORDER))
    parser.add_argument("--seeds", default=None, help="Sobrescribe semillas, por ejemplo 10,20,30,42,100")
    parser.add_argument("--log-every", type=int, default=0, help="N>0 añade un TRAIN_STEP cada N pasos después de los 5 pasos iniciales.")
    parser.add_argument("--run-id", default=EXPERIMENT_ID)
    parser.add_argument("--no-deterministic", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> base.ExperimentConfig:
    mode_config = dict(base.MODE_CONFIGS[args.mode])
    model_names = tuple(item.strip() for item in args.models.split(",") if item.strip())
    unknown = sorted(set(model_names) - set(MODEL_ORDER))
    if unknown or not model_names:
        raise ValueError(f"Modelos inválidos: {unknown or 'lista vacía'}")
    if model_names[0] != "lerp_fwht_dct":
        raise ValueError("La Regla de Oro exige ejecutar lerp_fwht_dct primero.")
    if args.seeds:
        mode_config["seeds"] = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
    if args.epochs is not None:
        mode_config["epochs"] = args.epochs
    if args.steps_per_epoch is not None:
        mode_config["steps_per_epoch"] = args.steps_per_epoch
    if args.log_every < 0:
        raise ValueError("--log-every debe ser >= 0.")
    return base.ExperimentConfig(
        experiment_id=args.run_id,
        mode=args.mode,
        model_names=model_names,
        device=args.device,
        deterministic=not args.no_deterministic,
        log_every=args.log_every,
        train_split=0.70,
        valid_split=0.15,
        random_basis_seed=331,
        **mode_config,
    )


def run_one_seed(
    model_name: str,
    seed: int,
    vocab_size: int,
    splits: dict[str, torch.Tensor],
    valid_batches: list[tuple[torch.Tensor, torch.Tensor]],
    test_batches: list[tuple[torch.Tensor, torch.Tensor]],
    config: base.ExperimentConfig,
    logger: base.TimestampLogger,
    device: torch.device,
) -> dict[str, Any]:
    global _ACTIVE_TRAINING_SEED
    _ACTIVE_TRAINING_SEED = seed
    base.configure_determinism(seed, config.deterministic)
    planned_bases = basis_metadata(model_name, config, seed)
    logger.log_json("BASIS_PLAN", {"model": model_name, "training_seed": seed, "layers": planned_bases})
    model = V331Model(vocab_size, config, model_name).to(device)
    architecture = base.describe_architecture(model, config)
    total_params = base.trainable_parameter_count(model)
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
        epoch_loss_total = epoch_forward_seconds = epoch_step_seconds = epoch_grad_norm_total = 0.0
        epoch_tokens = 0
        last_grad_norm = 0.0
        epoch_start = time.perf_counter()
        for step in range(1, config.steps_per_epoch + 1):
            global_step += 1
            current_lr = config.lr * min(1.0, global_step / warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = current_lr
            step_start = time.perf_counter()
            tokens, targets = base.sample_batch(splits["train"], config.batch_size, config.seq_len, train_generator, device)
            optimizer.zero_grad(set_to_none=True)
            forward_start = time.perf_counter()
            logits = model(tokens)
            forward_seconds = time.perf_counter() - forward_start
            epoch_forward_seconds += forward_seconds
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip))
            optimizer.step()
            step_seconds = time.perf_counter() - step_start
            epoch_step_seconds += step_seconds
            epoch_grad_norm_total += grad_norm
            last_grad_norm = grad_norm
            token_count = targets.numel()
            epoch_loss_total += loss.item() * token_count
            epoch_tokens += token_count
            if (epoch == 1 and step <= 5) or (config.log_every > 0 and step % config.log_every == 0):
                logger.log(
                    f"TRAIN_STEP | model={model_name} | seed={seed} | epoch={epoch}/{config.epochs} | "
                    f"step={step}/{config.steps_per_epoch} | global_step={global_step}/{total_steps} | "
                    f"loss={loss.item():.6f} | epoch_loss_so_far={epoch_loss_total / epoch_tokens:.6f} | "
                    f"lr={current_lr:.8f} | grad_norm={grad_norm:.6f} | tokens={token_count} | "
                    f"forward_seconds={forward_seconds:.4f} | step_seconds={step_seconds:.4f}"
                )

        valid_metrics = base.evaluate(model, valid_batches)
        epoch_wall_clock_seconds = time.perf_counter() - epoch_start
        epoch_record = {
            "epoch": epoch, "train_loss": epoch_loss_total / epoch_tokens, "train_tokens": epoch_tokens,
            "valid": valid_metrics, "epoch_wall_clock_seconds": epoch_wall_clock_seconds,
            "function_evaluation_seconds": epoch_forward_seconds + valid_metrics["function_evaluation_seconds"],
            "training_step_seconds": epoch_step_seconds,
            "internal_overhead_seconds": epoch_wall_clock_seconds - epoch_forward_seconds - valid_metrics["function_evaluation_seconds"],
            "mean_grad_norm": epoch_grad_norm_total / config.steps_per_epoch,
            "last_grad_norm": last_grad_norm, "final_lr": current_lr,
        }
        history.append(epoch_record)
        logger.log(
            f"EPOCH_END | model={model_name} | seed={seed} | epoch={epoch}/{config.epochs} | "
            f"train_loss={epoch_record['train_loss']:.6f} | valid_loss={valid_metrics['loss_mean']:.6f} | "
            f"valid_ppl={valid_metrics['ppl']:.4f} | valid_acc={valid_metrics['accuracy']:.6f} | "
            f"valid_sequences={valid_metrics['n_sequences']} | valid_loss_se_sequence={valid_metrics['loss_se_sequence']:.6f} | "
            f"epoch_seconds={epoch_wall_clock_seconds:.2f} | function_evaluation_seconds={epoch_record['function_evaluation_seconds']:.2f} | "
            f"internal_overhead_seconds={epoch_record['internal_overhead_seconds']:.2f} | "
            f"mean_grad_norm={epoch_record['mean_grad_norm']:.6f} | last_grad_norm={last_grad_norm:.6f} | final_lr={current_lr:.8f}"
        )
        if best_checkpoint is None or valid_metrics["loss_mean"] < best_checkpoint["valid"]["loss_mean"]:
            best_checkpoint = {"epoch": epoch, "valid": valid_metrics, "state_dict": copy.deepcopy(model.state_dict())}
            logger.log(f"CHECKPOINT | model={model_name} | seed={seed} | selected_epoch={epoch} | criterion=minimum_valid_loss | valid_loss={valid_metrics['loss_mean']:.6f}")

    assert best_checkpoint is not None
    model.load_state_dict(best_checkpoint.pop("state_dict"))
    test_metrics = base.evaluate(model, test_batches)
    elapsed = time.perf_counter() - run_start
    logger.log(
        f"TEST_RESULT | model={model_name} | seed={seed} | selected_epoch={best_checkpoint['epoch']} | "
        f"test_loss={test_metrics['loss_mean']:.6f} | test_ppl={test_metrics['ppl']:.4f} | test_acc={test_metrics['accuracy']:.6f} | "
        f"test_sequences={test_metrics['n_sequences']} | test_loss_se_sequence={test_metrics['loss_se_sequence']:.6f} | run_seconds={elapsed:.2f}"
    )
    _ACTIVE_TRAINING_SEED = None
    return {
        "model": model_name, "seed": seed, "params": total_params, "architecture": architecture,
        "basis_metadata": planned_bases, "history": history, "selected_checkpoint": best_checkpoint, "test": test_metrics,
        "wall_clock_seconds": elapsed,
        "function_evaluation_seconds": sum(item["function_evaluation_seconds"] for item in history),
        "internal_overhead_seconds": sum(item["internal_overhead_seconds"] for item in history),
    }


def paired_comparisons(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = (("lerp_fwht_dct", "lerp_random_pair"), ("lerp_random_pair", "lerp_random_tied"),
             ("lerp_fwht_random", "lerp_random_pair"), ("lerp_dct_random", "lerp_random_pair"),
             ("lerp_fwht_dct", "dense_ffn"))
    by_model = {name: sorted((run for run in seed_results if run["model"] == name), key=lambda run: run["seed"]) for name in MODEL_ORDER}
    report: dict[str, Any] = {}
    for left, right in pairs:
        if not by_model[left] or not by_model[right]:
            continue
        left_by_seed = {run["seed"]: run for run in by_model[left]}
        right_by_seed = {run["seed"]: run for run in by_model[right]}
        common = sorted(set(left_by_seed) & set(right_by_seed))
        deltas = [left_by_seed[seed]["test"]["loss_mean"] - right_by_seed[seed]["test"]["loss_mean"] for seed in common]
        mean, std, se = base.mean_and_se(deltas)
        report[f"{left}_minus_{right}"] = {"left": left, "right": right, "seeds": common, "deltas": deltas, "mean_delta": mean, "paired_sd": std, "paired_se": se, "two_se": 2 * se}
    return report


def append_ledger(summary: dict[str, Any], config: base.ExperimentConfig, logger: base.TimestampLogger) -> None:
    candidate = summary["lerp_fwht_dct"]
    entry = {"experiment_id": config.experiment_id, "fecha": datetime.now(timezone.utc).date().isoformat(),
             "familia": "two_branch_basis_ablation", "dataset": "Tiny Shakespeare char; temporal split 70/15/15",
             "n_eval": config.test_batches * config.batch_size, "metric_name": "test_loss_best_valid_checkpoint",
             "value": round(candidate["test_loss_mean_across_seeds"], 6), "SE": round(candidate["test_loss_se_across_seeds"], 6),
             "params": candidate["params"], "nivel_rigor": config.rigor_level, "etiqueta": "SEÑAL"}
    base.LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with base.LEDGER_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.log_json("LEDGER_ENTRY", entry)


def main() -> int:
    logger = base.TimestampLogger()
    try:
        started_utc = datetime.now(timezone.utc).isoformat()
        args = parse_args()
        config = build_config(args)
        device = torch.device(config.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Se solicitó CUDA pero torch.cuda.is_available() es False.")
        logger.log("=" * 96)
        logger.log(f"EXPERIMENT_START | {config.experiment_id} — Ablación causal de dos ramas y bases")
        logger.log("QUESTION | ¿FWHT+DCT-II mejoran a RandomPair a igual topología Lerp de dos ramas?")
        logger.log("CLAIM_BOUNDARY | Nivel 1/2 según modo; no declara ganadores ni transferencia a BPE/LLMs.")
        logger.log_json("METADATA", {"started_utc": started_utc, "argv": sys.argv, **base.hardware_metadata(device)})
        logger.log_json("FULL_CONFIG", asdict(config))
        logger.log("PROTOCOL | candidate primero; split temporal 70/15/15; checkpoint por validación; test una sola vez.")
        logger.log("PROTOCOL | RandomPair es el control crítico; RandomTied separa dos ramas de diversidad de base.")
        logger.log("=" * 96)
        corpus = base.load_tiny_shakespeare(logger)
        corpus_sha256 = hashlib.sha256(corpus.encode("utf-8")).hexdigest()
        tokens, stoi = base.encode_corpus(corpus)
        splits = base.split_tokens(tokens, config)
        logger.log(f"DATA_SPLIT | vocab_size={len(stoi)} | train_tokens={len(splits['train']):,} | valid_tokens={len(splits['valid']):,} | test_tokens={len(splits['test']):,} | corpus_sha256={corpus_sha256}")
        valid_batches = base.build_fixed_eval_batches(splits["valid"], config.batch_size, config.seq_len, config.valid_batches, 331_001, device)
        test_batches = base.build_fixed_eval_batches(splits["test"], config.batch_size, config.seq_len, config.test_batches, 331_002, device)
        logger.log(f"EVAL_PLAN | valid_sequences={config.valid_batches * config.batch_size} | test_sequences={config.test_batches * config.batch_size} | non_overlapping_fixed_windows=true")
        seed_results = [run_one_seed(name, seed, len(stoi), splits, valid_batches, test_batches, config, logger, device) for name in config.model_names for seed in config.seeds]
        summary = base.summarize_by_model(seed_results)
        paired = paired_comparisons(seed_results)
        logger.log_json("SUMMARY_BY_MODEL", summary)
        logger.log_json("PAIRED_COMPARISONS", paired)
        payload = {"experiment_id": config.experiment_id, "started_utc": started_utc,
                   "metadata": {"argv": sys.argv, **base.hardware_metadata(device)}, "config": asdict(config),
                   "dataset": {"name": "Tiny Shakespeare (character-level)", "source_url": base.TINY_SHAKESPEARE_URL,
                               "local_path": str(base.DATA_PATH), "sha256": corpus_sha256, "vocab_size": len(stoi),
                               "split_tokens": {name: len(split) for name, split in splits.items()}},
                   "question": "¿FWHT+DCT-II mejoran a RandomPair a igual topología Lerp de dos ramas?",
                   "seed_results": seed_results, "summary_by_model": summary, "paired_comparisons": paired}
        base.write_results(payload, config, logger)
        append_ledger(summary, config, logger)
        logger.log("EXPERIMENT_COMPLETE | status=success | next_step=inspeccionar_JSON_antes_de_redactar_findings")
        return 0
    except Exception as error:
        logger.log(f"EXPERIMENT_ERROR | type={type(error).__name__} | message={error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
