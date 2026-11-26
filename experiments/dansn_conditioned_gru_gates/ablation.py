"""Controlled ablations for where DANSN conditions a GRU.

Variants:
- readout_fusion: existing PersistentDANSNRNN baseline.
- reset_only: DANSN context enters only the GRU reset gate.
- update_only: DANSN context enters only the GRU update gate.
- full: DANSN context enters reset, update, and candidate state.

All gate-conditioned variants instantiate identical parameter shapes. Inactive
context projections are masked to zero, preserving the total parameter count
and isolating the location of graph conditioning.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Sequence

import torch

from agi import Brain
from benchmark_continual_learning import (
    TEST_TEXT,
    TRAIN_TEXT,
    PlainGRU,
    evaluate_plain_gru,
    train_plain_gru,
)
from persistent_dansn_rnn import (
    PersistentDANSNRNN,
    evaluate_prequential,
    train_stream,
)
from experiments.dansn_conditioned_gru_gates import DANSNConditionedGRU


ABLATIONS = ("reset_only", "update_only", "full")


def _parameter_count(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


@torch.no_grad()
def gate_diagnostics(model: DANSNConditionedGRU, token_ids: Sequence[int]) -> dict:
    """Measure reset/update gate distributions without revealing test targets."""
    if len(token_ids) < 2:
        raise ValueError("token_ids must contain at least two tokens")

    was_training = model.training
    model.eval()
    state = model.initial_state()
    history = [int(token_ids[0])]
    reset_values = []
    update_values = []
    readout_norms = []

    for raw in token_ids[:-1]:
        token = int(raw)
        _logits, state, trace = model.forward_step(token, history, state)
        reset_values.append(trace["reset_gate"].detach().reshape(-1).cpu())
        update_values.append(trace["update_gate"].detach().reshape(-1).cpu())
        readout_norms.append(
            float(torch.linalg.vector_norm(trace["graph_readout"]).cpu())
        )
        # Append only the observed current token. No target is exposed through
        # graph writes; this is a frozen-memory diagnostic pass.
        history.append(int(token_ids[len(history)]))

    model.train(was_training)

    def summarize(values):
        flat = torch.cat(values) if values else torch.zeros(1)
        return {
            "mean": float(flat.mean()),
            "std": float(flat.std(unbiased=False)),
            "saturation_below_0_05": float((flat < 0.05).float().mean()),
            "saturation_above_0_95": float((flat > 0.95).float().mean()),
        }

    return {
        "reset_gate": summarize(reset_values),
        "update_gate": summarize(update_values),
        "graph_readout_norm_mean": (
            sum(readout_norms) / len(readout_norms) if readout_norms else 0.0
        ),
        "diagnostic_tokens": len(reset_values),
    }


def _new_brain(vocab):
    return Brain(vocab=copy.deepcopy(vocab))


def _train_hybrid(vocab, train_ids, test_ids, *, seed, epochs,
                  hidden_size, chunk_size, window_size, learning_rate,
                  ablation: str):
    # Reset the random seed before every architecture. The three gate
    # ablations share identical tensor shapes and therefore begin with the
    # same initialization for a given seed.
    torch.manual_seed(seed + 101)
    brain = _new_brain(vocab)

    if ablation == "readout_fusion":
        model = PersistentDANSNRNN(
            brain.engine,
            len(vocab.word_to_id) + 1,
            hidden_size=hidden_size,
            window_size=window_size,
            graph_depth=3,
            message_rounds=1,
            max_active_nodes=128,
            max_memory_nodes=10_000,
        )
    elif ablation in ABLATIONS:
        model = DANSNConditionedGRU(
            brain.engine,
            len(vocab.word_to_id) + 1,
            hidden_size=hidden_size,
            window_size=window_size,
            graph_depth=3,
            message_rounds=1,
            max_active_nodes=128,
            max_memory_nodes=10_000,
            gate_ablation=ablation,
        )
    else:
        raise ValueError(f"unknown ablation: {ablation}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train_metrics = [
        train_stream(model, train_ids, optimizer, chunk_size)
        for _ in range(epochs)
    ]

    # Static evaluation and gate diagnostics each use isolated copies to avoid
    # one diagnostic pass changing LRU/access metadata for another.
    static_model = copy.deepcopy(model)
    static = evaluate_prequential(static_model, test_ids, update_graph=False)

    diagnostics = None
    if ablation in ABLATIONS:
        diagnostics_model = copy.deepcopy(model)
        diagnostics = gate_diagnostics(diagnostics_model, test_ids)

    # Adaptive evaluation scores each target before neural/graph memory writes.
    adaptive = train_stream(model, test_ids, optimizer, chunk_size)

    result = {
        "test_static": static,
        "test_adaptive": adaptive,
        "last_train": train_metrics[-1],
        "parameter_count": _parameter_count(model),
        "graph": model.graph_stats(),
    }
    if diagnostics is not None:
        result["gate_diagnostics"] = diagnostics
    return result


def run_benchmark(
    seed: int = 7,
    epochs: int = 3,
    hidden_size: int = 48,
    chunk_size: int = 8,
    window_size: int = 8,
    learning_rate: float = 2e-3,
) -> dict:
    if epochs < 1 or hidden_size < 1 or chunk_size < 1 or window_size < 2:
        raise ValueError("epochs/hidden_size/chunk_size must be positive; window_size >= 2")

    torch.manual_seed(seed)
    vocab_brain = Brain()
    train_ids = vocab_brain.vocab.tokenize(TRAIN_TEXT)
    test_ids = vocab_brain.vocab.tokenize(TEST_TEXT)
    vocab = vocab_brain.vocab
    vocab_size = len(vocab.word_to_id) + 1

    torch.manual_seed(seed + 7)
    plain = PlainGRU(vocab_size, hidden_size)
    plain_optimizer = torch.optim.AdamW(plain.parameters(), lr=learning_rate)
    plain_train = [
        train_plain_gru(plain, train_ids, plain_optimizer, chunk_size)
        for _ in range(epochs)
    ]
    plain_static = evaluate_plain_gru(plain, test_ids)
    plain_adaptive = train_plain_gru(
        plain, test_ids, plain_optimizer, chunk_size
    )
    results = {
        "plain_gru": {
            "test_static": plain_static,
            "test_adaptive": plain_adaptive,
            "last_train": plain_train[-1],
            "parameter_count": _parameter_count(plain),
        }
    }

    for name in ("readout_fusion", *ABLATIONS):
        results[name] = _train_hybrid(
            vocab,
            train_ids,
            test_ids,
            seed=seed,
            epochs=epochs,
            hidden_size=hidden_size,
            chunk_size=chunk_size,
            window_size=window_size,
            learning_rate=learning_rate,
            ablation=name,
        )

    return {
        "protocol": {
            "seed": seed,
            "epochs": epochs,
            "hidden_size": hidden_size,
            "chunk_size": chunk_size,
            "window_size": window_size,
            "train_tokens": len(train_ids),
            "test_tokens": len(test_ids),
            "static": "held-out targets scored with train-end model and graph memory frozen",
            "adaptive": "score each target before neural and graph updates",
            "gate_ablation_parameter_control": (
                "reset_only, update_only, and full use identical model shapes and "
                "parameter counts; inactive graph-context pathways are zero-masked"
            ),
            "initialization": (
                "reset_only, update_only, and full use the same seed-specific "
                "parameter initialization"
            ),
        },
        "results": results,
        "interpretation": (
            "Exploratory synthetic ablation pilot. Compare static with static and "
            "adaptive with adaptive. The small corpus and three-seed design do not "
            "establish general superiority."
        ),
    }


def run_multi_seed(
    seeds: Sequence[int] = (7, 13, 21),
    epochs: int = 3,
    hidden_size: int = 48,
    chunk_size: int = 8,
    window_size: int = 8,
    learning_rate: float = 2e-3,
) -> dict:
    seed_list = [int(seed) for seed in seeds]
    if not seed_list:
        raise ValueError("at least one seed is required")

    runs = [
        run_benchmark(
            seed=seed,
            epochs=epochs,
            hidden_size=hidden_size,
            chunk_size=chunk_size,
            window_size=window_size,
            learning_rate=learning_rate,
        )
        for seed in seed_list
    ]

    systems = ("plain_gru", "readout_fusion", *ABLATIONS)

    def stats(values):
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        return {
            "mean": mean,
            "population_std": math.sqrt(variance),
            "runs": len(values),
        }

    summary = {}
    for system in systems:
        result_rows = [run["results"][system] for run in runs]
        summary[system] = {
            "static_perplexity": stats([
                row["test_static"]["perplexity"] for row in result_rows
            ]),
            "adaptive_perplexity": stats([
                row["test_adaptive"]["perplexity"] for row in result_rows
            ]),
            "parameter_count": stats([
                float(row["parameter_count"]) for row in result_rows
            ]),
            "graph_nodes": stats([
                float(row.get("graph", {}).get("nodes", 0))
                for row in result_rows
            ]),
            "persistent_memory_nodes": stats([
                float(row.get("graph", {}).get("memory_nodes", 0))
                for row in result_rows
            ]),
        }
        if system in ABLATIONS:
            for gate_name in ("reset_gate", "update_gate"):
                for metric in (
                    "mean",
                    "std",
                    "saturation_below_0_05",
                    "saturation_above_0_95",
                ):
                    summary[system].setdefault("gate_diagnostics", {}).setdefault(
                        gate_name, {}
                    )[metric] = stats([
                        row["gate_diagnostics"][gate_name][metric]
                        for row in result_rows
                    ])
            summary[system]["gate_diagnostics"]["graph_readout_norm_mean"] = stats([
                row["gate_diagnostics"]["graph_readout_norm_mean"]
                for row in result_rows
            ])

    return {
        "protocol": {
            "seeds": seed_list,
            "epochs": epochs,
            "hidden_size": hidden_size,
            "chunk_size": chunk_size,
            "window_size": window_size,
        },
        "summary": summary,
        "runs": runs,
        "interpretation": (
            "Three-seed synthetic ablation pilot only. Gate-conditioned variants "
            "are parameter-matched with each other; readout-fusion and plain GRU "
            "are not parameter-matched to those variants."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="7,13,21")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--hidden-size", type=int, default=48)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    results = run_multi_seed(
        seeds=seeds,
        epochs=args.epochs,
        hidden_size=args.hidden_size,
        chunk_size=args.chunk_size,
        window_size=args.window_size,
        learning_rate=args.learning_rate,
    )
    rendered = json.dumps(results, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
