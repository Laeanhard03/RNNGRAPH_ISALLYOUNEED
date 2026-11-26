"""Matched-stream comparison of plain GRU, readout-fusion DANSN-GRU,
and DANSN-conditioned GRU gates.

This is a small synthetic pilot harness, not a general language benchmark.
All systems see the same chronological train/test token streams and seeds.
Static test results are scored before any test-time update. Adaptive results
score each target first, then perform that model's specified online update.
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
from experiments.dansn_conditioned_gru_gates import DANSNConditionedGRU
from persistent_dansn_rnn import (
    PersistentDANSNRNN,
    evaluate_prequential,
    train_stream,
)


def _parameter_count(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _new_hybrid(brain: Brain, vocab_size: int, hidden_size: int,
                window_size: int, gate_conditioned: bool):
    model_type = DANSNConditionedGRU if gate_conditioned else PersistentDANSNRNN
    return model_type(
        brain.engine,
        vocab_size,
        hidden_size=hidden_size,
        window_size=window_size,
        graph_depth=3,
        message_rounds=1,
        max_active_nodes=128,
        max_memory_nodes=10_000,
    )


def run_benchmark(
    epochs: int = 3,
    hidden_size: int = 48,
    chunk_size: int = 8,
    window_size: int = 8,
    learning_rate: float = 2e-3,
    seed: int = 7,
) -> dict:
    if epochs < 1 or hidden_size < 1 or chunk_size < 1 or window_size < 2:
        raise ValueError("epochs/hidden_size/chunk_size must be positive; window_size >= 2")

    torch.manual_seed(int(seed))
    vocabulary_brain = Brain()
    train_ids = vocabulary_brain.vocab.tokenize(TRAIN_TEXT)
    test_ids = vocabulary_brain.vocab.tokenize(TEST_TEXT)
    vocab = vocabulary_brain.vocab
    vocab_size = len(vocab.word_to_id) + 1

    # Plain GRU: ordinary parameter learning, no external graph memory.
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

    def train_hybrid(gate_conditioned: bool):
        # Each architecture receives its own empty graph and identical stream.
        brain = Brain(vocab=copy.deepcopy(vocab))
        model = _new_hybrid(
            brain, vocab_size, hidden_size, window_size, gate_conditioned
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        train_metrics = [
            train_stream(model, train_ids, optimizer, chunk_size)
            for _ in range(epochs)
        ]

        # Copy the complete system so static scoring cannot mutate LRU metadata
        # or consume any memory capacity before the adaptive pass.
        static_model = copy.deepcopy(model)
        static = evaluate_prequential(
            static_model, test_ids, update_graph=False
        )
        # Adaptive pass scores each target before neural and graph adaptation.
        adaptive = train_stream(model, test_ids, optimizer, chunk_size)
        return {
            "test_static": static,
            "test_adaptive": adaptive,
            "last_train": train_metrics[-1],
            "parameter_count": _parameter_count(model),
            "graph": model.graph_stats(),
        }

    readout_fusion = train_hybrid(gate_conditioned=False)
    gate_conditioned = train_hybrid(gate_conditioned=True)

    return {
        "protocol": {
            "seed": int(seed),
            "epochs": epochs,
            "hidden_size": hidden_size,
            "chunk_size": chunk_size,
            "window_size": window_size,
            "train_tokens": len(train_ids),
            "test_tokens": len(test_ids),
            "static": "held-out targets scored with learned model and train-end memory frozen",
            "adaptive": "each target scored before online neural/graph update",
            "parameter_counts": "reported per model; same hidden width, not parameter-matched",
        },
        "results": {
            "plain_gru": {
                "test_static": plain_static,
                "test_adaptive": plain_adaptive,
                "last_train": plain_train[-1],
                "parameter_count": _parameter_count(plain),
            },
            "readout_fusion_dansn_gru": readout_fusion,
            "gate_conditioned_dansn_gru": gate_conditioned,
        },
        "interpretation": (
            "Small synthetic pilot only. Compare static with static and adaptive "
            "with adaptive. Same hidden width and streams are used, but model "
            "parameter counts differ. Do not infer general superiority."
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
            epochs=epochs,
            hidden_size=hidden_size,
            chunk_size=chunk_size,
            window_size=window_size,
            learning_rate=learning_rate,
            seed=seed,
        )
        for seed in seed_list
    ]

    def stats(values):
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        return {"mean": mean, "population_std": math.sqrt(variance), "runs": len(values)}

    summary = {}
    for system in (
        "plain_gru",
        "readout_fusion_dansn_gru",
        "gate_conditioned_dansn_gru",
    ):
        static_key = "test_static"
        adaptive_key = "test_adaptive"
        if system == "plain_gru":
            static_key = "test_static"
            adaptive_key = "test_adaptive"

        static_values = [
            run["results"][system][static_key]["perplexity"] for run in runs
        ]
        adaptive_values = [
            run["results"][system][adaptive_key]["perplexity"] for run in runs
        ]
        summary[system] = {
            "static_perplexity": stats(static_values),
            "adaptive_perplexity": stats(adaptive_values),
            "parameter_count": stats([
                float(run["results"][system]["parameter_count"])
                for run in runs
            ]),
        }

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
            "Exploratory multi-seed synthetic pilot, not a general benchmark. "
            "Seed variation is not a confidence interval over real-world tasks."
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
