"""Larger-corpus benchmark for DANSN-conditioned GRU gate experiments.

Static test metrics are fully frozen. Prequential memory-only metrics score each
target before writing it to DANSN; neural parameters stay frozen during that
test-time adaptation. This separates model learning from persistent graph
memory adaptation.
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
    PlainGRU,
    evaluate_plain_gru,
    train_plain_gru,
)
from persistent_dansn_rnn import PersistentDANSNRNN, evaluate_prequential, train_stream
from experiments.dansn_conditioned_gru_gates.corpus import load_encoded_corpus
from experiments.dansn_conditioned_gru_gates.model import DANSNConditionedGRU
from experiments.dansn_conditioned_gru_gates.four_gate import DANSNFourGateGRU


def _parameter_count(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _new_hybrid(vocab, hidden_size: int, window_size: int, kind: str):
    brain = Brain(vocab=copy.deepcopy(vocab))
    common = {
        "engine": brain.engine,
        "vocab_size": len(brain.vocab.word_to_id) + 1,
        "hidden_size": hidden_size,
        "window_size": window_size,
        "graph_depth": 3,
        "message_rounds": 1,
        "max_active_nodes": 128,
        "max_memory_nodes": 20_000,
    }
    if kind == "readout_fusion":
        return PersistentDANSNRNN(**common)
    if kind == "gate_full":
        return DANSNConditionedGRU(**common, gate_ablation="full")
    if kind == "gate_update_only":
        return DANSNConditionedGRU(**common, gate_ablation="update_only")
    if kind == "four_gate":
        return DANSNFourGateGRU(**common, gate_ablation="full")
    raise ValueError(f"unknown hybrid kind: {kind}")


def run_one_seed(
    corpus,
    seed: int,
    epochs: int,
    hidden_size: int,
    chunk_size: int,
    window_size: int,
    learning_rate: float,
) -> dict:
    train_ids = corpus.train_ids
    validation_ids = corpus.validation_ids
    test_ids = corpus.test_ids
    vocab_size = len(corpus.brain.vocab.word_to_id) + 1

    # Plain GRU baseline.
    torch.manual_seed(seed)
    plain = PlainGRU(vocab_size, hidden_size)
    plain_optimizer = torch.optim.AdamW(plain.parameters(), lr=learning_rate)
    plain_train = [
        train_plain_gru(plain, train_ids, plain_optimizer, chunk_size)
        for _ in range(epochs)
    ]
    plain_validation = evaluate_plain_gru(plain, validation_ids)
    plain_test = evaluate_plain_gru(plain, test_ids)

    results = {
        "plain_gru": {
            "validation_static": plain_validation,
            "test_static": plain_test,
            "train_last_epoch": plain_train[-1],
            "parameter_count": _parameter_count(plain),
            "graph_memory_adaptation": "not applicable",
        }
    }

    # Hybrid baselines and gate placements use independent empty graph memories.
    for offset, kind in enumerate(
        ("readout_fusion", "gate_update_only", "gate_full", "four_gate"), start=1
    ):
        torch.manual_seed(seed + offset)
        model = _new_hybrid(
            corpus.brain.vocab, hidden_size, window_size, kind
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        train_metrics = [
            train_stream(model, train_ids, optimizer, chunk_size)
            for _ in range(epochs)
        ]

        validation_static = evaluate_prequential(
            copy.deepcopy(model), validation_ids, update_graph=False
        )
        test_static = evaluate_prequential(
            copy.deepcopy(model), test_ids, update_graph=False
        )
        # The model copy receives test-time graph writes only after each target
        # is scored. Its neural parameters do not change in this evaluation.
        test_memory_adaptive = evaluate_prequential(
            copy.deepcopy(model), test_ids, update_graph=True
        )

        results[kind] = {
            "validation_static": validation_static,
            "test_static": test_static,
            "test_prequential_memory_only": test_memory_adaptive,
            "train_last_epoch": train_metrics[-1],
            "parameter_count": _parameter_count(model),
            "graph_after_training": model.graph_stats(),
        }

    return {
        "seed": int(seed),
        "results": results,
    }


def _stats(values: Sequence[float]) -> dict:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "mean": mean,
        "population_std": math.sqrt(variance),
        "runs": len(values),
    }


def run_benchmark(
    corpus,
    seeds: Sequence[int] = (7,),
    epochs: int = 1,
    hidden_size: int = 48,
    chunk_size: int = 32,
    window_size: int = 8,
    learning_rate: float = 2e-3,
) -> dict:
    seed_list = [int(seed) for seed in seeds]
    if not seed_list:
        raise ValueError("at least one seed is required")
    if min(epochs, hidden_size, chunk_size, window_size) < 1 or window_size < 2:
        raise ValueError("invalid training configuration")

    runs = [
        run_one_seed(
            corpus,
            seed=seed,
            epochs=epochs,
            hidden_size=hidden_size,
            chunk_size=chunk_size,
            window_size=window_size,
            learning_rate=learning_rate,
        )
        for seed in seed_list
    ]

    systems = ("plain_gru", "readout_fusion", "gate_update_only", "gate_full", "four_gate")
    summary = {}
    for system in systems:
        summary[system] = {
            "validation_static_perplexity": _stats([
                run["results"][system]["validation_static"]["perplexity"]
                for run in runs
            ]),
            "test_static_perplexity": _stats([
                run["results"][system]["test_static"]["perplexity"]
                for run in runs
            ]),
            "parameter_count": _stats([
                float(run["results"][system]["parameter_count"])
                for run in runs
            ]),
        }
        if system != "plain_gru":
            summary[system]["test_prequential_memory_only_perplexity"] = _stats([
                run["results"][system]["test_prequential_memory_only"]["perplexity"]
                for run in runs
            ])

    return {
        "dataset": corpus.metadata,
        "protocol": {
            "seeds": seed_list,
            "epochs": epochs,
            "hidden_size": hidden_size,
            "chunk_size": chunk_size,
            "window_size": window_size,
            "static": (
                "validation/test scored with train-end neural parameters and "
                "DANSN graph/memory frozen"
            ),
            "prequential_memory_only": (
                "test targets scored before each DANSN graph/memory write; "
                "neural parameters remain frozen"
            ),
            "training": "all systems see the same ordered train token stream",
            "parameter_matching": (
                "same hidden width but different total parameter counts; "
                "counts are reported and comparisons are not parameter matched"
            ),
        },
        "summary": summary,
        "runs": runs,
        "interpretation": (
            "Real-corpus exploratory benchmark. Do not pool static and "
            "prequential metrics. A single corpus and a few seeds do not prove "
            "general superiority; repeat on multiple datasets and seeds."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("wikitext103", "pg19"), default="wikitext103")
    parser.add_argument("--train-tokens", type=int, default=100_000)
    parser.add_argument("--validation-tokens", type=int, default=20_000)
    parser.add_argument("--test-tokens", type=int, default=20_000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=48)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seeds", type=str, default="7")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--tfds-data-dir", type=str, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-results/dansn_gated_gru_real_corpus.json"),
    )
    args = parser.parse_args()

    corpus = load_encoded_corpus(
        dataset_name=args.dataset,
        train_tokens=args.train_tokens,
        validation_tokens=args.validation_tokens,
        test_tokens=args.test_tokens,
        cache_dir=args.cache_dir,
        tfds_data_dir=args.tfds_data_dir,
    )
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    result = run_benchmark(
        corpus=corpus,
        seeds=seeds,
        epochs=args.epochs,
        hidden_size=args.hidden_size,
        chunk_size=args.chunk_size,
        window_size=args.window_size,
        learning_rate=args.learning_rate,
    )
    rendered = json.dumps(result, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
