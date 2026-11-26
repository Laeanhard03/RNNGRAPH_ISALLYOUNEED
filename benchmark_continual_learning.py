"""Prequential comparison of GRU, DANSN, read-only hybrid and writable hybrid.

This is an experimental harness. It reports results; it does not assert that
any architecture is better. All models are scored target-before-update.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Dict, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from agi import Brain
from persistent_dansn_rnn import (
    PersistentDANSNRNN,
    evaluate_prequential,
    train_stream,
)


TRAIN_TEXT = """
a red fox lives near a blue lake.
the red fox watches the blue lake.
a blue bird flies over the lake.
the fox sees the blue bird.
the blue bird returns to the lake.
a red fox watches the blue bird.
a small boat crosses the blue lake.
the fox watches a small boat.
the bird flies over the small boat.
the red fox lives near the blue lake.
"""
TEST_TEXT = """
the blue bird watches the red fox.
a small boat crosses the lake.
the fox sees a blue bird.
the red fox watches a small boat.
"""


class PlainGRU(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.cell = nn.GRUCell(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size)

    def step(self, token_id: int, hidden=None):
        token = torch.tensor(token_id, dtype=torch.long, device=self.embedding.weight.device)
        emb = self.embedding(token)
        if hidden is None:
            hidden = emb.new_zeros(emb.shape)
        hidden = self.cell(emb, hidden)
        return self.output(hidden), hidden


def _perplexity(loss_sum: float, count: int) -> float:
    return math.exp(min(loss_sum / max(count, 1), 80.0))


def train_plain_gru(model, token_ids: Sequence[int], optimizer, chunk_size: int):
    model.train()
    hidden = None
    losses = []
    chunk = []
    optimizer.zero_grad(set_to_none=True)
    for index, token in enumerate(token_ids[:-1]):
        logits, hidden = model.step(int(token), hidden)
        target = torch.tensor([int(token_ids[index + 1])], dtype=torch.long,
                              device=logits.device)
        loss = F.cross_entropy(logits.unsqueeze(0), target)
        chunk.append(loss)
        losses.append(float(loss.detach().cpu()))
        if len(chunk) >= chunk_size or index == len(token_ids) - 2:
            torch.stack(chunk).mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            hidden = hidden.detach()
            chunk.clear()
    mean = sum(losses) / len(losses)
    return {"mean_loss": mean, "perplexity": math.exp(min(mean, 80.0))}


def train_read_only_hybrid(model, token_ids: Sequence[int], optimizer, chunk_size: int):
    """Train the RNN to read a prebuilt DANSN graph without graph/memory writes."""
    model.train()
    state = model.initial_state()
    history = [int(token_ids[0])]
    chunk = []
    losses = []
    optimizer.zero_grad(set_to_none=True)
    for index, raw in enumerate(token_ids[:-1]):
        token = int(raw)
        target = int(token_ids[index + 1])
        logits, state, _ = model.forward_step(token, history, state)
        loss = F.cross_entropy(
            logits.unsqueeze(0),
            torch.tensor([target], dtype=torch.long, device=model.device),
        )
        chunk.append(loss)
        losses.append(float(loss.detach().cpu()))
        history.append(target)
        if len(chunk) >= chunk_size or index == len(token_ids) - 2:
            torch.stack(chunk).mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            state = state.detached()
            chunk.clear()
    mean = sum(losses) / len(losses)
    return {"mean_loss": mean, "perplexity": math.exp(min(mean, 80.0))}


@torch.no_grad()
def evaluate_plain_gru(model, token_ids: Sequence[int]) -> Dict[str, float]:
    model.eval()
    hidden = None
    losses = []
    for index, token in enumerate(token_ids[:-1]):
        logits, hidden = model.step(int(token), hidden)
        target = torch.tensor([int(token_ids[index + 1])], dtype=torch.long,
                              device=logits.device)
        losses.append(float(F.cross_entropy(logits.unsqueeze(0), target).cpu()))
    mean = sum(losses) / max(len(losses), 1)
    return {"mean_loss": mean, "perplexity": math.exp(min(mean, 80.0)),
            "tokens": len(losses)}


@torch.no_grad()
def evaluate_dansn(brain: Brain, token_ids: Sequence[int], window_size: int,
                   adapt_after_scoring: bool) -> Dict[str, float]:
    engine, vocab = brain.engine, brain.vocab
    ignored = vocab.attention_ignored_ids()
    history = [int(token_ids[0])]
    losses = []
    covered = 0
    for index, raw in enumerate(token_ids[:-1]):
        token = int(raw)
        target = int(token_ids[index + 1])
        topics = [item for item in history if item not in ignored]
        distribution = engine.next_distribution(
            history, topics, vocab, window_size=window_size,
            temperature=1.0, top_k=0,
        )
        probability = distribution.probabilities().get(target, 0.0)
        covered += int(probability > 0.0)
        losses.append(-math.log(max(probability, 1e-12)))
        # Causal graph update: target enters only after its NLL is recorded.
        if adapt_after_scoring:
            context = history[-(window_size - 1):]
            engine.observe_sequence(tuple(context) + (target,))
        history.append(target)
    mean = sum(losses) / max(len(losses), 1)
    return {
        "mean_loss": mean,
        "perplexity": math.exp(min(mean, 80.0)),
        "tokens": len(losses),
        "candidate_coverage": covered / max(len(losses), 1),
        "graph_nodes": engine.stats()["nodes"],
    }


def run_benchmark(epochs=3, hidden_size=48, chunk_size=8, window_size=8,
                  learning_rate=2e-3, seed=7):
    torch.manual_seed(seed)
    vocab_brain = Brain()
    train_ids = vocab_brain.vocab.tokenize(TRAIN_TEXT)
    test_ids = vocab_brain.vocab.tokenize(TEST_TEXT)
    vocab = vocab_brain.vocab
    vocab_size = len(vocab.word_to_id) + 1

    # 1) Plain recurrent baseline.
    plain = PlainGRU(vocab_size, hidden_size)
    plain_optimizer = torch.optim.AdamW(plain.parameters(), lr=learning_rate)
    plain_train = [train_plain_gru(plain, train_ids, plain_optimizer, chunk_size)
                   for _ in range(epochs)]
    plain_static = evaluate_plain_gru(plain, test_ids)
    plain_adaptive = train_plain_gru(plain, test_ids, plain_optimizer, chunk_size)

    # 2) Canonical DANSN, prebuilt from training data. Static and adaptive
    # test runs share the same initial test-start graph because static scoring
    # is read-only and happens first.
    dansn_brain = Brain(vocab=copy.deepcopy(vocab))
    dansn_brain.engine.read_book(train_ids, window_size)
    dansn_static = evaluate_dansn(
        dansn_brain, test_ids, window_size, adapt_after_scoring=False
    )
    dansn_adaptive = evaluate_dansn(
        dansn_brain, test_ids, window_size, adapt_after_scoring=True
    )

    # 3) Read-only hybrid: prebuild graph from training data; train the RNN
    # while keeping graph/memory fixed. At test time report both frozen and
    # online-RNN adaptation, with no graph or persistent-memory writes.
    readonly_brain = Brain(vocab=copy.deepcopy(vocab))
    readonly_brain.engine.read_book(train_ids, window_size)
    readonly = PersistentDANSNRNN(
        readonly_brain.engine, vocab_size, hidden_size=hidden_size,
        window_size=window_size, graph_depth=3, message_rounds=1,
    )
    readonly_optimizer = torch.optim.AdamW(readonly.parameters(), lr=learning_rate)
    readonly_train = [
        train_read_only_hybrid(readonly, train_ids, readonly_optimizer, chunk_size)
        for _ in range(epochs)
    ]
    readonly_static = evaluate_prequential(readonly, test_ids, update_graph=False)
    readonly_adaptive = train_read_only_hybrid(
        readonly, test_ids, readonly_optimizer, chunk_size
    )

    # 4) Writable hybrid: graph starts empty and grows online during training.
    # Report frozen test-start performance, then score-before-update continual
    # adaptation with both neural updates and graph/state writes.
    coupled_brain = Brain(vocab=copy.deepcopy(vocab))
    coupled = PersistentDANSNRNN(
        coupled_brain.engine, vocab_size, hidden_size=hidden_size,
        window_size=window_size, graph_depth=3, message_rounds=1,
    )
    coupled_optimizer = torch.optim.AdamW(coupled.parameters(), lr=learning_rate)
    coupled_train = [
        train_stream(coupled, train_ids, coupled_optimizer, chunk_size)
        for _ in range(epochs)
    ]
    coupled_static = evaluate_prequential(coupled, test_ids, update_graph=False)
    coupled_adaptive = train_stream(
        coupled, test_ids, coupled_optimizer, chunk_size
    )

    return {
        "protocol": {
            "train_tokens": len(train_ids),
            "test_tokens": len(test_ids),
            "epochs": epochs,
            "hidden_size": hidden_size,
            "chunk_size": chunk_size,
            "window_size": window_size,
            "seed": seed,
            "static_test": "score without test-time graph or parameter updates",
            "adaptive_test": "score each target first, then apply each system's online update",
        },
        "results": {
            "plain_gru": {
                "test_static": plain_static,
                "test_adaptive_parameter_updates": plain_adaptive,
                "last_train": plain_train[-1],
            },
            "dansn_only": {
                "test_static_graph": dansn_static,
                "test_adaptive_graph_updates": dansn_adaptive,
            },
            "read_only_dansn_gru": {
                "test_static": readonly_static,
                "test_adaptive_parameter_updates_graph_frozen": readonly_adaptive,
                "last_train": readonly_train[-1],
                "graph_nodes": readonly.graph_stats()["nodes"],
            },
            "persistent_dansn_rnn": {
                "test_static": coupled_static,
                "test_adaptive_parameter_and_graph_updates": coupled_adaptive,
                "last_train": coupled_train[-1],
                "graph": coupled.graph_stats(),
            },
        },
        "interpretation": (
            "Small synthetic smoke benchmark only. Compare static metrics with static "
            "metrics and adaptive metrics with adaptive metrics. Use multiple seeds "
            "and a larger chronological corpus before drawing conclusions."
        ),
    }


def run_multi_seed(seeds: Sequence[int], epochs=3, hidden_size=48, chunk_size=8,
                   window_size=8, learning_rate=2e-3):
    """Run independent seeds and summarize static/adaptive perplexity."""
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
    paths = {
        "plain_gru": ("test_static", "test_adaptive_parameter_updates"),
        "dansn_only": ("test_static_graph", "test_adaptive_graph_updates"),
        "read_only_dansn_gru": (
            "test_static",
            "test_adaptive_parameter_updates_graph_frozen",
        ),
        "persistent_dansn_rnn": (
            "test_static",
            "test_adaptive_parameter_and_graph_updates",
        ),
    }

    def stats(values):
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        return {"mean": mean, "std": math.sqrt(variance), "runs": len(values)}

    summary = {}
    for system, (static_key, adaptive_key) in paths.items():
        static_values = [
            run["results"][system][static_key]["perplexity"] for run in runs
        ]
        adaptive_values = [
            run["results"][system][adaptive_key]["perplexity"] for run in runs
        ]
        summary[system] = {
            "static_perplexity": stats(static_values),
            "adaptive_perplexity": stats(adaptive_values),
        }
        if system == "dansn_only":
            summary[system]["static_candidate_coverage"] = stats([
                run["results"][system][static_key]["candidate_coverage"]
                for run in runs
            ])
            summary[system]["adaptive_candidate_coverage"] = stats([
                run["results"][system][adaptive_key]["candidate_coverage"]
                for run in runs
            ])

    return {
        "protocol": {
            "seeds": seed_list,
            "epochs": epochs,
            "hidden_size": hidden_size,
            "chunk_size": chunk_size,
            "window_size": window_size,
            "static_vs_adaptive": "summarized separately; never mix the two protocols",
        },
        "summary": summary,
        "runs": runs,
        "interpretation": (
            "Small synthetic multi-seed pilot. Mean/std describe seed variation only; "
            "they do not establish performance on natural language or broad generalization."
        ),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--hidden-size", type=int, default=48)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Optional comma-separated multi-seed run, e.g. 7,13,21",
    )
    parser.add_argument("--output", type=Path, default=Path("benchmark_results.json"))
    args = parser.parse_args()

    common = dict(
        epochs=args.epochs,
        hidden_size=args.hidden_size,
        chunk_size=args.chunk_size,
        window_size=args.window_size,
        learning_rate=args.learning_rate,
    )
    if args.seeds:
        seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
        result = run_multi_seed(seeds, **common)
    else:
        result = run_benchmark(seed=args.seed, **common)

    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
