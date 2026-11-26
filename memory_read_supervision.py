"""Memory-read supervised learning experiment for delayed DANSN-RNN recall.

This extends the existing delayed relational-recall task without changing the
canonical agi.py engine or the persistent DANSN-RNN core. An auxiliary
contrastive head learns to align the graph-read vector with the correct place
token embedding. Query labels contribute gradients only; they are never written
as graph edges or persistent memory vectors.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path
from typing import Dict, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from agi import Brain
from benchmark_continual_learning import train_read_only_hybrid
from delayed_relational_recall import (
    _make_hybrid,
    _rank_metrics,
    build_recall_data,
    evaluate_hybrid_queries,
    ingest_context_online,
    train_hybrid_query,
    warm_hybrid_state,
)
from persistent_dansn_rnn import PersistentDANSNRNN, train_stream


class MemoryReadAnswerHead(nn.Module):
    """Project a retrieved memory vector into the place-embedding space.

    The candidate set is the fixed vocabulary's place-token embeddings. A
    temperature-scaled cosine softmax supplies a contrastive classification
    objective and can also decode answers at test time.
    """

    def __init__(self, hidden_size: int, temperature: float = 0.1) -> None:
        super().__init__()
        if hidden_size < 1 or temperature <= 0:
            raise ValueError("hidden_size and temperature must be positive")
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.temperature = float(temperature)

    def forward(
        self,
        graph_readout: torch.Tensor,
        embedding_weight: torch.Tensor,
        candidate_ids: Sequence[int],
    ) -> torch.Tensor:
        if not candidate_ids:
            raise ValueError("candidate_ids must not be empty")
        projected = F.normalize(self.projection(graph_readout), dim=-1)
        candidates = F.embedding(
            torch.as_tensor(candidate_ids, dtype=torch.long, device=embedding_weight.device),
            embedding_weight,
        )
        candidates = F.normalize(candidates, dim=-1)
        return torch.matmul(projected, candidates.transpose(0, 1)) / self.temperature


def _place_candidates(vocab) -> Tuple[list[int], Dict[int, int]]:
    candidates = sorted(
        token_id
        for token, token_id in vocab.word_to_id.items()
        if token.startswith("place")
    )
    if len(candidates) < 2:
        raise ValueError("the recall vocabulary must contain at least two place tokens")
    return candidates, {token_id: index for index, token_id in enumerate(candidates)}


def train_memory_supervised_query(
    model: PersistentDANSNRNN,
    answer_head: MemoryReadAnswerHead,
    query: Tuple[int, int, int],
    optimizer: torch.optim.Optimizer,
    candidate_ids: Sequence[int],
    target_to_class: Dict[int, int],
    auxiliary_weight: float = 0.5,
) -> Tuple[float, float]:
    """Train answer decoding from the read vector, without writing the answer."""
    if auxiliary_weight < 0:
        raise ValueError("auxiliary_weight must be non-negative")
    where_id, person_id, target_id = query
    if target_id not in target_to_class:
        raise ValueError("query target is not in the place candidate set")

    model.train()
    answer_head.train()
    optimizer.zero_grad(set_to_none=True)

    state = model.initial_state()
    _ignored, state, _where_trace = model.forward_step(where_id, [where_id], state)
    next_logits, _state, trace = model.forward_step(person_id, [person_id], state)

    target = torch.tensor([target_id], dtype=torch.long, device=model.device)
    next_token_loss = F.cross_entropy(next_logits.unsqueeze(0), target)

    answer_logits = answer_head(
        trace["graph_readout"], model.embedding.weight, candidate_ids
    )
    answer_class = torch.tensor(
        [target_to_class[target_id]], dtype=torch.long, device=model.device
    )
    auxiliary_loss = F.cross_entropy(answer_logits.unsqueeze(0), answer_class)
    total_loss = next_token_loss + auxiliary_weight * auxiliary_loss

    # Crucially, this query label is never passed to observe_target().
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(model.parameters()) + list(answer_head.parameters()), 1.0
    )
    optimizer.step()

    return float(next_token_loss.detach().cpu()), float(auxiliary_loss.detach().cpu())


@torch.no_grad()
def evaluate_memory_answer_head(
    model: PersistentDANSNRNN,
    answer_head: MemoryReadAnswerHead,
    prefix_ids: Sequence[int],
    queries: Sequence[Tuple[int, int, int]],
    candidate_ids: Sequence[int],
    target_to_class: Dict[int, int],
    reset_recurrent_state: bool,
) -> Dict[str, float]:
    """Decode each answer from the graph-read vector without mutating memory."""
    model.eval()
    answer_head.eval()
    base_state = warm_hybrid_state(model, prefix_ids)
    losses, ranks = [], []
    correct = 0

    for where_id, person_id, target_id in queries:
        state = model.initial_state() if reset_recurrent_state else base_state.detached()
        _ignored, state, _where_trace = model.forward_step(where_id, [where_id], state)
        _next_logits, _state, trace = model.forward_step(person_id, [person_id], state)
        logits = answer_head(
            trace["graph_readout"], model.embedding.weight, candidate_ids
        )
        target_class = target_to_class[target_id]
        nll, rank, prediction = _rank_metrics(logits, target_class)
        losses.append(nll)
        ranks.append(rank)
        correct += int(prediction == target_class)

    count = max(len(queries), 1)
    return {
        "accuracy": correct / count,
        "mean_nll": sum(losses) / count,
        "mean_reciprocal_rank": sum(1.0 / rank for rank in ranks) / count,
        "queries": float(len(queries)),
    }


def _new_hybrid(vocab, vocab_size: int, hidden_size: int,
                window_size: int, graph_depth: int):
    brain = Brain(vocab=copy.deepcopy(vocab))
    model = _make_hybrid(
        brain.engine, vocab_size, hidden_size, window_size, graph_depth
    )
    return brain, model


def run_memory_supervision(
    seed: int = 7,
    subject_count: int = 24,
    distractor_count: int = 24,
    delays: Sequence[int] = (0, 8, 24),
    epochs: int = 3,
    hidden_size: int = 32,
    window_size: int = 9,
    graph_depth: int = 8,
    chunk_size: int = 16,
    learning_rate: float = 2e-3,
    auxiliary_weight: float = 0.5,
    temperature: float = 0.1,
) -> Dict[str, object]:
    if not delays or any(delay < 0 or delay > distractor_count for delay in delays):
        raise ValueError("delays must be non-empty and within [0, distractor_count]")
    torch.manual_seed(seed)
    data = build_recall_data(seed, subject_count, distractor_count)
    vocab = data["vocab"]
    vocab_size = len(vocab.word_to_id) + 1
    candidate_ids, target_to_class = _place_candidates(vocab)
    train_prefix = data["train_prefix_ids"]
    test_queries = data["test_query_ids"]

    # The read-only comparison learns the same next-token query objective.
    ro_brain, readonly = _new_hybrid(
        vocab, vocab_size, hidden_size, window_size, graph_depth
    )
    ro_brain.engine.read_book(train_prefix, window_size)
    readonly_optimizer = torch.optim.AdamW(readonly.parameters(), lr=learning_rate)

    # Persistent next-token-only baseline.
    _next_brain, next_only = _new_hybrid(
        vocab, vocab_size, hidden_size, window_size, graph_depth
    )
    next_optimizer = torch.optim.AdamW(next_only.parameters(), lr=learning_rate)

    # Same persistent architecture, with an auxiliary read-vector answer loss.
    _supervised_brain, supervised = _new_hybrid(
        vocab, vocab_size, hidden_size, window_size, graph_depth
    )
    answer_head = MemoryReadAnswerHead(hidden_size, temperature)
    supervised_optimizer = torch.optim.AdamW(
        list(supervised.parameters()) + list(answer_head.parameters()),
        lr=learning_rate,
    )

    next_query_losses = []
    auxiliary_query_losses = []
    for epoch in range(epochs):
        train_read_only_hybrid(
            readonly, train_prefix, readonly_optimizer, chunk_size
        )
        train_stream(next_only, train_prefix, next_optimizer, chunk_size)
        train_stream(supervised, train_prefix, supervised_optimizer, chunk_size)

        query_order = list(data["train_query_ids"])
        random.Random(seed + epoch + 101).shuffle(query_order)
        for query in query_order:
            next_query_losses.append(
                train_hybrid_query(
                    next_only, query, next_optimizer, write_memory=False
                )
            )
            next_loss, auxiliary_loss = train_memory_supervised_query(
                supervised,
                answer_head,
                query,
                supervised_optimizer,
                candidate_ids,
                target_to_class,
                auxiliary_weight,
            )
            auxiliary_query_losses.append(auxiliary_loss)

    readonly_weights = {
        key: value.detach().clone() for key, value in readonly.state_dict().items()
    }
    next_weights = {
        key: value.detach().clone() for key, value in next_only.state_dict().items()
    }
    supervised_weights = {
        key: value.detach().clone() for key, value in supervised.state_dict().items()
    }

    result: Dict[str, object] = {
        "protocol": {
            "seed": seed,
            "subjects": subject_count,
            "training_query_subjects": len(data["train_query_ids"]),
            "held_out_query_subjects": len(test_queries),
            "distractor_facts": distractor_count,
            "delays": list(delays),
            "epochs": epochs,
            "hidden_size": hidden_size,
            "window_size": window_size,
            "graph_depth": graph_depth,
            "auxiliary_weight": auxiliary_weight,
            "answer_temperature": temperature,
            "train_test_mappings_independently_randomized": True,
            "held_out_queries_never_used_for_training": True,
            "query_answer_labels_written_to_graph": False,
            "test_context_ingested_with_frozen_weights": True,
        },
        "training": {
            "next_token_only_query_nll_mean": (
                sum(next_query_losses) / max(len(next_query_losses), 1)
            ),
            "memory_supervised_auxiliary_nll_mean": (
                sum(auxiliary_query_losses) / max(len(auxiliary_query_losses), 1)
            ),
        },
        "results": {},
        "interpretation": (
            "Synthetic controlled comparison. The auxiliary head is trained to "
            "decode the read vector against place-token embeddings. No answer "
            "labels are written into graph topology or persistent memory."
        ),
    }

    for delay in delays:
        test_prefix = data["test_fact_ids"] + [
            token
            for row in data["distractor_ids"][:delay]
            for token in row
        ]

        ro_brain, ro_model = _new_hybrid(
            vocab, vocab_size, hidden_size, window_size, graph_depth
        )
        ro_brain.engine.read_book(test_prefix, window_size)
        ro_model.load_state_dict(readonly_weights)

        _next_test_brain, next_model = _new_hybrid(
            vocab, vocab_size, hidden_size, window_size, graph_depth
        )
        next_model.load_state_dict(next_weights)
        ingest_context_online(next_model, test_prefix)

        _supervised_test_brain, supervised_model = _new_hybrid(
            vocab, vocab_size, hidden_size, window_size, graph_depth
        )
        supervised_model.load_state_dict(supervised_weights)
        ingest_context_online(supervised_model, test_prefix)

        delay_result = {
            "read_only_next_token_reset": evaluate_hybrid_queries(
                ro_model, test_prefix, test_queries, True
            ),
            "read_only_next_token_continuous": evaluate_hybrid_queries(
                ro_model, test_prefix, test_queries, False
            ),
            "persistent_next_token_only_reset": evaluate_hybrid_queries(
                next_model, test_prefix, test_queries, True
            ),
            "persistent_next_token_only_continuous": evaluate_hybrid_queries(
                next_model, test_prefix, test_queries, False
            ),
            "persistent_memory_supervised_logits_reset": evaluate_hybrid_queries(
                supervised_model, test_prefix, test_queries, True
            ),
            "persistent_memory_supervised_logits_continuous": evaluate_hybrid_queries(
                supervised_model, test_prefix, test_queries, False
            ),
            "memory_read_head_readonly_reset": evaluate_memory_answer_head(
                ro_model, answer_head, test_prefix, test_queries,
                candidate_ids, target_to_class, True
            ),
            "memory_read_head_readonly_continuous": evaluate_memory_answer_head(
                ro_model, answer_head, test_prefix, test_queries,
                candidate_ids, target_to_class, False
            ),
            "memory_read_head_persistent_reset": evaluate_memory_answer_head(
                supervised_model, answer_head, test_prefix, test_queries,
                candidate_ids, target_to_class, True
            ),
            "memory_read_head_persistent_continuous": evaluate_memory_answer_head(
                supervised_model, answer_head, test_prefix, test_queries,
                candidate_ids, target_to_class, False
            ),
            "persistent_graph_nodes": supervised_model.graph_stats()["nodes"],
            "persistent_memory_paths": supervised_model.graph_stats()["memory_nodes"],
        }
        result["results"][str(delay)] = delay_result

    return result


def aggregate_runs(
    runs: Sequence[Dict[str, object]], seeds: Sequence[int]
) -> Dict[str, object]:
    if not runs:
        raise ValueError("at least one run is required")
    delays = list(runs[0]["results"].keys())
    metric_names = [
        key for key in runs[0]["results"][delays[0]]
        if key not in ("persistent_graph_nodes", "persistent_memory_paths")
    ]

    def summarize(values):
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        return {"mean": mean, "std": math.sqrt(variance), "runs": len(values)}

    summary = {}
    for delay in delays:
        summary[delay] = {}
        for metric_name in metric_names:
            sample = runs[0]["results"][delay][metric_name]
            summary[delay][metric_name] = {
                key: summarize([
                    float(run["results"][delay][metric_name][key])
                    for run in runs
                ])
                for key in ("accuracy", "mean_nll", "mean_reciprocal_rank")
            }
        for count_name in ("persistent_graph_nodes", "persistent_memory_paths"):
            summary[delay][count_name] = summarize([
                float(run["results"][delay][count_name]) for run in runs
            ])

    return {
        "seeds": [int(seed) for seed in seeds],
        "summary": summary,
        "runs": list(runs),
        "interpretation": (
            "Seed aggregation for a synthetic delayed-recall task only; it does "
            "not establish robust relational reasoning or broad generalization."
        ),
    }


def run_multi_seed(seeds: Sequence[int] = (7, 13, 21), **kwargs):
    if not seeds:
        raise ValueError("at least one seed is required")
    runs = [
        run_memory_supervision(seed=int(seed), **kwargs)
        for seed in seeds
    ]
    return aggregate_runs(runs, seeds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--subjects", type=int, default=24)
    parser.add_argument("--distractors", type=int, default=24)
    parser.add_argument("--delays", type=str, default="0,8,24")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--window-size", type=int, default=9)
    parser.add_argument("--graph-depth", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--auxiliary-weight", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument(
        "--output", type=Path, default=Path("memory-read-supervision-results.json")
    )
    args = parser.parse_args()

    common = dict(
        subject_count=args.subjects,
        distractor_count=args.distractors,
        delays=[
            int(value.strip())
            for value in args.delays.split(",")
            if value.strip()
        ],
        epochs=args.epochs,
        hidden_size=args.hidden_size,
        window_size=args.window_size,
        graph_depth=args.graph_depth,
        chunk_size=args.chunk_size,
        learning_rate=args.learning_rate,
        auxiliary_weight=args.auxiliary_weight,
        temperature=args.temperature,
    )
    if args.seeds:
        seeds = [
            int(value.strip())
            for value in args.seeds.split(",")
            if value.strip()
        ]
        output = run_multi_seed(seeds=seeds, **common)
    else:
        output = run_memory_supervision(seed=args.seed, **common)

    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
