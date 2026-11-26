"""Delayed two-hop relational recall benchmark for the DANSN-RNN experiment.

A model observes facts of the form:
    person owns vehicle . vehicle parked place .
After a configurable number of unrelated distractor facts, it receives
    where person
and must predict the correct place.

The test knowledge base is remapped independently from the training one.
Held-out subjects are never used as training queries. At test time, the
persistent hybrid absorbs the new fact stream by graph/memory writes with
neural weights frozen. We score with recurrent state both preserved and reset.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from agi import Brain, Vocabulary
from benchmark_continual_learning import PlainGRU, train_plain_gru, train_read_only_hybrid
from persistent_dansn_rnn import PersistentDANSNRNN, train_stream


def _permutation(size: int, rng: random.Random) -> List[int]:
    values = list(range(size))
    rng.shuffle(values)
    return values


def _fact_text(subject_count: int, vehicle_map: Sequence[int],
               place_map: Sequence[int], rng: random.Random) -> str:
    order = list(range(subject_count))
    rng.shuffle(order)
    rows = []
    for subject in order:
        vehicle = vehicle_map[subject]
        place = place_map[vehicle]
        rows.append(
            f"person{subject:02d} owns vehicle{vehicle:02d} . "
            f"vehicle{vehicle:02d} parked place{place:02d} ."
        )
    return " ".join(rows)


def _distractor_rows(count: int, offset: int = 0) -> List[str]:
    rows = []
    for index in range(offset, offset + count):
        rows.append(
            f"traveler{index:03d} visits island{index:03d} . "
            f"island{index:03d} borders sector{index:03d} ."
        )
    return rows


def _query_rows(indices: Iterable[int], place_map: Sequence[int],
                vehicle_map: Sequence[int]) -> str:
    rows = []
    for subject in indices:
        vehicle = vehicle_map[subject]
        place = place_map[vehicle]
        rows.append(f"where person{subject:02d} place{place:02d} .")
    return " ".join(rows)


def build_recall_data(
    seed: int = 7,
    subject_count: int = 24,
    distractor_count: int = 48,
    train_fraction: float = 0.75,
):
    if subject_count < 4 or distractor_count < 0:
        raise ValueError("subject_count must be >= 4 and distractor_count >= 0")
    if not 0.5 <= train_fraction < 1.0:
        raise ValueError("train_fraction must be in [0.5, 1)")

    train_rng = random.Random(seed)
    test_rng = random.Random(seed + 100_003)
    train_vehicle = _permutation(subject_count, train_rng)
    train_place = _permutation(subject_count, train_rng)
    test_vehicle = _permutation(subject_count, test_rng)
    test_place = _permutation(subject_count, test_rng)

    split = max(2, min(subject_count - 1, int(subject_count * train_fraction)))
    train_queries = list(range(split))
    test_queries = list(range(split, subject_count))

    train_facts = _fact_text(
        subject_count, train_vehicle, train_place, random.Random(seed + 1)
    )
    test_facts = _fact_text(
        subject_count, test_vehicle, test_place, random.Random(seed + 2)
    )
    distractors = _distractor_rows(distractor_count)
    train_query_text = _query_rows(train_queries, train_place, train_vehicle)

    # Build one fixed vocabulary from both knowledge bases. Test mappings and
    # entities are not new vocabulary items; only their relationships differ.
    vocab_text = " ".join([
        train_facts,
        test_facts,
        " ".join(distractors),
        train_query_text,
        " ".join(f"where person{i:02d}" for i in test_queries),
    ])
    vocab = Vocabulary()
    vocab.tokenize(vocab_text)

    def encode(text: str) -> List[int]:
        return vocab.tokenize(text)

    return {
        "vocab": vocab,
        "train_facts_text": train_facts,
        "test_facts_text": test_facts,
        "distractors": distractors,
        "train_query_indices": train_queries,
        "test_query_indices": test_queries,
        "train_place_map": train_place,
        "train_vehicle_map": train_vehicle,
        "test_place_map": test_place,
        "test_vehicle_map": test_vehicle,
        "train_prefix_ids": encode(train_facts + " " + " ".join(distractors)),
        "test_fact_ids": encode(test_facts),
        "distractor_ids": [encode(row) for row in distractors],
        "train_query_ids": [
            (
                vocab.word_to_id["where"],
                vocab.word_to_id[f"person{subject:02d}"],
                vocab.word_to_id[f"place{train_place[train_vehicle[subject]]:02d}"],
            )
            for subject in train_queries
        ],
        "test_query_ids": [
            (
                vocab.word_to_id["where"],
                vocab.word_to_id[f"person{subject:02d}"],
                vocab.word_to_id[f"place{test_place[test_vehicle[subject]]:02d}"],
            )
            for subject in test_queries
        ],
        "train_ids": encode(train_facts + " " + " ".join(distractors) + " " + train_query_text),
    }


def train_plain_query(model: PlainGRU, query: Tuple[int, int, int],
                      optimizer: torch.optim.Optimizer) -> float:
    where_id, person_id, target_id = query
    model.train()
    optimizer.zero_grad(set_to_none=True)
    _ignored, hidden = model.step(where_id, None)
    logits, _hidden = model.step(person_id, hidden)
    loss = F.cross_entropy(
        logits.unsqueeze(0),
        torch.tensor([target_id], dtype=torch.long, device=logits.device),
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach().cpu())


def train_hybrid_query(
    model: PersistentDANSNRNN,
    query: Tuple[int, int, int],
    optimizer: torch.optim.Optimizer,
    write_memory: bool,
) -> float:
    where_id, person_id, target_id = query
    model.train()
    optimizer.zero_grad(set_to_none=True)
    state = model.initial_state()
    _ignored, state, _ = model.forward_step(where_id, [where_id], state)
    logits, state, _trace = model.forward_step(person_id, [person_id], state)
    loss = F.cross_entropy(
        logits.unsqueeze(0),
        torch.tensor([target_id], dtype=torch.long, device=model.device),
    )

    # The answer is revealed only after its prediction loss has been computed.
    if write_memory:
        model.observe_target([person_id], target_id, state)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if write_memory:
        model.detach_memory()
    return float(loss.detach().cpu())


@torch.no_grad()
def ingest_context_online(
    model: PersistentDANSNRNN, token_ids: Sequence[int]
):
    """Absorb observed facts causally; no optimizer update is performed."""
    if len(token_ids) < 2:
        return model.initial_state()
    model.eval()
    state = model.initial_state()
    history = [int(token_ids[0])]
    for index in range(len(token_ids) - 1):
        token = int(token_ids[index])
        target = int(token_ids[index + 1])
        _logits, state, _trace = model.forward_step(token, history, state)
        model.observe_target(history, target, state)
        history.append(target)
    _logits, state, _trace = model.forward_step(int(token_ids[-1]), history, state)
    return state


@torch.no_grad()
def warm_plain_state(model: PlainGRU, token_ids: Sequence[int]):
    model.eval()
    state = None
    for token in token_ids:
        _logits, state = model.step(int(token), state)
    return state


@torch.no_grad()
def warm_hybrid_state(model: PersistentDANSNRNN, token_ids: Sequence[int]):
    model.eval()
    state = model.initial_state()
    history: List[int] = []
    for raw in token_ids:
        token = int(raw)
        history.append(token)
        _logits, state, _trace = model.forward_step(token, history, state)
    return state


def _rank_metrics(logits: torch.Tensor, target_id: int) -> Tuple[float, int, float]:
    log_probs = torch.log_softmax(logits, dim=-1)
    nll = float(-log_probs[target_id].cpu())
    rank = int((logits > logits[target_id]).sum().item()) + 1
    return nll, rank, int(torch.argmax(logits).item())


@torch.no_grad()
def evaluate_plain_queries(
    model: PlainGRU,
    prefix_ids: Sequence[int],
    queries: Sequence[Tuple[int, int, int]],
) -> Dict[str, float]:
    base_state = warm_plain_state(model, prefix_ids)
    losses, ranks, correct = [], [], 0
    for where_id, person_id, target_id in queries:
        state = None if base_state is None else base_state.detach().clone()
        _ignored, state = model.step(where_id, state)
        logits, _state = model.step(person_id, state)
        nll, rank, prediction = _rank_metrics(logits, target_id)
        losses.append(nll)
        ranks.append(rank)
        correct += int(prediction == target_id)
    count = max(len(queries), 1)
    return {
        "accuracy": correct / count,
        "mean_nll": sum(losses) / count,
        "mean_reciprocal_rank": sum(1.0 / rank for rank in ranks) / count,
        "queries": float(len(queries)),
    }


@torch.no_grad()
def evaluate_hybrid_queries(
    model: PersistentDANSNRNN,
    prefix_ids: Sequence[int],
    queries: Sequence[Tuple[int, int, int]],
    reset_recurrent_state: bool,
) -> Dict[str, float]:
    base_state = warm_hybrid_state(model, prefix_ids)
    losses, ranks = [], []
    correct = 0
    target_reachable = 0
    target_memory_visible = 0

    for where_id, person_id, target_id in queries:
        state = model.initial_state() if reset_recurrent_state else base_state.detached()
        _ignored, state, _where_trace = model.forward_step(where_id, [where_id], state)
        logits, _state, trace = model.forward_step(person_id, [person_id], state)
        nll, rank, prediction = _rank_metrics(logits, target_id)
        losses.append(nll)
        ranks.append(rank)
        correct += int(prediction == target_id)
        active = trace["active_paths"]
        target_reachable += int(any(path and path[-1] == target_id for path in active))
        target_memory_visible += int(any(path in model.node_memory for path in active))

    count = max(len(queries), 1)
    return {
        "accuracy": correct / count,
        "mean_nll": sum(losses) / count,
        "mean_reciprocal_rank": sum(1.0 / rank for rank in ranks) / count,
        "target_reachable_fraction": target_reachable / count,
        "target_memory_visible_fraction": target_memory_visible / count,
        "queries": float(len(queries)),
    }


def _make_hybrid(engine, vocab_size: int, hidden_size: int, window_size: int,
                 graph_depth: int, max_active_nodes: int = 256):
    return PersistentDANSNRNN(
        engine,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        window_size=window_size,
        graph_depth=graph_depth,
        message_rounds=1,
        max_active_nodes=max_active_nodes,
        max_memory_nodes=100_000,
    )


def run_delayed_recall(
    seed: int = 7,
    subject_count: int = 24,
    distractor_count: int = 48,
    delays: Sequence[int] = (0, 8, 24, 48),
    epochs: int = 3,
    hidden_size: int = 48,
    window_size: int = 9,
    graph_depth: int = 8,
    chunk_size: int = 16,
    learning_rate: float = 2e-3,
) -> Dict[str, object]:
    if not delays or any(delay < 0 or delay > distractor_count for delay in delays):
        raise ValueError("delays must be non-empty and within [0, distractor_count]")
    torch.manual_seed(seed)
    data = build_recall_data(seed, subject_count, distractor_count)
    vocab = data["vocab"]
    vocab_size = len(vocab.word_to_id) + 1

    train_prefix = data["train_prefix_ids"]
    test_queries = data["test_query_ids"]

    # Plain GRU: facts/distractors train the recurrent model; query examples
    # train the answer interface on training subjects only.
    plain = PlainGRU(vocab_size, hidden_size)
    plain_optimizer = torch.optim.AdamW(plain.parameters(), lr=learning_rate)

    # Read-only hybrid: graph topology is batch-built from the training facts;
    # gradients train the recurrent/query modules, but training queries do not
    # write persistent node vectors.
    readonly_brain = Brain(vocab=copy.deepcopy(vocab))
    readonly_brain.engine.read_book(train_prefix, window_size)
    readonly = _make_hybrid(readonly_brain.engine, vocab_size, hidden_size,
                            window_size, graph_depth)
    readonly_optimizer = torch.optim.AdamW(readonly.parameters(), lr=learning_rate)

    # Writable hybrid: DANSN graph and persistent vectors are grown causally
    # during the training fact stream.
    writable_brain = Brain(vocab=copy.deepcopy(vocab))
    writable = _make_hybrid(writable_brain.engine, vocab_size, hidden_size,
                            window_size, graph_depth)
    writable_optimizer = torch.optim.AdamW(writable.parameters(), lr=learning_rate)

    training_history = {"plain_query_loss": [], "readonly_query_loss": [],
                        "writable_query_loss": []}
    for _epoch in range(epochs):
        train_plain_gru(plain, train_prefix, plain_optimizer, chunk_size)
        train_read_only_hybrid(readonly, train_prefix, readonly_optimizer, chunk_size)
        train_stream(writable, train_prefix, writable_optimizer, chunk_size)

        query_order = list(data["train_query_ids"])
        random.Random(seed + _epoch + 9).shuffle(query_order)
        for query in query_order:
            training_history["plain_query_loss"].append(
                train_plain_query(plain, query, plain_optimizer)
            )
            training_history["readonly_query_loss"].append(
                train_hybrid_query(readonly, query, readonly_optimizer, write_memory=False)
            )
            # Query labels train the retrieval/readout weights but are not written
            # back as a shortcut person->place edge. Memory must be retrieved from
            # the two-hop facts, not from an answer copied during supervision.
            training_history["writable_query_loss"].append(
                train_hybrid_query(writable, query, writable_optimizer, write_memory=False)
            )

    readonly_weights = {key: value.detach().clone() for key, value in readonly.state_dict().items()}
    writable_weights = {key: value.detach().clone() for key, value in writable.state_dict().items()}

    results: Dict[str, object] = {
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
            "test_mapping_is_independent": True,
            "held_out_queries_never_used_for_training": True,
            "test_context_absorbed_without_optimizer_updates": True,
            "target_scored_before_any_query_answer_write": True,
        },
        "training": {
            "plain_query_nll_mean": sum(training_history["plain_query_loss"]) / max(len(training_history["plain_query_loss"]), 1),
            "readonly_query_nll_mean": sum(training_history["readonly_query_loss"]) / max(len(training_history["readonly_query_loss"]), 1),
            "writable_query_nll_mean": sum(training_history["writable_query_loss"]) / max(len(training_history["writable_query_loss"]), 1),
        },
        "results": {},
        "interpretation": (
            "Synthetic two-hop delayed-recall experiment. Test knowledge-base mappings "
            "are independently permuted. Test queries are held out from query supervision. "
            "Use multiple seeds and larger subject/distractor counts before drawing conclusions."
        ),
    }

    distractor_texts = data["distractors"]
    for delay in delays:
        test_prefix = data["test_fact_ids"] + [
            token for row in data["distractor_ids"][:delay] for token in row
        ]

        # Plain GRU receives the same observed test prefix, but has no external
        # graph. Reset-state evaluation tests memory without recurrent carry.
        plain_metrics = evaluate_plain_queries(plain, test_prefix, test_queries)

        # Read-only graph baseline batch-builds topology from the observed test
        # prefix. Persistent node vectors are absent.
        ro_brain = Brain(vocab=copy.deepcopy(vocab))
        ro_brain.engine.read_book(test_prefix, window_size)
        ro_model = _make_hybrid(ro_brain.engine, vocab_size, hidden_size,
                                window_size, graph_depth)
        ro_model.load_state_dict(readonly_weights)
        ro_reset = evaluate_hybrid_queries(ro_model, test_prefix, test_queries, True)
        ro_continuous = evaluate_hybrid_queries(ro_model, test_prefix, test_queries, False)

        # Writable hybrid receives exactly the same facts sequentially, with
        # weights frozen; every target is revealed only after forward_step.
        wr_brain = Brain(vocab=copy.deepcopy(vocab))
        wr_model = _make_hybrid(wr_brain.engine, vocab_size, hidden_size,
                                window_size, graph_depth)
        wr_model.load_state_dict(writable_weights)
        ingest_state = ingest_context_online(wr_model, test_prefix)
        wr_reset = evaluate_hybrid_queries(wr_model, test_prefix, test_queries, True)
        wr_continuous = evaluate_hybrid_queries(wr_model, test_prefix, test_queries, False)

        # Plain recurrent state is evaluated both with and without the prefix.
        plain_reset_model = plain
        plain_reset = evaluate_plain_queries(
            plain_reset_model, [], test_queries
        )
        # evaluate_plain_queries on [] starts with zero hidden state; it is the
        # recurrent-only control for the state-reset condition.
        results[str(delay)] = {
            "delay_distractor_facts": delay,
            "plain_gru_reset": plain_reset,
            "plain_gru_continuous": plain_metrics,
            "readonly_dansn_gru_reset": ro_reset,
            "readonly_dansn_gru_continuous": ro_continuous,
            "persistent_dansn_rnn_reset": wr_reset,
            "persistent_dansn_rnn_continuous": wr_continuous,
            "persistent_graph_nodes": wr_model.graph_stats()["nodes"],
            "persistent_memory_paths": wr_model.graph_stats()["memory_nodes"],
        }

    return results


def run_multi_seed(
    seeds: Sequence[int] = (7, 13, 21),
    **kwargs,
) -> Dict[str, object]:
    runs = [run_delayed_recall(seed=int(seed), **kwargs) for seed in seeds]
    delays = list(runs[0]["results"].keys())
    metric_keys = [
        "plain_gru_reset", "plain_gru_continuous",
        "readonly_dansn_gru_reset", "readonly_dansn_gru_continuous",
        "persistent_dansn_rnn_reset", "persistent_dansn_rnn_continuous",
    ]

    def summarize(values: Sequence[float]) -> Dict[str, float]:
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        return {"mean": mean, "std": math.sqrt(variance), "runs": len(values)}

    summary = {}
    for delay in delays:
        summary[delay] = {}
        for metric in metric_keys:
            sample = runs[0]["results"][delay][metric]
            keys = ["accuracy", "mean_nll", "mean_reciprocal_rank"]
            for optional in ("target_reachable_fraction", "target_memory_visible_fraction"):
                if optional in sample:
                    keys.append(optional)
            summary[delay][metric] = {
                key: summarize([run["results"][delay][metric][key] for run in runs])
                for key in keys
            }
    return {
        "seeds": list(seeds),
        "summary": summary,
        "runs": runs,
        "interpretation": (
            "Synthetic pilot only. Mean/std describe seed variation on this generated "
            "task; they are not evidence of broad language or factual generalization."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--subjects", type=int, default=24)
    parser.add_argument("--distractors", type=int, default=48)
    parser.add_argument("--delays", type=str, default="0,8,24,48")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--hidden-size", type=int, default=48)
    parser.add_argument("--window-size", type=int, default=9)
    parser.add_argument("--graph-depth", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--output", type=Path, default=Path("delayed-recall-results.json"))
    args = parser.parse_args()
    delays = [int(item.strip()) for item in args.delays.split(",") if item.strip()]
    common = dict(
        subject_count=args.subjects,
        distractor_count=args.distractors,
        delays=delays,
        epochs=args.epochs,
        hidden_size=args.hidden_size,
        window_size=args.window_size,
        graph_depth=args.graph_depth,
        chunk_size=args.chunk_size,
        learning_rate=args.learning_rate,
    )
    if args.seeds:
        seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
        result = run_multi_seed(seeds=seeds, **common)
    else:
        result = run_delayed_recall(seed=args.seed, **common)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
