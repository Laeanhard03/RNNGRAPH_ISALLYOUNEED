"""One-shot associative sequence-recall benchmark for DANSN-GRU variants.

The task trains a controller on one set of key/value associations, then gives it
new held-out key/value pairs at evaluation time. Each pair is revealed once,
followed by distractors and a query. Recurrent hidden state is reset at test
start; DANSN graph and persistent node memory are retained. Query targets are
scored before they are revealed or written.

This benchmark evaluates both graph-level storage/retrieval and neural
prediction. It is intentionally a path-association task, not a claim that the
current trie-like DANSN already performs abstract semantic multi-hop reasoning.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import torch
import torch.nn.functional as F

from agi import Brain
from benchmark_continual_learning import PlainGRU, train_plain_gru
from persistent_dansn_rnn import PersistentDANSNRNN, train_stream
from experiments.dansn_conditioned_gru_gates.model import DANSNConditionedGRU
from experiments.dansn_conditioned_gru_gates.four_gate import DANSNFourGateGRU


@dataclass
class RecallDataset:
    brain: Brain
    train_tokens: list[int]
    test_tokens: list[int]
    support_paths: dict[int, tuple[int, ...]]
    query_paths: dict[int, tuple[int, ...]]
    query_delays: dict[int, int]
    metadata: dict
    preview: dict


def _append_noise(tokens: list[int], rng: random.Random,
                  noise_ids: Sequence[int], count: int) -> None:
    tokens.extend(rng.choice(noise_ids) for _ in range(count))


def generate_dataset(
    seed: int = 7,
    train_keys: int = 16,
    heldout_keys: int = 12,
    value_count: int = 16,
    train_episodes: int = 8,
    train_delay: int = 12,
    test_delay: int = 64,
) -> RecallDataset:
    """Generate train associations and disjoint one-shot held-out associations."""
    if min(train_keys, heldout_keys, train_episodes, train_delay, test_delay) < 1:
        raise ValueError("key counts, episodes, and delays must be positive")
    if value_count < max(train_keys, heldout_keys):
        raise ValueError("value_count must be >= both key counts")

    rng = random.Random(int(seed))
    train_key_words = [f"trainkey{i:03d}" for i in range(train_keys)]
    test_key_words = [f"heldoutkey{i:03d}" for i in range(heldout_keys)]
    value_words = [f"value{i:03d}" for i in range(value_count)]
    noise_words = ["noisea", "noiseb", "noisec", "noised", "noisee", "noisef"]
    control_words = ["recallctx", "noiseleft", "noiseright"]
    all_words = control_words + train_key_words + test_key_words + value_words + noise_words

    brain = Brain()
    brain.vocab.tokenize(" ".join(all_words))
    ids = brain.vocab.word_to_id
    ctx = ids["recallctx"]
    noise_ids = [ids[word] for word in noise_words]

    train_values = rng.sample(value_words, train_keys)
    test_values = rng.sample(value_words, heldout_keys)
    train_mapping_words = dict(zip(train_key_words, train_values))
    test_mapping_words = dict(zip(test_key_words, test_values))
    train_mapping = {
        ids[key]: ids[value] for key, value in train_mapping_words.items()
    }
    test_mapping = {
        ids[key]: ids[value] for key, value in test_mapping_words.items()
    }

    train_stream: list[int] = []
    for _episode in range(train_episodes):
        support_order = list(train_key_words)
        rng.shuffle(support_order)
        for key in support_order:
            train_stream.extend([ctx, ids[key], ids[train_mapping_words[key]]])

        _append_noise(train_stream, rng, noise_ids, train_delay)

        query_order = list(train_key_words)
        rng.shuffle(query_order)
        for key in query_order:
            train_stream.extend([ctx, ids[key], ids[train_mapping_words[key]]])
            _append_noise(train_stream, rng, noise_ids, rng.randint(1, 3))

        # Expose held-out key embeddings during training without exposing their
        # recallctx->key association or their eventual key->value mapping.
        exposure_order = list(test_key_words)
        rng.shuffle(exposure_order)
        for key in exposure_order:
            train_stream.extend([ids["noiseleft"], ids[key], ids["noiseright"]])

    test_stream: list[int] = []
    support_paths: dict[int, tuple[int, ...]] = {}
    query_paths: dict[int, tuple[int, ...]] = {}
    query_delays: dict[int, int] = {}
    support_positions_by_key: dict[int, int] = {}

    support_order = list(test_key_words)
    rng.shuffle(support_order)
    for key in support_order:
        start = len(test_stream)
        value_id = test_mapping[ids[key]]
        test_stream.extend([ctx, ids[key], value_id])
        key_position = start + 1
        path = (ctx, ids[key], value_id)
        support_paths[key_position] = path
        support_positions_by_key[ids[key]] = key_position

    _append_noise(test_stream, rng, noise_ids, test_delay)

    query_order = list(test_key_words)
    rng.shuffle(query_order)
    for key in query_order:
        _append_noise(test_stream, rng, noise_ids, rng.randint(1, 4))
        start = len(test_stream)
        value_id = test_mapping[ids[key]]
        test_stream.extend([ctx, ids[key], value_id])
        key_position = start + 1
        path = (ctx, ids[key], value_id)
        query_paths[key_position] = path
        query_delays[key_position] = key_position - support_positions_by_key[ids[key]]

    if len(train_stream) < 2 or len(test_stream) < 2:
        raise RuntimeError("generated streams are unexpectedly short")

    decode = brain.vocab.id_to_word
    metadata = {
        "seed": int(seed),
        "train_keys": len(train_key_words),
        "heldout_keys": len(test_key_words),
        "value_vocabulary": len(value_words),
        "training_episodes": int(train_episodes),
        "train_delay_tokens": int(train_delay),
        "test_delay_tokens": int(test_delay),
        "train_stream_tokens": len(train_stream),
        "test_stream_tokens": len(test_stream),
        "query_count": len(query_paths),
        "mean_query_delay_tokens": (
            sum(query_delays.values()) / max(len(query_delays), 1)
        ),
        "vocab_size": len(ids) + 1,
        "value_token_ids": [ids[word] for word in value_words],
        "train_value_coverage_count": len(set(train_values)),
        "all_value_classes_seen_in_training": set(train_values) == set(value_words),
        "heldout_values_seen_in_training": set(test_values).issubset(set(train_values)),
        "train_associations": train_mapping_words,
        "heldout_associations": test_mapping_words,
        "test_key_ids_disjoint_from_train": not (
            set(train_mapping).intersection(test_mapping)
        ),
    }
    preview = {
        "test_sequence_first_120_tokens": [
            decode[token_id] for token_id in test_stream[:120]
        ],
        "heldout_associations": test_mapping_words,
        "query_count": len(query_paths),
    }
    return RecallDataset(
        brain=brain,
        train_tokens=train_stream,
        test_tokens=test_stream,
        support_paths=support_paths,
        query_paths=query_paths,
        query_delays=query_delays,
        metadata=metadata,
        preview=preview,
    )


def _make_hybrid(brain: Brain, hidden_size: int, window_size: int,
                 system: str):
    common = {
        "engine": brain.engine,
        "vocab_size": len(brain.vocab.word_to_id) + 1,
        "hidden_size": hidden_size,
        "window_size": window_size,
        "graph_depth": 3,
        "message_rounds": 1,
        "max_active_nodes": 128,
        "max_memory_nodes": 10_000,
    }
    if system == "readout_fusion":
        return PersistentDANSNRNN(**common)
    if system == "three_path_gate_gru":
        return DANSNConditionedGRU(**common, gate_ablation="full")
    four_gate = {
        "four_gate": "full",
        "four_gate_no_memory_injection": "no_memory_injection",
        "four_gate_no_persistent_write": "no_persistent_write",
        "four_gate_no_graph_evidence": "no_graph_evidence",
    }
    if system in four_gate:
        return DANSNFourGateGRU(**common, gate_ablation=four_gate[system])
    raise ValueError(f"unknown hybrid system: {system}")


@torch.no_grad()
def _evaluate_plain(model: PlainGRU, dataset: RecallDataset) -> dict:
    model.eval()
    hidden = None
    losses: list[float] = []
    correct = 0
    value_only_correct = 0
    reset_state_correct = 0
    reset_value_only_correct = 0
    target_value_probs: list[float] = []
    value_probability_mass: list[float] = []
    target_value_ranks: list[float] = []
    count = 0
    value_ids = torch.tensor(
        dataset.metadata["value_token_ids"], dtype=torch.long,
        device=model.embedding.weight.device
    )
    for index, token in enumerate(dataset.test_tokens[:-1]):
        logits, hidden = model.step(int(token), hidden)
        if index in dataset.query_paths:
            target = int(dataset.test_tokens[index + 1])
            target_tensor = torch.tensor([target], dtype=torch.long, device=logits.device)
            losses.append(float(F.cross_entropy(logits.unsqueeze(0), target_tensor).cpu()))
            correct += int(int(logits.argmax().item()) == target)

            value_logits = logits.index_select(0, value_ids)
            target_positions = (value_ids == target).nonzero(as_tuple=False)
            target_position = int(target_positions[0].item())
            value_only_correct += int(int(value_ids[value_logits.argmax()].item()) == target)
            value_probs = torch.softmax(value_logits, dim=0)
            target_value_probs.append(float(value_probs[target_position].cpu()))
            value_probability_mass.append(
                float(torch.softmax(logits, dim=0).index_select(0, value_ids).sum().cpu())
            )
            target_value_ranks.append(
                float(1 + (value_logits > value_logits[target_position]).sum().item())
            )

            reset_logits, _ = model.step(int(token), None)
            reset_state_correct += int(int(reset_logits.argmax().item()) == target)
            reset_value_logits = reset_logits.index_select(0, value_ids)
            reset_value_only_correct += int(
                int(value_ids[reset_value_logits.argmax()].item()) == target
            )
            count += 1
    mean_loss = sum(losses) / max(len(losses), 1)
    return {
        "query_count": count,
        "query_top1_accuracy": correct / max(count, 1),
        "query_value_only_top1_accuracy": value_only_correct / max(count, 1),
        "query_reset_state_top1_accuracy": reset_state_correct / max(count, 1),
        "query_reset_state_value_only_top1_accuracy": reset_value_only_correct / max(count, 1),
        "query_target_value_probability_mean": (
            sum(target_value_probs) / len(target_value_probs) if target_value_probs else 0.0
        ),
        "query_value_probability_mass_mean": (
            sum(value_probability_mass) / len(value_probability_mass)
            if value_probability_mass else 0.0
        ),
        "query_target_value_rank_mean": (
            sum(target_value_ranks) / len(target_value_ranks)
            if target_value_ranks else 0.0
        ),
        "query_mean_nll": mean_loss,
        "query_perplexity": math.exp(min(mean_loss, 80.0)),
        "graph_active_path_coverage": None,
        "persistent_vector_coverage": None,
        "support_graph_write_rate": None,
        "support_vector_write_rate": None,
        "mean_write_gate_at_support": None,
        "mean_memory_gate_at_query": None,
        "mean_effective_memory_mix_at_query": None,
        "mean_persistent_memory_readout_norm_at_query": None,
        "mean_memory_output_context_norm_at_query": None,
        "mean_pointer_mix_at_query": None,
        "mean_pointer_candidate_count_at_query": None,
        "mean_pointer_target_probability_at_query": None,
        "mean_update_gate_at_query": None,
        "mean_reset_gate_at_query": None,
        "mean_query_delay_tokens": dataset.metadata["mean_query_delay_tokens"],
        "graph_stats": None,
    }

@torch.no_grad()
def _evaluate_hybrid(model, dataset: RecallDataset) -> dict:
    """Reset RNN state, retain trained DANSN memory, then score support/query stream."""
    model.eval()
    state = model.initial_state()
    history = [int(dataset.test_tokens[0])]
    query_losses: list[float] = []
    query_correct = 0
    active_paths = 0
    memory_paths = 0
    support_graph_writes = 0
    support_vector_writes = 0
    write_gate_support: list[float] = []
    memory_gate_query: list[float] = []
    effective_memory_mix_query: list[float] = []
    persistent_memory_norm_query: list[float] = []
    memory_output_norm_query: list[float] = []
    pointer_mix_query: list[float] = []
    pointer_candidate_count_query: list[float] = []
    pointer_target_probability_query: list[float] = []
    update_gate_query: list[float] = []
    reset_gate_query: list[float] = []
    value_only_correct = 0
    reset_state_correct = 0
    reset_value_only_correct = 0
    target_value_probs: list[float] = []
    value_probability_mass: list[float] = []
    target_value_ranks: list[float] = []
    value_ids = torch.tensor(
        dataset.metadata["value_token_ids"], dtype=torch.long, device=model.device
    )

    for index, raw_token in enumerate(dataset.test_tokens[:-1]):
        token = int(raw_token)
        target = int(dataset.test_tokens[index + 1])
        if history[-1] != token:
            raise RuntimeError("test stream history lost causal alignment")

        logits, state, trace = model.forward_step(token, history, state)

        if index in dataset.query_paths:
            expected_path = dataset.query_paths[index]
            query_losses.append(float(F.cross_entropy(
                logits.unsqueeze(0),
                torch.tensor([target], dtype=torch.long, device=logits.device),
            ).cpu()))
            query_correct += int(int(logits.argmax().item()) == target)
            value_logits = logits.index_select(0, value_ids)
            target_positions = (value_ids == target).nonzero(as_tuple=False)
            target_position = int(target_positions[0].item())
            value_only_correct += int(
                int(value_ids[value_logits.argmax()].item()) == target
            )
            value_probs = torch.softmax(value_logits, dim=0)
            target_value_probs.append(float(value_probs[target_position].cpu()))
            value_probability_mass.append(
                float(torch.softmax(logits, dim=0).index_select(0, value_ids).sum().cpu())
            )
            target_value_ranks.append(
                float(1 + (value_logits > value_logits[target_position]).sum().item())
            )
            pointer_distribution = trace["persistent_token_distribution"]
            pointer_mix_query.append(
                float(trace["effective_pointer_mix"].detach().cpu())
            )
            pointer_candidate_count_query.append(
                float(torch.count_nonzero(pointer_distribution).item())
            )
            pointer_target_probability_query.append(
                float(pointer_distribution[target].detach().cpu())
            )

            # Reset only the temporary controller state while keeping DANSN
            # graph topology and persistent vectors available for this query.
            reset_logits, _reset_state, _reset_trace = model.forward_step(
                token, history, model.initial_state()
            )
            reset_state_correct += int(int(reset_logits.argmax().item()) == target)
            reset_value_logits = reset_logits.index_select(0, value_ids)
            reset_value_only_correct += int(
                int(value_ids[reset_value_logits.argmax()].item()) == target
            )

            active_paths += int(expected_path in trace.get("active_paths", []))
            memory_paths += int(expected_path in model.node_memory)

            if "memory_injection_gate" in trace:
                memory_gate_query.append(
                    float(trace["memory_injection_gate"].mean().detach().cpu())
                )
                effective_memory_mix_query.append(
                    float(trace["effective_memory_mix"].mean().detach().cpu())
                )
                persistent_memory_norm_query.append(
                    float(torch.linalg.vector_norm(
                        trace["persistent_memory_readout"]
                    ).detach().cpu())
                )
                memory_output_norm_query.append(
                    float(torch.linalg.vector_norm(
                        trace["memory_output_context"]
                    ).detach().cpu())
                )
                update_gate_query.append(
                    float(trace["update_gate"].mean().detach().cpu())
                )
                reset_gate_query.append(
                    float(trace["reset_gate"].mean().detach().cpu())
                )

        is_support = index in dataset.support_paths
        if is_support and "persistent_write_gate" in trace:
            write_gate_support.append(
                float(trace["persistent_write_gate"].mean().detach().cpu())
            )

        # Every target is revealed only after its loss is computed. This is
        # online memory adaptation, but model parameters remain frozen.
        model.observe_target(history, target, state)

        if is_support:
            expected_path = dataset.support_paths[index]
            support_graph_writes += int(expected_path in model._path_nodes)
            support_vector_writes += int(expected_path in model.node_memory)

        history.append(target)

    count = len(query_losses)
    mean_loss = sum(query_losses) / max(count, 1)
    graph_stats = model.graph_stats()
    return {
        "query_count": count,
        "query_top1_accuracy": query_correct / max(count, 1),
        "query_value_only_top1_accuracy": value_only_correct / max(count, 1),
        "query_reset_state_top1_accuracy": reset_state_correct / max(count, 1),
        "query_reset_state_value_only_top1_accuracy": reset_value_only_correct / max(count, 1),
        "query_target_value_probability_mean": (
            sum(target_value_probs) / len(target_value_probs)
            if target_value_probs else 0.0
        ),
        "query_value_probability_mass_mean": (
            sum(value_probability_mass) / len(value_probability_mass)
            if value_probability_mass else 0.0
        ),
        "query_target_value_rank_mean": (
            sum(target_value_ranks) / len(target_value_ranks)
            if target_value_ranks else 0.0
        ),
        "query_mean_nll": mean_loss,
        "query_perplexity": math.exp(min(mean_loss, 80.0)),
        "graph_active_path_coverage": active_paths / max(count, 1),
        "persistent_vector_coverage": memory_paths / max(count, 1),
        "support_graph_write_rate": support_graph_writes / max(
            len(dataset.support_paths), 1
        ),
        "support_vector_write_rate": support_vector_writes / max(
            len(dataset.support_paths), 1
        ),
        "mean_write_gate_at_support": (
            sum(write_gate_support) / len(write_gate_support)
            if write_gate_support else None
        ),
        "mean_memory_gate_at_query": (
            sum(memory_gate_query) / len(memory_gate_query)
            if memory_gate_query else None
        ),
        "mean_effective_memory_mix_at_query": (
            sum(effective_memory_mix_query) / len(effective_memory_mix_query)
            if effective_memory_mix_query else None
        ),
        "mean_persistent_memory_readout_norm_at_query": (
            sum(persistent_memory_norm_query) / len(persistent_memory_norm_query)
            if persistent_memory_norm_query else None
        ),
        "mean_memory_output_context_norm_at_query": (
            sum(memory_output_norm_query) / len(memory_output_norm_query)
            if memory_output_norm_query else None
        ),
        "mean_pointer_mix_at_query": (
            sum(pointer_mix_query) / len(pointer_mix_query)
            if pointer_mix_query else None
        ),
        "mean_pointer_candidate_count_at_query": (
            sum(pointer_candidate_count_query) / len(pointer_candidate_count_query)
            if pointer_candidate_count_query else None
        ),
        "mean_pointer_target_probability_at_query": (
            sum(pointer_target_probability_query) / len(pointer_target_probability_query)
            if pointer_target_probability_query else None
        ),
        "mean_update_gate_at_query": (
            sum(update_gate_query) / len(update_gate_query)
            if update_gate_query else None
        ),
        "mean_reset_gate_at_query": (
            sum(reset_gate_query) / len(reset_gate_query)
            if reset_gate_query else None
        ),
        "mean_query_delay_tokens": dataset.metadata["mean_query_delay_tokens"],
        "graph_stats": graph_stats,
    }


def _parameter_count(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def run_one_seed(
    seed: int,
    epochs: int = 2,
    hidden_size: int = 32,
    chunk_size: int = 64,
    window_size: int = 4,
    train_keys: int = 16,
    heldout_keys: int = 12,
    value_count: int = 16,
    train_episodes: int = 8,
    train_delay: int = 12,
    test_delay: int = 64,
) -> dict:
    if min(epochs, hidden_size, chunk_size) < 1:
        raise ValueError("epochs, hidden_size, and chunk_size must be positive")
    dataset = generate_dataset(
        seed=seed,
        train_keys=train_keys,
        heldout_keys=heldout_keys,
        value_count=value_count,
        train_episodes=train_episodes,
        train_delay=train_delay,
        test_delay=test_delay,
    )
    vocab_size = len(dataset.brain.vocab.word_to_id) + 1
    systems = (
        "plain_gru",
        "readout_fusion",
        "three_path_gate_gru",
        "four_gate",
        "four_gate_no_memory_injection",
        "four_gate_no_persistent_write",
        "four_gate_no_graph_evidence",
    )
    results = {}
    for offset, system in enumerate(systems):
        # Four-gate ablations have identical parameter shapes and share the
        # same initialization for this seed; only their selected path differs.
        init_seed = seed + 400 if system.startswith("four_gate") else seed + 31 * (offset + 1)
        torch.manual_seed(init_seed)
        if system == "plain_gru":
            model = PlainGRU(vocab_size, hidden_size)
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
            train_metrics = [
                train_plain_gru(model, dataset.train_tokens, optimizer, chunk_size)
                for _ in range(epochs)
            ]
            evaluation = _evaluate_plain(model, dataset)
        else:
            brain = Brain(vocab=copy.deepcopy(dataset.brain.vocab))
            model = _make_hybrid(brain, hidden_size, window_size, system)
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
            train_metrics = [
                train_stream(model, dataset.train_tokens, optimizer, chunk_size)
                for _ in range(epochs)
            ]
            evaluation = _evaluate_hybrid(model, dataset)
        results[system] = {
            **evaluation,
            "parameter_count": _parameter_count(model),
            "train_last_epoch": train_metrics[-1],
        }

    return {
        "seed": int(seed),
        "dataset": dataset.metadata,
        "results": results,
        "dataset_preview": dataset.preview,
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
    seeds: Sequence[int] = (7, 13, 21),
    **configuration,
) -> dict:
    seed_list = [int(seed) for seed in seeds]
    if not seed_list:
        raise ValueError("at least one seed is required")
    runs = [run_one_seed(seed=seed, **configuration) for seed in seed_list]
    systems = tuple(runs[0]["results"])
    summary = {}
    for system in systems:
        summary[system] = {
            "query_top1_accuracy": _stats([
                run["results"][system]["query_top1_accuracy"] for run in runs
            ]),
            "query_mean_nll": _stats([
                run["results"][system]["query_mean_nll"] for run in runs
            ]),
            "query_value_only_top1_accuracy": _stats([
                run["results"][system]["query_value_only_top1_accuracy"] for run in runs
            ]),
            "query_reset_state_top1_accuracy": _stats([
                run["results"][system]["query_reset_state_top1_accuracy"] for run in runs
            ]),
            "query_reset_state_value_only_top1_accuracy": _stats([
                run["results"][system]["query_reset_state_value_only_top1_accuracy"]
                for run in runs
            ]),
            "query_target_value_probability_mean": _stats([
                run["results"][system]["query_target_value_probability_mean"]
                for run in runs
            ]),
            "query_value_probability_mass_mean": _stats([
                run["results"][system]["query_value_probability_mass_mean"]
                for run in runs
            ]),
            "query_target_value_rank_mean": _stats([
                run["results"][system]["query_target_value_rank_mean"]
                for run in runs
            ]),
            "graph_active_path_coverage": _stats([
                float(run["results"][system]["graph_active_path_coverage"] or 0.0)
                for run in runs
            ]),
            "persistent_vector_coverage": _stats([
                float(run["results"][system]["persistent_vector_coverage"] or 0.0)
                for run in runs
            ]),
            "support_graph_write_rate": _stats([
                float(run["results"][system]["support_graph_write_rate"] or 0.0)
                for run in runs
            ]),
            "support_vector_write_rate": _stats([
                float(run["results"][system]["support_vector_write_rate"] or 0.0)
                for run in runs
            ]),
            "parameter_count": _stats([
                float(run["results"][system]["parameter_count"]) for run in runs
            ]),
        }
        for gate_key in (
            "mean_write_gate_at_support",
            "mean_memory_gate_at_query",
            "mean_update_gate_at_query",
            "mean_reset_gate_at_query",
        ):
            values = [
                run["results"][system][gate_key]
                for run in runs
                if run["results"][system][gate_key] is not None
            ]
            if values:
                summary[system][gate_key] = _stats(values)
        for memory_key in (
            "mean_effective_memory_mix_at_query",
            "mean_persistent_memory_readout_norm_at_query",
            "mean_memory_output_context_norm_at_query",
            "mean_pointer_mix_at_query",
            "mean_pointer_candidate_count_at_query",
            "mean_pointer_target_probability_at_query",
        ):
            values = [
                run["results"][system][memory_key]
                for run in runs
                if run["results"][system][memory_key] is not None
            ]
            if values:
                summary[system][memory_key] = _stats(values)

    first = runs[0]
    return {
        "task": (
            "one-shot held-out key/value association recall after a distractor "
            "delay; test recurrent state resets while DANSN memory persists"
        ),
        "protocol": {
            **configuration,
            "seeds": seed_list,
            "training": (
                "train on stable associations for training keys; held-out keys "
                "appear during training only in unrelated noise contexts"
            ),
            "evaluation": (
                "reveal each held-out key/value pair once, then delay with "
                "distractors and query in shuffled order; score before target reveal"
            ),
            "parameter_updates_during_test": False,
            "recurrent_state_at_test_start": "reset",
            "dANSN_graph_and_memory_at_test_start": "retained from training",
            "metric_scope": (
                "query accuracy measures neural top-1 response; path and vector "
                "coverage measure graph storage/retrieval separately"
            ),
            "parameter_matching": (
                "four-gate ablations share parameter shapes and initialization; "
                "other architectures are not parameter-matched"
            ),
        },
        "dataset": first["dataset"],
        "summary": summary,
        "runs": runs,
        "interpretation": (
            "Synthetic mechanism test, not a real-language benchmark. High path "
            "coverage with low answer accuracy means graph storage/retrieval works "
            "but the neural readout/controller has not learned to decode it. "
            "Low path coverage indicates a storage, write-gate, or retrieval issue. "
            "Repeat with more seeds, values, delays, and interference patterns."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="7,13,21")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--train-keys", type=int, default=16)
    parser.add_argument("--heldout-keys", type=int, default=12)
    parser.add_argument("--value-count", type=int, default=16)
    parser.add_argument("--train-episodes", type=int, default=8)
    parser.add_argument("--train-delay", type=int, default=12)
    parser.add_argument("--test-delay", type=int, default=64)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-results/associative_recall_results.json"),
    )
    parser.add_argument(
        "--data-output",
        type=Path,
        default=Path("benchmark-results/associative_recall_dataset.json"),
    )
    args = parser.parse_args()
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    configuration = {
        "epochs": args.epochs,
        "hidden_size": args.hidden_size,
        "chunk_size": args.chunk_size,
        "window_size": args.window_size,
        "train_keys": args.train_keys,
        "heldout_keys": args.heldout_keys,
        "value_count": args.value_count,
        "train_episodes": args.train_episodes,
        "train_delay": args.train_delay,
        "test_delay": args.test_delay,
    }
    result = run_benchmark(seeds=seeds, **configuration)
    rendered = json.dumps(result, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")

    sample = generate_dataset(seed=seeds[0], **{
        key: configuration[key]
        for key in (
            "train_keys", "heldout_keys", "value_count", "train_episodes",
            "train_delay", "test_delay"
        )
    })
    data_rendered = json.dumps({
        "metadata": sample.metadata,
        "preview": sample.preview,
        "train_tokens": sample.train_tokens,
        "test_tokens": sample.test_tokens,
        "support_positions": {
            str(key): list(value) for key, value in sample.support_paths.items()
        },
        "query_positions": {
            str(key): list(value) for key, value in sample.query_paths.items()
        },
        "query_delays": sample.query_delays,
    }, indent=2)
    args.data_output.parent.mkdir(parents=True, exist_ok=True)
    args.data_output.write_text(data_rendered + "\n", encoding="utf-8")
    print(rendered)
    print(f"\nWrote {args.output} and {args.data_output}")


if __name__ == "__main__":
    main()
