"""Context-dependent branch selection for the DANSN four-gate GRU.

Every held-out key has several persistent outgoing answer branches. A cue is
processed into the recurrent state before the same key context is queried.
Queries expose all branches to the pointer; only the recurrent context should
make the desired branch receive the highest score. Test targets are scored
without neural updates or target-dependent writes.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

from agi import Brain
from experiments.dansn_conditioned_gru_gates.four_gate import DANSNFourGateGRU


@dataclass(frozen=True)
class BranchQuery:
    key_id: int
    cue_id: int
    target_id: int
    alternate_cue_id: int
    alternate_target_id: int
    distractor_ids: tuple[int, ...]

    @property
    def input_tokens(self) -> tuple[int, ...]:
        # The answer is deliberately absent from the model input.
        return (self.cue_id, *self.distractor_ids)


@dataclass
class ContextualBranchDataset:
    brain: Brain
    train_tokens: list[int]
    train_query_positions: frozenset[int]
    train_records: list[list[tuple[int, tuple[int, ...], int, bool, bool]]]
    test_supports: list[tuple[int, int, int]]
    queries: list[BranchQuery]
    context_id: int
    noise_ids: tuple[int, ...]
    metadata: dict


def _assign_branches(
    keys: Sequence[int],
    cues: Sequence[int],
    values: Sequence[int],
    rng: random.Random,
) -> dict[tuple[int, int], int]:
    """Give every key distinct cue-conditioned values with balanced coverage."""
    if len(cues) > len(values):
        raise ValueError("cue_count must not exceed value_count")
    value_order = list(values)
    rng.shuffle(value_order)
    mapping = {}
    for key_index, key_id in enumerate(keys):
        for cue_index, cue_id in enumerate(cues):
            value_index = (key_index * len(cues) + cue_index) % len(values)
            mapping[(int(key_id), int(cue_id))] = int(value_order[value_index])
    return mapping


def generate_dataset(
    seed: int = 7,
    train_keys: int = 16,
    heldout_keys: int = 8,
    cue_count: int = 3,
    value_count: int = 12,
    train_episodes: int = 4,
    train_delay: int = 8,
    test_delay: int = 8,
) -> ContextualBranchDataset:
    """Create train mappings and one-shot multi-branch held-out associations."""
    if min(train_keys, heldout_keys, cue_count, value_count, train_episodes) < 1:
        raise ValueError("key counts, cue/value counts, and episodes must be positive")
    if cue_count > value_count:
        raise ValueError("cue_count must not exceed value_count")
    if train_keys * cue_count < value_count:
        raise ValueError("train_keys * cue_count must cover every value class")
    if train_delay < 0 or test_delay < 0:
        raise ValueError("delays must be non-negative")

    rng = random.Random(int(seed))
    cue_words = [f"cue{i:02d}" for i in range(cue_count)]
    train_key_words = [f"trainkey{i:03d}" for i in range(train_keys)]
    heldout_key_words = [f"heldoutkey{i:03d}" for i in range(heldout_keys)]
    value_words = [f"value{i:03d}" for i in range(value_count)]
    noise_words = [
        "noisea", "noiseb", "noisec", "noised", "noisee", "noisef",
        "noiseg", "noiseh",
    ]
    control_words = ["branchctx", "noiseleft", "noiseright"]
    all_words = (
        cue_words + train_key_words + heldout_key_words + value_words
        + noise_words + control_words
    )

    brain = Brain()
    brain.vocab.tokenize(" ".join(all_words))
    vocab = brain.vocab.word_to_id
    cue_ids = [vocab[word] for word in cue_words]
    train_key_ids = [vocab[word] for word in train_key_words]
    heldout_key_ids = [vocab[word] for word in heldout_key_words]
    value_ids = [vocab[word] for word in value_words]
    noise_ids = tuple(vocab[word] for word in noise_words)
    context_id = vocab["branchctx"]

    train_mapping = _assign_branches(
        train_key_ids, cue_ids, value_ids, random.Random(seed + 101)
    )
    heldout_mapping = _assign_branches(
        heldout_key_ids, cue_ids, value_ids, random.Random(seed + 202)
    )

    train_tokens: list[int] = []
    train_query_positions: set[int] = set()
    train_records: list[list[tuple[int, tuple[int, ...], int, bool, bool]]] = []

    def add_record(events: list[tuple[int, tuple[int, ...], int, bool, bool]]) -> None:
        """Append an independent recurrent episode and its causal token stream."""
        if not events:
            return
        offset = len(train_tokens)
        # Each event's target is exactly the next event's input. Store one
        # initial input followed by the revealed targets, without duplication.
        train_tokens.append(int(events[0][0]))
        for event_index, (token_id, _graph_history, target_id, supervise, _write) in enumerate(events):
            if int(token_id) != int(train_tokens[offset + event_index]):
                raise RuntimeError("training record is not next-token aligned")
            if supervise:
                train_query_positions.add(offset + event_index)
            train_tokens.append(int(target_id))
        train_records.append(events)

    for _episode in range(train_episodes):
        support_rows = [
            (cue_id, key_id, train_mapping[(key_id, cue_id)])
            for key_id in train_key_ids
            for cue_id in cue_ids
        ]
        rng.shuffle(support_rows)
        for cue_id, key_id, target_id in support_rows:
            # Cue enters recurrent state, but DANSN sees a separate shared
            # key path. All cue-conditioned values therefore share (context,key).
            add_record([
                (cue_id, (cue_id,), context_id, False, True),
                (context_id, (context_id,), key_id, False, True),
                (key_id, (context_id, key_id), target_id, False, True),
            ])

        query_rows = list(support_rows)
        rng.shuffle(query_rows)
        for cue_id, key_id, target_id in query_rows:
            delay = rng.randint(0, train_delay)
            distractors = [rng.choice(noise_ids) for _ in range(delay)]
            query_inputs = [cue_id, *distractors, context_id, key_id]
            graph_histories = [
                (token_id,) for token_id in [cue_id, *distractors, context_id]
            ] + [(context_id, key_id)]
            query_targets = [*distractors, context_id, key_id, target_id]
            events = [
                (
                    int(token_id),
                    tuple(int(x) for x in graph_history),
                    int(next_token),
                    event_index == len(query_inputs) - 1,
                    False,
                )
                for event_index, (token_id, graph_history, next_token) in enumerate(
                    zip(query_inputs, graph_histories, query_targets)
                )
            ]
            add_record(events)

        # Expose held-out key embeddings without exposing their branch relation.
        exposure_rows = list(heldout_key_ids)
        rng.shuffle(exposure_rows)
        for key_id in exposure_rows:
            add_record([
                (vocab["noiseleft"], (vocab["noiseleft"],), key_id, False, False),
                (key_id, (key_id,), vocab["noiseright"], False, False),
            ])

    test_supports = [
        (cue_id, key_id, heldout_mapping[(key_id, cue_id)])
        for key_id in heldout_key_ids
        for cue_id in cue_ids
    ]
    rng.shuffle(test_supports)

    queries: list[BranchQuery] = []
    query_rows = [
        (key_id, cue_index, cue_id)
        for key_id in heldout_key_ids
        for cue_index, cue_id in enumerate(cue_ids)
    ]
    rng.shuffle(query_rows)
    for key_id, cue_index, cue_id in query_rows:
        alternate_cue = cue_ids[(cue_index + 1) % len(cue_ids)]
        delay_tokens = tuple(
            rng.choice(noise_ids) for _ in range(test_delay)
        )
        queries.append(BranchQuery(
            key_id=key_id,
            cue_id=cue_id,
            target_id=heldout_mapping[(key_id, cue_id)],
            alternate_cue_id=alternate_cue,
            alternate_target_id=heldout_mapping[(key_id, alternate_cue)],
            distractor_ids=delay_tokens,
        ))

    train_value_coverage = {
        train_mapping[(key_id, cue_id)]
        for key_id in train_key_ids
        for cue_id in cue_ids
    }
    metadata = {
        "seed": int(seed),
        "train_keys": int(train_keys),
        "heldout_keys": int(heldout_keys),
        "cue_count": int(cue_count),
        "value_count": int(value_count),
        "training_episodes": int(train_episodes),
        "train_query_delay_max": int(train_delay),
        "test_query_delay": int(test_delay),
        "test_delay_is_within_training_range": int(test_delay) <= int(train_delay),
        "train_token_count": len(train_tokens),
        "train_query_supervision_positions": len(train_query_positions),
        "training_records": len(train_records),
        "graph_context_separated_from_recurrent_cue": True,
        "heldout_support_count": len(test_supports),
        "heldout_query_count": len(queries),
        "candidate_branches_per_key": int(cue_count),
        "train_and_heldout_key_ids_disjoint": (
            set(train_key_ids).isdisjoint(heldout_key_ids)
        ),
        "all_value_classes_seen_in_training": (
            len(train_value_coverage) == len(value_ids)
        ),
        "heldout_values_seen_in_training": set(
            row[2] for row in test_supports
        ).issubset(train_value_coverage),
        "train_mapping": {
            f"{key}:{cue}": value
            for (key, cue), value in train_mapping.items()
        },
        "heldout_mapping": {
            f"{key}:{cue}": value
            for (key, cue), value in heldout_mapping.items()
        },
    }
    return ContextualBranchDataset(
        brain=brain,
        train_tokens=train_tokens,
        train_query_positions=frozenset(train_query_positions),
        train_records=train_records,
        test_supports=test_supports,
        queries=queries,
        context_id=context_id,
        noise_ids=noise_ids,
        metadata=metadata,
    )


@torch.no_grad()
def ingest_heldout_supports(
    model: DANSNFourGateGRU,
    dataset: ContextualBranchDataset,
) -> dict:
    """Score each support target from its cue/context, then write it to DANSN."""
    written = 0
    for cue_id, key_id, target_id in dataset.test_supports:
        state = model.initial_state()
        # Cue is recurrent context only. DANSN's key path is intentionally
        # identical for all cue branches: (context,key).
        support_events = (
            (cue_id, (cue_id,), dataset.context_id),
            (dataset.context_id, (dataset.context_id,), key_id),
            (key_id, (dataset.context_id, key_id), target_id),
        )
        for token_id, graph_history, next_token in support_events:
            _logits, state, _trace = model.forward_step(
                int(token_id), graph_history, state
            )
            # Each known support continuation is observed only after its input
            # has been scored, matching the training support-record protocol.
            model.observe_target(graph_history, int(next_token), state)
        expected_path = (dataset.context_id, key_id, target_id)
        written += int(expected_path in model.node_memory)
    return {
        "support_count": len(dataset.test_supports),
        "persistent_vector_writes": written,
        "persistent_vector_write_coverage": (
            written / max(len(dataset.test_supports), 1)
        ),
    }


@torch.no_grad()
def _run_query(
    model: DANSNFourGateGRU,
    dataset: ContextualBranchDataset,
    cue_id: int,
    key_id: int,
    distractor_ids: Sequence[int],
):
    state = model.initial_state()
    logits = None
    trace = None
    # Recurrent state sees the full cue/distractor sequence. DANSN history is
    # intentionally separate and ends at the shared (context,key) parent.
    for token_id, graph_history in [
        (cue_id, (cue_id,)),
        *[(int(token), (int(token),)) for token in distractor_ids],
        (dataset.context_id, (dataset.context_id,)),
        (key_id, (dataset.context_id, key_id)),
    ]:
        logits, state, trace = model.forward_step(
            int(token_id), graph_history, state
        )
    if logits is None or trace is None:
        raise RuntimeError("query contained no input tokens")
    return logits, (dataset.context_id, key_id), trace


def _normalized_contextual_objective(
    nll_losses: Sequence[torch.Tensor],
    pointer_losses: Sequence[torch.Tensor],
    pointer_loss_weight: float,
) -> torch.Tensor:
    """Average token and selector objectives over their own eligible sets."""
    if not nll_losses:
        raise ValueError("at least one next-token loss is required")
    objective = torch.stack(list(nll_losses)).mean()
    if pointer_losses:
        objective = objective + float(pointer_loss_weight) * torch.stack(
            list(pointer_losses)
        ).mean()
    return objective


def train_contextual_stream(
    model: DANSNFourGateGRU,
    dataset: ContextualBranchDataset,
    optimizer: torch.optim.Optimizer,
    *,
    chunk_size: int = 64,
    grad_clip: float = 1.0,
    pointer_loss_weight: float = 1.0,
) -> dict:
    """Train on independent cue episodes with shared DANSN key paths.

    The cue and distractors update the recurrent state, but each event supplies
    a separate DANSN graph-history key. At query key positions, all previously
    written alternatives share the same (context,key) parent, allowing the
    auxiliary pointer loss to rank among the already stored continuations.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    if pointer_loss_weight < 0:
        raise ValueError("pointer_loss_weight must be non-negative")
    model._validate_ids(dataset.train_tokens)

    # Keep the two objectives separately normalized. Selector supervision
    # applies only to a minority of events; averaging (NLL + selector loss)
    # across every event would dilute its gradient by the query-event fraction.
    chunk_nll_losses = []
    chunk_pointer_losses = []
    observed_nll = []
    observed_pointer_losses = []
    combined_losses = []
    pointer_supervision_examples = 0
    pointer_candidate_counts = []
    updates = 0
    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_events = sum(len(record) for record in dataset.train_records)
    event_index = 0
    for record in dataset.train_records:
        state = model.initial_state()
        for token, graph_history, target, supervise_pointer, write_target in record:
            logits, state, trace = model.forward_step(
                int(token), graph_history, state
            )
            target_tensor = torch.tensor(
                [int(target)], dtype=torch.long, device=model.device
            )
            nll_loss = F.cross_entropy(logits.unsqueeze(0), target_tensor)
            selector_loss = logits.new_zeros(())

            if supervise_pointer:
                pointer = trace["persistent_token_distribution"]
                candidate_count = int(torch.count_nonzero(pointer).item())
                pointer_candidate_counts.append(float(candidate_count))
                target_probability = pointer[int(target)]
                if candidate_count > 0 and float(
                    target_probability.detach().cpu()
                ) > 0.0:
                    selector_loss = -torch.log(
                        target_probability.clamp_min(1e-9)
                    )
                    observed_pointer_losses.append(
                        float(selector_loss.detach().cpu())
                    )
                    pointer_supervision_examples += 1

            chunk_nll_losses.append(nll_loss)
            if selector_loss.requires_grad and float(
                selector_loss.detach().cpu()
            ) > 0.0:
                chunk_pointer_losses.append(selector_loss)
            observed_nll.append(float(nll_loss.detach().cpu()))
            combined_losses.append(float(
                nll_loss.detach().cpu()
                + pointer_loss_weight * selector_loss.detach().cpu()
            ))

            # Both ordinary next-token loss and candidate-ranking loss are
            # computed before the target is revealed to persistent DANSN memory.
            if write_target:
                model.observe_target(graph_history, int(target), state)
            event_index += 1

            if len(chunk_nll_losses) >= chunk_size or event_index == total_events:
                chunk_loss = _normalized_contextual_objective(
                    chunk_nll_losses,
                    chunk_pointer_losses,
                    pointer_loss_weight,
                )
                chunk_loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                state = state.detached()
                model.detach_memory()
                chunk_nll_losses.clear()
                chunk_pointer_losses.clear()
                updates += 1

    return {
        "mean_loss": sum(observed_nll) / max(len(observed_nll), 1),
        "mean_nll": sum(observed_nll) / max(len(observed_nll), 1),
        "mean_combined_loss": sum(combined_losses) / max(len(combined_losses), 1),
        "mean_pointer_loss": (
            sum(observed_pointer_losses) / len(observed_pointer_losses)
            if observed_pointer_losses else 0.0
        ),
        "pointer_supervision_examples": float(pointer_supervision_examples),
        "mean_training_pointer_candidates": (
            sum(pointer_candidate_counts) / len(pointer_candidate_counts)
            if pointer_candidate_counts else 0.0
        ),
        "tokens": float(len(observed_nll)),
        "updates": float(updates),
    }



@torch.no_grad()
def evaluate_model(
    model: DANSNFourGateGRU,
    dataset: ContextualBranchDataset,
) -> dict:
    model.eval()
    support_metrics = ingest_heldout_supports(model, dataset)
    value_ids = torch.tensor(
        sorted(set(dataset.metadata["train_mapping"].values())),
        dtype=torch.long,
        device=model.device,
    )

    correct = 0
    value_only_correct = 0
    pointer_correct = 0
    reset_pointer_correct = 0
    counterfactual_pointer_correct = 0
    pointer_target_probs: list[float] = []
    mixed_target_probs: list[float] = []
    reset_pointer_target_probs: list[float] = []
    candidate_counts: list[float] = []
    branch_entropies: list[float] = []
    vector_coverage = 0
    memory_mix: list[float] = []
    pointer_mix: list[float] = []
    query_count = len(dataset.queries)

    for query in dataset.queries:
        logits, history, trace = _run_query(
            model, dataset, query.cue_id, query.key_id, query.distractor_ids
        )
        target = int(query.target_id)
        predicted = int(logits.argmax().item())
        correct += int(predicted == target)
        value_logits = logits.index_select(0, value_ids)
        value_prediction = int(value_ids[value_logits.argmax()].item())
        value_only_correct += int(value_prediction == target)

        probabilities = torch.softmax(logits, dim=0)
        mixed_target_probs.append(float(probabilities[target].cpu()))

        pointer = trace["persistent_token_distribution"]
        candidate_count = int(torch.count_nonzero(pointer).item())
        candidate_counts.append(float(candidate_count))
        pointer_target_probs.append(float(pointer[target].cpu()))
        if candidate_count:
            pointer_correct += int(int(pointer.argmax().item()) == target)
            positive = pointer[pointer > 0]
            branch_entropies.append(float(
                -(positive * positive.clamp_min(1e-12).log()).sum().cpu()
            ))
        else:
            branch_entropies.append(0.0)
        expected_path = (dataset.context_id, query.key_id, target)
        vector_coverage += int(
            expected_path in trace.get("persistent_memory_paths", [])
        )
        memory_mix.append(float(
            trace["effective_memory_mix"].mean().detach().cpu()
        ))
        pointer_mix.append(float(
            trace["effective_pointer_mix"].detach().cpu()
        ))

        # Counterfactual: same key and graph candidates, different cue. This
        # checks whether the context can switch selection to another branch.
        _wrong_logits, _wrong_history, wrong_trace = _run_query(
            model,
            dataset,
            query.alternate_cue_id,
            query.key_id,
            query.distractor_ids,
        )
        wrong_pointer = wrong_trace["persistent_token_distribution"]
        if torch.count_nonzero(wrong_pointer).item():
            counterfactual_pointer_correct += int(
                int(wrong_pointer.argmax().item()) == query.alternate_target_id
            )

        # Diagnostic: hold the entire observed history constant but remove the
        # recurrent state immediately before the key's retrieval step.
        reset_logits, _reset_state, reset_trace = model.forward_step(
            query.key_id, history, model.initial_state()
        )
        reset_pointer = reset_trace["persistent_token_distribution"]
        reset_pointer_target_probs.append(float(reset_pointer[target].cpu()))
        if torch.count_nonzero(reset_pointer).item():
            reset_pointer_correct += int(
                int(reset_pointer.argmax().item()) == target
            )

    denominator = max(query_count, 1)
    return {
        **support_metrics,
        "query_count": query_count,
        "query_top1_accuracy": correct / denominator,
        "query_value_only_top1_accuracy": value_only_correct / denominator,
        "pointer_top1_accuracy": pointer_correct / denominator,
        "pointer_target_probability_mean": (
            sum(pointer_target_probs) / denominator
        ),
        "mixed_target_probability_mean": (
            sum(mixed_target_probs) / denominator
        ),
        "reset_state_pointer_top1_accuracy": reset_pointer_correct / denominator,
        "reset_state_pointer_target_probability_mean": (
            sum(reset_pointer_target_probs) / denominator
        ),
        "counterfactual_pointer_top1_accuracy": (
            counterfactual_pointer_correct / denominator
        ),
        "mean_pointer_candidate_count": (
            sum(candidate_counts) / denominator
        ),
        "mean_pointer_branch_entropy": (
            sum(branch_entropies) / denominator
        ),
        "persistent_vector_query_coverage": vector_coverage / denominator,
        "mean_effective_memory_mix": sum(memory_mix) / denominator,
        "mean_effective_pointer_mix": sum(pointer_mix) / denominator,
        "mean_context_pointer_gain": (
            sum(pointer_target_probs) - sum(reset_pointer_target_probs)
        ) / denominator,
        "graph_stats": model.graph_stats(),
    }


def _parameter_count(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def run_one_seed(
    seed: int,
    epochs: int = 4,
    hidden_size: int = 32,
    chunk_size: int = 64,
    learning_rate: float = 2e-3,
    train_keys: int = 16,
    heldout_keys: int = 8,
    cue_count: int = 3,
    value_count: int = 12,
    train_episodes: int = 4,
    train_delay: int = 8,
    test_delay: int = 8,
    pointer_loss_weight: float = 1.0,
) -> dict:
    if min(epochs, hidden_size, chunk_size) < 1:
        raise ValueError("epochs, hidden_size, and chunk_size must be positive")
    dataset = generate_dataset(
        seed=seed,
        train_keys=train_keys,
        heldout_keys=heldout_keys,
        cue_count=cue_count,
        value_count=value_count,
        train_episodes=train_episodes,
        train_delay=train_delay,
        test_delay=test_delay,
    )

    systems = {
        "four_gate_full": "full",
        "four_gate_no_pointer": "no_pointer",
        "four_gate_no_memory_injection": "no_memory_injection",
        "four_gate_no_persistent_write": "no_persistent_write",
        "four_gate_no_graph_evidence": "no_graph_evidence",
    }
    results = {}
    for system_name, ablation in systems.items():
        # Identical initial weights across four-gate ablations for this seed.
        torch.manual_seed(int(seed) + 1701)
        brain = Brain(vocab=copy.deepcopy(dataset.brain.vocab))
        model = DANSNFourGateGRU(
            brain.engine,
            len(brain.vocab.word_to_id) + 1,
            hidden_size=hidden_size,
            window_size=4,
            graph_depth=3,
            message_rounds=1,
            max_active_nodes=128,
            max_memory_nodes=10_000,
            gate_ablation=ablation,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        train_metrics = [
            train_contextual_stream(
                model,
                dataset,
                optimizer,
                chunk_size=chunk_size,
                pointer_loss_weight=pointer_loss_weight,
            )
            for _ in range(epochs)
        ]
        model.eval()
        evaluation = evaluate_model(model, dataset)
        results[system_name] = {
            **evaluation,
            "parameter_count": _parameter_count(model),
            "last_train_mean_loss": train_metrics[-1]["mean_nll"],
            "last_train_mean_pointer_loss": train_metrics[-1]["mean_pointer_loss"],
            "last_train_pointer_supervision_examples": train_metrics[-1]["pointer_supervision_examples"],
            "last_train_mean_pointer_candidates": train_metrics[-1]["mean_training_pointer_candidates"],
        }

    return {
        "seed": int(seed),
        "dataset": dataset.metadata,
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
    seeds: Sequence[int] = (7, 13, 21),
    **configuration,
) -> dict:
    seed_list = [int(seed) for seed in seeds]
    if not seed_list:
        raise ValueError("at least one seed is required")
    runs = [
        run_one_seed(seed=seed, **configuration)
        for seed in seed_list
    ]
    metrics = (
        "query_top1_accuracy",
        "query_value_only_top1_accuracy",
        "pointer_top1_accuracy",
        "pointer_target_probability_mean",
        "mixed_target_probability_mean",
        "reset_state_pointer_top1_accuracy",
        "reset_state_pointer_target_probability_mean",
        "counterfactual_pointer_top1_accuracy",
        "mean_pointer_candidate_count",
        "mean_pointer_branch_entropy",
        "persistent_vector_write_coverage",
        "persistent_vector_query_coverage",
        "mean_effective_memory_mix",
        "mean_effective_pointer_mix",
        "mean_context_pointer_gain",
        "last_train_mean_loss",
        "last_train_mean_pointer_loss",
        "last_train_pointer_supervision_examples",
        "last_train_mean_pointer_candidates",
        "parameter_count",
    )
    summary = {}
    for system_name in runs[0]["results"]:
        summary[system_name] = {
            metric: _stats([
                float(run["results"][system_name][metric])
                for run in runs
            ])
            for metric in metrics
        }

    return {
        "task": (
            "context-dependent selection among multiple persistent outgoing "
            "DANSN branches for the same key"
        ),
        "protocol": {
            **configuration,
            "seeds": seed_list,
            "candidate_branches_per_key": configuration.get("cue_count", 3),
            "pointer_loss_weight": configuration.get("pointer_loss_weight", 1.0),
            "training": (
                "train on multiple cue->value branches per training key; "
                "query supervision requires using the cue to select among "
                "already stored outgoing candidates; an auxiliary pointer "
                "negative-log-probability loss is applied only when the correct "
                "target is already one of the direct stored candidates"
            ),
            "heldout_support": (
                "one-shot writes for held-out keys after training; each key "
                "receives several candidate continuations under the same "
                "(context,key) parent"
            ),
            "test": (
                "each query starts with a fresh recurrent state, processes its "
                "cue and distractors, then context and key; score before "
                "target reveal; no neural parameter updates or target writes"
            ),
            "counterfactual": (
                "replay the same key and graph candidates with a different cue; "
                "the pointer should select the alternative cue's stored value"
            ),
            "reset_diagnostic": (
                "re-score at the key with the recurrent state zeroed but the "
                "same input history and persistent graph retained"
            ),
            "ablation_meaning": {
                "no_pointer": (
                    "preserve DANSN memory injection into recurrent hidden state "
                    "and decoder vector skip; disable only pointer probability mix"
                ),
                "no_memory_injection": (
                    "disable memory gate influence on hidden state, decoder skip, "
                    "and pointer probability mix"
                ),
                "no_persistent_write": "disable graph and persistent-vector writes",
                "no_graph_evidence": "zero graph evidence features supplied to controller",
            },
        },
        "summary": summary,
        "runs": runs,
        "interpretation": (
            "This is a controlled synthetic mechanism test. A high candidate "
            "count confirms ambiguity is present; high pointer accuracy and "
            "counterfactual cue accuracy indicate contextual branch selection. "
            "If full succeeds but no_pointer fails, the direct pointer is carrying "
            "the decision. If no_pointer also succeeds, the recurrent state and "
            "memory-vector decoder can resolve branches without pointer copying. "
            "High reset-state accuracy would weaken the claim that recurrent cue "
            "state is required. Do not infer general language or semantic reasoning."
        ),
    }


def _dataset_payload(dataset: ContextualBranchDataset) -> dict:
    return {
        "metadata": dataset.metadata,
        "train_tokens": dataset.train_tokens,
        "train_query_positions": sorted(dataset.train_query_positions),
        "train_records": [
            [
                {
                    "token_id": token,
                    "graph_history": list(graph_history),
                    "target_id": target,
                    "supervise_pointer": supervise,
                    "write_target": write_target,
                }
                for token, graph_history, target, supervise, write_target in record
            ]
            for record in dataset.train_records
        ],
        "heldout_supports": [
            {"cue_id": cue, "key_id": key, "target_id": target}
            for cue, key, target in dataset.test_supports
        ],
        "queries": [
            {
                "key_id": query.key_id,
                "cue_id": query.cue_id,
                "target_id": query.target_id,
                "alternate_cue_id": query.alternate_cue_id,
                "alternate_target_id": query.alternate_target_id,
                "input_tokens": list(query.input_tokens),
                "distractor_ids": list(query.distractor_ids),
            }
            for query in dataset.queries
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="7,13,21")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--train-keys", type=int, default=16)
    parser.add_argument("--heldout-keys", type=int, default=8)
    parser.add_argument("--cue-count", type=int, default=3)
    parser.add_argument("--value-count", type=int, default=12)
    parser.add_argument("--train-episodes", type=int, default=4)
    parser.add_argument("--train-delay", type=int, default=8)
    parser.add_argument("--test-delay", type=int, default=8)
    parser.add_argument("--pointer-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-results/contextual_branching_results.json"),
    )
    parser.add_argument(
        "--data-output",
        type=Path,
        default=Path("benchmark-results/contextual_branching_dataset.json"),
    )
    args = parser.parse_args()
    seeds = [
        int(item.strip()) for item in args.seeds.split(",") if item.strip()
    ]
    configuration = {
        "epochs": args.epochs,
        "hidden_size": args.hidden_size,
        "chunk_size": args.chunk_size,
        "learning_rate": args.learning_rate,
        "train_keys": args.train_keys,
        "heldout_keys": args.heldout_keys,
        "cue_count": args.cue_count,
        "value_count": args.value_count,
        "train_episodes": args.train_episodes,
        "train_delay": args.train_delay,
        "test_delay": args.test_delay,
        "pointer_loss_weight": args.pointer_loss_weight,
    }
    results = run_benchmark(seeds=seeds, **configuration)
    rendered = json.dumps(results, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")

    sample = generate_dataset(seed=seeds[0], **{
        key: configuration[key]
        for key in (
            "train_keys", "heldout_keys", "cue_count", "value_count",
            "train_episodes", "train_delay", "test_delay",
        )
    })
    args.data_output.parent.mkdir(parents=True, exist_ok=True)
    args.data_output.write_text(
        json.dumps(_dataset_payload(sample), indent=2) + "\n",
        encoding="utf-8",
    )
    print(rendered)
    print(f"\nWrote {args.output} and {args.data_output}")


if __name__ == "__main__":
    main()
