"""Tests for the ambiguous, context-dependent branch-selection task."""
import copy

import torch

from agi import Brain
from experiments.dansn_conditioned_gru_gates.contextual_branching import (
    _run_query,
    generate_dataset,
    ingest_heldout_supports,
)
from experiments.dansn_conditioned_gru_gates.four_gate import DANSNFourGateGRU


def _make_model(dataset, hidden_size=16):
    brain = Brain(vocab=copy.deepcopy(dataset.brain.vocab))
    model = DANSNFourGateGRU(
        brain.engine,
        len(brain.vocab.word_to_id) + 1,
        hidden_size=hidden_size,
        window_size=4,
        graph_depth=3,
        message_rounds=1,
        max_active_nodes=64,
        max_memory_nodes=500,
    )
    # Deterministically open the write gate so this test isolates retrieval.
    with torch.no_grad():
        model.four_gate_cell.controller.weight.zero_()
        model.four_gate_cell.controller.bias.zero_()
        model.four_gate_cell.controller.bias[
            3 * hidden_size:4 * hidden_size
        ].fill_(20.0)
        model.four_gate_cell.controller.bias[
            2 * hidden_size:3 * hidden_size
        ].fill_(2.0)
    return model


def test_contextual_dataset_gives_each_heldout_key_multiple_distinct_branches():
    dataset = generate_dataset(
        seed=211,
        train_keys=6,
        heldout_keys=4,
        cue_count=3,
        value_count=8,
        train_episodes=2,
        train_delay=2,
        test_delay=4,
    )
    assert dataset.metadata["train_and_heldout_key_ids_disjoint"] is True
    assert dataset.metadata["all_value_classes_seen_in_training"] is True
    assert dataset.metadata["heldout_query_count"] == 12
    for key_id in {key for _cue, key, _value in dataset.test_supports}:
        values = {
            value for _cue, key, value in dataset.test_supports if key == key_id
        }
        assert len(values) == 3


def test_context_query_inputs_do_not_contain_the_answer_token():
    dataset = generate_dataset(
        seed=223,
        train_keys=6,
        heldout_keys=3,
        cue_count=3,
        value_count=8,
        train_episodes=1,
        train_delay=1,
        test_delay=3,
    )
    for query in dataset.queries:
        assert query.target_id not in query.input_tokens
        assert query.alternate_target_id not in query.input_tokens
        assert query.target_id != query.alternate_target_id


def test_same_key_exposes_multiple_persistent_pointer_candidates():
    torch.manual_seed(227)
    dataset = generate_dataset(
        seed=227,
        train_keys=6,
        heldout_keys=2,
        cue_count=3,
        value_count=8,
        train_episodes=1,
        train_delay=1,
        test_delay=3,
    )
    model = _make_model(dataset)
    support = ingest_heldout_supports(model, dataset)
    assert support["persistent_vector_write_coverage"] == 1.0

    query = dataset.queries[0]
    logits, _history, trace = _run_query(
        model,
        dataset,
        query.cue_id,
        query.key_id,
        query.distractor_ids,
    )
    pointer = trace["persistent_token_distribution"]
    assert torch.isfinite(logits).all()
    assert trace["pointer_available"] is True
    assert torch.count_nonzero(pointer).item() == 3
    assert torch.isclose(pointer.sum(), torch.tensor(1.0), atol=1e-6)
    for _cue, key_id, value_id in dataset.test_supports:
        if key_id == query.key_id:
            assert (dataset.context_id, key_id, value_id) in trace[
                "persistent_memory_paths"
            ]
