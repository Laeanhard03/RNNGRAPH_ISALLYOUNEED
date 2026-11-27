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
    assert dataset.metadata["graph_context_separated_from_recurrent_cue"] is True

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


def test_default_training_delay_covers_default_evaluation_delay():
    dataset = generate_dataset(
        seed=229,
        train_keys=6,
        heldout_keys=3,
        cue_count=3,
        value_count=8,
        train_episodes=1,
    )
    assert dataset.metadata["train_query_delay_max"] == 8
    assert dataset.metadata["test_query_delay"] == 8
    assert dataset.metadata["test_delay_is_within_training_range"] is True


def test_training_query_positions_point_to_key_and_revealed_value():
    dataset = generate_dataset(
        seed=233,
        train_keys=6,
        heldout_keys=3,
        cue_count=3,
        value_count=8,
        train_episodes=2,
        train_delay=2,
        test_delay=3,
    )
    expected_count = 2 * 6 * 3
    assert len(dataset.train_query_positions) == expected_count
    for position in dataset.train_query_positions:
        key_id = dataset.train_tokens[position]
        target_id = dataset.train_tokens[position + 1]
        assert dataset.train_tokens[position - 1] == dataset.context_id
        assert any(
            int(mapping_key.split(":")[0]) == key_id
            and target_id == value_id
            for mapping_key, value_id in dataset.metadata[
                "train_mapping"
            ].items()
        )


def test_auxiliary_pointer_loss_trains_only_stored_query_candidates():
    from experiments.dansn_conditioned_gru_gates.contextual_branching import (
        train_contextual_stream,
    )

    torch.manual_seed(239)
    dataset = generate_dataset(
        seed=239,
        train_keys=6,
        heldout_keys=2,
        cue_count=3,
        value_count=8,
        train_episodes=2,
        train_delay=2,
        test_delay=3,
    )
    model = _make_model(dataset)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
    before = model.query_projection.weight.detach().clone()

    metrics = train_contextual_stream(
        model,
        dataset,
        optimizer,
        chunk_size=32,
        pointer_loss_weight=1.0,
    )

    assert metrics["pointer_supervision_examples"] == len(
        dataset.train_query_positions
    )
    assert metrics["mean_training_pointer_candidates"] == 3.0
    assert metrics["mean_pointer_loss"] > 0.0
    assert torch.isfinite(torch.tensor(metrics["mean_pointer_loss"]))
    assert not torch.allclose(before, model.query_projection.weight.detach())


def test_cue_is_recurrent_context_but_does_not_split_dansn_candidate_parent():
    torch.manual_seed(241)
    dataset = generate_dataset(
        seed=241,
        train_keys=6,
        heldout_keys=2,
        cue_count=3,
        value_count=8,
        train_episodes=1,
        train_delay=2,
        test_delay=3,
    )
    model = _make_model(dataset)
    ingest_heldout_supports(model, dataset)

    key_id = dataset.queries[0].key_id
    stored_values = {
        value_id
        for _cue, support_key, value_id in dataset.test_supports
        if support_key == key_id
    }
    assert len(stored_values) == 3
    assert all(
        (dataset.context_id, key_id, value_id) in model.node_memory
        for value_id in stored_values
    )

    query_a = next(q for q in dataset.queries if q.key_id == key_id)
    _logits, _history, trace = _run_query(
        model,
        dataset,
        query_a.cue_id,
        key_id,
        query_a.distractor_ids,
    )
    assert torch.count_nonzero(
        trace["persistent_token_distribution"]
    ).item() == 3


def test_training_query_records_are_read_only_after_supports_are_written():
    dataset = generate_dataset(
        seed=251,
        train_keys=6,
        heldout_keys=2,
        cue_count=3,
        value_count=8,
        train_episodes=2,
        train_delay=2,
        test_delay=3,
    )
    query_records = [
        record
        for record in dataset.train_records
        if any(event[3] for event in record)
    ]
    support_records = [
        record
        for record in dataset.train_records
        if all(event[4] for event in record) and len(record) == 3
    ]

    assert len(query_records) == 2 * 6 * 3
    assert len(support_records) == 2 * 6 * 3
    assert all(not event[4] for record in query_records for event in record)
    assert all(event[4] for record in support_records for event in record)
    assert all(
        record[-1][3] is True and record[-1][4] is False
        for record in query_records
    )


def test_heldout_support_ingestion_writes_shared_parent_memory():
    torch.manual_seed(257)
    dataset = generate_dataset(
        seed=257,
        train_keys=6,
        heldout_keys=2,
        cue_count=3,
        value_count=8,
        train_episodes=1,
        train_delay=1,
        test_delay=2,
    )
    model = _make_model(dataset)
    ingest_heldout_supports(model, dataset)

    key_id = dataset.queries[0].key_id
    assert (dataset.context_id, key_id) in model.node_memory
    for cue_id, support_key, value_id in dataset.test_supports:
        if support_key == key_id:
            assert (dataset.context_id, key_id, value_id) in model.node_memory
