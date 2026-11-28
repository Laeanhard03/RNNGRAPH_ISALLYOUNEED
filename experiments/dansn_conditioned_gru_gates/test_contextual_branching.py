"""Tests for the ambiguous, context-dependent branch-selection task."""
import copy

import torch

from agi import Brain
from experiments.dansn_conditioned_gru_gates.contextual_branching import (
    _run_query,
    generate_dataset,
    ingest_heldout_supports,
    _normalized_contextual_objective,
    evaluate_training_key_diagnostics,
    train_contextual_stream,
    trace_query_trajectory,
    collect_cue_trajectory_diagnostics,
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
    assert trace["pointer_scores"].shape == (3,)
    assert trace["pointer_candidate_ids"].shape == (3,)
    assert set(trace["pointer_candidate_ids"].tolist()) == {
        value for _cue, support_key, value in dataset.test_supports
        if support_key == query.key_id
    }
    assert trace["previous_hidden"].shape == (model.hidden_size,)
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


def test_selector_loss_is_not_diluted_by_unrelated_token_events():
    """Selector loss is averaged over eligible branch queries, not all tokens."""
    nll_losses = [
        torch.tensor(2.0, requires_grad=True) for _ in range(64)
    ]
    pointer_losses = [
        torch.tensor(1.1, requires_grad=True) for _ in range(5)
    ]

    objective = _normalized_contextual_objective(
        nll_losses,
        pointer_losses,
        pointer_loss_weight=1.0,
    )

    assert torch.allclose(objective.detach(), torch.tensor(3.1), atol=1e-6)
    objective.backward()
    assert torch.allclose(
        pointer_losses[0].grad, torch.tensor(1.0 / 5), atol=1e-6
    )
    assert torch.allclose(
        nll_losses[0].grad, torch.tensor(1.0 / 64), atol=1e-6
    )


def test_nonlinear_pointer_scorer_receives_candidate_ranking_gradient():
    torch.manual_seed(263)
    dataset = generate_dataset(
        seed=263,
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
    query = dataset.queries[0]
    state = model.initial_state()
    events = [
        (query.cue_id, (query.cue_id,)),
        *[(int(token), (int(token),)) for token in query.distractor_ids],
        (dataset.context_id, (dataset.context_id,)),
        (query.key_id, (dataset.context_id, query.key_id)),
    ]
    trace = None
    for token_id, graph_history in events:
        _logits, state, trace = model.forward_step(
            int(token_id), graph_history, state
        )

    assert trace is not None
    assert trace["pointer_available"] is True
    pointer = trace["persistent_token_distribution"]
    assert pointer[query.target_id].item() > 0.0
    selector_loss = -torch.log(pointer[query.target_id].clamp_min(1e-9))
    selector_loss.backward()

    scorer_gradient = model.pointer_scorer[0].weight.grad
    assert scorer_gradient is not None
    assert torch.isfinite(scorer_gradient).all()
    assert torch.linalg.vector_norm(scorer_gradient).item() > 0.0


def test_recurrent_query_state_is_traceable_and_cue_sensitive():
    torch.manual_seed(269)
    dataset = generate_dataset(
        seed=269,
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
    distractors = dataset.queries[0].distractor_ids
    cue_ids = sorted({cue for cue, support_key, _value in dataset.test_supports
                      if support_key == key_id})
    states = []
    for cue_id in cue_ids:
        _logits, _history, trace = _run_query(
            model, dataset, cue_id, key_id, distractors
        )
        states.append(trace["previous_hidden"])
    assert len(states) == 3
    assert all(torch.isfinite(state).all() for state in states)
    assert any(
        torch.linalg.vector_norm(states[i] - states[j]).item() > 1e-6
        for i in range(len(states))
        for j in range(i + 1, len(states))
    )


def test_time_resolved_cue_trace_reports_gate_and_state_changes():
    torch.manual_seed(271)
    dataset = generate_dataset(
        seed=271,
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
    query = dataset.queries[0]

    trajectory = trace_query_trajectory(
        model,
        dataset,
        query.cue_id,
        query.key_id,
        query.distractor_ids,
    )
    assert [row["stage"] for row in trajectory] == [
        "cue",
        "distractor_1",
        "distractor_2",
        "distractor_3",
        "context",
        "key",
    ]
    required_metrics = (
        "reset_gate_mean",
        "update_gate_mean",
        "memory_injection_gate_mean",
        "effective_memory_mix_mean",
        "reset_retention_ratio",
        "carry_retention_ratio",
        "hidden_norm",
        "recurrent_hidden_norm",
    )
    for row in trajectory:
        assert torch.isfinite(row["hidden"]).all()
        assert torch.isfinite(row["previous_hidden"]).all()
        for metric in required_metrics:
            assert torch.isfinite(torch.tensor(row[metric]))

    diagnostics = collect_cue_trajectory_diagnostics(model, dataset)
    assert diagnostics["matched_distractors_per_key"] is True
    assert diagnostics["keys_measured"] == 2
    assert [row["stage"] for row in diagnostics["stages"]] == [
        "cue",
        "distractor_1",
        "distractor_2",
        "distractor_3",
        "context",
        "key",
    ]
    key_metrics = diagnostics["stages"][-1]
    assert torch.isfinite(torch.tensor(
        key_metrics["previous_hidden_cosine"]
    ))
    assert torch.isfinite(torch.tensor(
        key_metrics["pointer_query_l2"]
    ))


def test_protected_context_makes_selector_query_depend_on_cue_after_matched_delay():
    torch.manual_seed(277)
    dataset = generate_dataset(
        seed=277,
        train_keys=6,
        heldout_keys=2,
        cue_count=3,
        value_count=8,
        train_episodes=1,
        train_delay=2,
        test_delay=4,
    )
    baseline = _make_model(dataset)
    model = DANSNFourGateGRU(
        baseline.engine,
        baseline.embedding.num_embeddings,
        hidden_size=baseline.hidden_size,
        window_size=baseline.window_size,
        graph_depth=baseline.graph_depth,
        message_rounds=baseline.message_rounds,
        max_active_nodes=baseline.max_active_nodes,
        max_memory_nodes=baseline.max_memory_nodes,
        transition_variant="protected_context",
    )
    with torch.no_grad():
        model.four_gate_cell.controller.weight.zero_()
        model.four_gate_cell.controller.bias.zero_()
        model.four_gate_cell.controller.bias[
            3 * model.hidden_size:4 * model.hidden_size
        ].fill_(20.0)
        model.four_gate_cell.controller.bias[
            2 * model.hidden_size:3 * model.hidden_size
        ].fill_(2.0)
    ingest_heldout_supports(model, dataset)
    query = next(q for q in dataset.queries if q.key_id == dataset.queries[0].key_id)
    cue_ids = sorted({
        cue for cue, key, _value in dataset.test_supports if key == query.key_id
    })
    traces = [
        _run_query(model, dataset, cue, query.key_id, query.distractor_ids)[2]
        for cue in cue_ids
    ]
    assert all(trace["pointer_available"] for trace in traces)
    assert all(trace["protected_context_enabled"].item() == 1.0 for trace in traces)
    assert any(
        torch.linalg.vector_norm(
            traces[i]["pointer_query"] - traces[j]["pointer_query"]
        ).item() > 1e-4
        for i in range(len(traces))
        for j in range(i + 1, len(traces))
    )


def test_training_key_diagnostics_uses_same_three_candidate_query_protocol():
    torch.manual_seed(281)
    dataset = generate_dataset(
        seed=281,
        train_keys=6,
        heldout_keys=2,
        cue_count=3,
        value_count=8,
        train_episodes=1,
        train_delay=2,
        test_delay=3,
    )
    model = _make_model(dataset)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    train_contextual_stream(
        model, dataset, optimizer, chunk_size=32, pointer_loss_weight=1.0
    )
    diagnostics = evaluate_training_key_diagnostics(model, dataset)
    assert diagnostics["query_count"] == 18
    assert diagnostics["mean_candidate_count"] == 3.0
    assert diagnostics["persistent_vector_query_coverage"] == 1.0
    assert 0.0 <= diagnostics["pointer_top1_accuracy"] <= 1.0
    assert 0.0 <= diagnostics["counterfactual_pointer_top1_accuracy"] <= 1.0
    assert torch.isfinite(torch.tensor(diagnostics["mean_target_score_margin"]))


def test_context_write_variant_ingests_cue_bearing_candidate_vectors():
    torch.manual_seed(283)
    dataset = generate_dataset(
        seed=283,
        train_keys=6,
        heldout_keys=2,
        cue_count=3,
        value_count=8,
        train_episodes=1,
        train_delay=2,
        test_delay=3,
    )
    baseline = _make_model(dataset)
    model = DANSNFourGateGRU(
        baseline.engine,
        baseline.embedding.num_embeddings,
        hidden_size=baseline.hidden_size,
        window_size=baseline.window_size,
        graph_depth=baseline.graph_depth,
        message_rounds=baseline.message_rounds,
        max_active_nodes=baseline.max_active_nodes,
        max_memory_nodes=baseline.max_memory_nodes,
        transition_variant="protected_context_write",
    )
    with torch.no_grad():
        model.four_gate_cell.controller.weight.zero_()
        model.four_gate_cell.controller.bias.zero_()
        model.four_gate_cell.controller.bias[
            3 * model.hidden_size:4 * model.hidden_size
        ].fill_(20.0)
        model.four_gate_cell.controller.bias[
            2 * model.hidden_size:3 * model.hidden_size
        ].fill_(2.0)
    ingest_heldout_supports(model, dataset)
    query = dataset.queries[0]
    candidates = [
        path for path in model.node_memory
        if len(path) == 3 and path[:2] == (dataset.context_id, query.key_id)
    ]
    assert len(candidates) == 3
    vectors = [model.node_memory[path] for path in candidates]
    assert all(torch.isfinite(vector).all() for vector in vectors)
    assert any(
        torch.linalg.vector_norm(vectors[i] - vectors[j]).item() > 1e-6
        for i in range(len(vectors))
        for j in range(i + 1, len(vectors))
    )
    _logits, _history, trace = _run_query(
        model, dataset, query.cue_id, query.key_id, query.distractor_ids
    )
    assert trace["context_write_enabled"].item() == 1.0
    assert torch.count_nonzero(trace["persistent_token_distribution"]).item() == 3

def test_query_aligned_candidate_write_stores_the_pre_key_query_state():
    """The persistent candidate must describe the state used to score its edge."""
    torch.manual_seed(293)
    dataset = generate_dataset(
        seed=293,
        train_keys=6,
        heldout_keys=2,
        cue_count=3,
        value_count=8,
        train_episodes=1,
        train_delay=2,
        test_delay=3,
    )
    baseline = _make_model(dataset)
    model = DANSNFourGateGRU(
        baseline.engine,
        baseline.embedding.num_embeddings,
        hidden_size=baseline.hidden_size,
        window_size=baseline.window_size,
        graph_depth=baseline.graph_depth,
        message_rounds=baseline.message_rounds,
        max_active_nodes=baseline.max_active_nodes,
        max_memory_nodes=baseline.max_memory_nodes,
        transition_variant="protected_context_query_aligned_write",
    )
    with torch.no_grad():
        model.four_gate_cell.controller.weight.zero_()
        model.four_gate_cell.controller.bias.zero_()
        model.four_gate_cell.controller.bias[
            3 * model.hidden_size:4 * model.hidden_size
        ].fill_(20.0)
        model.four_gate_cell.controller.bias[
            2 * model.hidden_size:3 * model.hidden_size
        ].fill_(2.0)

    query = dataset.queries[0]
    state = model.initial_state()
    for token_id, graph_history in [
        (query.cue_id, (query.cue_id,)),
        (dataset.context_id, (dataset.context_id,)),
        (query.key_id, (dataset.context_id, query.key_id)),
    ]:
        _logits, state, trace = model.forward_step(
            int(token_id), graph_history, state
        )

    assert trace["context_write_enabled"].item() == 1.0
    assert state.write_query_hidden is not None
    assert state.write_query_context is not None
    assert torch.equal(state.write_query_hidden, trace["previous_hidden"])
    assert torch.equal(
        state.write_query_context, trace["previous_context_hidden"]
    )

    target_id = int(query.target_id)
    expected_input = torch.cat((
        trace["previous_hidden"],
        trace["previous_context_hidden"],
        model.embedding(torch.tensor(target_id, dtype=torch.long)),
        state.readout,
    ))
    expected_candidate = model.memory_norm(
        torch.tanh(model.write_candidate(expected_input))
    )
    model.observe_target(
        (dataset.context_id, query.key_id), target_id, state
    )
    stored = model.node_memory[
        (dataset.context_id, query.key_id, target_id)
    ]
    assert torch.allclose(stored, expected_candidate, atol=1e-6, rtol=1e-6)

def test_pointer_query_key_embedding_ablation_keeps_recurrent_key_input():
    """Removing key identity from ranking must not remove the key from recurrence."""
    torch.manual_seed(307)
    dataset = generate_dataset(
        seed=307,
        train_keys=6,
        heldout_keys=2,
        cue_count=3,
        value_count=8,
        train_episodes=1,
        train_delay=2,
        test_delay=3,
    )
    baseline = _make_model(dataset)
    model = DANSNFourGateGRU(
        baseline.engine,
        baseline.embedding.num_embeddings,
        hidden_size=baseline.hidden_size,
        window_size=baseline.window_size,
        graph_depth=baseline.graph_depth,
        message_rounds=baseline.message_rounds,
        max_active_nodes=baseline.max_active_nodes,
        max_memory_nodes=baseline.max_memory_nodes,
        transition_variant="protected_context_no_key_embedding",
    )
    h = model.hidden_size
    with torch.no_grad():
        model.query_projection.weight.zero_()
        model.query_projection.weight[:, :h].copy_(torch.eye(h))
        model.query_projection.weight[:, h:].zero_()
        model.query_projection.bias.zero_()
    key_a = torch.linspace(-0.9, 0.9, h)
    key_b = torch.linspace(0.8, -0.8, h)
    selection = torch.linspace(-0.4, 0.6, h)
    query_a, query_token_a = model._pointer_query(key_a, selection)
    query_b, query_token_b = model._pointer_query(key_b, selection)
    assert torch.count_nonzero(query_token_a).item() == 0
    assert torch.count_nonzero(query_token_b).item() == 0
    assert torch.allclose(query_a, selection)
    assert torch.allclose(query_b, selection)

    query = dataset.queries[0]
    _logits, _state, trace = model.forward_step(
        int(query.key_id),
        (dataset.context_id, query.key_id),
        model.initial_state(),
    )
    actual_key_embedding = model.embedding(
        torch.tensor(query.key_id, dtype=torch.long)
    )
    assert torch.allclose(trace["recurrent_input_embedding"], actual_key_embedding)
    assert torch.count_nonzero(trace["pointer_query_token_embedding"]).item() == 0
    assert trace["pointer_key_embedding_enabled"].item() == 0.0


def test_key_embedding_enabled_pointer_query_remains_key_sensitive():
    torch.manual_seed(311)
    dataset = generate_dataset(
        seed=311,
        train_keys=6,
        heldout_keys=2,
        cue_count=3,
        value_count=8,
        train_episodes=1,
        train_delay=2,
        test_delay=3,
    )
    model = _make_model(dataset)
    h = model.hidden_size
    with torch.no_grad():
        model.query_projection.weight.zero_()
        model.query_projection.weight[:, :h].copy_(torch.eye(h))
        model.query_projection.weight[:, h:].zero_()
        model.query_projection.bias.zero_()
    key_a = torch.linspace(-0.9, 0.9, h)
    key_b = torch.linspace(0.8, -0.8, h)
    selection = torch.zeros(h)
    query_a, key_input_a = model._pointer_query(key_a, selection)
    query_b, key_input_b = model._pointer_query(key_b, selection)
    assert torch.allclose(key_input_a, key_a)
    assert torch.allclose(key_input_b, key_b)
    assert not torch.allclose(query_a, query_b)

