"""Tests for the DANSN four-gate GRU architecture."""
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from agi import Brain
from experiments.dansn_conditioned_gru_gates.four_gate import (
    DANSNFourGateCell,
    DANSNFourGateGRU,
    FourGateState,
    train_stream,
)


def make_brain_and_stream():
    brain = Brain()
    ids = brain.vocab.tokenize(
        "red fox jumps over blue lake red fox jumps over blue lake"
    )
    return brain, ids


def make_model(brain, **overrides):
    config = dict(
        hidden_size=16,
        window_size=4,
        graph_depth=2,
        message_rounds=1,
        max_active_nodes=24,
        max_memory_nodes=200,
    )
    config.update(overrides)
    return DANSNFourGateGRU(
        brain.engine, len(brain.vocab.word_to_id) + 1, **config
    )


def test_cell_emits_four_distinct_vector_gates_and_valid_state():
    torch.manual_seed(11)
    cell = DANSNFourGateCell(hidden_size=12)
    x = torch.randn(12)
    h = torch.randn(12)
    memory = torch.randn(12)
    evidence = torch.rand(5)

    hidden, gates = cell(x, h, memory, evidence)

    for name in (
        "update_gate",
        "reset_gate",
        "memory_injection_gate",
        "persistent_write_gate",
    ):
        value = gates[name]
        assert value.shape == (12,)
        assert torch.all(value >= 0.0)
        assert torch.all(value <= 1.0)
    assert hidden.shape == (12,)


def test_forward_calculates_four_gates_without_writing_graph_memory():
    torch.manual_seed(13)
    brain, ids = make_brain_and_stream()
    model = make_model(brain)
    nodes_before = model.graph_stats()["nodes"]

    logits, state, trace = model.forward_step(ids[0], [ids[0]])

    assert logits.shape == (len(brain.vocab.word_to_id) + 1,)
    assert torch.isfinite(logits).all()
    assert isinstance(state, FourGateState)
    assert state.write_gate is not None
    assert trace["graph_evidence"].shape == (5,)
    assert model.graph_stats()["nodes"] == nodes_before
    assert model.graph_stats()["memory_nodes"] == 0


def test_memory_injection_gate_controls_memory_contribution():
    torch.manual_seed(17)
    cell = DANSNFourGateCell(hidden_size=8)
    x = torch.randn(8)
    h = torch.randn(8)
    memory = torch.randn(8)
    evidence = torch.rand(5)

    hidden_full, gates_full = cell(x, h, memory, evidence)
    hidden_no_memory, gates_no_memory = cell(
        x, h, memory, evidence, memory_enabled=False
    )

    assert torch.equal(
        gates_no_memory["memory_injection_gate"], torch.zeros_like(memory)
    )
    assert not torch.allclose(hidden_full, hidden_no_memory)


def test_write_gate_zero_skips_graph_and_persistent_vector_writes():
    torch.manual_seed(19)
    brain, ids = make_brain_and_stream()
    model = make_model(brain)
    _, state, _ = model.forward_step(ids[0], [ids[0]])
    state.write_gate = torch.zeros_like(state.write_gate)

    before_nodes = model.graph_stats()["nodes"]
    before_frequency = float(model.engine.root.frequency)
    path = model.observe_target([ids[0]], ids[1], state)

    assert model.graph_stats()["nodes"] == before_nodes
    assert float(model.engine.root.frequency) == before_frequency
    assert model.graph_stats()["memory_nodes"] == 0
    assert path not in model.node_memory


def test_confidence_zero_prevents_graph_and_persistent_writes_exactly():
    torch.manual_seed(23)
    brain, ids = make_brain_and_stream()
    model = make_model(brain)
    _, state, _ = model.forward_step(ids[0], [ids[0]])
    before_nodes = model.graph_stats()["nodes"]
    before_frequency = float(model.engine.root.frequency)

    model.observe_target([ids[0]], ids[1], state, confidence=0.0)

    assert model.graph_stats()["nodes"] == before_nodes
    assert float(model.engine.root.frequency) == before_frequency
    assert model.graph_stats()["memory_nodes"] == 0


def test_write_gate_strength_scales_graph_path_reinforcement():
    torch.manual_seed(27)
    brain, ids = make_brain_and_stream()
    model = make_model(brain)
    _, state, _ = model.forward_step(ids[0], [ids[0]])
    state.write_gate = torch.full_like(state.write_gate, 0.4)

    model.observe_target([ids[0]], ids[1], state, confidence=0.5)

    observed_path = tuple(ids[:2])
    assert observed_path in model._path_nodes
    assert abs(float(model._path_nodes[observed_path].frequency) - 0.2) < 1e-6
    assert observed_path in model.node_memory


def test_write_gate_is_applied_after_target_scoring_and_controls_write_strength():
    torch.manual_seed(29)
    brain, ids = make_brain_and_stream()
    model = make_model(brain)
    _, state, _ = model.forward_step(ids[0], [ids[0]])
    state.write_gate = torch.ones_like(state.write_gate)

    path = model.observe_target([ids[0]], ids[1], state, confidence=0.8)

    assert path in model.node_memory
    assert model.node_memory_writes[path] == 1
    assert model.node_memory_quality[path] > 0.0


def test_loss_backpropagates_through_all_four_gate_paths():
    torch.manual_seed(31)
    brain, ids = make_brain_and_stream()
    model = make_model(brain)

    # Step one produces a write gate before ids[1] is revealed.
    _, first_state, first_trace = model.forward_step(ids[0], [ids[0]])
    assert torch.isfinite(first_trace["persistent_write_gate"]).all()
    model.observe_target([ids[0]], ids[1], first_state)

    # The following prediction retrieves the written vector, so its loss can
    # backpropagate through the earlier persistent-write gate as well.
    logits, _state, trace = model.forward_step(
        ids[1], ids[:2], first_state
    )
    target = torch.tensor([ids[2]], dtype=torch.long)
    loss = F.cross_entropy(logits.unsqueeze(0), target)
    loss.backward()

    for name in (
        "update_gate",
        "reset_gate",
        "memory_injection_gate",
        "persistent_write_gate",
    ):
        assert torch.isfinite(trace[name]).all()

    grad = model.four_gate_cell.controller.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert torch.linalg.vector_norm(grad).item() > 0
    write_grad = grad[3 * model.hidden_size:4 * model.hidden_size]
    assert torch.linalg.vector_norm(write_grad).item() > 0


def test_training_updates_parameters_and_uses_four_gate_state_detach():
    torch.manual_seed(37)
    brain, ids = make_brain_and_stream()
    model = make_model(brain)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    before = model.four_gate_cell.controller.weight.detach().clone()

    metrics = train_stream(model, ids * 2, optimizer, chunk_size=3)

    assert metrics["tokens"] == float(len(ids * 2) - 1)
    assert metrics["updates"] > 0
    assert torch.isfinite(torch.tensor(metrics["mean_loss"]))
    assert not torch.equal(before, model.four_gate_cell.controller.weight.detach())
    assert model.graph_stats()["nodes"] > 0
    assert all(not value.requires_grad for value in model.node_memory.values())


def test_checkpoint_round_trip_preserves_four_gate_weights_and_memory():
    torch.manual_seed(41)
    brain, ids = make_brain_and_stream()
    model = make_model(brain)
    _, state, _ = model.forward_step(ids[0], [ids[0]])
    state.write_gate = torch.ones_like(state.write_gate)
    model.observe_target([ids[0]], ids[1], state)

    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = Path(tmp) / "four_gate.pt"
        model.save_checkpoint(checkpoint)
        restored = make_model(brain)
        restored.load_checkpoint(checkpoint)

    assert set(restored.node_memory) == set(model.node_memory)
    for key in model.node_memory:
        assert torch.equal(restored.node_memory[key], model.node_memory[key])
    for key, value in model.state_dict().items():
        assert torch.equal(restored.state_dict()[key], value)


def test_four_gate_checkpoint_rejects_other_architectures():
    import pytest

    brain, _ = make_brain_and_stream()
    model = make_model(brain)
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = Path(tmp) / "wrong.pt"
        torch.save(
            {"architecture": "dansn_conditioned_gru_gates_v1"},
            checkpoint,
        )
        with pytest.raises(ValueError, match="four-gate"):
            model.load_checkpoint(checkpoint)


def test_four_gate_ablations_preserve_shapes_and_disable_selected_paths():
    torch.manual_seed(43)
    models = []
    streams = []
    for ablation in (
        "full",
        "no_memory_injection",
        "no_persistent_write",
        "no_graph_evidence",
    ):
        brain, ids = make_brain_and_stream()
        models.append(make_model(brain, gate_ablation=ablation))
        streams.append((brain, ids))

    reference_shapes = {
        name: tuple(value.shape)
        for name, value in models[0].state_dict().items()
    }
    for model in models[1:]:
        assert {
            name: tuple(value.shape)
            for name, value in model.state_dict().items()
        } == reference_shapes

    _, no_memory_state, no_memory_trace = models[1].forward_step(
        streams[1][1][0], [streams[1][1][0]]
    )
    assert torch.count_nonzero(
        no_memory_trace["memory_injection_gate"]
    ).item() == 0

    _, no_write_state, _ = models[2].forward_step(
        streams[2][1][0], [streams[2][1][0]]
    )
    assert torch.count_nonzero(no_write_state.write_gate).item() == 0
    before_nodes = models[2].graph_stats()["nodes"]
    before_frequency = float(models[2].engine.root.frequency)
    models[2].observe_target(
        [streams[2][1][0]], streams[2][1][1], no_write_state
    )
    assert models[2].graph_stats()["nodes"] == before_nodes
    assert float(models[2].engine.root.frequency) == before_frequency
    assert models[2].graph_stats()["memory_nodes"] == 0

    _, _, no_evidence_trace = models[3].forward_step(
        streams[3][1][0], [streams[3][1][0]]
    )
    assert torch.count_nonzero(no_evidence_trace["graph_evidence"]).item() == 0
