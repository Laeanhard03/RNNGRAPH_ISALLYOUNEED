"""Tests for the isolated DANSN-conditioned GRU gate experiment."""
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from agi import Brain
from experiments.dansn_conditioned_gru_gates.model import (
    DANSNConditionedGRU,
    DANSNConditionedGRUCell,
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
    return DANSNConditionedGRU(
        brain.engine, len(brain.vocab.word_to_id) + 1, **config
    )


def test_graph_context_changes_reset_and_update_gate_values():
    torch.manual_seed(13)
    cell = DANSNConditionedGRUCell(input_size=8, hidden_size=8)
    inputs = torch.randn(8)
    hidden = torch.randn(8)
    no_context = torch.zeros(8)
    retrieved_context = torch.randn(8)

    _, gates_without = cell(inputs, hidden, no_context)
    _, gates_with = cell(inputs, hidden, retrieved_context)

    assert not torch.allclose(
        gates_without["update_gate"], gates_with["update_gate"]
    )
    assert not torch.allclose(
        gates_without["reset_gate"], gates_with["reset_gate"]
    )


def test_prediction_is_read_only_then_dansn_context_conditions_gru_gates():
    torch.manual_seed(7)
    brain, ids = make_brain_and_stream()
    model = make_model(brain)
    before = model.graph_stats()

    first_logits, state, first_trace = model.forward_step(ids[0], [ids[0]])
    assert first_logits.shape == (len(brain.vocab.word_to_id) + 1,)
    assert torch.isfinite(first_logits).all()
    assert model.graph_stats()["nodes"] == before["nodes"]
    assert model.graph_stats()["memory_nodes"] == 0
    assert first_trace["matched_path"] is None

    written_path = model.observe_target([ids[0]], ids[1], state)
    assert model.graph_stats()["nodes"] > before["nodes"]
    assert model.graph_stats()["memory_nodes"] == 1

    logits, next_state, trace = model.forward_step(ids[1], ids[:2], state)
    assert logits.shape == first_logits.shape
    assert torch.isfinite(logits).all()
    assert trace["matched_path"] == written_path
    assert trace["active_paths"]
    assert torch.linalg.vector_norm(trace["graph_readout"]).item() > 0

    for gate_name in ("update_gate", "reset_gate"):
        gate = trace[gate_name]
        assert gate.shape == (model.hidden_size,)
        assert torch.all(gate >= 0) and torch.all(gate <= 1)

    assert next_state.hidden.shape == (model.hidden_size,)


def test_loss_backpropagates_through_graph_conditioned_gates():
    torch.manual_seed(19)
    brain, ids = make_brain_and_stream()
    model = make_model(brain)

    _, first_state, _ = model.forward_step(ids[0], [ids[0]])
    model.observe_target([ids[0]], ids[1], first_state)

    logits, _state, trace = model.forward_step(ids[1], ids[:2], first_state)
    target = torch.tensor([ids[2]], dtype=torch.long)
    loss = F.cross_entropy(logits.unsqueeze(0), target)
    loss.backward()

    grad = model.gate_cell.context_projection.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert torch.linalg.vector_norm(grad).item() > 0
    assert torch.linalg.vector_norm(trace["graph_readout"]).item() > 0


def test_training_stream_updates_gate_parameters_and_persistent_memory():
    torch.manual_seed(23)
    brain, ids = make_brain_and_stream()
    model = make_model(brain)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    before = model.gate_cell.context_projection.weight.detach().clone()

    metrics = train_stream(model, ids * 2, optimizer, chunk_size=3)

    assert metrics["tokens"] == float(len(ids * 2) - 1)
    assert metrics["updates"] > 0
    assert torch.isfinite(torch.tensor(metrics["mean_loss"]))
    assert not torch.equal(
        before, model.gate_cell.context_projection.weight.detach()
    )
    assert model.graph_stats()["memory_nodes"] > 0
    assert all(not value.requires_grad for value in model.node_memory.values())


def test_checkpoint_round_trip_preserves_gate_weights_and_sparse_memory():
    torch.manual_seed(29)
    brain, ids = make_brain_and_stream()
    model = make_model(brain)
    _, state, _ = model.forward_step(ids[0], [ids[0]])
    model.observe_target([ids[0]], ids[1], state)

    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = Path(tmp) / "dansn_conditioned_gru.pt"
        model.save_checkpoint(checkpoint)

        restored = make_model(brain)
        restored.load_checkpoint(checkpoint)

    assert set(restored.node_memory) == set(model.node_memory)
    for key in model.node_memory:
        assert torch.equal(restored.node_memory[key], model.node_memory[key])
    for key, value in model.state_dict().items():
        assert torch.equal(restored.state_dict()[key], value)


if __name__ == "__main__":
    raise SystemExit(
        "Run with: python -m pytest experiments/dansn_conditioned_gru_gates/test_model.py"
    )


def test_context_ablation_routes_graph_context_to_only_selected_gate_paths():
    torch.manual_seed(37)
    inputs = torch.randn(8)
    hidden = torch.randn(8)
    no_context = torch.zeros(8)
    graph_context = torch.randn(8)

    reset_cell = DANSNConditionedGRUCell(
        input_size=8, hidden_size=8, conditioning_mode="reset_only"
    )
    _, reset_without = reset_cell(inputs, hidden, no_context)
    _, reset_with = reset_cell(inputs, hidden, graph_context)
    assert not torch.allclose(
        reset_without["reset_gate"], reset_with["reset_gate"]
    )
    assert torch.equal(
        reset_without["update_gate"], reset_with["update_gate"]
    )

    update_cell = DANSNConditionedGRUCell(
        input_size=8, hidden_size=8, conditioning_mode="update_only"
    )
    _, update_without = update_cell(inputs, hidden, no_context)
    _, update_with = update_cell(inputs, hidden, graph_context)
    assert not torch.allclose(
        update_without["update_gate"], update_with["update_gate"]
    )
    assert torch.equal(
        update_without["reset_gate"], update_with["reset_gate"]
    )
    # With reset unchanged and no direct candidate context, candidate is equal.
    assert torch.equal(update_without["candidate"], update_with["candidate"])


def test_gate_ablation_mode_is_saved_and_must_match_on_restore():
    import pytest

    brain, ids = make_brain_and_stream()
    reset_model = make_model(brain, gate_ablation="reset_only")
    _, state, _ = reset_model.forward_step(ids[0], [ids[0]])
    reset_model.observe_target([ids[0]], ids[1], state)

    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = Path(tmp) / "reset_only.pt"
        reset_model.save_checkpoint(checkpoint)

        same_mode = make_model(brain, gate_ablation="reset_only")
        same_mode.load_checkpoint(checkpoint)
        assert same_mode.gate_ablation == "reset_only"

        wrong_mode = make_model(brain, gate_ablation="update_only")
        with pytest.raises(ValueError, match="gate_ablation"):
            wrong_mode.load_checkpoint(checkpoint)
