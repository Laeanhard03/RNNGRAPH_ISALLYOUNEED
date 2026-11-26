"""Tests for persistent DANSN-RNN causal memory and quality safeguards."""
import math
import tempfile
from pathlib import Path

import torch

from agi import Brain
from persistent_dansn_rnn import (
    PersistentDANSNRNN,
    evaluate_prequential,
    train_stream,
)


def make_brain_and_stream():
    brain = Brain()
    ids = brain.vocab.tokenize(
        "[user] red fox jumps over blue lake red fox jumps over blue lake."
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
    return PersistentDANSNRNN(
        brain.engine, len(brain.vocab.word_to_id) + 1, **config
    )


def test_predict_is_read_only_then_target_grows_graph_and_memory():
    brain, ids = make_brain_and_stream()
    model = make_model(brain)
    before = model.graph_stats()

    logits, state, trace = model.forward_step(ids[0], [ids[0]])
    assert logits.shape == (len(brain.vocab.word_to_id) + 1,)
    assert torch.isfinite(logits).all()
    assert model.graph_stats()["nodes"] == before["nodes"]
    assert model.graph_stats()["memory_nodes"] == 0
    assert trace["matched_path"] is None

    written_path = model.observe_target([ids[0]], ids[1], state)
    after = model.graph_stats()
    assert after["nodes"] > before["nodes"]
    assert after["memory_nodes"] == 1
    assert written_path in model.node_memory

    logits2, _state2, trace2 = model.forward_step(ids[1], ids[:2], state)
    assert torch.isfinite(logits2).all()
    assert trace2["matched_path"] == written_path
    assert torch.linalg.vector_norm(trace2["graph_readout"]).item() > 0


def test_online_training_updates_parameters_and_persistent_graph():
    torch.manual_seed(7)
    brain, ids = make_brain_and_stream()
    model = make_model(brain)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    before = model.controller.weight_ih.detach().clone()

    metrics = train_stream(model, ids * 2, optimizer, chunk_size=3)

    assert metrics["tokens"] == float(len(ids * 2) - 1)
    assert metrics["updates"] > 0
    assert math.isfinite(metrics["mean_loss"])
    assert math.isfinite(metrics["perplexity"])
    assert model.graph_stats()["nodes"] > 0
    assert model.graph_stats()["memory_nodes"] > 0
    assert not torch.equal(before, model.controller.weight_ih.detach())
    assert all(not value.requires_grad for value in model.node_memory.values())


def test_prequential_evaluation_scores_before_graph_update_and_keeps_weights_fixed():
    brain, ids = make_brain_and_stream()
    model = make_model(brain)
    before_parameters = [p.detach().clone() for p in model.parameters()]
    before_nodes = model.graph_stats()["nodes"]

    metrics = evaluate_prequential(model, ids, update_graph=True)

    assert metrics["tokens"] == float(len(ids) - 1)
    assert math.isfinite(metrics["mean_loss"])
    assert math.isfinite(metrics["perplexity"])
    assert model.graph_stats()["nodes"] > before_nodes
    for old, new in zip(before_parameters, model.parameters()):
        assert torch.equal(old, new.detach())


def test_zero_confidence_does_not_overwrite_existing_memory():
    brain, ids = make_brain_and_stream()
    model = make_model(brain)
    _, state, _ = model.forward_step(ids[0], [ids[0]])
    key = model.observe_target([ids[0]], ids[1], state, confidence=1.0)
    before = model.node_memory[key].detach().clone()
    before_quality = model.node_memory_quality[key]
    before_writes = model.node_memory_writes[key]

    altered_state = type(state)(
        hidden=torch.full_like(state.hidden, 7.0),
        readout=torch.full_like(state.readout, -3.0),
    )
    model.observe_target([ids[0]], ids[1], altered_state, confidence=0.0)

    assert torch.equal(model.node_memory[key], before)
    assert model.node_memory_quality[key] == before_quality
    assert model.node_memory_writes[key] == before_writes


def test_lru_eviction_preserves_recently_retrieved_memory():
    brain, ids = make_brain_and_stream()
    model = make_model(brain, max_memory_nodes=2)

    _, state_a, _ = model.forward_step(ids[0], [ids[0]])
    key_a = model.observe_target([ids[0]], ids[1], state_a)
    _, state_b, _ = model.forward_step(ids[2], [ids[2]])
    key_b = model.observe_target([ids[2]], ids[3], state_b)
    assert key_a in model.node_memory and key_b in model.node_memory

    # Reading context A makes it the most recently used memory.
    model.forward_step(ids[1], [ids[0], ids[1]])
    _, state_c, _ = model.forward_step(ids[4], [ids[4]])
    key_c = model.observe_target([ids[4]], ids[5], state_c)

    assert key_a in model.node_memory
    assert key_c in model.node_memory
    assert key_b not in model.node_memory
    assert len(model.node_memory) == 2


def test_graph_memory_checkpoint_roundtrip_with_quality_metadata():
    brain, ids = make_brain_and_stream()
    model = make_model(brain)
    _, state, _ = model.forward_step(ids[0], [ids[0]])
    key = model.observe_target([ids[0]], ids[1], state, confidence=0.75)

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        vocab_path = root / "vocab.bin"
        graph_path = root / "graph.bin"
        model_path = root / "model.pt"
        brain.save(str(vocab_path), str(graph_path))
        model.save_checkpoint(model_path)

        restored_brain = Brain.load(str(vocab_path), str(graph_path))
        restored = make_model(restored_brain)
        restored.load_checkpoint(model_path)

        assert restored.node_memory.keys() == model.node_memory.keys()
        assert restored.node_memory_quality[key] == model.node_memory_quality[key]
        assert restored.node_memory_writes[key] == model.node_memory_writes[key]
        assert torch.allclose(restored.node_memory[key], model.node_memory[key])
        logits, _state, trace = restored.forward_step(ids[1], ids[:2])
        assert torch.isfinite(logits).all()
        assert trace["matched_path"] in restored.node_memory


def test_version_one_checkpoint_loads_with_default_quality_metadata():
    brain, ids = make_brain_and_stream()
    model = make_model(brain)
    _, state, _ = model.forward_step(ids[0], [ids[0]])
    model.observe_target([ids[0]], ids[1], state)

    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "v1.pt"
        model.save_checkpoint(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        checkpoint["format_version"] = 1
        checkpoint.pop("node_memory_quality", None)
        checkpoint.pop("node_memory_last_access", None)
        checkpoint.pop("node_memory_writes", None)
        checkpoint.pop("memory_clock", None)
        torch.save(checkpoint, path)

        restored = make_model(brain)
        restored.load_checkpoint(path)
        assert restored.node_memory
        assert all(value == 1.0 for value in restored.node_memory_quality.values())


def test_unseen_context_falls_back_to_controller_without_crashing():
    brain, ids = make_brain_and_stream()
    model = make_model(brain)
    logits, _state, trace = model.forward_step(ids[-1], [ids[-1]])
    assert torch.isfinite(logits).all()
    assert trace["active_paths"] == []
    assert torch.equal(trace["graph_readout"], torch.zeros(model.hidden_size))


if __name__ == "__main__":
    checks = [
        test_predict_is_read_only_then_target_grows_graph_and_memory,
        test_online_training_updates_parameters_and_persistent_graph,
        test_prequential_evaluation_scores_before_graph_update_and_keeps_weights_fixed,
        test_zero_confidence_does_not_overwrite_existing_memory,
        test_lru_eviction_preserves_recently_retrieved_memory,
        test_graph_memory_checkpoint_roundtrip_with_quality_metadata,
        test_version_one_checkpoint_loads_with_default_quality_metadata,
        test_unseen_context_falls_back_to_controller_without_crashing,
    ]
    for check in checks:
        check()
        print("PASS", check.__name__)
    print(f"\n{len(checks)}/{len(checks)} passed")
