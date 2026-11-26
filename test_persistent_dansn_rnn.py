"""Tests for the coupled persistent DANSN-RNN experiment."""
import math
import tempfile
from pathlib import Path

import torch

from agi import Brain
from persistent_dansn_rnn import PersistentDANSNRNN, evaluate_prequential, train_stream


def make_brain_and_stream():
    brain = Brain()
    # Build the fixed vocabulary but intentionally leave the graph empty.
    ids = brain.vocab.tokenize("[user] red fox jumps over blue lake red fox jumps over blue lake.")
    return brain, ids


def make_model(brain, **overrides):
    config = dict(hidden_size=16, window_size=4, graph_depth=2,
                  message_rounds=1, max_active_nodes=24, max_memory_nodes=200)
    config.update(overrides)
    return PersistentDANSNRNN(brain.engine, len(brain.vocab.word_to_id) + 1, **config)


def test_predict_is_read_only_then_target_grows_graph_and_memory():
    brain, ids = make_brain_and_stream()
    model = make_model(brain)
    before = model.graph_stats()
    logits, state, trace = model.forward_step(ids[0], [ids[0]])
    assert logits.shape == (len(brain.vocab.word_to_id) + 1,)
    assert torch.isfinite(logits).all()
    assert model.graph_stats() == before
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


def test_graph_and_memory_checkpoint_roundtrip():
    brain, ids = make_brain_and_stream()
    model = make_model(brain)
    _logits, state, _trace = model.forward_step(ids[0], [ids[0]])
    model.observe_target([ids[0]], ids[1], state)
    expected = {key: value.detach().clone() for key, value in model.node_memory.items()}

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        vocab_path, graph_path, model_path = root / "vocab.bin", root / "graph.bin", root / "model.pt"
        brain.save(str(vocab_path), str(graph_path))
        model.save_checkpoint(model_path)
        restored_brain = Brain.load(str(vocab_path), str(graph_path))
        restored = make_model(restored_brain)
        restored.load_checkpoint(model_path)

        assert restored.node_memory.keys() == expected.keys()
        for key, value in expected.items():
            assert torch.allclose(restored.node_memory[key], value)
        logits, _state, trace = restored.forward_step(ids[1], ids[:2])
        assert torch.isfinite(logits).all()
        assert trace["matched_path"] in restored.node_memory


def test_unknown_context_falls_back_to_controller():
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
        test_graph_and_memory_checkpoint_roundtrip,
        test_unknown_context_falls_back_to_controller,
    ]
    for check in checks:
        check()
        print("PASS", check.__name__)
    print(f"\n{len(checks)}/{len(checks)} passed")
