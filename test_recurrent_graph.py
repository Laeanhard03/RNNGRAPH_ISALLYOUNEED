"""Tests for the optional recurrent graph-state block (requires PyTorch)."""

import torch

from agi import Brain
from recurrent_graph import RecurrentGraphBlock


def make_brain():
    brain = Brain()
    brain.absorb_text("\n".join([
        "[user] what is a dog? [ai] a dog is an animal.",
        "[user] what is a cat? [ai] a cat is an animal.",
        "[user] a dog chases a cat. [ai] the cat runs away.",
    ]))
    return brain


def test_shapes_and_sparse_states():
    brain = make_brain()
    model = RecurrentGraphBlock(brain.engine, len(brain.vocab.word_to_id) + 1,
                                hidden_size=24, graph_depth=2, message_rounds=2)
    ids = brain.vocab.tokenize("[user] what is a dog? [ai]")
    logits, state = model.forward_sequence(ids)
    assert logits.shape == (len(ids), len(brain.vocab.word_to_id) + 1)
    assert torch.isfinite(logits).all()
    assert state.hidden.shape == (24,)
    assert state.node_states


def test_gradient_flows_through_controller_and_graph():
    brain = make_brain()
    model = RecurrentGraphBlock(brain.engine, len(brain.vocab.word_to_id) + 1,
                                hidden_size=16, graph_depth=2, message_rounds=2)
    ids = brain.vocab.tokenize("[user] what is a dog? [ai]")
    logits, _ = model.forward_sequence(ids)
    logits.square().mean().backward()
    assert model.controller.weight_ih.grad is not None
    assert model.node_update.weight_ih.grad is not None
    assert model.message_projection.weight.grad is not None
    assert model.embedding.weight.grad is not None


def test_state_can_continue_across_chunks():
    brain = make_brain()
    model = RecurrentGraphBlock(brain.engine, len(brain.vocab.word_to_id) + 1,
                                hidden_size=16, graph_depth=2)
    ids = brain.vocab.tokenize("[user] what is a dog? [ai]")
    _, state_a = model.forward_sequence(ids[:4])
    logits_b, state_b = model.forward_sequence(ids[4:], state_a)
    assert logits_b.shape[0] == len(ids) - 4
    assert state_b.hidden is not None
    assert state_b.node_states


def test_empty_sequence():
    brain = make_brain()
    model = RecurrentGraphBlock(brain.engine, len(brain.vocab.word_to_id) + 1, hidden_size=8)
    logits, state = model.forward_sequence([])
    assert logits.shape == (0, len(brain.vocab.word_to_id) + 1)
    assert state.hidden is None


if __name__ == "__main__":
    tests = [test_shapes_and_sparse_states, test_gradient_flows_through_controller_and_graph,
             test_state_can_continue_across_chunks, test_empty_sequence]
    for test in tests:
        test()
        print("PASS", test.__name__)
    print(f"\n{len(tests)}/{len(tests)} passed")
