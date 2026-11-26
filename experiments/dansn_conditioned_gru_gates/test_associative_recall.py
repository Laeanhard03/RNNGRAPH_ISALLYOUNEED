"""Tests for the generated one-shot associative recall task."""
import torch

from agi import Brain
from experiments.dansn_conditioned_gru_gates.associative_recall import (
    generate_dataset,
)
from experiments.dansn_conditioned_gru_gates.four_gate import DANSNFourGateGRU


def test_generated_train_and_heldout_keys_are_disjoint_and_reproducible():
    first = generate_dataset(
        seed=101, train_keys=6, heldout_keys=5, value_count=8,
        train_episodes=2, train_delay=3, test_delay=8,
    )
    second = generate_dataset(
        seed=101, train_keys=6, heldout_keys=5, value_count=8,
        train_episodes=2, train_delay=3, test_delay=8,
    )

    assert first.metadata == second.metadata
    assert first.train_tokens == second.train_tokens
    assert first.test_tokens == second.test_tokens
    assert first.metadata["test_key_ids_disjoint_from_train"] is True
    assert len(first.query_paths) == 5
    assert len(first.support_paths) == 5
    assert set(first.query_paths.values()) == set(first.support_paths.values())


def test_query_positions_follow_support_and_target_is_not_in_prefix():
    dataset = generate_dataset(
        seed=103, train_keys=5, heldout_keys=5, value_count=7,
        train_episodes=2, train_delay=3, test_delay=9,
    )
    for position, expected_path in dataset.query_paths.items():
        assert dataset.test_tokens[position] != expected_path[-1]
        assert dataset.test_tokens[position + 1] == expected_path[-1]
        assert expected_path in dataset.support_paths.values()
        assert dataset.query_delays[position] >= dataset.metadata["test_delay_tokens"]


def test_four_gate_stores_association_and_retrieves_it_after_distractors():
    brain = Brain()
    ctx, key, value, noise_a, noise_b = brain.vocab.tokenize(
        "recallctx keyalpha valueomega noisea noiseb"
    )
    model = DANSNFourGateGRU(
        brain.engine,
        len(brain.vocab.word_to_id) + 1,
        hidden_size=16,
        window_size=4,
        graph_depth=3,
        message_rounds=1,
        max_active_nodes=32,
        max_memory_nodes=100,
    )

    # Support: predict the value from the key, then reveal and store it.
    _, support_state, _ = model.forward_step(key, [ctx, key])
    support_state.write_gate = torch.ones_like(support_state.write_gate)
    model.observe_target([ctx, key], value, support_state)

    expected_path = (ctx, key, value)
    assert expected_path in model._path_nodes
    assert expected_path in model.node_memory

    # The recurrent state is deliberately reset. Only DANSN's persistent graph
    # and its node vector survive the intervening distractor tokens.
    query_history = [ctx, key, value, noise_a, noise_b, ctx, key]
    query_state = model.initial_state()
    logits, _state, trace = model.forward_step(key, query_history, query_state)

    assert torch.isfinite(logits).all()
    assert expected_path in trace["active_paths"]
    assert expected_path in model.node_memory


def test_training_covers_every_value_class_for_a_balanced_task():
    dataset = generate_dataset(
        seed=107, train_keys=8, heldout_keys=6, value_count=8,
        train_episodes=2, train_delay=3, test_delay=8,
    )
    assert dataset.metadata["all_value_classes_seen_in_training"] is True
    assert dataset.metadata["heldout_values_seen_in_training"] is True
    assert len(dataset.metadata["value_token_ids"]) == 8
    assert dataset.metadata["train_value_coverage_count"] == 8
