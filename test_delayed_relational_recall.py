"""Tests for delayed two-hop relational recall and causal memory ingestion."""
import copy
import math

import torch

from agi import Brain
from delayed_relational_recall import (
    build_recall_data,
    evaluate_hybrid_queries,
    ingest_context_online,
    _make_hybrid,
)


def test_train_and_test_query_subjects_are_disjoint_and_mappings_are_remapped():
    data = build_recall_data(seed=19, subject_count=12, distractor_count=8)
    train = set(data["train_query_indices"])
    held_out = set(data["test_query_indices"])
    assert train.isdisjoint(held_out)
    assert train | held_out == set(range(12))
    assert data["train_vehicle_map"] != data["test_vehicle_map"]
    assert data["train_place_map"] != data["test_place_map"]
    assert len(data["test_query_ids"]) == len(held_out)


def test_fact_chain_contains_two_hops_and_target_is_not_in_query_prompt():
    data = build_recall_data(seed=23, subject_count=12, distractor_count=4)
    vocab = data["vocab"]
    for where_id, person_id, target_id in data["test_query_ids"]:
        assert target_id not in (where_id, person_id)
        assert vocab.id_to_word[where_id] == "where"
        assert vocab.id_to_word[person_id].startswith("person")
        assert vocab.id_to_word[target_id].startswith("place")


def test_online_context_ingestion_grows_graph_and_persistent_memory_without_weight_updates():
    data = build_recall_data(seed=29, subject_count=12, distractor_count=8)
    brain = Brain(vocab=copy.deepcopy(data["vocab"]))
    model = _make_hybrid(
        brain.engine, len(data["vocab"].word_to_id) + 1,
        hidden_size=16, window_size=9, graph_depth=8,
    )
    before_parameters = [parameter.detach().clone() for parameter in model.parameters()]
    before = model.graph_stats()

    context = data["test_fact_ids"] + [
        token for row in data["distractor_ids"][:8] for token in row
    ]
    ingest_context_online(model, context)

    after = model.graph_stats()
    assert after["nodes"] > before["nodes"]
    assert after["memory_nodes"] > 0
    for old, new in zip(before_parameters, model.parameters()):
        assert torch.equal(old, new.detach())
    assert all(not vector.requires_grad for vector in model.node_memory.values())


def test_subject_prefix_can_reach_the_two_hop_location_in_graph():
    data = build_recall_data(seed=31, subject_count=12, distractor_count=8)
    brain = Brain(vocab=copy.deepcopy(data["vocab"]))
    model = _make_hybrid(
        brain.engine, len(data["vocab"].word_to_id) + 1,
        hidden_size=16, window_size=9, graph_depth=8,
    )
    ingest_context_online(model, data["test_fact_ids"])

    for _where_id, person_id, target_id in data["test_query_ids"]:
        start = model._longest_context_path([person_id])
        assert start == (person_id,)
        active, _incoming = model._active_subgraph(start)
        assert any(path[-1] == target_id for path in active if path)
        # The full two-hop memory vector must remain under this subject path,
        # so querying the subject can actually retrieve the stored state.
        assert any(
            path and path[0] == person_id and target_id in path
            for path in model.node_memory
        )


def test_query_evaluation_scores_without_mutating_graph_or_persistent_vectors():
    data = build_recall_data(seed=37, subject_count=12, distractor_count=8)
    brain = Brain(vocab=copy.deepcopy(data["vocab"]))
    model = _make_hybrid(
        brain.engine, len(data["vocab"].word_to_id) + 1,
        hidden_size=16, window_size=9, graph_depth=8,
    )
    ingest_context_online(model, data["test_fact_ids"])
    before_nodes = model.graph_stats()["nodes"]
    before_memory = {
        path: value.detach().clone() for path, value in model.node_memory.items()
    }

    reset_metrics = evaluate_hybrid_queries(
        model, data["test_fact_ids"], data["test_query_ids"], True
    )
    continuous_metrics = evaluate_hybrid_queries(
        model, data["test_fact_ids"], data["test_query_ids"], False
    )

    for metrics in (reset_metrics, continuous_metrics):
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert math.isfinite(metrics["mean_nll"])
        assert math.isfinite(metrics["mean_reciprocal_rank"])
        assert metrics["queries"] == len(data["test_query_ids"])
    assert model.graph_stats()["nodes"] == before_nodes
    assert model.graph_stats()["memory_nodes"] == len(before_memory)
    for path, value in before_memory.items():
        assert torch.equal(model.node_memory[path], value)


if __name__ == "__main__":
    checks = [
        test_train_and_test_query_subjects_are_disjoint_and_mappings_are_remapped,
        test_fact_chain_contains_two_hops_and_target_is_not_in_query_prompt,
        test_online_context_ingestion_grows_graph_and_persistent_memory_without_weight_updates,
        test_subject_prefix_can_reach_the_two_hop_location_in_graph,
        test_query_evaluation_scores_without_mutating_graph_or_persistent_vectors,
    ]
    for check in checks:
        check()
        print("PASS", check.__name__)
    print(f"\n{len(checks)}/{len(checks)} passed")
