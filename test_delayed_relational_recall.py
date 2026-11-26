"""Tests for delayed two-hop relational recall and causal memory ingestion."""
import copy
import math

import torch

from agi import Brain
from delayed_relational_recall import (
    aggregate_multi_seed_results,
    build_recall_data,
    evaluate_hybrid_queries,
    ingest_context_online,
    link_shared_entity_paths,
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


def test_interleaved_fact_generator_separates_relation_edges():
    data = build_recall_data(
        seed=47, subject_count=12, distractor_count=8, interleave_facts=2
    )
    text = data["test_facts_text"]
    first_relation_end = text.index(" .")
    gap_start = text.index("traveler000 visits island000")
    gap_end = text.index("traveler001 visits island001")
    second_relation_start = text.index(" parked place")
    assert first_relation_end < gap_start < gap_end < second_relation_start
    assert len(data["interleaved_distractors"]) == 2
    assert len(data["distractor_ids"]) == 6
    assert data["interleave_facts"] == 2


def test_unlinked_trie_stores_both_facts_but_cannot_join_their_context_paths():
    data = build_recall_data(
        seed=53, subject_count=12, distractor_count=8, interleave_facts=2
    )
    vocab = data["vocab"]
    brain = Brain(vocab=copy.deepcopy(vocab))
    model = _make_hybrid(
        brain.engine, len(vocab.word_to_id) + 1,
        hidden_size=16, window_size=9, graph_depth=8,
    )
    ingest_context_online(model, data["test_fact_ids"])

    owns_id = vocab.word_to_id["owns"]
    parked_id = vocab.word_to_id["parked"]
    for subject in data["test_query_indices"]:
        person_id = vocab.word_to_id[f"person{subject:02d}"]
        vehicle_id = vocab.word_to_id[
            f"vehicle{data['test_vehicle_map'][subject]:02d}"
        ]
        place_id = vocab.word_to_id[
            f"place{data['test_place_map'][data['test_vehicle_map'][subject]]:02d}"
        ]
        assert (person_id, owns_id, vehicle_id) in model._path_nodes
        assert any(
            any(path[index:index + 3] == (vehicle_id, parked_id, place_id)
                for index in range(max(len(path) - 2, 0)))
            for path in model._path_nodes
        )
        start = model._longest_context_path([person_id])
        active, _incoming = model._active_subgraph(start)
        assert any(path and path[-1] == vehicle_id for path in active)
        assert not any(path and path[-1] == place_id for path in active)

    metrics = evaluate_hybrid_queries(
        model, data["test_fact_ids"], data["test_query_ids"], True
    )
    assert metrics["target_reachable_fraction"] == 0.0
    assert metrics["target_bearing_memory_visible_fraction"] == 0.0


def test_observed_entity_linking_joins_paths_and_makes_memory_visible():
    data = build_recall_data(
        seed=59, subject_count=12, distractor_count=8, interleave_facts=2
    )
    vocab = data["vocab"]
    brain = Brain(vocab=copy.deepcopy(vocab))
    model = _make_hybrid(
        brain.engine, len(vocab.word_to_id) + 1,
        hidden_size=16, window_size=9, graph_depth=8,
    )
    ingest_context_online(model, data["test_fact_ids"])
    stats = link_shared_entity_paths(model, data["test_fact_ids"], vocab)

    assert stats["observed_first_hop_facts"] == 12
    assert stats["observed_second_hop_facts"] == 12
    assert stats["cross_links_added"] > 0
    assert stats["memory_aliases_added"] > 0
    assert stats["unlinked_entities"] == 0

    metrics = evaluate_hybrid_queries(
        model, data["test_fact_ids"], data["test_query_ids"], True
    )
    assert metrics["target_reachable_fraction"] == 1.0
    assert metrics["target_memory_visible_fraction"] > 0.0
    assert metrics["target_bearing_memory_visible_fraction"] == 1.0


def test_entity_linking_does_not_change_weights_or_add_query_answer_writes():
    data = build_recall_data(
        seed=61, subject_count=12, distractor_count=8, interleave_facts=2
    )
    vocab = data["vocab"]
    brain = Brain(vocab=copy.deepcopy(vocab))
    model = _make_hybrid(
        brain.engine, len(vocab.word_to_id) + 1,
        hidden_size=16, window_size=9, graph_depth=8,
    )
    ingest_context_online(model, data["test_fact_ids"])
    params = [parameter.detach().clone() for parameter in model.parameters()]
    memory_before_link = {
        path: value.detach().clone() for path, value in model.node_memory.items()
    }

    # Only facts are supplied to the linker; held-out query tuples are separate.
    stats = link_shared_entity_paths(model, data["test_fact_ids"], vocab)

    for old, new in zip(params, model.parameters()):
        assert torch.equal(old, new.detach())
    assert set(memory_before_link).issubset(model.node_memory)
    assert stats["memory_aliases_added"] > 0
    for path, value in memory_before_link.items():
        assert torch.equal(model.node_memory[path], value)


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



def test_multi_seed_aggregator_reports_delay_metrics_and_memory_counts():
    metric = {
        "accuracy": 0.5,
        "mean_nll": 1.5,
        "mean_reciprocal_rank": 0.75,
        "target_reachable_fraction": 1.0,
        "target_memory_visible_fraction": 0.8,
    }
    run_a = {
        "results": {
            "8": {
                "plain_gru_reset": metric,
                "plain_gru_continuous": metric,
                "readonly_dansn_gru_reset": metric,
                "readonly_dansn_gru_continuous": metric,
                "persistent_dansn_rnn_reset": metric,
                "persistent_dansn_rnn_continuous": metric,
                "persistent_graph_nodes": 100,
                "persistent_memory_paths": 20,
            }
        }
    }
    run_b = {
        "results": {
            "8": {
                "plain_gru_reset": {**metric, "accuracy": 1.0, "mean_nll": 0.5},
                "plain_gru_continuous": metric,
                "readonly_dansn_gru_reset": metric,
                "readonly_dansn_gru_continuous": metric,
                "persistent_dansn_rnn_reset": metric,
                "persistent_dansn_rnn_continuous": metric,
                "persistent_graph_nodes": 200,
                "persistent_memory_paths": 40,
            }
        }
    }
    result = aggregate_multi_seed_results([run_a, run_b], [7, 13])
    summary = result["summary"]["8"]
    assert result["seeds"] == [7, 13]
    assert summary["plain_gru_reset"]["accuracy"]["mean"] == 0.75
    assert math.isclose(summary["plain_gru_reset"]["mean_nll"]["mean"], 1.0)
    assert summary["persistent_graph_nodes"]["mean"] == 150.0
    assert summary["persistent_memory_paths"]["mean"] == 30.0


if __name__ == "__main__":
    checks = [
        test_train_and_test_query_subjects_are_disjoint_and_mappings_are_remapped,
        test_fact_chain_contains_two_hops_and_target_is_not_in_query_prompt,
        test_interleaved_fact_generator_separates_relation_edges,
        test_online_context_ingestion_grows_graph_and_persistent_memory_without_weight_updates,
        test_subject_prefix_can_reach_the_two_hop_location_in_graph,
        test_unlinked_trie_stores_both_facts_but_cannot_join_their_context_paths,
        test_observed_entity_linking_joins_paths_and_makes_memory_visible,
        test_entity_linking_does_not_change_weights_or_add_query_answer_writes,
        test_query_evaluation_scores_without_mutating_graph_or_persistent_vectors,
        test_multi_seed_aggregator_reports_delay_metrics_and_memory_counts,
    ]
    for check in checks:
        check()
        print("PASS", check.__name__)
    print(f"\n{len(checks)}/{len(checks)} passed")
