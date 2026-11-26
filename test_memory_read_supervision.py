"""Regression tests for auxiliary memory-read supervision."""
import copy
import math

import torch

from agi import Brain
from delayed_relational_recall import build_recall_data, ingest_context_online, _make_hybrid
from memory_read_supervision import (
    MemoryReadAnswerHead,
    _place_candidates,
    aggregate_runs,
    run_memory_supervision,
    train_memory_supervised_query,
)


def test_memory_read_head_aligns_graph_readout_to_place_embeddings():
    torch.manual_seed(3)
    head = MemoryReadAnswerHead(hidden_size=12, temperature=0.2)
    readout = torch.randn(12, requires_grad=True)
    embeddings = torch.randn(30, 12, requires_grad=True)
    candidate_ids = [3, 7, 11, 13]
    logits = head(readout, embeddings, candidate_ids)
    assert logits.shape == (4,)
    loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), torch.tensor([2]))
    loss.backward()
    assert readout.grad is not None
    assert torch.isfinite(readout.grad).all()
    assert head.projection.weight.grad is not None
    assert embeddings.grad is not None


def test_memory_supervised_query_updates_weights_without_writing_answer_to_graph():
    torch.manual_seed(5)
    data = build_recall_data(seed=41, subject_count=12, distractor_count=8)
    brain = Brain(vocab=copy.deepcopy(data["vocab"]))
    model = _make_hybrid(
        brain.engine,
        len(data["vocab"].word_to_id) + 1,
        hidden_size=16,
        window_size=9,
        graph_depth=8,
    )
    # The model has observed the training facts and has retrievable path memory.
    ingest_context_online(model, data["train_prefix_ids"])
    candidates, target_to_class = _place_candidates(data["vocab"])
    head = MemoryReadAnswerHead(hidden_size=16, temperature=0.2)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()), lr=1e-3
    )

    before_stats = model.graph_stats()
    before_memory = {
        path: value.detach().clone() for path, value in model.node_memory.items()
    }
    before_head = head.projection.weight.detach().clone()
    query = data["train_query_ids"][0]

    next_loss, auxiliary_loss = train_memory_supervised_query(
        model, head, query, optimizer, candidates, target_to_class,
        auxiliary_weight=0.5,
    )

    assert math.isfinite(next_loss)
    assert math.isfinite(auxiliary_loss)
    assert not torch.equal(before_head, head.projection.weight.detach())
    after_stats = model.graph_stats()
    assert after_stats["nodes"] == before_stats["nodes"]
    assert after_stats["memory_nodes"] == before_stats["memory_nodes"]
    assert set(model.node_memory) == set(before_memory)
    for path, value in before_memory.items():
        assert torch.equal(model.node_memory[path], value)


def test_small_memory_supervision_run_reports_all_comparison_systems():
    result = run_memory_supervision(
        seed=43,
        subject_count=8,
        distractor_count=4,
        delays=(0, 2),
        epochs=1,
        hidden_size=8,
        window_size=9,
        graph_depth=8,
        chunk_size=16,
    )
    assert result["protocol"]["query_answer_labels_written_to_graph"] is False
    assert set(result["results"]) == {"0", "2"}
    metrics = result["results"]["0"]
    assert "read_only_next_token_reset" in metrics
    assert "persistent_next_token_only_reset" in metrics
    assert "memory_read_head_persistent_reset" in metrics
    assert "memory_read_head_readonly_reset" in metrics
    assert 0.0 <= metrics["memory_read_head_persistent_reset"]["accuracy"] <= 1.0
    assert math.isfinite(metrics["memory_read_head_persistent_reset"]["mean_nll"])


def test_multi_seed_aggregator_summarizes_memory_head_metrics():
    metric = {
        "accuracy": 0.25,
        "mean_nll": 2.0,
        "mean_reciprocal_rank": 0.5,
    }
    run_a = {
        "results": {
            "0": {
                "memory_read_head_persistent_reset": metric,
                "persistent_graph_nodes": 100,
                "persistent_memory_paths": 20,
            }
        }
    }
    run_b = {
        "results": {
            "0": {
                "memory_read_head_persistent_reset": {
                    **metric, "accuracy": 0.75, "mean_nll": 1.0
                },
                "persistent_graph_nodes": 200,
                "persistent_memory_paths": 40,
            }
        }
    }
    summary = aggregate_runs([run_a, run_b], [7, 13])["summary"]["0"]
    assert summary["memory_read_head_persistent_reset"]["accuracy"]["mean"] == 0.5
    assert summary["persistent_graph_nodes"]["mean"] == 150.0
    assert summary["persistent_memory_paths"]["mean"] == 30.0


if __name__ == "__main__":
    checks = [
        test_memory_read_head_aligns_graph_readout_to_place_embeddings,
        test_memory_supervised_query_updates_weights_without_writing_answer_to_graph,
        test_small_memory_supervision_run_reports_all_comparison_systems,
        test_multi_seed_aggregator_summarizes_memory_head_metrics,
    ]
    for check in checks:
        check()
        print("PASS", check.__name__)
    print(f"\n{len(checks)}/{len(checks)} passed")
