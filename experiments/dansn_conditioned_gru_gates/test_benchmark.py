"""Smoke tests for the matched three-way benchmark."""
import math

from experiments.dansn_conditioned_gru_gates.benchmark import run_benchmark


def test_three_way_benchmark_smoke_returns_static_and_adaptive_metrics():
    result = run_benchmark(
        epochs=1,
        hidden_size=8,
        chunk_size=4,
        window_size=4,
        seed=31,
    )

    expected = {
        "plain_gru",
        "readout_fusion_dansn_gru",
        "gate_conditioned_dansn_gru",
    }
    assert set(result["results"]) == expected

    for name, metrics in result["results"].items():
        assert metrics["parameter_count"] > 0
        for mode in ("test_static", "test_adaptive"):
            assert math.isfinite(metrics[mode]["mean_loss"])
            assert math.isfinite(metrics[mode]["perplexity"])
            assert metrics[mode]["perplexity"] > 0

    assert (
        result["results"]["gate_conditioned_dansn_gru"]["graph"]["memory_nodes"]
        > 0
    )
