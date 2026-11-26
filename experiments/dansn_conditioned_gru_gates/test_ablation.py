"""Tests for DANSN gate-ablation diagnostics and benchmark controls."""
import math

from experiments.dansn_conditioned_gru_gates.ablation import (
    ABLATIONS,
    run_benchmark,
)


def test_gate_ablation_benchmark_smoke_and_parameter_matching():
    result = run_benchmark(
        seed=41,
        epochs=1,
        hidden_size=8,
        chunk_size=4,
        window_size=4,
    )

    expected = {"plain_gru", "readout_fusion", *ABLATIONS}
    assert set(result["results"]) == expected

    counts = {
        result["results"][name]["parameter_count"]
        for name in ABLATIONS
    }
    assert len(counts) == 1

    for name in ABLATIONS:
        row = result["results"][name]
        for mode in ("test_static", "test_adaptive"):
            assert math.isfinite(row[mode]["mean_loss"])
            assert math.isfinite(row[mode]["perplexity"])
        diagnostics = row["gate_diagnostics"]
        for gate in ("reset_gate", "update_gate"):
            assert 0.0 <= diagnostics[gate]["mean"] <= 1.0
            assert diagnostics[gate]["std"] >= 0.0
            assert 0.0 <= diagnostics[gate]["saturation_below_0_05"] <= 1.0
            assert 0.0 <= diagnostics[gate]["saturation_above_0_95"] <= 1.0

    # Graph topology and persistent state are still exercised by every variant.
    assert all(
        result["results"][name]["graph"]["memory_nodes"] > 0
        for name in ABLATIONS
    )
