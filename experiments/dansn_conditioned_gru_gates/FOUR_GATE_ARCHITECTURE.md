# DANSN Four-Gate GRU (DG-GRU)

**Status:** isolated experimental variant on **feature/dg-gru-four-gate-control**. This branch does not alter the canonical DANSN graph engine or the prior two/three-gate experiments.

## Architecture

At each sequence step, current input and previous hidden state query the existing bounded DANSN context neighborhood. The graph reader returns a differentiable memory readout m_t, plus bounded structural evidence phi_t:

1. Context/suffix match present.
2. Active-neighborhood occupancy.
3. Normalized path frequency.
4. Persistent-memory coverage.
5. Mean quality of retrieved persistent memory.

A learned controller receives [x_t; h_(t-1); m_t; phi_t] and generates four vector-valued sigmoid gates:

- z_t, update: blend prior recurrent state and candidate.
- r_t, reset: filter the previous state used by the candidate.
- g_mem, memory injection: scale retrieved DANSN memory entering the candidate.
- g_write, persistent write: scale the post-score memory-vector write to the selected post-target context path.

The cell equations are:

```text
[z_t, r_t, g_mem_t, g_write_t]
    = sigmoid(Controller(x_t, h_(t-1), m_t, phi_t))

candidate_t = tanh(
    W_x x_t
  + W_h (r_t * h_(t-1))
  + W_m (g_mem_t * m_t)
)

h_t = (1 - z_t) * h_(t-1) + z_t * candidate_t
```

The fourth gate is stored in the recurrent state when the prediction is made. The target is scored first. Only then is the target revealed and the saved write gate applied to the candidate persistent-memory vector. The write gate cannot inspect the target it is about to score.

## What the persistent-write gate controls

The same write gate controls both forms of DANSN storage after the target has been scored:

1. **Graph topology reinforcement:** the mean write-gate strength, multiplied by the supplied confidence, becomes a fractional frequency increment for each observed edge on the context-to-target path. This changes association strength without modifying the canonical agi.py engine.
2. **Persistent node-vector update:** the per-dimension write-gate vector controls the blend between the previous node-memory vector and the new candidate state.

A configurable threshold suppresses negligible writes. A zero write gate or zero confidence performs neither structural path reinforcement nor persistent-vector storage. Graph-frequency increments are non-differentiable; gradients flow through the per-dimension persistent-vector blend during truncated BPTT.

## Implementation

- **four_gate.py:** DANSNFourGateCell, DANSNFourGateGRU, and FourGateState.
- Reuses the existing DANSN graph reader, bounded active-neighborhood selection, persistent node-memory store, graph growth, quality/recency bookkeeping, and truncated-BPTT training/evaluation utilities.
- Uses a unique checkpoint architecture marker to prevent silent cross-loading with earlier DANSN-GRU checkpoints.
- **test_four_gate.py:** checks gate ranges, memory-injection ablation, read-before-write causality, graph-write thresholding, training gradients, persistent writes, and checkpoint isolation.

## Ablations

The model includes shape-preserving ablations:

- full
- no_memory_injection
- no_persistent_write
- no_graph_evidence

These keep parameter shapes fixed and set the selected information path to zero, allowing controlled follow-up comparisons.

## Run

From the repository root:

```bash
python -m pytest -q experiments/dansn_conditioned_gru_gates/
```

Use the real-corpus runner:

```bash
python -m experiments.dansn_conditioned_gru_gates.large_corpus_benchmark \
  --dataset wikitext103 \
  --train-tokens 100000 --validation-tokens 20000 --test-tokens 20000 \
  --epochs 1 --seeds 7 \
  --output benchmark-results/wikitext103_four_gate_dggru.json
```

The benchmark includes plain GRU, readout-fusion DANSN-GRU, update-only conditioned GRU, full three-path gate-conditioned GRU, the full four-gate DG-GRU, and three four-gate ablations (no memory injection, no persistent write, and no graph evidence). Static test metrics and score-before-write prequential memory-only metrics are separate. Parameter counts are reported, not assumed to match. The four-gate variants retain identical parameter shapes; the ablations zero the selected path rather than deleting its parameters.

## Important limitations

Graph-neighborhood selection and discrete topology updates are non-differentiable. Gradients flow through the selected-node readout, controller, four gates, candidate computation, and persistent vectors during truncated BPTT. Graph evidence is a heuristic structural summary, not a calibrated probability of retrieval correctness. The persistent write gate controls both fractional graph-edge frequency reinforcement and persistent vector writes after target scoring; it does not make discrete edge creation differentiable.

This architecture is a research hypothesis. Do not claim it outperforms readout-fusion or three-gate variants until multi-seed real-corpus and delayed/interleaved relational-recall results support that claim.


## Verification status

The GitHub Actions run [45878580368](https://github.com/Laeanhard03/RNNGRAPH_ISALLYOUNEED/actions/runs/45878580368) passed the full experiment test directory: **24 passed**. It reported one harmless PyTorch warning because NumPy is not installed in the minimal CI test environment.

The bounded WikiText-103 workflow [45878304021](https://github.com/Laeanhard03/RNNGRAPH_ISALLYOUNEED/actions/runs/45878304021) is still running from an earlier commit that predates the fractional topology-reinforcement change described above. Treat it as an earlier pipeline smoke run, not as an evaluation of the final four-gate write behavior. A final real-corpus result for the weighted graph/vector-write implementation remains outstanding.
