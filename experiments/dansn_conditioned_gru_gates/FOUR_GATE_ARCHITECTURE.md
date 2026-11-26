# DANSN Four-Gate GRU (DG-GRU)

**Status:** isolated experimental variant on \`feature/dg-gru-four-gate-control\`. This branch does not alter the canonical DANSN graph engine or the prior two/three-gate experiments.

## Architecture

At each sequence step, current input and previous hidden state query the existing bounded DANSN context neighborhood. The graph reader returns a differentiable memory readout \(m_t\), plus bounded structural evidence \(\phi_t\):

1. context/suffix match present;
2. active-neighborhood occupancy;
3. normalized path frequency;
4. persistent-memory coverage;
5. mean quality of retrieved persistent memory.

A learned controller receives \([x_t; h_{t-1}; m_t; \phi_t]\) and generates four vector-valued sigmoid gates:

- \(z_t\), update: blend prior recurrent state and candidate;
- \(r_t\), reset: filter the previous state used by the candidate;
- \(g_t^{mem}\), memory injection: scale retrieved DANSN memory entering the candidate;
- \(g_t^{write}\), persistent write: scale the post-score memory-vector write to the selected post-target context path.

The cell equations are:

\[
z_t,r_t,g_t^{mem},g_t^{write}
=\sigma(\operatorname{Controller}(x_t,h_{t-1},m_t,\phi_t))
\]

\[
\tilde h_t=\tanh(W_xx_t+W_h(r_t\odot h_{t-1})
+W_m(g_t^{mem}\odot m_t))
\]

\[
h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde h_t
\]

The fourth gate is stored in the recurrent state when the prediction is made. The target is scored first. Only then is the target revealed and the saved write gate applied to the candidate persistent-memory vector. Thus the write gate cannot use the answer it is about to score.

## Graph writes vs. persistent-vector writes

The observed token sequence still grows/reinforces discrete DANSN topology after scoring, as in the prior persistent-memory experiment. \(g_t^{write}\) specifically controls whether/how strongly a learned state vector is stored on the post-target context path. A configurable threshold suppresses negligible writes. A zero-confidence observation never writes a persistent vector.

## Implementation

- \`four_gate.py\`: \`DANSNFourGateCell\`, \`DANSNFourGateGRU\`, and \`FourGateState\`.
- Reuses the existing DANSN graph reader, bounded active-neighborhood selection, persistent node-memory store, graph growth, quality/recency bookkeeping, and truncated-BPTT training/evaluation utilities.
- Unique checkpoint architecture marker prevents silent cross-loading with earlier DANSN-GRU checkpoints.
- \`test_four_gate.py\`: checks gate ranges, memory injection ablation, read-before-write causality, graph-write thresholding, training gradients, persistent writes, and checkpoint isolation.

## Ablations

The model includes shape-preserving ablations:
- \`full\`
- \`no_memory_injection\`
- \`no_persistent_write\`
- \`no_graph_evidence\`

These keep the parameter shapes fixed and set the selected information path to zero, allowing controlled follow-up comparisons.

## Run

From the repository root:

    python -m pytest -q experiments/dansn_conditioned_gru_gates/

Use with the real-corpus runner:

    python -m experiments.dansn_conditioned_gru_gates.large_corpus_benchmark \
      --dataset wikitext103 \
      --train-tokens 100000 --validation-tokens 20000 --test-tokens 20000 \
      --epochs 1 --seeds 7 \
      --output benchmark-results/wikitext103_four_gate_dggru.json

The benchmark includes plain GRU, readout-fusion DANSN-GRU, update-only conditioned GRU, full three-path gate-conditioned GRU, and the four-gate DG-GRU. Static test metrics and score-before-write prequential memory-only metrics are separate. Parameter counts are reported, not assumed to match.

## Important limitations

The graph neighborhood selection and discrete topology updates are non-differentiable. Gradients flow through the selected-node readout, controller, four gates, candidate computation, and persistent vectors during truncated BPTT. Graph evidence is a heuristic structural summary, not a calibrated probability of retrieval correctness. The persistent write gate controls vector writes, not whether observed token edges are added to the DANSN topology.

This architecture is a research hypothesis. Do not claim it outperforms the previous readout-fusion or three-gate variants until multi-seed real-corpus and delayed/interleaved relational-recall results support that claim.
