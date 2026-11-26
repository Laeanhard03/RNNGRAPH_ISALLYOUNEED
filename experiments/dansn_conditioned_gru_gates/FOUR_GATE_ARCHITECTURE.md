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


## Targeted synthetic mechanism test

The follow-on branch also includes `associative_recall.py`. It generates deterministic training associations and a disjoint held-out key set. Each held-out association is introduced at test time, followed by distractor tokens and a shuffled recall query. The RNN hidden state is reset before test while DANSN topology and persistent memory from training remain. Test-time neural parameters are frozen, and every query loss is recorded before the target is revealed.

The evaluator reports two distinct outcomes:

- **Graph mechanism:** whether the expected context path is activated and its persistent vector exists at query time; whether support writes create the intended graph path/vector.
- **Predictive mechanism:** top-1 value accuracy and query negative log-likelihood after delay.

The comparison includes plain GRU, readout-fusion, three-path gate-conditioned GRU, full DG-GRU, and four-gate ablations. This synthetic benchmark tests one-shot sequence association and delayed retrieval only. It does not test general semantic idea formation or multi-hop reasoning across distinct graph branches.


## Retrieval suffix backoff correction — 2026-11-27

The first synthetic delayed-recall run found a context-selection failure shared by the persistent DANSN recurrent adapters: a newly observed, longer suffix ending at the current query key could be terminal (no answer continuation) and shadow a shorter suffix that had the stored key/value continuation. The path and persistent vector had been written, but the readout neighborhood did not include that stored child.

`PersistentDANSNRNN._longest_context_path()` now prefers the longest matching suffix whose graph node has outgoing continuations or has its own persistent vector. It falls back to the longest existing terminal suffix only when no more useful suffix exists. A regression test simulates the real stream order, including observing the query marker and key after distractors, then checks that retrieval backs off to the association prefix and activates its value child.

This is a retrieval correctness fix to the shared experimental recurrent adapter; canonical `agi.py` remains unchanged. The synthetic benchmark is rerun after this correction. Do not use the first run's 0% active-path coverage as a model-capability conclusion; it exposed the matcher bug.


## First corrected synthetic mechanism result — 2026-11-27

Run [45890601026](https://github.com/Laeanhard03/RNNGRAPH_ISALLYOUNEED/actions/runs/45890601026) completed successfully; the experiment suite passed **29 tests** (one non-fatal missing-NumPy warning). The corrected matcher yielded 100% expected active-path coverage and 100% persistent-vector coverage for all write-enabled systems, while the no-persistent-write control showed 0% for both.

Despite successful storage and path activation, all systems scored **0% query top-1 accuracy** across three seeds. Mean query NLL was 4.233 (plain GRU), 4.547 (readout fusion), 4.207 (three-path gate GRU), 5.111 (full four-gate DG-GRU), 5.222 (no memory injection), 4.249 (no persistent write), and 5.118 (no graph evidence). The task is therefore not yet solved end-to-end: DANSN exposes the expected stored path, but the neural decoder has not converted that retrieved representation into the correct output token. No superiority claim is justified.

The next diagnostic iteration ensures every possible answer token appears during supervised training and adds value-only ranking/probability metrics plus per-query recurrent-state reset scores. This distinguishes a decoder failure from output-vocabulary competition and from reliance on transient hidden state.

## Four-gate state-coupling correction — 2026-11-27

The first balanced diagnostic showed that the previous memory path was too weakly coupled to recurrence: persistent memory entered only inside the candidate activation, then its effect was multiplied again by the update gate. The reset gate also filtered the candidate input but did not clear old hidden content from the carried-state interpolation. Full DG-GRU therefore remained vulnerable to distractor-contaminated hidden state.

The corrected four-gate cell explicitly combines two state sources:

1. **Recurrent proposal**
   [
   h_t^{\mathrm{rec}}=(1-z_t)\odot(r_t\odot h_{t-1})+
   z_t\odot\tanh(W_xx_t+W_h(r_t\odot h_{t-1}))
   ]
2. **Persistent-memory proposal**
   [
   m_t^{\mathrm{proj}}=\tanh(W_m\,\mathrm{ReadDANSN}(x_t,h_{t-1},G))
   ]
3. **Memory-controlled state fusion**
   [
   \alpha_t=g_t^{\mathrm{mem}}\odot a_t,qquad
   h_t=(1-\alpha_t)\odot h_t^{\mathrm{rec}}+
   \alpha_t\odot m_t^{\mathrm{proj}}
   ]
   where (a_t) indicates whether a persistent DANSN vector was actually retrieved.

This is a key change: DANSN memory is now an independent source of recurrent-state content and is not attenuated a second time by the update gate. The reset gate now controls both candidate conditioning and hidden-state retention, allowing it to discard distractor state when graph context indicates a different active association.

The direct persistent read is separate from the broad transient graph-neighborhood readout. It first attends to stored vectors on immediate outgoing continuations of the matched context; if none exist, it falls back to other memory-bearing active paths. Each stored vector is keyed using the child-node embedding plus the vector itself and receives learned query attention with graph-frequency and memory-quality priors. The ordinary graph readout remains available to the output head and write path.

Ablations remain parameter-shape matched: no memory injection sets the fusion gate to zero; no persistent write prevents storage; no graph evidence zeros structural controller features while actual memory availability remains separately detected. Checkpoint marker is now `dansn_four_gate_gru_v2` because the state-transition semantics changed.


## Memory-gate initialization

The first direct state-fusion run measured mean effective memory mix of 0.084; the model learned to keep the memory gate nearly closed, and full DG-GRU only modestly improved NLL over its no-memory-injection ablation. The memory gate's initial bias is therefore now 0.0 (initial sigmoid near 0.5) so retrieved persistent vectors can contribute from early training. The persistent-write gate remains initialized at -2.0 to keep writes conservative until learned. This is an initialization hypothesis being tested, not a guaranteed performance improvement.
