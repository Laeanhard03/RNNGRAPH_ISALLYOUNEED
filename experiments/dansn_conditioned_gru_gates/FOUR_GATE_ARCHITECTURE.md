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

Ablations remain parameter-shape matched: no memory injection sets the fusion gate to zero; no persistent write prevents storage; no graph evidence zeros structural controller features while actual memory availability remains separately detected. Checkpoint marker is now `dansn_four_gate_gru_v4` because the state-transition semantics changed.


## Memory-gate initialization

The first direct state-fusion run measured mean effective memory mix of 0.084; the model learned to keep the memory gate nearly closed, and full DG-GRU only modestly improved NLL over its no-memory-injection ablation. The memory gate's initial bias is therefore now 0.0 (initial sigmoid near 0.5) so retrieved persistent vectors can contribute from early training. The persistent-write gate remains initialized at -2.0 to keep writes conservative until learned. This is an initialization hypothesis being tested, not a guaranteed performance improvement.


## Joint recurrent–associative decoder channel

The initial open-memory run raised the mean effective memory mix to 0.356 and improved full-model NLL, but value-only top-1 remained 19.4%. The recurrent state was using DANSN, yet the output head still received only `[hidden ; broad_graph_readout]`. To avoid making the prediction decoder reconstruct all persistent-association detail from the fused hidden state, the v3 model adds a gated skip channel:

\[
q_t^{mem}=\alpha_t\odot ReadDANSN_t^{persistent}
\]

\[
logits_t=W_o[\,h_t\;;\;g_t^{broad}\;;\;q_t^{mem}\,]+b_o
\]

The same effective memory mix (\alpha_t) now controls both (a) DANSN's contribution to recurrent hidden state and (b) the separate associative-memory channel seen by the output decoder. This makes the decoder jointly observe recurrent state, transient graph context, and persistent associative memory. The no-memory-injection ablation sets both the hidden fusion and decoder skip to zero. Checkpoint marker is now `dansn_four_gate_gru_v4`.


## V4 memory-gated next-token pointer

The v3 balanced recall result showed a useful associative effect (NLL 4.164 vs 4.477 for the no-memory ablation; whole-vocabulary top-1 5.56% vs 2.78%) but value-only top-1 remained low at 11.1%. A dense learned decoder should not have to infer an exact token identity solely from a compressed memory vector when DANSN already stores a discrete next-edge association.

V4 therefore derives a token pointer distribution from persistent vectors stored on the matched context's immediate outgoing graph edges. The attention scores already used for persistent-vector retrieval are scattered into a vocabulary-sized distribution over the child token IDs:
[
p_t^{DANSN}(v)=sum_{j: token(path_j)=v} a_{t,j}
]
Only immediate continuation edges with persistent vectors contribute to this pointer. If there is no such edge, pointer availability is false and the pointer mix is zero. Memory-only fallback paths can still contribute to vector state fusion but cannot create a next-token pointer.

The memory gate controls the probability mixture:
[
\beta_t=mean(\alpha_t)
]
[
p_t(v)=(1-\beta_t)\,softmax(logits_t^{GRU})(v)+\beta_t\,p_t^{DANSN}(v)
]
[
logits_t(v)=\log(clamp(p_t(v),\epsilon))
]
Thus the model receives DANSN in three cooperating places: recurrent-state fusion, a direct decoder vector channel, and an explicit associative next-token distribution. Pointer topology selection remains discrete; the attention scores, mixture gate, GRU, vector read, and neural distribution are differentiable. This path is causal because it reads only edges written after earlier targets were scored.

The no-memory-injection ablation zeros both the hidden/decoder vector routes and the pointer mixture. The no-persistent-write ablation has no stored continuation vectors and therefore no pointer. Checkpoint marker is now `dansn_four_gate_gru_v4`.


## V4 causal associative-recall result — 2026-11-27

The corrected seven-system benchmark completed successfully in [GitHub Actions run 45896936662](https://github.com/Laeanhard03/RNNGRAPH_ISALLYOUNEED/actions/runs/45896936662). The full experiment suite passed **41 tests, 1 non-fatal warning in 5.28s**.

Balanced task: 16 training keys cover all 16 output-value classes; 12 disjoint held-out keys; unseen held-out key/value pairings; 64 distractors; mean support→query separation 116.4 tokens; 12 queries per seed; 3 seeds. Neural parameters are frozen during evaluation, recurrent state is reset once at test start, the graph and persistent vectors are retained, and every target is scored before it is written.

Three-seed mean metrics:
| System | Exact top-1 | Value-only top-1 | NLL | Pointer mix | Pointer target prob. |
|---|---:|---:|---:|---:|---:|
| Plain GRU | 0% | 0% | 4.310 | — | — |
| Readout-fusion DANSN | 0% | 27.8% | 4.199 | — | — |
| Three-path gate GRU | 0% | 8.3% | 4.320 | — | — |
| Full four-gate v4 | **100%** | **100%** | **0.385** | 0.679 | 1.0 |
| Four-gate, no memory injection | 2.78% | 5.56% | 4.477 | 0 | 1.0 |
| Four-gate, no persistent write | 2.78% | 11.1% | 4.038 | 0 | 0 |
| Four-gate, no graph evidence | **100%** | **100%** | 0.396 | 0.671 | 1.0 |

The full model had 100% active expected-path coverage and persistent-vector coverage. The pointer had one persistent candidate on average and assigned the true answer probability 1.0. Full four-gate also reached 100% value-only top-1 when the recurrent hidden state was reset at each individual query, confirming the stored DANSN association could answer without relying on transient hidden-state carry.

Critical interpretation:
- Full model vs no-memory-injection ablation shows the memory routes are essential: exact top-1 fell from 100% to 2.78% and NLL worsened 0.385→4.477 when both memory state fusion and pointer mixture were disabled.
- No-persistent-write also collapsed exact top-1 to 2.78% with 0 pointer candidates, demonstrating that the pointer depends on persistent storage.
- No-graph-evidence retained 100% recall; for this simple single-outgoing-association task, the graph-evidence feature vector is not necessary. The test supports memory-gated persistent retrieval, not the independent usefulness of graph evidence/reset/update gates.
- The readout-fusion baseline reached 27.8% value-only top-1, so V4 clearly beats it on this specific task, but it is not parameter matched and the v4 system adds an explicit symbolic next-token pointer.
- Each query had exactly one persistent outgoing answer candidate (pointer candidate count 1). This is a controlled one-shot key/value recall task, not yet ambiguous multi-branch disambiguation, multi-hop reasoning, free-form language, or abstract semantic learning.

Result artifacts: GitHub Actions artifact `dansn-associative-recall-results`, artifact ID 15154461731. Full run/JSON and data can be downloaded from the workflow page. Do not generalize the 100% result beyond the tested single-candidate delayed-association setting.


## Context-dependent multi-branch recall — first result and matched-delay follow-up

A harder task stores three different value continuations under the same \`(branchctx, key)\` parent. The model receives a cue, then distractors, then the shared context and key. The query target is not part of the input. Each held-out key has three one-shot support writes. The recurrent state is reset at each query start, while persistent DANSN memory is retained. The evaluation also changes the cue for the same key (counterfactual branch switch) and repeats the key lookup with the recurrent state zeroed.

The no-pointer ablation disables only the vocabulary-pointer probability mixture; it retains DANSN vector fusion into the recurrent hidden state and decoder skip channel. This isolates direct pointer copying from the vector-memory route.

### First three-seed result — delay mismatch

The initial benchmark [completed successfully in Actions run 45917748328](https://github.com/Laeanhard03/RNNGRAPH_ISALLYOUNEED/actions/runs/45917748328), with artifact ID 15163315694. It used 16 training keys, 8 held-out keys, 3 cues/branches per key, 12 value classes, 4 training episodes, and 72 held-out queries across seeds 7/13/21. Every query exposed exactly 3 persistent pointer candidates.

| Metric | Full V4 | No pointer | No memory injection | No persistent write | No graph evidence |
|---|---:|---:|---:|---:|---:|
| Output top-1 | 33.33% | 6.94% | 20.83% | 8.33% | 33.33% |
| Value-only top-1 | 33.33% | 13.89% | 20.83% | 9.72% | 33.33% |
| Raw pointer top-1 | 33.33% | 33.33% | 33.33% | 0% | 33.33% |
| Counterfactual cue selection | 31.94% | 31.94% | 33.33% | 0% | 31.94% |
| Pointer top-1 with hidden state reset | 33.33% | 33.33% | 33.33% | 0% | 33.33% |

Full V4 wrote and retrieved all held-out vectors (100% coverage), with a mean of 3 candidates/query and mean pointer target probability 0.3318. But selection remained at the 3-way chance rate (33.33%). Changing the cue did not reliably switch the chosen branch (31.94%), resetting the recurrent state did not reduce pointer accuracy (33.33%), and the carried-state target-probability gain was -0.0015. This run therefore demonstrates multi-branch storage and retrieval coverage, but **does not demonstrate cue-conditioned selection or a useful recurrent-state contribution**.

The no-pointer system's output top-1 was 6.94%; therefore the vector-to-hidden and decoder-vector routes did not independently solve branch selection in this configuration. No-graph-evidence matched full V4, so the graph-evidence vector is still not demonstrated to help here. No-persistent-write produced no candidates, as expected.

### Delay-distribution confound and corrected run

The first benchmark trained with distractor gaps up to 3 tokens but evaluated at 8 tokens. The cue therefore had to survive a longer, unseen delay. This makes the first result an out-of-distribution delay test rather than a clean matched-delay test.

The script's default maximum training delay has now been changed to 8 tokens, matching the evaluation delay. Dataset metadata records whether the test delay is inside the training range, and a regression test protects the default. The model architecture and ablation definitions are unchanged. The matched-delay rerun is [Actions run 45918540578](https://github.com/Laeanhard03/RNNGRAPH_ISALLYOUNEED/actions/runs/45918540578); a duplicate run at the latest test commit is [Actions run 45918553271](https://github.com/Laeanhard03/RNNGRAPH_ISALLYOUNEED/actions/runs/45918553271). Results are pending at the time of this edit.

The expanded test suite passed **46 tests, 1 warning in 3.95s**. Do not interpret the first run as a definitive architecture failure until the matched-delay benchmark completes. Even a successful matched-delay result would remain a synthetic three-branch task, not evidence of general language or semantic reasoning.


## Matched-delay branch-selection result — 2026-11-27

The corrected three-seed benchmark completed successfully in [Actions run 45918553271](https://github.com/Laeanhard03/RNNGRAPH_ISALLYOUNEED/actions/runs/45918553271), with artifact ID 15164135696. The expanded experiment suite passed **46 tests, 1 warning in 3.95s**. The corrected training and evaluation delays are both 8 tokens (training delay range 0–8, query delay 8), and metadata confirms that the evaluation delay is within the training range.

Configuration: 16 train keys, 8 held-out keys, 3 cue-conditioned branches per key, 12 value classes, 4 training episodes, 4 epochs, seeds 7/13/21, 24 held-out queries per seed (72 total). Every query has exactly three stored candidates; query target is not in the input; the recurrent state is reset at each query while DANSN graph and persistent vectors remain.

| Three-seed mean | Full V4 | No pointer | No memory injection | No persistent write | No graph evidence |
|---|---:|---:|---:|---:|---:|
| Output top-1 | 33.33% | 11.11% | 20.83% | 8.33% | 33.33% |
| Value-only top-1 | 33.33% | 16.67% | 20.83% | 8.33% | 33.33% |
| Raw pointer top-1 | 36.11% | 34.72% | 33.33% | 0% | 33.33% |
| Counterfactual cue selection | 31.94% | 30.56% | 33.33% | 0% | 33.33% |
| Pointer top-1 with hidden state reset | 33.33% | 33.33% | 33.33% | 0% | 33.33% |

Full V4 detailed measurements: pointer target probability 0.32935 (uniform chance 0.33333); mixed target probability 0.24428; mean candidate count 3.0; branch entropy 1.0163 (maximum ln(3)=1.0986); persistent vector write and query coverage 100%; effective memory/pointer mix 0.7022; reset-state pointer accuracy 33.33%; context pointer gain -0.00399.

Interpretation: even after matching train/test delay, output top-1 remains at three-way chance. Raw pointer top-1 is 36.11%, only slightly above chance and not accompanied by counterfactual cue switching (31.94%) or a benefit from carried recurrent state (reset-state pointer accuracy 33.33%; context pointer gain slightly negative). Thus this experiment **does not demonstrate reliable cue-conditioned branch selection or recurrent-state contribution**. It does demonstrate successful storage and candidate discovery: all three branches are written and retrieved for every key/query.

The no-pointer ablation retains memory-to-hidden and decoder-vector paths but obtains only 11.11% output top-1. No-graph-evidence matches full output accuracy, so graph-evidence features have not demonstrated an independent benefit in this task. No-persistent-write has zero stored candidates and zero pointer coverage, as expected.

### Selector diagnosis and next experiment

The present bottleneck is candidate scoring, not graph storage or candidate discovery. The existing persistent read scores each candidate using a dot product between a query projected from [token embedding; previous hidden] and a key projected from [child embedding; stored vector], plus structural priors. But the scores remain close to uniform, suggesting the controller's recurrent cue representation is not being exploited by this selector.

Next proposed change: add explicit pointer-candidate ranking supervision during training query steps, only when the correct target is already among the stored direct candidates. Score the target before calling observe_target(); apply an auxiliary negative-log-probability loss to the pointer distribution, then continue with the existing causal post-score write. This tests whether the existing recurrent query can learn to rank branches before introducing a larger additive-attention scorer. Keep no-pointer as a control that receives the same selector training but suppresses only pointer mixing at inference. Preserve the no-memory, no-write, no-graph-evidence, and reset-state comparisons.

The result remains a synthetic 3-branch experiment, not evidence of natural-language or general semantic reasoning. The matched-delay artifact is GitHub Actions artifact ID 15164135696 (dansn-contextual-branch-results).


## Auxiliary pointer-ranking trainer — validation status — 2026-11-27

The contextual benchmark now includes an auxiliary ranking objective at marked training query positions. It applies \`-log(pointer[target])\` only if the correct target is already represented among the direct persistent candidates. The usual vocabulary next-token loss is retained. Both losses are computed before \`observe_target()\`; the target is revealed and written only after scoring. The same selector supervision is used for all eligible ablations, while \`no_pointer\` disables only the pointer output mixture at inference.

A new training-position test checks that the supervised current token is the key and the next token is its stored target. A gradient test checks that selector training updates the query projection. The first CI attempt found that the trainer accidentally inherited an evaluation-only \`no_grad\` decorator. This was corrected in commit \`cc03c3b870ecf30e39c7ec483bdf9b4d83175cc4\`.

Corrected Actions run [45919882859](https://github.com/Laeanhard03/RNNGRAPH_ISALLYOUNEED/actions/runs/45919882859) has passed dependency installation and the complete experiment test step. Its three-seed benchmark is still running; no performance improvement is claimed yet.

## Selector-supervision audit and corrected graph-context protocol — 2026-11-27

The completed auxiliary-selector benchmark [45919882859](https://github.com/Laeanhard03/RNNGRAPH_ISALLYOUNEED/actions/runs/45919882859) passed CI, but artifact 15164019295 revealed that the new loss had **zero eligible pointer-supervision examples** in every system and all three seeds. Mean pointer loss was 0.0 and supervised-example count was 0. Therefore that benchmark did not actually test the intended auxiliary ranking objective; its accuracy must not be interpreted as evidence for or against the ranking loss.

The data-path audit found that the cue was being included in the same linear graph history that formed persistent DANSN paths. The intended task requires the cue to influence only the recurrent hidden state, while all answer continuations for a key share the DANSN parent (context,key). The earlier flat-history stream did not enforce this separation consistently.

Corrective implementation on the existing branch:

- Training data is now stored as independent recurrent episodes/events, each carrying an explicit DANSN graph-history key separately from the recurrent input sequence.
- In support episodes, the model processes cue → context → key through recurrence, while DANSN graph histories are respectively [cue], [context], and [context,key]. Only after the key prediction is scored is the support value written under (context,key,value).
- In query episodes, the recurrent state processes cue and distractors before context and key. At the key lookup, DANSN receives only (context,key), so all three values are direct candidates under one shared parent.
- Held-out support ingestion and evaluation now use the same separation. Reset-state diagnostics preserve the same graph-history path while removing the recurrent cue state.
- Training logs pointer candidate count and the number of truly supervised selector examples. The tests now require the expected number of eligible examples and exactly three candidates at those query positions; they also verify that all cue branches share the same stored DANSN parent path.

Corrected validation was triggered on commits 5332745b91c9c5ec1b6733ef9fce34b3695e3996 and d3c38993cb7e2ff77555a72341fb1adcb70a68f7. Actions runs [45925675394](https://github.com/Laeanhard03/RNNGRAPH_ISALLYOUNEED/actions/runs/45925675394) and [45925700464](https://github.com/Laeanhard03/RNNGRAPH_ISALLYOUNEED/actions/runs/45925700464) are the contextual benchmark validations; results are pending at the time this note was written. Do not make a new performance claim until an artifact shows nonzero selector-supervision examples and candidate count three at training query positions.


### Train/evaluation write-parity correction

A further audit found one more train/test mismatch: earlier selector-training query records wrote observed query-prefix tokens into DANSN, while held-out evaluation query prefixes were read-only. The selector could therefore see a query-specific graph write during training that was not available at evaluation.

The latest implementation now distinguishes two per-event flags: pointer-supervision eligibility and whether the event's known next token is written to DANSN after scoring. Support records write each observed support continuation after prediction. Query-training records are read-only, matching held-out query evaluation; the auxiliary pointer loss remains active at the query key only. Held-out support ingestion was updated to process and write cue→context, context→key, and key→value in the same causal order as training support records.

Additional tests verify: support events write; query events do not write; each query still receives pointer supervision; and held-out support ingestion writes both the shared (context,key) parent state and all three child branch vectors. The latest strict validation on this policy is [Actions run 45926132236](https://github.com/Laeanhard03/RNNGRAPH_ISALLYOUNEED/actions/runs/45926132236); it is pending. The earlier benchmark runs 45925675394 and 45925700464 used the previous query-write policy and must be treated as protocol-debug runs, not final evidence.


### Final protocol CI status

The strict final-code Actions run [45926132236](https://github.com/Laeanhard03/RNNGRAPH_ISALLYOUNEED/actions/runs/45926132236) has passed the complete four-gate/contextual-branch test step, including shared-parent candidate coverage, nonzero selector supervision, training query read-only policy, and held-out support write parity. The separate loader workflow [45926132263](https://github.com/Laeanhard03/RNNGRAPH_ISALLYOUNEED/actions/runs/45926132263) also passed. The three-seed benchmark in run 45926132236 remains in its training/evaluation step; no result artifact is available yet. Branch compare: 120 commits ahead of main, 0 behind; canonical agi.py unchanged.


## Final matched-protocol selector result and recurrent-preservation correction — 2026-11-27

The strict final-protocol benchmark in [Actions run 45926132236](https://github.com/Laeanhard03/RNNGRAPH_ISALLYOUNEED/actions/runs/45926132236) completed successfully and uploaded artifact 15166972788. The contextual test step passed. The artifact confirms that selector supervision is now active: each seed/system with writes receives 192 supervised training query positions and exactly 3 candidates at each, with mean auxiliary pointer loss about 1.1057 (close to the uniform three-way loss ln(3) = 1.0986).

However, the full four-gate model remains at chance across the three seeds: output top-1 33.33%, pointer top-1 33.33%, counterfactual cue selection 33.33%, and reset-state pointer top-1 33.33%. Mean pointer target probability is 0.3333328; the mean context pointer gain is effectively zero. Persistent write and query coverage remain 100%, with 3 candidates/query. This validates storage, retrieval, and the supervision path, but **does not validate cue-conditioned branch selection**.

A key diagnostic is that the effective memory mix at query time averages 0.985. In the previous cell, that coefficient directly blended the retrieved memory proposal over the recurrent transition, allowing a saturated memory gate to replace nearly all of the cue-bearing hidden state at shared context/key steps. The pointer selector therefore had little useful recurrent cue signal to rank candidates with.

### Recurrent-preserving memory injection — implemented, benchmark pending

The cell now distinguishes the controller's full effective memory mix from the coefficient used to blend persistent memory into the recurrent hidden state:

- <code>effective_memory_mix = memory_gate × availability</code> remains the full signal for the decoder memory skip and direct pointer probability mixture.
- <code>hidden_memory_mix = 0.5 × effective_memory_mix</code> bounds one-step persistent-state replacement so at least half of the recurrent transition remains in the hidden state.
- The resulting hidden state remains a blend of the recurrent transition and the persistent memory proposal; DANSN is still a direct state source, but cannot erase the working context in one step.

Added a regression test that forces the memory gate to saturate and verifies that the hidden state moves toward retrieved memory while staying aligned with the previous recurrent state. The updated code and test are committed on the existing experimental branch. The next contextual benchmark must determine whether this preserves the cue strongly enough to improve pointer ranking and counterfactual selection. No improvement is claimed before that run completes.


## Recurrent-preserving fusion benchmark — 2026-11-27

The bounded hidden-state memory blend passed the strict contextual unit tests and all three triggered GitHub workflows on commit `6089815dc42f854109588472a33848368247335c`. The contextual multi-branch benchmark completed in [Actions run 45932844216](https://github.com/Laeanhard03/RNNGRAPH_ISALLYOUNEED/actions/runs/45932844216), artifact 15169398947.

Three-seed full-model results remained at chance:
- output top-1: 33.33%;
- pointer top-1: 33.33%;
- counterfactual cue selection: 33.33%;
- reset-hidden pointer top-1: 33.33%;
- pointer target probability: 0.333334;
- candidate count: 3.0/query;
- persistent write and query coverage: 100%.

The hidden-state injection coefficient was bounded to at most 0.5 of the full effective memory mix, but this did not increase cue-dependent ranking. Thus, the overwrite hypothesis was plausible and the residual safeguard is useful, but it was not sufficient to explain or fix the selector failure.

### Second training-protocol issue: selector gradient dilution

A further audit found that the training loop appended `nll_loss + pointer_loss_weight × selector_loss` to a list for every token and then averaged that list over a chunk. Only key-query events carry selector supervision (192 examples among 2,391 total training events in the default protocol). This unintentionally scales the selector objective by the small fraction of eligible events, so a weight of 1.0 does not mean equal weighting relative to the token objective.

The trainer now accumulates two loss sets independently:
- next-token NLL, averaged over every event in the chunk;
- pointer-ranking loss, averaged only over eligible supervised branch-query events in the same chunk.

The chunk objective is `mean(NLL) + pointer_loss_weight × mean(selector_loss over eligible queries)`. Chunks with no eligible selector queries use only the NLL term. This preserves causal predict-before-write ordering and does not alter inference or graph storage.

Added a mathematical gradient test with 64 token losses and 5 selector losses. It verifies the selector gradients are averaged over 5 eligible examples (1/5 each), rather than diluted over all 64 events. The corrected benchmark is now running in [Actions run 45981519637](https://github.com/Laeanhard03/RNNGRAPH_ISALLYOUNEED/actions/runs/45981519637); its performance result remains pending until the artifact is inspected.

This correction tests the optimizer objective scaling before changing the candidate-scoring architecture. If ranking remains near `ln(3)` after this normalization, next isolate the recurrent cue representation and introduce an explicit additive candidate scorer with separate query, stored-candidate, and interaction terms. Do not claim contextual recall until counterfactual cue selection exceeds 33.33% and reset-state performance declines.


## Nonlinear context-conditioned candidate scorer — implemented, benchmark pending — 2026-11-27

The previous selector's query/key dot product remained at the uniform three-branch baseline even after (a) separating recurrent cue input from shared DANSN graph history, (b) correcting read-only query/write parity, (c) activating selector supervision, (d) separately normalizing selector loss, and (e) retaining a recurrent residual during memory injection.

The selector now uses an explicit nonlinear pairwise scorer:
`score(q,k) = MLP([q; k; q ⊙ k]) + structural_prior`

- (q) is projected from the current token embedding and previous recurrent hidden state.
- (k) is projected from the candidate child embedding plus its persisted DANSN path state.
- The elementwise interaction (q ⊙ k) gives the MLP an explicit cue/candidate compatibility channel.
- The resulting scores still softmax over the direct outgoing DANSN candidates, then feed the pointer distribution. Candidate discovery, persistent writes, and causal score-before-write order are unchanged.

### Diagnostics added

The model trace now exposes pointer candidate IDs/scores, projected query, and previous recurrent hidden state. Held-out evaluation records:
- mean target candidate score margin against the strongest alternative;
- pairwise cosine similarity and L2 distance between recurrent hidden states for distinct cues on the same key, with the distractor sequence held identical.

The matched-distractor diagnostic is important: comparing cue-conditioned hidden states under different noise sequences would confound the cue with distractor variation. The final diagnostic replays each cue for each held-out key using the same distractor tokens.

Regression tests verify candidate score/ID alignment, nonzero finite gradients into the nonlinear scorer from ranking loss, and cue sensitivity of the recurrent query state under matched distractors. Checkpoint architecture marker advanced to `dansn_four_gate_gru_v5`, with a round-trip test checking the marker, so older v4 checkpoints are not mistaken for the nonlinear-selector model.

Commits:
- scorer: `e8e3583cc068975503ef5aa39c75fdc87c638ccf`
- cue-state and score-margin diagnostics: `107eb7800541381f2603e2ddffad28847f246623`
- matched-distractor diagnostic: `beee6eea514dcd62abf0100f5a13298962ea9796`
- checkpoint marker v5: `50abf452d033b2ab82156aa99de2378cc1bd86c9`
- checkpoint regression test: `d860b6d99077e6c79ebc7921126f13d4c3a2bcfd`

The latest complete code/test revision is being validated by [GitHub Actions run 45987516539](https://github.com/Laeanhard03/RNNGRAPH_ISALLYOUNEED/actions/runs/45987516539). Earlier runs on the scorer implementation passed the nonlinear scorer's gradient and cue-sensitivity tests; the benchmark stage is still running. Do not claim a recall improvement until the artifact shows above-chance pointer/counterfactual selection and a positive target-score margin.
