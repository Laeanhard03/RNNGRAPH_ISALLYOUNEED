# DANSN–RNN Persistent Memory: Online Learning Protocol

**Status:** Design proposal v0.1 with a first executable experimental adapter committed on 2026-11-26. The code is a prototype; no task-level performance claim is established.

**Experiment repository:** `Laeanhard03/RNNGRAPH_ISALLYOUNEED`  
**Canonical baseline:** `agi.py` remains unchanged.  
**Original project record:** [Deep Attention Neuron-Synapse Network](https://app.notion.com/p/424b91b4535a81738702e1f0b21c440d)

## 1. Objective

Build one coupled recurrent learner in which the RNN and DANSN participate in every sequence step:

- The recurrent controller maintains temporal state and predicts the next token.
- DANSN retrieves previously observed contextual associations and persistent state annotations.
- The controller's current representation helps write useful state into the relevant DANSN context nodes.
- New observations can extend/reinforce graph paths immediately, without requiring a gradient step to create a discrete edge.
- Prediction loss backpropagates through the differentiable read/recurrent/prediction path so the RNN learns how to use the growing graph memory.

This is not a read-only memory sidecar and not a replacement of the original DANSN implementation. It is a separate experiment around the existing graph engine.

## 2. State model

At time t, maintain:

- `G_t = (V_t, E_t, c_t)`: DANSN topology and observed edge/node counts.
- `M_t = {m_v}`: persistent vector annotations associated with graph context nodes.
- `h_t`: RNN hidden state for the active sequence.
- `theta`: trainable neural parameters: token embeddings, recurrent controller, query/key/value projections, write/update gates, and prediction head.

Keep these notions distinct:

- Graph topology/counts are non-parametric memory updated by observation.
- Persistent node vectors store learned contextual representations.
- RNN hidden state is the current sequence's working state.
- Temporary activations inside a graph read/message-passing operation are not automatically persistent.

### Stable graph-memory identity

The current DANSN trie has path-specific neurons: the same token can occur in different nodes under different prefixes. The experiment should initially attach memory to the *context path*, not only to a token ID.

Do not use Python `id(node)` as a persisted key. It changes across reloads. Use a stable path key (for example a tuple of token IDs from the root, or a persistent node ID with a migration/version policy). This allows the state for a context-specific neuron to be restored after the graph is loaded.

## 3. Causal one-step learning protocol

The essential invariant is **predict before write**. Never expose the current target to graph retrieval before computing its prediction loss.

Given current observed token x_t, previous controller state h_(t-1), and graph/memory from prior observations:

### A. Retrieve prior context

Form a query from the current token and prior recurrent state:

```
q_t = W_q [e(x_t); h_(t-1)]
K_t = Retrieve(G_(t-1), context_<=t, q_t)
```

Candidate selection should first use DANSN's existing longest-suffix and bounded BFS structure. Within that bounded candidate set, use learned differentiable attention to rank/read persistent node vectors. A proposed score is:

```
score(t,v) = dot(W_q q_t, W_k m_v) / sqrt(d)
             + lambda_s * structural_score(G, v)
```

The structural score can include existing path/edge frequency and hop-depth discount. Avoid scanning the entire graph on every token.

Readout:

```
beta_t = softmax(score(t, v) for v in K_t)
r_t = sum_v beta_(t,v) * W_v m_v
```

If no relevant memory exists, use a zero/default readout and let the controller operate normally.

### B. Update the RNN and predict

```
h_t = GRU([e(x_t); r_t], h_(t-1))
logits_t = W_o [h_t; r_t]
p_t = softmax(logits_t)
L_t = cross_entropy(logits_t, x_(t+1))
```

The prediction is made from the graph state that existed before x_(t+1) was observed.

### C. Reveal the target and grow DANSN

Only after calculating L_t may x_(t+1) be revealed to the learner:

1. Record the observed transition from the relevant context suffix to x_(t+1).
2. Create missing path nodes/edges or increment existing frequencies in G.
3. Identify the terminal context node(s) for the observed path.
4. Build a write candidate from the controller's representation and the newly observed association.

The first implementation should write to the terminal context-specific node for the observed suffix-plus-target path. Do not broadcast every hidden state to all retrieved nodes; that would contaminate unrelated contexts. A later controlled experiment can test weighted writes to a small number of related nodes.

### D. Write persistent state

For a selected node v, form a candidate memory and write gate:

```
z_t = tanh(W_write [h_t; e(x_(t+1)); r_t])
g_t = sigmoid(W_gate  [h_t; e(x_(t+1)); r_t])
m_v <- (1 - g_t) * m_v + g_t * z_t
```

A confidence/relevance factor can further scale g_t. New nodes start from a defined initialization (initially zeros) and acquire memory when a transition is observed.

This write is an online memory operation. It does not require an optimizer step to append a new edge or record a new observed association.

At the next sequence step, the updated graph path and memory can be retrieved immediately.

## 4. How backpropagation interacts with graph writes

There are two learning channels and they must not be conflated.

### Fast non-parametric update

- Create/extend discrete graph edges after the target has been revealed.
- Update observation counts.
- Update persistent node memory through the explicit gated write rule.
- These operations can occur after each observed token or short example.

### Gradient-based neural update

Prediction loss backpropagates through:

- the prediction head;
- the fused RNN state;
- GRU/controller operations;
- graph query, key, value and readout projections;
- any differentiable state-write operations retained in the current truncated training window.

Discrete graph topology changes are not ordinary differentiable parameters. Do not pretend that a gradient creates a trie edge.

### Truncated online training

Do not retain an autograd graph across the model's entire lifetime.

Initial training policy:

1. Process a short chunk of sequential tokens with causal prediction-before-write.
2. Accumulate next-token losses over that chunk.
3. Run backpropagation through the chunk (truncated BPTT) and update `theta`.
4. Detach the controller state at the chunk boundary.
5. Commit persistent graph-memory vectors as detached values, then continue with the next chunk.

Within a chunk, a differentiable working overlay may preserve gradients through writes and reads that occur in that same chunk. At the boundary, detach/commit the memory so old graph history does not remain attached to the autograd graph.

For the simplest first prototype, graph topology writes and persistent memory writes can be treated as explicit non-gradient updates, while loss gradients train the controller and read/readout projections. A follow-up ablation can add differentiable within-chunk writes and an auxiliary memory-write objective.

The system can update the RNN every small chunk rather than perform a full offline retraining run whenever new data arrives. This is online/continual adaptation, not a guarantee that one gradient step is always sufficient.

## 5. Perplexity and evaluation without leakage

Evaluate on a chronological stream in prequential order:

1. At token t, predict x_(t+1) using only G_t and M_t built from tokens already revealed.
2. Record negative log-likelihood for x_(t+1).
3. Reveal x_(t+1).
4. Update graph topology and persistent state.
5. Continue.

Compute perplexity from the recorded pre-update losses. Never insert a target into DANSN before scoring its prediction. Do not evaluate on training-time graph state that has already absorbed the evaluation target.

Perplexity may improve as useful context accumulates, but graph growth can also create noisy associations and worsen predictions. Improvement is a testable hypothesis, not a guaranteed property.

## 6. Graph semantics and "ideas"

The current DANSN trie stores context-specific token paths and co-occurrence/continuation evidence. It does not automatically discover abstract semantic ideas or know that two different phrases mean the same thing.

For v0.1:
- attach persistent vectors to context-path neurons;
- let the RNN's learned query retrieve related nodes within a bounded candidate set;
- do not merge nodes or assert semantic identity based only on vector similarity.

A later experiment may add explicit concept/entity nodes or cross-context links. That requires its own confidence, provenance, and false-link evaluation.

## 7. Important implementation constraints

- **Dynamic vocabulary:** current `Vocabulary` assigns new IDs, but the PyTorch embedding/output layers have fixed dimensions. A streaming implementation needs a stable subword/byte vocabulary or a carefully tested embedding/output expansion strategy.
- **Graph mutation:** the existing recurrent prototype snapshots topology. New edges will not be visible until the active topology index is incrementally updated or refreshed.
- **Persistent IDs:** use stable path/node keys, not Python object identities.
- **Memory capacity:** maintain sparse states only for nodes that receive writes; enforce a memory budget and explicit eviction/decay policy.
- **Data boundaries:** do not let sliding windows cross unrelated documents or examples unless that continuity is intended.
- **Contradictory observations:** frequencies alone can reinforce conflicting statements. Track provenance/recency or design confidence and revision mechanisms before claiming factual memory.
- **Serialization:** persist graph, vocabulary, node-memory state, architecture/version metadata, and any optimizer checkpoint needed for resumption. Define safe loading and migration.
- **Original project integrity:** do not edit canonical `agi.py` for this experiment. Implement an adapter/sidecar in this experimental repository.

## 8. First validation matrix

Compare on exactly the same chronological stream and parameter budget where practical:

1. Plain GRU language model.
2. Original DANSN graph predictor.
3. GRU + read-only DANSN retrieval (existing prototype family).
4. Coupled GRU + persistent DANSN node-memory writes (this proposal).

Report:
- prequential next-token NLL/perplexity by stream segment;
- delayed recall and multi-hop relational recall;
- retention on earlier examples after later information arrives;
- robustness to distractors and contradictory facts;
- graph/node-memory size, retrieval latency, and tokens/second;
- ablations without graph growth, without persistent vectors, and without graph readout feedback.

Start with a small synthetic controlled corpus and held-out sequences, then move to a real text stream only after the causal evaluation harness is correct.

## 9. First implementation status — 2026-11-26

The separate adapter `persistent_dansn_rnn.py` now implements the first version of this protocol:

- Stable context-path tuples map persistent vectors to DANSN trie nodes.
- New online observations extend the graph path index incrementally.
- The recurrent controller retrieves bounded DANSN neighborhoods through differentiable attention and transient gated message passing.
- The target is scored before `observe_target` mutates graph topology or persistent memory.
- After reveal, the adapter increments/creates the observed path and writes one gated vector to the post-target context path.
- `train_stream` applies truncated BPTT over short chunks while graph and memory writes happen after each target.
- `evaluate_prequential` scores each target before graph adaptation and leaves neural parameters unchanged.
- Model weights and path-keyed memory can be checkpointed separately from the Brain graph/vocabulary.

Supporting files:
- `persistent_dansn_rnn.py`
- `test_persistent_dansn_rnn.py`
- `train_persistent_demo.py`
- Updated `README.md`

## 10. Memory-quality safeguards and comparison harness

The persistent adapter now includes:
- Per-context confidence metadata that lightly biases retrieval toward trusted memory vectors.
- Confidence-scaled writes. A zero-confidence write is an exact no-op for the stored vector.
- LRU memory-cap eviction, with lower quality and lower write count as tie-breakers.
- Persisted quality, last-access, and write-count metadata in checkpoint format v2. Format v1 checkpoints load with default quality metadata.
- Immediate path-index updates during graph writes, so newly created paths can be matched by the next recurrent step without a full graph rebuild.

The comparison harness `benchmark_continual_learning.py` evaluates four systems on a small chronological synthetic corpus:
1. Plain GRU.
2. Original DANSN-only predictor.
3. Read-only DANSN-GRU (graph prebuilt from the training stream; graph/state writes are disabled).
4. Writable persistent DANSN-RNN.

It now reports two test protocols for each system:
- **Static held-out:** score the test stream without test-time graph or parameter updates.
- **Adaptive prequential:** score each target first, then apply that system's online update (GRU optimizer update, DANSN graph update, read-only-hybrid GRU update with fixed graph, or writable-hybrid GRU plus graph/memory update).

Compare static metrics only against static metrics, and adaptive metrics only against adaptive metrics. The DANSN-only and writable hybrid variants score a target before updating the graph with it. The harness remains a small synthetic smoke/research comparison, not a benchmark claim; larger chronological corpora and multiple seeds are necessary.

## 11. Remaining implementation limits and verification

- Eight adapter tests cover causal predict-before-write, immediate graph/memory growth, online parameter updates, zero-confidence write safety, LRU eviction, checkpoint quality metadata, v1 checkpoint migration, and unknown-context fallback.
- GitHub Actions run [45857528186](https://github.com/Laeanhard03/RNNGRAPH_ISALLYOUNEED/actions/runs/45857528186) completed successfully against the actual repository `agi.py`: compilation, 14/14 canonical DANSN tests, 4/4 earlier recurrent tests, 8/8 persistent-memory tests, three-seed benchmark smoke test, and result artifact upload all passed.
- The three-seed synthetic pilot (seeds 7, 13, 21) reported static held-out PPL means: GRU 16.333 ± 0.711; DANSN-only 528.270 ± 0.000; read-only DANSN-GRU 14.269 ± 1.054; persistent DANSN-RNN 14.946 ± 0.611. Adaptive prequential PPL means: GRU 15.829 ± 0.701; DANSN-only 566.323 ± 0.000; read-only DANSN-GRU 13.611 ± 0.967; persistent DANSN-RNN 11.084 ± 0.533. The writable hybrid led the adaptive metric but did not lead static held-out performance.
- These remain smoke results from one tiny repetitive synthetic corpus, 29 predictions per seed, one training pass and unmatched parameter counts. Seed variation does not replace evaluation across distinct datasets, delayed recall tasks, longer streams, or matched compute budgets. Do not claim broad superiority.
- The prototype still uses a fixed vocabulary, one post-target context write, a simple LRU heuristic rather than learned utility-based eviction, and a single-stream training API.
- Persistent vectors live in a sparse external map keyed by stable graph paths rather than fields added to the original `Neuron` class.
- Dynamic vocabulary expansion, provenance/contradiction handling, abstract idea/concept merging, batching, and larger multi-corpus comparisons remain future work.
- The canonical `agi.py` file has not been modified. The adapter is an isolated experiment and must remain separate until validated.


## 12. Delayed two-hop relational recall experiment

The new `delayed_relational_recall.py` experiment tests whether a trained controller can use DANSN memory to answer a relationship lookup after unrelated material has intervened.

### Task structure

The observed knowledge base contains two-hop chains:

```text
person owns vehicle . vehicle parked place .
```

After the primary facts, the stream contains a variable number of unrelated two-hop distractor facts. The held-out query is:

```text
where person
```

The next-token target is the correct `place`. The test knowledge base uses independently randomized person-to-vehicle and vehicle-to-place mappings. Only a subset of person IDs is used for query supervision during training; the remaining query subjects are held out. Entity vocabulary is shared and fixed so this experiment tests relationship retrieval rather than vocabulary expansion.

### Systems and protocol

- **Plain GRU:** trained on the same fact/distractor stream and query supervision, with no external graph.
- **Read-only DANSN-GRU:** its graph topology is built from the observed test facts, but it has no persistent node vectors.
- **Persistent DANSN-RNN:** it absorbs the new test fact stream one target at a time. Each target is exposed to `observe_target` only after the corresponding forward step. Neural weights are frozen during test-context ingestion.

All systems are evaluated with recurrent state both preserved and reset before the query. The reset-state comparison asks whether the graph can serve as memory independently of the active recurrent state. Query scoring itself does not write the answer into memory.

Metrics: exact answer accuracy, answer NLL, mean reciprocal rank, whether the correct target is reachable in the retrieved neighborhood, and whether a persistent-memory path is among retrieved paths.

### Important scope boundary

The two edges in the main chain are adjacent in the serialized fact stream (`person owns vehicle . vehicle parked place`); the distractors occur after the chain and before the query. This is a delayed two-hop recall test over the current path-based DANSN representation. It does **not** yet test arbitrary joining of two facts separated by intervening unrelated facts, or general semantic concept linking. A separate cross-link/entity-node experiment would be needed for that stronger claim.

### Files and automated checks

- `delayed_relational_recall.py` — dataset generation, online fact ingestion, query evaluation, delay levels, and multi-seed aggregation.
- `test_delayed_relational_recall.py` — held-out split, remapping, causal ingestion, two-hop target reachability, and read-only query-evaluation tests.
- GitHub Actions run [45862480231](https://github.com/Laeanhard03/RNNGRAPH_ISALLYOUNEED/actions/runs/45862480231) passed compilation, 14/14 canonical DANSN tests, 4/4 earlier recurrent tests, 8/8 persistent-memory tests, the language-model benchmark smoke test, 6/6 delayed-recall tests (including the multi-seed aggregation regression), the three-seed delayed-recall benchmark, and artifact uploads.
- Larger delayed-recall pilot configuration: 24 subjects, 18 train-supervised query subjects, 6 held-out query subjects per seed, 24 distractor fact pairs, delays 0/8/24, 3 epochs, hidden size 32, and seeds 7/13/21.
- With recurrent state reset, plain GRU top-1 accuracy was 0%; read-only DANSN-GRU was 0%; persistent DANSN-RNN was 5.56% ± 7.86 percentage points (one correct answer among 18 held-out queries). Mean NLL/MRR were 4.970 ± 0.085 / 0.034 ± 0.011 for GRU, 4.183 ± 0.296 / 0.127 ± 0.033 for read-only hybrid, and 4.164 ± 0.099 / 0.153 ± 0.093 for persistent hybrid. Target reachability and persistent-memory visibility were both 100% for the writable hybrid.
- With recurrent state preserved, persistent DANSN-RNN had 0% accuracy at all delays and worse mean NLL than the read-only hybrid (5.081 vs 4.854 at delay 0; 5.267 vs 4.837 at delay 8; 5.260 vs 4.914 at delay 24). This suggests the current memory readout can be visible and modestly help after a recurrent reset, but does not yet integrate reliably with continuous recurrent state.
- The results are preliminary: one generated fact grammar, three seeds, 18 held-out queries total, fixed shared entity vocabulary, and one contiguous serialized two-hop fact chain followed by distractors. This does not establish robust relational reasoning or arbitrary fact joining. See [DELAYED_RELATIONAL_RECALL_PILOT_RESULTS.md](https://github.com/Laeanhard03/RNNGRAPH_ISALLYOUNEED/blob/main/DELAYED_RELATIONAL_RECALL_PILOT_RESULTS.md) for full per-delay metrics and limitations.
