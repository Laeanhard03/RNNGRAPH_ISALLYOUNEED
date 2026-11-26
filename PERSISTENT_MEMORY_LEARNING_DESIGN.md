# DANSN–RNN Persistent Memory: Online Learning Protocol

**Status:** Design proposal v0.1 — documentation only. No implementation or performance claim is made by this document.

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

## 9. Known limitations of the current prototype

The current `recurrent_graph.py` is not yet this design:
- its `node_states` live in a sequence state dictionary rather than a durable graph-memory store;
- graph topology is snapshotted and does not automatically reflect online edge creation;
- it does not implement causal online training, a task-level optimizer loop, or persistence of node vectors;
- its existing tests check shapes/gradients/state carry-over, not continual learning or perplexity.

Therefore this document defines the next implementation target; it does not describe features already implemented.
