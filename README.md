# RNNGRAPH_ISALLYOUNEED

Research code for the **Deep Attention Neuron-Synapse Network (DANSN)** and separate recurrent-memory experiments.

## Canonical graph engine

The original graph engine remains in `agi.py`. It is a token-sequence graph generator with trie-like synapse paths, longest-suffix context backoff, fuzzy continuation overlap, BFS multi-hop attention, and autoregressive candidate sampling.

**The canonical `agi.py` engine is not modified by the experiments below.**

Original project scripts:
- `agi.py` — canonical graph engine and `Brain` wrapper.
- `main.py` — interactive terminal interface.
- `build_dataset.py` — ChatterBot corpus preparation.
- `hf_tokenized.py` — Hugging Face conversation export.
- `learn_experiment.py` — sparse topic-to-candidate feedback weights.
- `feedback.py` — graph-native feedback through topic gains and synapse counts.
- `feedback_experiment.py` — feedback evaluation harness.
- `test_agi.py` — original engine regression tests.

## Earlier recurrent graph prototype

`recurrent_graph.py` contains an optional GRU controller with temporary sparse graph-node states and gated message passing. It snapshots graph topology and is not a continual learner.

## Persistent DANSN-RNN experiment

`persistent_dansn_rnn.py` implements a coupled recurrent loop:

1. The current token and previous GRU state query a bounded DANSN context neighborhood.
2. DANSN persistent context-path vectors are read using differentiable attention and graph-structure evidence.
3. The graph readout is fused into the GRU update and next-token prediction.
4. The target is scored **before** it is revealed to graph memory.
5. After target reveal, DANSN extends/reinforces the observed context-to-target path and writes a gated state vector to the post-target context path.
6. `train_stream` backpropagates next-token loss through short chunks using truncated BPTT; graph topology and memory writes happen every token.
7. `evaluate_prequential` records pre-update losses before adapting graph/memory. It does not change neural parameters.

### Important implementation details

- Persistent neuron memory is stored in a sparse dictionary keyed by stable token-path tuples, not Python object IDs.
- The path index is updated incrementally as graph paths are observed, so new nodes are retrievable without rebuilding the whole index.
- Only the post-target context node receives the initial write; memory is not broadcast to unrelated nodes.
- The first version uses a fixed vocabulary. Build the vocabulary before constructing the model or map unseen tokens to an UNK ID.
- Save the DANSN graph and vocabulary separately with `Brain.save()`; save neural parameters and sparse node memory with `model.save_checkpoint()`. Restore the matching graph/vocabulary before loading the model checkpoint.
- Model memory is sparse and bounded by an LRU cap with quality/write-count tie-breakers. Provenance, contradiction handling, learned utility-based eviction, and dynamic vocabulary are future work.
- Discrete graph-edge creation is a non-differentiable memory operation. Backpropagation trains the recurrent controller and differentiable query/readout/write modules; graph edges are created after observations.
- This is a research prototype, not evidence of better language modeling. Lower prequential perplexity is a hypothesis to test, not a guarantee.

### Run

```bash
pip install -r requirements-recurrent.txt
python train_persistent_demo.py
python test_persistent_dansn_rnn.py
python benchmark_continual_learning.py --epochs 3 --seeds 7,13,21
```

The demo uses a small synthetic corpus and is only a smoke test. It is not a benchmark. The test suite checks causal predict-before-write behavior, immediate graph growth, persistent memory retrieval, online parameter updates, prequential evaluation without weight updates, checkpoint round-tripping, and unknown-context fallback.

### Persistence and resume

```python
from agi import Brain
from persistent_dansn_rnn import PersistentDANSNRNN

brain = Brain.load("vocab.bin", "graph.bin")
model = PersistentDANSNRNN(
    brain.engine,
    vocab_size=len(brain.vocab.word_to_id) + 1,
    hidden_size=48,
    window_size=8,
    graph_depth=3,
    message_rounds=1,
)
model.load_checkpoint("persistent_memory.pt")
```

Use the same architecture configuration when restoring the model checkpoint. Graph/vocabulary and model/node-memory files are separate artifacts.

## Evaluation plan

Compare on the same chronological stream:
1. Plain GRU.
2. Original DANSN.
3. GRU with read-only DANSN retrieval.
4. GRU with persistent writable DANSN context memory.

Measure prequential next-token NLL/perplexity, delayed and multi-hop recall, retention, distractor sensitivity, graph/memory size, and retrieval latency. Keep target leakage out of every evaluation.

## Current status

- Original DANSN test suite: 14/14 passed in the earlier development session.
- Earlier recurrent prototype: 4/4 unit tests passed in the earlier development session.
- Persistent DANSN-RNN adapter tests are included; task-level benchmarks and superiority claims remain unestablished.

## Memory quality and comparison harness

The persistent adapter stores per-context memory quality and access metadata. Low-confidence writes are blended less strongly, quality provides a small retrieval prior, and capacity eviction removes the least-recently-used entry (ties favor lower-quality/less-used memories). A zero-confidence write leaves the existing vector unchanged. Checkpoint format v2 stores these metadata fields and still loads v1 checkpoints with default quality values.

Run `python benchmark_continual_learning.py --epochs 3 --seeds 7,13,21` to compare a plain GRU, original DANSN, a read-only DANSN-GRU, and the writable persistent DANSN-RNN on a small synthetic chronological stream. The harness reports both static held-out NLL/perplexity and prequential adaptive NLL/perplexity for all four systems, plus graph candidate coverage. Multi-seed mode summarizes mean and population standard deviation across independent seeds. Static scoring runs on a copy so it cannot mutate memory-access metadata used by the subsequent adaptive run. Static metrics are compared only with static metrics; adaptive metrics compare each system after score-before-update online adaptation. It is a smoke/research comparison, not a benchmark claim; use multiple seeds and larger held-out data before drawing conclusions. The one-seed and three-seed synthetic pilot results are recorded in [BENCHMARK_PILOT_RESULTS.md](BENCHMARK_PILOT_RESULTS.md), including the static/adaptive trade-off and limitations.
