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


## Delayed two-hop relational recall experiment

`delayed_relational_recall.py` tests whether the trained recurrent controller can use DANSN's growing graph and persistent context vectors to answer a two-hop lookup after distractors:

```text
person owns vehicle . vehicle parked place .
... unrelated facts ...
where person -> predict place
```

The training and test knowledge bases use independently randomized person-to-vehicle and vehicle-to-place mappings. Only a subset of subjects receives query supervision during training; the remaining subjects are held out. At test time, a fresh graph absorbs the remapped facts sequentially with neural weights frozen. We evaluate the answer with recurrent state preserved and reset. The read-only hybrid receives the same test facts as graph topology but has no persistent node vectors; the plain GRU receives no external graph. Query supervision trains the readout/controller but deliberately does not write the known answer as a shortcut edge; test answers must be retrieved from the two-hop fact path.

Run:

```bash
python test_delayed_relational_recall.py
python delayed_relational_recall.py --subjects 24 --distractors 48 --delays 0,8,24,48 --epochs 3
python delayed_relational_recall.py --subjects 24 --distractors 24 --delays 0,8,24 --epochs 3 --hidden-size 32 --window-size 9 --graph-depth 8 --chunk-size 16 --seeds 7,13,21
```

Metrics include exact answer accuracy, mean negative log-likelihood, mean reciprocal rank, target reachability within the retrieved graph neighborhood, and whether a persistent-memory path was among retrieved nodes. Static query scoring does not reveal the answer to the model before scoring it. This is a controlled synthetic memory task, not a natural-language benchmark.

The three-seed pilot and limitations are recorded in [DELAYED_RELATIONAL_RECALL_PILOT_RESULTS.md](DELAYED_RELATIONAL_RECALL_PILOT_RESULTS.md). In the recurrent-state-reset condition, the persistent hybrid reached 5.56% ± 7.86% top-1 accuracy (one correct answer in 18 held-out queries) versus 0% for the read-only hybrid, with slightly better NLL/MRR. With recurrent state preserved, the persistent hybrid underperformed the read-only hybrid at all three distractor delays. The correct target was reachable and a stored memory path was visible in all writable-hybrid queries, so the current bottleneck is translating retrieved state into the right answer. This is a weak pilot signal, not reliable recall or proof of generalization.


## Memory-read supervised learning experiment

`memory_read_supervision.py` adds a controlled follow-up to the delayed two-hop recall task. It trains a small auxiliary head to align the DANSN graph-read vector with the correct place-token embedding:

```text
scores(place_i) = cosine(W_memory · graph_readout, embedding(place_i)) / temperature
loss = next_token_loss + auxiliary_weight · memory_answer_loss
```

The query answer is used only as a training label after the query forward pass. It is **not** written into graph topology or persistent memory. At test time, the knowledge base is ingested with neural weights frozen, and the answer head must decode from the retrieved graph-read vector.

The comparison includes:
- Read-only DANSN-GRU trained with the next-token query objective.
- Persistent DANSN-RNN trained with next-token loss only.
- Persistent DANSN-RNN trained with next-token loss plus memory-read supervision.
- A matched-weight ablation that evaluates the learned answer head with graph topology but without persistent memory vectors.

Run the focused tests and multi-seed experiment:

```bash
python test_memory_read_supervision.py
python memory_read_supervision.py --subjects 24 --distractors 24 --delays 0,8,24 --epochs 3 --hidden-size 32 --window-size 9 --graph-depth 8 --seeds 7,13,21
```

This is still a fixed-vocabulary synthetic task with held-out subjects and independently remapped test relationships. Results must be interpreted as a controlled prototype experiment, not evidence of robust relational reasoning. The canonical `agi.py` and persistent-memory core are unchanged.


### Pilot result (3 seeds)

The full result is in [MEMORY_READ_SUPERVISION_PILOT_RESULTS.md](MEMORY_READ_SUPERVISION_PILOT_RESULTS.md). The final tie-aware CI run [45865069032](https://github.com/Laeanhard03/RNNGRAPH_ISALLYOUNEED/actions/runs/45865069032) passed all tests and the three-seed benchmark. The persistent memory-read answer head reached 11.11% ± 7.86 pp accuracy with recurrent state reset (2 correct out of 18 queries), versus 0% for the matched topology-only read-head ablation; with recurrent state preserved it reached 5.56% ± 7.86 pp (1/18). Its mean reciprocal rank improved over the uniform ablation, but mean NLL remained worse than uniform guessing (3.694–4.155 versus 3.178). The main next-token output head still had 0% continuous-state accuracy. This is a weak ranking signal, not reliable recall or calibrated answer selection; do not promote the auxiliary head into the core yet.


### Interleaved relational facts and exact-entity bridge

The delayed-recall harness now has an optional hard mode:

```text
person owns vehicle
... unrelated complete fact pairs ...
vehicle parked place
where person -> predict place
```

Use `--interleave-facts N` to insert N unrelated fact pairs between the complete first-hop and second-hop fact groups. The original contiguous task remains the default (`--interleave-facts 0`). Training and test knowledge bases both use the same serialization format but independently randomized relationships.

For the interleaved condition, the harness trains a matched pair of persistent models:
- **Unlinked trie:** ordinary DANSN path growth and persistent writes, with no associative bridge.
- **Exact-entity linked:** after facts are observed, join the context-specific `person owns vehicle` path to the observed `vehicle parked place` branch when the exact vehicle token ID matches. Alias the already-observed target-bearing memory vector to the joined path. The linker reads facts only and never reads query labels or writes an answer edge.

Run the hard mode:

```bash
python test_delayed_relational_recall.py
python delayed_relational_recall.py --subjects 24 --distractors 24 --interleave-facts 8 --delays 0,8 --epochs 3 --hidden-size 32 --window-size 9 --graph-depth 8 --seeds 7,13,21
```

This is a deliberately narrow structural probe: it tests exact repeated entity-token linking, not semantic entity resolution, aliases, pronouns, or paraphrase. The bridge is an experimental operation in the recall harness; it has not been promoted into `persistent_dansn_rnn.py` or the canonical `agi.py`.


The interleaved evaluation now reports both generic active-memory visibility and `target_bearing_memory_visible_fraction`, which counts only memory vectors on active paths ending in the correct target token. Use the target-bearing metric when determining whether the answer-bearing state is actually exposed; generic path visibility alone is not sufficient.


### Interleaved relational-recall pilot result

The 24-subject, three-seed interleaved experiment completed in GitHub Actions [45868295697](https://github.com/Laeanhard03/RNNGRAPH_ISALLYOUNEED/actions/runs/45868295697). With 8 unrelated fact pairs inserted between the two relation groups, the unlinked persistent trie had 0% target reachability and 0% target-bearing memory visibility. The observed exact-entity bridge linked all 24 vehicle relations per seed and raised both diagnostics to 100% at delays 0 and 8. However, the linked model still got 0/18 held-out queries correct at both delays. Continuous-state NLL improved modestly (4.993 to 4.828 at delay 0; 4.877 to 4.651 at delay 8), without a top-1 accuracy gain. This proves a narrow structural path-joining capability, not reliable relational answering. The complete setup, metrics and limitations are in [DELAYED_RELATIONAL_RECALL_PILOT_RESULTS.md](DELAYED_RELATIONAL_RECALL_PILOT_RESULTS.md). The exact-token linker remains an experiment; neither `agi.py` nor `persistent_dansn_rnn.py` was changed.


## DANSN-conditioned GRU on larger corpora

A follow-on experiment to `feature/dansn-conditioned-gru-gates` lives on
`feature/dansn-gated-gru-real-corpora`. It preserves the canonical
`agi.py` engine and the earlier synthetic gate experiment.

The new branch adds:
- Streaming WikiText-103 and optional PG-19 corpus loaders.
- Train-only vocabulary construction with a fixed UNK mapping for validation/test.
- A larger-corpus comparison of plain GRU, readout-fusion DANSN-GRU,
  update-gate-only DANSN-GRU, and fully gate-conditioned DANSN-GRU.
- Separate frozen/static and score-before-write DANSN-memory evaluation.
- Offline corpus tests and GitHub Actions workflows.

See [the dataset and run guide](experiments/dansn_conditioned_gru_gates/DATASETS.md).
Dataset binaries are downloaded/cached at run time and are not committed here.
Start with bounded token budgets before attempting a full PG-19 run.


## Four-gate DANSN-GRU experiment

The follow-on branch **feature/dg-gru-four-gate-control** adds a distinct four-gate DG-GRU variant with DANSN-conditioned update, reset, memory-injection, and persistent-write gates. The write gate is computed before target reveal; after scoring, it scales both graph-path frequency reinforcement and the persistent node-vector update. A zero write gate suppresses both. See [FOUR_GATE_ARCHITECTURE.md](experiments/dansn_conditioned_gru_gates/FOUR_GATE_ARCHITECTURE.md). It also adds this variant to the real-corpus comparison runner. The branch is experimental; no performance improvement is assumed.
