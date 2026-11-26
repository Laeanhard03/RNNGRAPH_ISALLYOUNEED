# RNNGRAPH_ISALLYOUNEED

Research code for the **Deep Attention Neuron-Synapse Network** and an optional
RNN-controlled recurrent graph-state block.

## Canonical graph engine

The original graph engine remains in `agi.py`. It is a token-sequence graph
generator with trie-like synapse paths, longest-suffix context backoff,
fuzzy continuation overlap, BFS multi-hop attention, and autoregressive
candidate sampling. It is not a Transformer or a pretrained language model.

Original project scripts:
- `agi.py` — canonical graph engine and `Brain` wrapper.
- `main.py` — interactive terminal interface.
- `build_dataset.py` — ChatterBot corpus preparation.
- `hf_tokenized.py` — Hugging Face conversation export.
- `learn_experiment.py` — sparse topic-to-candidate feedback weights.
- `feedback.py` — graph-native feedback through topic gains and synapse counts.
- `feedback_experiment.py` — feedback evaluation harness.
- `test_agi.py` — original engine regression tests.

## Optional recurrent graph-state block

`recurrent_graph.py` introduces a separate PyTorch module that combines:
- a GRU controller for temporal sequence state;
- shared token embeddings;
- sparse temporary state vectors for context-specific graph neurons;
- gated message passing along synapse edges in a retrieved local subgraph;
- query-weighted graph readout and a gated residual into the controller state.

It uses the canonical engine's suffix lookup to choose an active neighborhood.
The graph topology is a snapshot: call `model.refresh_graph(engine)` after
changing the underlying graph.

This is a research prototype, not a trained language model. It does not yet
include a complete training loop, batching, optimizer, dataset pipeline,
benchmark comparison, or end-to-end gradient learning of discrete graph
retrieval. Train it on a task before interpreting its logits as useful output.

### Install and test

The original engine uses the Python standard library. The recurrent module
requires PyTorch:

```bash
pip install -r requirements-recurrent.txt
python test_agi.py
python test_recurrent_graph.py
```

The recurrent tests check tensor shapes, finite outputs, gradients through
controller and graph-update parameters, state carry-over between chunks, and
empty-sequence handling. They are unit tests, not evidence of task-level
reasoning or performance gains.

## Research direction

The chosen direction is an **RNN-controlled recurrent graph-state network**:
the RNN handles temporal control while graph neurons maintain sequence-local
states and exchange gated messages through persistent synaptic structure.
Keep the original `agi.py` engine as the baseline. Validate against a plain
GRU and established gated graph-recurrent baselines before integrating this
prototype into the canonical engine.

## Current status

- Original engine regression suite: 14/14 passed in the development session.
- Recurrent graph-state unit tests: 4/4 passed in the development session.
- No task-level benchmark or superiority claim has been established.
