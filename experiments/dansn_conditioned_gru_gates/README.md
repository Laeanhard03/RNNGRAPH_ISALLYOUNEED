# DANSN-Conditioned GRU Gates

**Status:** isolated research prototype on branch feature/dansn-conditioned-gru-gates. This is not the canonical DANSN engine and is not yet benchmark-validated.

## Hypothesis

Condition the GRU's reset gate, update gate, and candidate-state calculation directly on a differentiable readout from DANSN's active graph neighborhood.

This differs from the existing PersistentDANSNRNN baseline, which reads DANSN context and fuses it into the controller/input path after the standard GRU update. Here, graph context participates in the gate preactivations themselves.

## Gate equations

    z_t = sigmoid(W_z x_t + U_z h_(t-1) + V_z c_t)
    r_t = sigmoid(W_r x_t + U_r h_(t-1) + V_r c_t)
    n_t = tanh(W_n x_t + U_n (r_t * h_(t-1)) + V_n c_t)
    h_t = (1 - z_t) * h_(t-1) + z_t * n_t

The sigmoid gates remain trainable, differentiable control signals. DANSN does not directly choose binary gate values. It supplies context that learned projections transform into evidence for the gates.

## Per-token causal flow

1. Embed the current token and query DANSN using that embedding plus the previous hidden state.
2. Select a bounded graph neighborhood using the existing DANSN context/suffix retrieval.
3. Compute a differentiable attention readout from the selected node states and persistent memory.
4. Feed the readout into the reset gate, update gate, and candidate-state calculation.
5. Produce next-token logits from the new hidden state and graph readout.
6. Score the target before revealing it to DANSN.
7. Only after scoring, call observe_target() to extend/reinforce the graph path and perform the existing confidence-gated persistent memory write.

This preserves the existing predict-before-write invariant.

## Implementation

- model.py defines DANSNConditionedGRUCell and DANSNConditionedGRU.
- The model reuses PersistentDANSNRNN graph indexing, sparse active-neighborhood retrieval, persistent context-path memory, memory quality metadata, confidence-gated writes, and capacity management.
- It replaces the earlier standard GRUCell plus post-GRU fusion gate with explicit graph-conditioned GRU gate equations.
- The output head receives [h_t ; c_t], allowing the graph readout to inform prediction as well as gate computation.
- The checkpoint includes an architecture marker so this model cannot silently load a checkpoint from a different recurrent architecture.
- test_model.py covers gate conditioning, causal read-before-write behavior, graph-context gradients, online training, and checkpoint restoration.

The canonical agi.py and existing persistent_dansn_rnn.py are not modified by this experiment.

## Run

From the repository root, with the recurrent dependencies installed:

    python -m pytest experiments/dansn_conditioned_gru_gates/test_model.py

Minimal example:

    import torch
    from agi import Brain
    from experiments.dansn_conditioned_gru_gates import (
        DANSNConditionedGRU,
        train_stream,
    )

    brain = Brain()
    token_ids = brain.vocab.tokenize(
        "red fox jumps over blue lake red fox jumps over blue lake"
    )

    model = DANSNConditionedGRU(
        brain.engine,
        vocab_size=len(brain.vocab.word_to_id) + 1,
        hidden_size=48,
        window_size=8,
        graph_depth=3,
        message_rounds=1,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    metrics = train_stream(model, token_ids * 4, optimizer, chunk_size=8)
    print(metrics)

    brain.save("vocab.bin", "graph.bin")
    model.save_checkpoint("dansn_conditioned_gru.pt")

Restore the matching Brain graph/vocabulary first, construct the same architecture configuration, and then call model.load_checkpoint(...).

## Required comparisons

Compare under identical chronological data, splits, seeds, sequence budgets, hidden widths, and ideally matched parameter counts:

1. Plain GRU.
2. Existing GRU + DANSN readout fusion (PersistentDANSNRNN).
3. DANSN-conditioned GRU gates (this experiment).
4. Optionally, separate graph reads for reset and update gates.

Measure static held-out next-token NLL/perplexity separately from score-before-update prequential NLL/perplexity; delayed/interleaved multi-hop recall; held-out entity/relationship generalization; distractor and retrieval-noise sensitivity; forgetting/interference; memory growth; and latency. Ablate graph context from each gate separately.

## Risks and prior-art boundary

- Incorrect graph retrieval can corrupt gate decisions. Add and evaluate confidence-aware fallback/residual behavior before relying on it.
- The active graph selection is discrete; gradients flow through attention/readout and gate projections over selected candidates, not through discrete node selection itself.
- Preserve causal ordering: no target-bearing graph write may occur before that target's prediction loss is computed.
- Keep persistent DANSN write gates conceptually separate from GRU reset/update gates: one controls durable graph memory writes, the others control temporary recurrent state.
- Graph-informed GRU gates have precedents in gated graph neural networks. Any contribution must come from DANSN-specific context retrieval, persistent context-specific memory, online structural updates, or demonstrated empirical benefit—not from the broad combination alone.

Treat this as an untested hypothesis until controlled results support it. Do not promote it into the existing persistent-memory core or change the canonical DANSN engine based on unit tests alone.


## Matched three-way benchmark

Run a same-stream comparison of plain GRU, the existing readout-fusion DANSN-GRU, and this gate-conditioned DANSN-GRU:

    python -m pytest -q experiments/dansn_conditioned_gru_gates/

    python -m experiments.dansn_conditioned_gru_gates.benchmark \
      --epochs 3 --seeds 7,13,21 --hidden-size 48 \
      --chunk-size 8 --window-size 8 \
      --output experiments/dansn_conditioned_gru_gates/benchmark_results.json

The harness trains each system on the same chronological synthetic corpus and evaluates a common held-out stream. It reports frozen/static test metrics separately from score-before-update adaptive metrics, plus parameter counts and graph-memory size. The three systems use the same hidden width, but their total parameter counts are not identical; the harness reports this rather than implying a parameter-matched comparison.

Start with the one-epoch benchmark smoke test in CI. Treat the multi-seed run as an exploratory pilot. The corpus is deliberately small and synthetic; follow up with longer and harder delayed/interleaved relational-recall tests before drawing conclusions.


## Gate-location ablations

The gate_ablation option now supports three configurations:

- reset_only: graph context contributes to the reset-gate preactivation only. It can influence the candidate indirectly through the reset-filtered previous hidden state.
- update_only: graph context contributes to the update-gate preactivation only. The reset gate and candidate have no direct graph-context input.
- full: graph context contributes to reset gate, update gate, and candidate state.

Example:

    model = DANSNConditionedGRU(
        brain.engine,
        vocab_size=len(brain.vocab.word_to_id) + 1,
        hidden_size=48,
        gate_ablation="reset_only",
    )

The three variants instantiate identical parameter shapes. The graph-context projection has separate reset, update, and candidate output slices; inactive slices are exactly zero-masked and receive no gradient. This preserves equal total parameter counts across the three gate ablations.

## Run the ablation suite

    python -m pytest -q experiments/dansn_conditioned_gru_gates/

    python -m experiments.dansn_conditioned_gru_gates.ablation \
      --epochs 3 --seeds 7,13,21 --hidden-size 48 \
      --chunk-size 8 --window-size 8 \
      --output experiments/dansn_conditioned_gru_gates/ablation_results.json

The ablation harness compares:
1. Plain GRU.
2. Existing readout-fusion DANSN-GRU.
3. DANSN-conditioned reset gate only.
4. DANSN-conditioned update gate only.
5. Full DANSN-conditioned reset + update + candidate paths.

All use the same synthetic chronological stream and training/evaluation protocol. Each gate-conditioned variant starts from identical seed-specific parameter initialization. Static test perplexity is computed on a copy with graph memory frozen; adaptive perplexity scores each target before online neural/graph updates.

For each gate ablation, gate_diagnostics reports reset/update activation mean, standard deviation, fractions below 0.05 and above 0.95, and mean graph-readout norm. These diagnostics are collected in a separate frozen pass on a model copy so they do not alter adaptive evaluation state.

This is a small synthetic ablation pilot. The gate-conditioned variants are parameter-matched with each other, but the plain GRU and readout-fusion baseline have different parameter counts. Interpret static and adaptive metrics separately.


## One-shot associative recall mechanism test

The dedicated synthetic benchmark `associative_recall.py` checks whether the coupled recurrent/graph system can store a newly revealed key/value sequence and recall it after distractors. Training uses stable associations for one key set; held-out keys appear during training only in unrelated noise contexts. During evaluation, new held-out key/value pairs are revealed once, followed by a distractor delay and shuffled recall queries. The recurrent state resets at test start while the trained DANSN graph and persistent memory are retained. Each query is scored before its target is revealed.

Run:

    python -m pytest -q experiments/dansn_conditioned_gru_gates/
    python -m experiments.dansn_conditioned_gru_gates.associative_recall --seeds 7,13,21 --epochs 2

Compared systems: plain GRU, readout-fusion DANSN-RNN, three-path gate-conditioned GRU, full four-gate DG-GRU, and the three four-gate ablations. Report both neural query accuracy/NLL and graph path/vector-memory coverage. High graph coverage with low neural accuracy means storage/retrieval succeeds but the learned readout does not yet decode it. This is a synthetic path-association mechanism test, not a real-language or abstract semantic-reasoning benchmark.


### Retrieval matcher correction

The first delayed-recall run exposed a terminal-suffix shadowing bug in the shared experimental graph reader. The matcher could prefer a longer, newly created suffix with no continuation over a shorter prefix that contains the learned association. The reader now prefers the longest suffix with outgoing graph continuations; only when none exists does it fall back to a memory-only suffix, then any existing suffix. A regression test reproduces the actual causal sequence. The post-fix synthetic run is the one to use when assessing recall behavior.


### First corrected synthetic result — 2026-11-27

GitHub Actions run [45890601026](https://github.com/Laeanhard03/RNNGRAPH_ISALLYOUNEED/actions/runs/45890601026) passed **29 tests** and completed the three-seed delayed-recall benchmark. After the continuation-first suffix matcher fix, graph active-path coverage increased from 0% to **100%** and persistent-vector coverage remained **100%** for all writing-enabled systems. The no-persistent-write ablation had 0% graph/vector coverage as expected.

However, query top-1 accuracy remained **0% for every system**, including the full four-gate model. Mean query NLL: plain GRU 4.233; readout-fusion 4.547; three-path gate GRU 4.207; full four-gate 5.111; no-memory-injection 5.222; no-persistent-write 4.249; no-graph-evidence 5.118. This means graph-path storage and retrieval are now structurally confirmed, but the network's neural output head does not yet decode the retrieved association into the correct value. Do not claim the full architecture works end-to-end or outperforms the simpler baselines.

The next diagnostic run strengthens the task so all 16 possible output-value classes occur in supervised training (16 training keys, 12 held-out keys), then reports value-restricted top-1, probability mass assigned to value tokens, target rank among values, and a second query score with the temporary RNN hidden state reset at each query while DANSN memory is retained. These isolate output decoding from general vocabulary competition and recurrent carry-over.


### Four-gate state coupling

The four-gate cell now has a direct DANSN-to-hidden-state path. The reset gate controls both candidate conditioning and how much prior hidden state survives. The memory gate independently blends an attended persistent DANSN vector with the recurrent proposal, rather than placing memory only inside the candidate activation (where update-gate multiplication weakened it). Direct memory retrieval prioritizes stored vectors on outgoing continuations of the matched context, separate from the broad transient graph readout. The current checkpoint marker is `dansn_four_gate_gru_v4`.


The memory-injection gate now starts with bias 0.0 (initial sigmoid near 0.5) rather than -2.0, because the first direct-fusion run learned an effective injection mean of only 0.084. Persistent-write bias remains -2.0 so storage remains conservative. A test guards this initialization. The change is under evaluation; no gain is assumed.


### Joint recurrent–associative output

The v3 decoder receives three channels: recurrent hidden state, broad graph-neighborhood readout, and the same memory-gated persistent-vector readout that participates in hidden-state fusion. The memory gate therefore influences both the next recurrent state and the output decoder. The no-memory-injection ablation zeros both routes. Checkpoint marker: `dansn_four_gate_gru_v4`.


### V4 associative next-token pointer

The four-gate output now also receives a DANSN next-token distribution over immediate outgoing association edges that carry persistent vectors. Those edge-attention weights are accumulated by child token ID into a vocabulary distribution, then mixed with the neural GRU softmax using the mean effective memory gate. The pointer only exists when a persistent next-edge was stored before the current query target is scored. The same memory gate therefore controls memory fusion into hidden state, the decoder vector skip, and the pointer probability mixture. Checkpoint marker: `dansn_four_gate_gru_v4`.


### V4 causal associative recall result

The V4 pointer benchmark completed: [GitHub Actions run 45896936662](https://github.com/Laeanhard03/RNNGRAPH_ISALLYOUNEED/actions/runs/45896936662). The test suite passed **41 tests**. On the balanced three-seed delayed key/value task, full four-gate v4 achieved **100% exact and value-only top-1**, NLL 0.385, mean pointer mix 0.679, and true-target pointer probability 1.0. Query recall also stayed 100% when the recurrent hidden state was reset for each query.

The no-memory-injection ablation fell to 2.78% exact top-1 / 5.56% value-only; no-persistent-write fell to 2.78% exact / 11.1% value-only. This shows persistent writes and the gated memory routes are necessary on this task. The no-graph-evidence ablation also reached 100%, so the explicit graph-evidence features are not needed for this simple setting. The readout-fusion baseline achieved 27.8% value-only top-1.

Important scope: each query had one persistent outgoing answer candidate. This validates one-shot delayed key/value path recall with distractors and hidden-state resets; it does not yet validate ambiguous branching, multi-hop recall, natural language, or semantic idea formation. The v4 model is successful on this controlled mechanism test, but broader capability remains unproven.
