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
