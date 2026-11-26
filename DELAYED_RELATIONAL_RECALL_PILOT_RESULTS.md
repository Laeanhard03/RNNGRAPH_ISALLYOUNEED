# Delayed Two-Hop Relational Recall — Pilot Results

**Status:** Preliminary synthetic pilot. The task and CI pass, but answer accuracy remains low; this is not evidence of reliable relational reasoning.

## Run and configuration

- GitHub Actions run: https://github.com/Laeanhard03/RNNGRAPH_ISALLYOUNEED/actions/runs/45862480231
- Tested head commit: `183ed9b93a4c10975252a44441109ea26c0421d5`
- Seeds: 7, 13, 21
- Subjects: 24 per generated knowledge base
- Query supervision: 18 subjects per seed
- Held-out query subjects: 6 per seed (18 held-out queries across all seeds)
- Distractor facts: 24 unrelated two-hop fact pairs
- Delays: 0, 8, and 24 distractor fact pairs after the primary fact chains
- Training: 3 epochs, hidden size 32, BPTT chunk 16
- DANSN context window: 9 tokens; graph depth: 8
- Train/test knowledge bases use independently randomized person-to-vehicle and vehicle-to-place mappings.
- Test fact ingestion is online and causal; neural weights are frozen during ingestion. Queries are scored without writing their answer back to graph memory.
- Query evaluation runs with recurrent state reset and with the recurrent state preserved from the absorbed context stream.

## Task boundary

Primary facts are serialized as a contiguous two-hop chain:

```text
person owns vehicle . vehicle parked place .
... unrelated distractor facts ...
where person -> predict place
```

This tests delayed lookup after distractors that follow the complete fact chain. It does **not** yet test arbitrary interleaving of distractors between the two edges, or general abstract concept linking/merging.

## Results — recurrent state reset

This is the key condition for testing whether external DANSN memory can supply information without relying on the active RNN state. The same aggregate appeared at delays 0, 8 and 24 because the queried subject's own trie branch remained directly addressable.

| System | Top-1 accuracy | Mean NLL | Mean reciprocal rank |
|---|---:|---:|---:|
| Plain GRU | 0.0% ± 0.0 | 4.970 ± 0.085 | 0.034 ± 0.011 |
| Read-only DANSN-GRU | 0.0% ± 0.0 | 4.183 ± 0.296 | 0.127 ± 0.033 |
| Persistent DANSN-RNN | **5.56% ± 7.86** | **4.164 ± 0.099** | **0.153 ± 0.093** |

The persistent hybrid reached the correct target within the retrieved neighborhood for all queries, and a persistent-memory path was visible in the retrieval set for all queries. The read-only graph also reached the target in all queries, but had no persistent vectors.

The persistent hybrid's top-1 result corresponds to only one correct answer across 18 held-out queries. Its NLL/MRR improvements over the read-only hybrid are small and uncertain.

## Results — recurrent state preserved

The continuous condition keeps the RNN state accumulated while reading test facts and distractors.

| Distractor delay | System | Top-1 accuracy | Mean NLL | MRR |
|---:|---|---:|---:|---:|
| 0 | Plain GRU | 0.0% | 5.393 ± 0.068 | 0.034 ± 0.003 |
| 0 | Read-only DANSN-GRU | 0.0% | 4.854 ± 0.451 | 0.069 ± 0.034 |
| 0 | Persistent DANSN-RNN | 0.0% | 5.081 ± 0.243 | 0.051 ± 0.015 |
| 8 | Plain GRU | 0.0% | 5.416 ± 0.107 | 0.032 ± 0.004 |
| 8 | Read-only DANSN-GRU | 0.0% | 4.837 ± 0.357 | 0.085 ± 0.047 |
| 8 | Persistent DANSN-RNN | 0.0% | 5.267 ± 0.140 | 0.041 ± 0.015 |
| 24 | Plain GRU | 0.0% | 5.385 ± 0.099 | 0.033 ± 0.003 |
| 24 | Read-only DANSN-GRU | 5.56% ± 7.86 | 4.914 ± 0.392 | 0.101 ± 0.073 |
| 24 | Persistent DANSN-RNN | 0.0% | 5.260 ± 0.133 | 0.043 ± 0.018 |

In this condition, the writable persistent hybrid underperformed the read-only hybrid. That suggests the current persistent write/read path may be interfering with the RNN's continuous working state, even though the retrieved neighborhood contains the stored target path.

## Memory and graph growth

| Distractor delay | Mean graph nodes | Mean persistent memory paths |
|---:|---:|---:|
| 0 | 1,552 | 191 |
| 8 | 2,090 | 255 |
| 24 | 3,162 | 383 |

Memory visibility and graph reachability are not the same as correct answer selection. The model can retrieve the right neighborhood yet still fail to rank the corresponding place token first.

## Interpretation

The experiment demonstrates:
- The new test knowledge base can be absorbed into the DANSN graph without gradient updates to neural weights.
- Stable subject-rooted paths can still reach the relevant two-hop answer after unrelated facts have been added.
- The persistent hybrid exposes memory vectors to the query neighborhood in every measured query.
- Reset-state evaluation shows a small positive signal over the read-only hybrid in NLL, MRR and one top-1 answer.

It does **not** demonstrate reliable recall:
- Persistent top-1 accuracy is only 1/18 across seeds in reset mode.
- With recurrent state preserved, persistent DANSN-RNN is worse than the read-only hybrid on NLL and MRR at all three delays.
- Three seeds on one generated task are not a robust dataset-level evaluation.
- The task serializes each two-hop chain contiguously; it does not yet require joining separately occurring facts with arbitrary intervening text.

## Engineering conclusion and next step

The bottleneck is now more specific than graph reachability: **the retrieved persistent state is present, but the learned readout/output path does not reliably convert it into the correct answer token.**

The next experiment should add an explicit, causally valid memory-read objective:
1. Train on a larger set of independent two-hop chains and query subjects.
2. Keep the answer out of the prompt and do not write the answer during query supervision.
3. Add a contrastive/auxiliary loss that aligns a retrieved context vector with the correct answer embedding after the answer is scored.
4. Compare memory-read supervision against next-token-only training and the read-only hybrid.
5. Add distractors between the two facts, not just after the combined chain.
6. Report query accuracy, NLL, MRR, target reachability, memory visibility, and state-reset versus continuous-state results separately.

Keep `agi.py` unchanged until the persistent-memory mechanism demonstrates repeatable gains on this harder task.


## Interleaved two-hop recall and exact-entity bridge pilot — 2026-11-27

The next harder test has now been run on three seeds. Rather than keeping the two relation edges adjacent, the test stream is:

```text
person owns vehicle
... 8 unrelated two-hop fact pairs ...
vehicle parked place
... optional trailing distractors ...
where person -> predict place
```

### Configuration and causal safeguards

- Seeds 7, 13, 21.
- 24 people, 24 vehicles and 24 places; 18 query-supervised subjects and 6 held-out subjects per seed (18 held-out queries per delay across seeds).
- 8 unrelated distractor fact pairs inserted between the two relation groups; additional 16 distractor pairs are available as trailing delay material.
- Delayed conditions: 0 and 8 trailing distractor pairs.
- Three epochs; hidden size 32; window size 9; graph depth 8; chunk size 16.
- Train/test relationship mappings are independently randomized.
- Test facts are ingested causally with neural weights frozen. Link construction receives only observed fact tokens, never query/answer labels. Query targets are not written into the graph.
- Comparison: plain GRU, read-only DANSN-GRU, persistent DANSN-RNN with an ordinary unlinked trie, and a matched persistent model with an exact-token entity bridge.

### What the entity bridge does

The original trie stores context-specific paths. The first-hop context `person owns vehicle` and second-hop context `vehicle parked place` are therefore separate branches when unrelated facts are interleaved between them. The experiment adds one explicit cross-context edge when the exact same vehicle token ID occurs in both observed facts, then aliases the already-written target-bearing memory vector onto the joined path. This is a task-specific exact-identity rule, not learned semantic entity resolution.

Across all three seeds, the linker observed 24 first-hop and 24 second-hop facts per run, added 24 cross-links and 24 memory aliases, and left zero entities unlinked.

### Aggregate results (mean ± population standard deviation)

At delay 0 (no additional trailing distractors):

| System | Recurrent state | Accuracy | Mean NLL | MRR | Target reachable | Target-bearing memory visible |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Plain GRU | Reset | 5.56% ± 7.86 pp | 4.799 ± 0.055 | 0.127 ± 0.062 | — | — |
| Read-only DANSN-GRU | Reset | 5.56% ± 7.86 pp | 4.363 ± 0.194 | 0.149 ± 0.104 | 0% | — |
| Persistent DANSN-RNN, unlinked | Reset | 0% | 4.181 ± 0.270 | 0.100 ± 0.035 | 0% | 0% |
| Persistent DANSN-RNN, entity-linked | Reset | 0% | 4.133 ± 0.305 | 0.097 ± 0.032 | 100% | 100% |
| Persistent DANSN-RNN, unlinked | Preserved | 0% | 4.993 ± 0.285 | 0.056 ± 0.012 | 0% | 0% |
| Persistent DANSN-RNN, entity-linked | Preserved | 0% | 4.828 ± 0.130 | 0.062 ± 0.014 | 100% | 100% |

At delay 8 (eight additional trailing distractor pairs):

| System | Recurrent state | Accuracy | Mean NLL | MRR | Target reachable | Target-bearing memory visible |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Persistent DANSN-RNN, unlinked | Reset | 0% | 4.181 ± 0.270 | 0.100 ± 0.035 | 0% | 0% |
| Persistent DANSN-RNN, entity-linked | Reset | 0% | 4.133 ± 0.305 | 0.097 ± 0.032 | 100% | 100% |
| Persistent DANSN-RNN, unlinked | Preserved | 0% | 4.877 ± 0.377 | 0.063 ± 0.016 | 0% | 0% |
| Persistent DANSN-RNN, entity-linked | Preserved | 0% | 4.651 ± 0.348 | 0.068 ± 0.013 | 100% | 100% |
| Read-only DANSN-GRU | Preserved | 0% | 5.242 ± 0.470 | 0.049 ± 0.022 | 0% | — |

The key finding is structural, not yet an answer-accuracy win:
- Ordinary persistent trie traversal retained unrelated active memory paths in every query, but could not reach the correct place or expose its target-bearing memory (0% for both diagnostics).
- The exact-entity bridge made the correct target reachable and its corresponding stored memory vector visible in **100%** of held-out queries at both delays.
- Despite this, the linked model's top-1 answer accuracy remained **0/18 at both delays**. NLL and MRR improved modestly over the unlinked persistent model in the continuous-state condition, but this did not translate into a correct answer.
- The reset-state MRR at delay 0 was slightly lower for the linked model than the unlinked model. Therefore, do not describe linking as a general performance improvement.

The unlinked persistent model had 2,090 graph nodes and 255 memory paths at delay 0, increasing to 2,626 nodes and 319 memory paths at delay 8. The linker added 24 cross-links and 24 memory aliases per seed; all observed entities were linked.

### Verification

GitHub Actions [45868295697](https://github.com/Laeanhard03/RNNGRAPH_ISALLYOUNEED/actions/runs/45868295697) completed successfully on commit `7525758f61462e3a5dcb019c72c337c47504ce28`. It passed compilation, canonical DANSN tests (14/14), earlier recurrent tests (4/4), persistent-memory tests (8/8), the existing continual-learning benchmark, the original delayed-recall tests and benchmark, the expanded interleaved relational tests and three-seed benchmark, memory-read supervision tests (5/5), the three-seed memory-read benchmark, and artifact uploads.

A separate diagnostic now reports `target_bearing_memory_visible_fraction`, which only counts memory paths whose final token is the correct target. This is stricter than generic active-memory visibility and avoids mistaking unrelated memory activity for retrieval of the answer-bearing state.

### Engineering conclusion and next step

This result isolates two distinct capabilities:
1. **Structural fact joining:** exact repeated-entity linking can bridge context-specific trie branches and expose the correct memory vector.
2. **Answer decoding:** the existing learned readout/output path still does not select the correct answer, even when the correct target is reachable and its memory vector is active.

The next experiment should instrument attention weights and contribution scores on the exact target-bearing path, then test a query-time candidate/pointer decoder that scores the reachable place nodes directly. The decoder must use only the observed graph and current query, not the held-out label. Compare it against the existing vocabulary output layer and keep the exact-entity link as an explicit ablation. Do not merge the bridge or a new decoder into the persistent-memory core until its gains repeat across larger independent knowledge bases.

Canonical `agi.py` (SHA `4abc472fd62e3d2807dc9709894d9cb9ca6d12dc`) and `persistent_dansn_rnn.py` (SHA `e789868ea7b406eb2213f7f7b91d90d2eddb1174`) remain unchanged.
