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
