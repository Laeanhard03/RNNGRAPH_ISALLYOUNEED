# Memory-Read Supervision Pilot Results

**Status:** CI passed; exploratory signal only, not validated relational recall.

## Experiment

The experiment trains an auxiliary answer head to map the DANSN graph-read vector to the correct place-token embedding. Its objective is:

```text
L = L_next_token + 0.5 * L_memory_answer
```

The answer head compares normalized projected readout vectors against the 24 possible place-token embeddings using temperature 0.1. Query labels contribute gradients after the query forward pass. They are never passed to `observe_target`, so they cannot create answer shortcut edges or write answer-specific persistent vectors.

The benchmark compares:
- Read-only DANSN-GRU.
- Persistent DANSN-RNN trained with next-token loss only.
- Persistent DANSN-RNN trained with next-token loss plus memory-read supervision.
- The auxiliary read head with persistent vectors available and a matched-weight topology-only ablation with the same learned neural parameters but no persistent vectors.

At test time, facts are ingested online with neural weights frozen. The query answer is scored without graph or memory writes.

## Configuration

- Seeds: 7, 13, 21.
- 24 people, vehicles and places in each generated knowledge base.
- 18 training-supervised query subjects and 6 held-out subjects per seed.
- 18 held-out query examples across the three seeds.
- Train and test relationship mappings independently randomized.
- 24 unrelated two-hop distractor fact pairs; delays 0, 8 and 24.
- 3 training epochs, hidden size 32, context window 9, graph depth 8, chunk size 16.
- Auxiliary loss weight 0.5; contrastive temperature 0.1.
- Fixed shared vocabulary; the two relation edges are contiguous in the serialized fact stream and distractors follow the chain.

Metrics below are mean ± population standard deviation across the three seeds. Exact percentages correspond to six queries per seed.

## Memory-read answer head

There are 24 candidate places. A uniform answer distribution has NLL `ln(24) = 3.178`, top-1 accuracy 4.17% in expectation, and tie-aware MRR 0.08.

| Distractor delay | Recurrent state | Persistent vectors | Accuracy | Mean NLL | Tie-aware MRR |
| ---: | --- | --- | ---: | ---: | ---: |
| 0 | Reset | Yes | 11.11% ± 7.86 pp | 3.904 ± 0.346 | 0.195 ± 0.091 |
| 0 | Preserved | Yes | 5.56% ± 7.86 pp | 4.155 ± 0.236 | 0.147 ± 0.074 |
| 8 | Preserved | Yes | 5.56% ± 7.86 pp | 3.703 ± 0.469 | 0.199 ± 0.099 |
| 24 | Preserved | Yes | 5.56% ± 7.86 pp | 3.694 ± 0.472 | 0.188 ± 0.115 |
| 0, 8, 24 | Reset | No (matched read-only ablation) | 0% | 3.178 ± 0.000 | 0.080 ± 0.000 |
| 0, 8, 24 | Preserved | No (matched read-only ablation) | 0% | 3.178 ± 0.000 | 0.080 ± 0.000 |

Across all delays, the persistent read head made 2 correct predictions out of 18 with recurrent state reset. With recurrent state preserved, it made 1 correct prediction out of 18 at each delay. The topology-only read-head ablation remained uniform.

**Important calibration result:** although the persistent read head's MRR exceeds the uniform ablation's MRR, its mean NLL (3.694–4.155) is worse than uniform NLL (3.178). The head sometimes ranks the answer higher, but when wrong it is too confident. This does not establish reliable retrieval.

## Main next-token output head

| Condition | Accuracy | Mean NLL | MRR |
| --- | ---: | ---: | ---: |
| Persistent next-token-only, reset | 0% | 4.177 ± 0.252 | 0.085 ± 0.008 |
| Persistent memory-supervised, reset | 5.56% ± 7.86 pp | 4.454 ± 0.077 | 0.153 ± 0.072 |
| Persistent next-token-only, preserved (delay 8) | 0% | 5.185 ± 0.377 | 0.031 ± 0.017 |
| Persistent memory-supervised, preserved (delay 8) | 0% | 4.948 ± 0.343 | 0.038 ± 0.020 |
| Persistent next-token-only, preserved (delay 24) | 0% | 5.433 ± 0.283 | 0.027 ± 0.010 |
| Persistent memory-supervised, preserved (delay 24) | 0% | 4.964 ± 0.335 | 0.039 ± 0.022 |

The auxiliary objective modestly improved MRR and NLL for the integrated next-token logits in the continuous-state condition at delays 8 and 24, but top-1 accuracy remained zero. In the reset condition, mean NLL was worse than the next-token-only persistent baseline. The added objective therefore did not resolve the main decoder's answer-selection problem.

## Graph and memory growth

| Distractor delay | Graph nodes (mean) | Persistent memory paths (mean) |
| ---: | ---: | ---: |
| 0 | 1,552 | 191 |
| 8 | 2,090 | 255 |
| 24 | 3,162 | 383 |

The topology and persistent path store grow as additional distractor facts are observed. This is a capacity/efficiency consideration for later work; the present experiment does not show that all stored paths are useful.

## Verification

GitHub Actions run [45865069032](https://github.com/Laeanhard03/RNNGRAPH_ISALLYOUNEED/actions/runs/45865069032) completed successfully on commit `cd3142494aec4f65e6a695beb875a6b198b3fa73`. It passed:
- Python compilation.
- Canonical DANSN tests (14/14).
- Earlier recurrent adapter tests (4/4).
- Persistent-memory adapter tests (8/8).
- Three-seed continual-learning benchmark smoke test.
- Delayed relational recall tests (6/6) and benchmark.
- Memory-read supervised learning tests (5/5), including average-rank handling for tied scores.
- Three-seed memory-read supervision benchmark and artifact upload.

The MRR implementation was corrected before this final run: tied candidate scores now use average rank. The earlier run `45864522825` had an MRR bug for all-tied read-only predictions and must not be used for MRR conclusions.

## Interpretation and next step

This experiment provides a weak signal that persistent graph-read vectors contain information useful for ranking held-out answers. It does not yet provide reliable answer selection or calibrated confidence: only 2/18 reset-state queries and 1/18 continuous-state queries were correct, and the answer head's NLL was worse than uniform.

Do **not** promote this auxiliary head into the core architecture yet. The next research step should improve the memory representation and decoder together, then test on larger independent knowledge bases with many more held-out queries. Add adversarial/interleaved relational edges so the model must connect facts that are not adjacent in the stream, test distractor placement before and between the two edges, and inspect attention on the precise target-bearing paths. Keep the canonical `agi.py` and persistent-memory core unchanged until a repeatable benefit is established.
