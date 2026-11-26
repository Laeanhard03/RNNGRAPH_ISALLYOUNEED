# Persistent DANSN-RNN Pilot Results

**Status:** Preliminary synthetic smoke benchmark only. Three random seeds were run, but the corpus remains tiny and repetitive; this is not a validated performance claim.

## Run configuration

- GitHub Actions run: https://github.com/Laeanhard03/RNNGRAPH_ISALLYOUNEED/actions/runs/45856819843
- Code commit tested: `88727c986bf37b1c81c96c53b9464d456f01a2e4`
- Seed: 7
- Training stream: 80 tokens
- Test stream: 30 tokens (29 next-token predictions)
- Training passes: 1
- Hidden size: 48
- Truncated-BPTT chunk size: 8
- DANSN context window: 8
- Test protocol: target is scored before any online adaptation.

## Static held-out test

No test-time parameter or graph updates are made.

| System | Test NLL | Perplexity |
|---|---:|---:|
| Plain GRU | 2.7515 | 15.6654 |
| DANSN-only | 6.2696 | 528.2702 |
| Read-only DANSN-GRU | 2.7186 | 15.1597 |
| Persistent DANSN-RNN | 2.6792 | 14.5740 |

DANSN-only candidate coverage was 79.31%. Targets absent from the graph's candidate set received a small probability floor for finite NLL, which strongly penalizes missing support.

## Adaptive prequential test

Each target is scored first. Then each system applies its own update:
- Plain GRU: parameter update via truncated BPTT.
- DANSN-only: graph path/frequency update.
- Read-only DANSN-GRU: RNN parameter update; graph remains frozen.
- Persistent DANSN-RNN: RNN parameter update plus DANSN graph and persistent-memory write.

| System | Adaptive NLL | Perplexity |
|---|---:|---:|
| Plain GRU | 2.7171 | 15.1367 |
| DANSN-only | 6.3392 | 566.3232 |
| Read-only DANSN-GRU | 2.6725 | 14.4761 |
| Persistent DANSN-RNN | 2.3735 | 10.7348 |

Persistent hybrid graph state after the run: 575 graph nodes and 108 persistent memory paths. DANSN-only candidate coverage remained 79.31%.

## Interpretation

This pilot is directionally encouraging:
- On static held-out scoring, the persistent hybrid was modestly better than the read-only hybrid and plain GRU in this run.
- During adaptive prequential processing, the writable hybrid reached lower perplexity than the GRU-only and frozen-graph hybrid baselines on this short stream.
- The DANSN-only predictor had limited candidate coverage and very high perplexity, showing why a neural output path may be needed to handle tokens not proposed by the graph.

However:
- Only one synthetic corpus, one seed, one training pass, and 29 test predictions were used.
- The models have not been parameter-count matched.
- Adaptive test results measure online adaptation, not zero-shot generalization.
- The tiny corpus is highly repetitive and not representative of natural language.
- This does not establish that DANSN-RNN outperforms a GRU or language model generally.

## Next required experiment

Run multiple seeds and larger chronological datasets; add delayed relational-recall tasks; match training budgets; report static and adaptive metrics separately; and inspect whether improvements persist after distractors and contradictory observations. Keep causal predict-before-write evaluation and do not use target information before scoring.


## Three-seed smoke comparison

- GitHub Actions run: https://github.com/Laeanhard03/RNNGRAPH_ISALLYOUNEED/actions/runs/45857528186
- Tested workflow commit: `826e4179986978d96fc3fd72c0f460c4bd5ae5b3`
- Seeds: 7, 13, 21
- Per seed: 80 training tokens, 30 test tokens (29 next-token predictions), one training pass, hidden size 48, chunk size 8, context window 8.
- Static scoring runs on a deep-copied model/memory state. It does not mutate the adaptive run's LRU access metadata.

### Mean and population standard deviation across seeds

| System | Static held-out PPL | Adaptive prequential PPL |
|---|---:|---:|
| Plain GRU | 16.333 ± 0.711 | 15.829 ± 0.701 |
| DANSN-only | 528.270 ± 0.000 | 566.323 ± 0.000 |
| Read-only DANSN-GRU | **14.269 ± 1.054** | 13.611 ± 0.967 |
| Persistent DANSN-RNN | 14.946 ± 0.611 | **11.084 ± 0.533** |

DANSN-only candidate coverage was 79.31% in all three runs. DANSN-only's zero standard deviation reflects that the baseline is deterministic on this fixed corpus, not broad robustness.

### Interpretation

The writable persistent hybrid had the lowest adaptive perplexity in all three seeds as reflected in the aggregate, while it did not lead the static held-out comparison. The read-only hybrid had a slightly lower static perplexity. This suggests the graph write mechanism may be helping short-horizon online adaptation on this synthetic stream, while persistent writes may introduce a static prediction trade-off. It does not prove the mechanism generalizes.

The experiment is still very small: one short repetitive corpus, only 29 test predictions per seed, one training pass, unbalanced parameter counts, and adaptive test-time learning. The three seeds measure initialization variation, not dataset variation or language generalization. The persistent model is not yet a validated continual learner.

## Next required experiment

Use several distinct chronological corpora and delayed multi-hop relational-recall tasks; increase test length substantially; match parameter and update budgets; include distractors and contradictory updates; and report static/adaptive performance separately. Preserve target-before-write evaluation.
