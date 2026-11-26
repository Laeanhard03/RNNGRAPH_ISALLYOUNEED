# Persistent DANSN-RNN Pilot Results

**Status:** Preliminary smoke benchmark only. Not a validated performance claim.

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
