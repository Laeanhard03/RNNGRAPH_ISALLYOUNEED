# DANSN-Conditioned GRU Gate Pilot Results

**Status:** first successful three-seed synthetic pilot; exploratory only.  
**Branch:** feature/dansn-conditioned-gru-gates  
**Workflow:** [GitHub Actions run 45872590298](https://github.com/Laeanhard03/RNNGRAPH_ISALLYOUNEED/actions/runs/45872590298)  
**Evaluated commit:** 1154193b5bc7e2b0bb3ab22f5b202ced04fb5312

## Protocol

- Seeds: 7, 13, 21.
- Three training epochs per seed.
- Hidden width 48; truncated-BPTT chunk size 8; DANSN context window 8.
- All systems use the same small chronological synthetic train/test streams from the repository's existing continual-learning benchmark.
- Static evaluation scores the held-out stream without test-time parameter or graph updates.
- Adaptive evaluation scores each target first, then performs the system's online updates. Hybrid models update both neural parameters and persistent DANSN memory.
- Same hidden width and token streams are used, but total parameter counts are not matched.

## Aggregate results

Perplexity (lower is better), mean ± population standard deviation over three seeds:

| System | Static test PPL | Adaptive test PPL | Parameters |
|---|---:|---:|---:|
| Plain GRU | 10.5832 ± 0.5641 | 10.2457 ± 0.5439 | 16,052 |
| Readout-fusion DANSN-GRU | 8.4185 ± 0.3022 | **5.1861 ± 0.0637** | 86,660 |
| Gate-conditioned DANSN-GRU | 9.0151 ± 0.0997 | 5.7563 ± 0.3162 | 79,556 |

## Per-seed results

| Seed | System | Static PPL | Adaptive PPL |
|---:|---|---:|---:|
| 7 | Plain GRU | 9.8681 | 9.5517 |
| 7 | Readout-fusion DANSN-GRU | 8.7164 | 5.1021 |
| 7 | Gate-conditioned DANSN-GRU | 8.8903 | 5.6882 |
| 13 | Plain GRU | 10.6343 | 10.3052 |
| 13 | Readout-fusion DANSN-GRU | 8.5351 | 5.1997 |
| 13 | Gate-conditioned DANSN-GRU | 9.1342 | 6.1732 |
| 21 | Plain GRU | 11.2470 | 10.8801 |
| 21 | Readout-fusion DANSN-GRU | 8.0041 | 5.2564 |
| 21 | Gate-conditioned DANSN-GRU | 9.0210 | 5.4077 |

## Interpretation

1. The gate-conditioned model improves over the plain GRU on this synthetic stream: its mean static PPL is about 14.8% lower, and its mean adaptive PPL is about 43.8% lower.
2. It does **not** outperform the existing readout-fusion hybrid. Its static PPL is about 7.1% higher, and its adaptive PPL is about 11.0% higher than readout fusion.
3. The gate-conditioned model has about 8.2% fewer trainable parameters than the readout-fusion hybrid (79,556 vs. 86,660), but this is not a parameter-matched comparison.
4. Both hybrid systems ended each seed with the same graph size (575 nodes) and 108 persistent memory paths. The difference in this pilot is therefore not explained by one model simply growing a larger graph.
5. This is a small synthetic pilot with three seeds, not evidence of general language-model performance, robust long-range reasoning, or statistical superiority.

Static metrics must only be compared with static metrics; adaptive metrics must only be compared with adaptive metrics. Adaptive evaluation includes online neural and persistent-memory updates and is not zero-shot generalization.

## Recommended next experiment

Do not promote the gate-conditioned model over the existing readout-fusion baseline on this result. Add diagnostics and controlled ablations to locate the bottleneck:

- Measure reset/update gate activation distributions, saturation, and variation when graph context is present versus absent.
- Ablate graph context from the reset gate, update gate, and candidate-state path separately.
- Test confidence-aware scaling/fallback for weak or noisy graph retrieval.
- Repeat on larger delayed/interleaved relational-recall tasks and use matched parameter budgets where feasible.

Keep this model isolated on its feature branch. The canonical agi.py and existing persistent_dansn_rnn.py remain unchanged.
