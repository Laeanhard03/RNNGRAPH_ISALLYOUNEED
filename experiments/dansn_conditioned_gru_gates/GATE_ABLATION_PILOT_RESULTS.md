# DANSN GRU Gate-Location Ablation Results

**Status:** completed exploratory three-seed synthetic pilot.  
**Branch:** feature/dansn-conditioned-gru-gates  
**GitHub Actions:** [run 45875149046](https://github.com/Laeanhard03/RNNGRAPH_ISALLYOUNEED/actions/runs/45875149046)  
**Evaluated source commit:** 533eba009558cea9a86a2df6dddf5ae09df4498b

## Protocol

- Seeds: 7, 13, 21.
- Three epochs, hidden width 48, truncated-BPTT chunk size 8, DANSN context window 8.
- Same chronological synthetic training/test stream for all systems.
- Static test: score held-out targets with train-end parameters and graph memory frozen.
- Adaptive test: score each target before the model performs its neural and graph-memory updates.
- Reset-only, update-only, and full gate models have identical parameter counts and start from the same seed-specific initialization. Inactive graph-context projection slices are zero-masked.
- Plain GRU and readout-fusion DANSN-GRU are comparison baselines, not parameter matched to the gate variants.

## Aggregate perplexity

Lower is better. Mean ± population standard deviation over three seeds.

| System | Static test PPL | Adaptive test PPL | Parameters |
|---|---:|---:|---:|
| Plain GRU | 10.4873 ± 0.4389 | 10.1222 ± 0.4333 | 16,052 |
| Readout-fusion DANSN-GRU | 9.1901 ± 0.5603 | 5.4898 ± 0.3573 | 86,660 |
| DANSN reset gate only | 8.5121 ± 0.4251 | 5.4839 ± 0.3405 | 79,556 |
| DANSN update gate only | 8.3381 ± 0.4352 | 5.3885 ± 0.3549 | 79,556 |
| DANSN reset + update + candidate | **8.2640 ± 0.4915** | **5.3130 ± 0.3911** | 79,556 |

Compared with the readout-fusion baseline in this run:
- Reset-only: 7.38% lower static PPL, 0.11% lower adaptive PPL.
- Update-only: 9.27% lower static PPL, 1.84% lower adaptive PPL.
- Full: 10.08% lower static PPL, 3.22% lower adaptive PPL.

The gate-conditioned variants use 8.20% fewer trainable parameters than the readout-fusion baseline. The three gate variants use exactly the same parameter count.

## Per-seed results

| Seed | System | Static PPL | Adaptive PPL |
|---:|---|---:|---:|
| 7 | Plain GRU | 9.9866 | 9.6202 |
| 7 | Readout fusion | 9.1834 | 5.3298 |
| 7 | Reset-only | 7.9498 | 5.0800 |
| 7 | Update-only | 7.7642 | 4.9770 |
| 7 | Full | **7.7053** | **4.9350** |
| 13 | Plain GRU | 11.0554 | 10.6774 |
| 13 | Readout fusion | 9.8797 | 5.9849 |
| 13 | Reset-only | 8.9775 | 5.9128 |
| 13 | Update-only | **8.8175** | **5.8432** |
| 13 | Full | 8.9015 | 5.8516 |
| 21 | Plain GRU | 10.4198 | 10.0691 |
| 21 | Readout fusion | 8.5073 | 5.1547 |
| 21 | Reset-only | 8.6088 | 5.4588 |
| 21 | Update-only | 8.4327 | 5.3453 |
| 21 | Full | **8.1852** | **5.1524** |

## Gate activation diagnostics

These diagnostics are measured during a separate frozen-memory pass on the held-out stream. The values below are the mean across seeds of each seed's gate activation statistics.

| Ablation | Reset mean | Update mean | Reset std | Update std |
|---|---:|---:|---:|---:|
| Reset-only | 0.5437 | 0.5509 | 0.1809 | 0.1788 |
| Update-only | 0.5276 | 0.5698 | 0.1694 | 0.1945 |
| Full | 0.5375 | 0.5864 | 0.1844 | 0.1926 |

Gate saturation was rare in all variants:
- Reset gate: below 0.05 was 0% in every variant; above 0.95 was about 0.024% for reset-only and full, 0% for update-only.
- Update gate: below 0.05 ranged from 0% to about 0.048%; above 0.95 ranged from 0% to about 0.096%.

Mean graph-readout norm was 2.526 (reset-only), 2.549 (update-only), and 2.471 (full). All hybrid variants ended with 575 graph nodes and 108 persistent memory paths per seed.

## Interpretation

1. **Update-only appears to carry most of the benefit in this pilot.** It improves substantially over the plain GRU and is close to full conditioning.
2. **Full conditioning has the best aggregate perplexity**, but the difference from update-only is small: about 0.9% lower static PPL and 1.4% lower adaptive PPL. That gap is smaller than the across-seed standard deviations, and three seeds are insufficient for a strong conclusion.
3. **Reset-only is the weakest of the three gate placements for adaptive perplexity.** Its adaptive mean is nearly identical to the readout-fusion baseline in this run.
4. None of the gates saturates frequently. This gives no evidence that gate saturation is the cause of the remaining performance differences.
5. The gate-conditioned variants have 8.2% fewer parameters than readout fusion. Because parameter counts differ between those baselines, their comparison is not capacity matched.
6. This is a small, repetitive synthetic corpus. It does not establish robust language modeling, long-range relational reasoning, or general superiority. The previous three-way pilot had different initialization/protocol details and gave a different ordering; do not pool the two pilot runs as if they were one experiment.

## Research decision and next steps

Keep the architecture experimental. The update-only variant deserves focused follow-up because it gives most of the observed gain with a simpler information path. Do not yet remove reset/candidate conditioning from the full model.

Next:
1. Add an update-only vs full comparison with more seeds and a longer, less repetitive corpus.
2. Add separate reset+update-without-candidate and candidate-only ablations. The current full-vs-update-only gap combines reset and candidate effects, so it cannot identify which of those two pathways contributes.
3. Test graph-readout confidence scaling or a learned fallback when retrieval is weak/noisy.
4. Repeat on delayed/interleaved multi-hop relational recall, measuring exact answer accuracy, retrieval reachability, forgetting/interference, latency, and memory use.
5. Only promote a gate layout if it replicates on harder tasks and fair parameter/compute budgets.

Canonical agi.py and existing persistent_dansn_rnn.py remain unchanged. This report records evidence for choosing the next experiment, not a production architecture decision.
