# Real-corpus datasets for DANSN-conditioned GRU experiments

This branch adds repeatable loaders and a benchmark runner for larger public language-modeling corpora. Dataset binaries are **not stored in this Git repository**: WikiText-103 is streamed from its upstream dataset repository, and PG-19 is downloaded/cached through TensorFlow Datasets. Only code, dataset provenance, configuration, and benchmark summaries belong here.

## Dataset options

| Dataset | Scale | Why use it | Source |
|---|---:|---|---|
| WikiText-103 raw | Over 100 million tokens; upstream download is about 190 MB | General language modeling with original casing, punctuation, numbers, and full Wikipedia articles | [Salesforce/wikitext on Hugging Face](https://huggingface.co/datasets/Salesforce/wikitext), [project repository](https://github.com/salesforce/wikitext) |
| PG-19 | 1,973,136,207 training tokens; 28,602 training books; roughly 11.74 GB download | Very long document contexts and persistent-memory stress testing | [Official Google DeepMind GitHub project](https://github.com/google-deepmind/pg19), [TensorFlow Datasets implementation](https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/datasets/pg19) |

WikiText-103 is derived from Wikipedia and its dataset card identifies a Creative Commons Attribution-ShareAlike license. Follow the upstream attribution and share-alike terms. PG-19 is based on Project Gutenberg books published before 1919. The PG-19 project metadata identifies Apache-2.0; check the upstream dataset documentation and individual source-book status before redistributing any extracted text.

PG-19 is intentionally not the default download: its full split is large and has historical-language/bias limitations. Start with bounded token limits and increase them only after checking disk, RAM, and graph-memory use.

## Installation

For WikiText-103:

    pip install -r requirements-recurrent.txt
    pip install -r requirements-corpus.txt

For PG-19, use a Linux environment with enough free storage (the prepared cache can exceed the archive size):

    pip install -r requirements-pg19.txt

## Run a larger real-corpus comparison

From the repository root:

    python -m pytest -q experiments/dansn_conditioned_gru_gates/

    python -m experiments.dansn_conditioned_gru_gates.large_corpus_benchmark \
      --dataset wikitext103 \
      --train-tokens 100000 --validation-tokens 20000 --test-tokens 20000 \
      --epochs 1 --seeds 7 \
      --output benchmark-results/wikitext103_dggru.json

The runner compares:

1. Plain GRU.
2. Existing readout-fusion DANSN-GRU.
3. DANSN-conditioned update-gate-only GRU.
4. Full DANSN-conditioned reset + update + candidate GRU.

All systems receive the same chronological training stream and train-only vocabulary. Validation/test out-of-vocabulary words map to one fixed UNK token; validation/test never expand the vocabulary.

For PG-19, begin with a small subset:

    python -m experiments.dansn_conditioned_gru_gates.large_corpus_benchmark \
      --dataset pg19 --train-tokens 100000 \
      --validation-tokens 20000 --test-tokens 20000 \
      --epochs 1 --seeds 7

Use the full corpus only after the bounded run is stable. The PG-19 loader prepares the upstream TFDS dataset, so it can require substantially more storage than the selected token budget suggests.

## Evaluation protocol

- Static validation/test: no parameter updates and no DANSN graph/memory writes.
- Prequential memory-only test: score each target first, then write the observed target to DANSN; freeze neural parameters for the entire test stream.
- Training: each model learns on the same ordered train tokens. The DANSN hybrids use truncated BPTT and confidence-gated persistent memory writes.
- Report static and adaptive/prequential metrics separately. Do not compare a static metric against a prequential metric.
- Report total parameter counts. Same hidden width does not make the model sizes identical.
- Repeat with multiple seeds and both datasets before making claims about superiority.

The token budgets are explicit controls, not claims that a small sample represents the whole upstream corpus. Increase them in stages (for example, 100k, 500k, 1m, then larger) and record the exact limits in each result artifact.
