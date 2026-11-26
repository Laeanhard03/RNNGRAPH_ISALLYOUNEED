"""Dataset adapters for real-corpus DANSN-GRU experiments.

Raw corpora are downloaded into the user's cache by the upstream libraries;
they are never committed to this repository. Vocabulary is built on train
only and frozen before validation/test encoding.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Optional


UNKNOWN_TOKEN = "<unk>"


class FrozenWordLookup(dict):
    """Dictionary whose missing-token lookup returns one fixed UNK ID."""

    def __init__(self, values: dict[str, int], unknown_id: int) -> None:
        super().__init__(values)
        self.unknown_id = int(unknown_id)

    def get(self, key, default=None):
        return super().get(key, self.unknown_id)

    def __deepcopy__(self, memo):
        import copy
        clone = type(self)(copy.deepcopy(dict(self), memo), self.unknown_id)
        memo[id(self)] = clone
        return clone


@dataclass
class EncodedCorpus:
    brain: Any
    train_ids: list[int]
    validation_ids: list[int]
    test_ids: list[int]
    metadata: dict[str, Any]


def freeze_vocabulary(vocab) -> int:
    """Add an UNK token and prevent validation/test from growing the vocabulary."""
    if UNKNOWN_TOKEN in vocab.word_to_id:
        unknown_id = int(vocab.word_to_id[UNKNOWN_TOKEN])
    else:
        unknown_id = int(vocab.next_id)
        vocab.next_id += 1
        vocab.word_to_id[UNKNOWN_TOKEN] = unknown_id
        vocab.id_to_word[unknown_id] = UNKNOWN_TOKEN
    vocab.word_to_id = FrozenWordLookup(dict(vocab.word_to_id), unknown_id)
    return unknown_id


def encode_documents(
    documents: Iterable[str],
    vocab,
    max_tokens: Optional[int] = None,
) -> list[int]:
    """Encode documents in order, optionally stopping at an exact token budget."""
    if max_tokens is not None and max_tokens < 1:
        raise ValueError("max_tokens must be positive or None")

    token_ids: list[int] = []
    for document in documents:
        if not isinstance(document, str) or not document.strip():
            continue
        encoded = vocab.tokenize(document)
        if max_tokens is None:
            token_ids.extend(encoded)
        else:
            remaining = max_tokens - len(token_ids)
            if remaining <= 0:
                break
            token_ids.extend(encoded[:remaining])
            if len(token_ids) >= max_tokens:
                break
    return token_ids


def _iter_wikitext103(split: str, cache_dir: Optional[str] = None) -> Iterator[str]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "WikiText-103 requires the optional corpus dependencies. "
            "Install with: pip install -r requirements-corpus.txt"
        ) from exc

    kwargs: dict[str, Any] = {"split": split, "streaming": True}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        **kwargs,
    )
    for row in dataset:
        text = row.get("text", "")
        if isinstance(text, str) and text.strip():
            yield text


def _iter_pg19(split: str, data_dir: Optional[str] = None) -> Iterator[str]:
    try:
        import tensorflow_datasets as tfds
    except ImportError as exc:
        raise RuntimeError(
            "PG-19 requires TensorFlow and TensorFlow Datasets. "
            "See requirements-pg19.txt and DATASETS.md."
        ) from exc

    dataset = tfds.load(
        "pg19",
        split=split,
        shuffle_files=False,
        data_dir=data_dir,
    )
    for row in tfds.as_numpy(dataset):
        text = row.get("book_text", "")
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="replace")
        if isinstance(text, str) and text.strip():
            yield text


def load_encoded_corpus(
    dataset_name: str = "wikitext103",
    train_tokens: int = 100_000,
    validation_tokens: int = 20_000,
    test_tokens: int = 20_000,
    cache_dir: Optional[str] = None,
    tfds_data_dir: Optional[str] = None,
) -> EncodedCorpus:
    """Load train/validation/test streams with a train-only fixed vocabulary.

    Supported datasets:
      - wikitext103: streamed from Salesforce/wikitext through HF Datasets.
      - pg19: loaded through the official TensorFlow Datasets builder.

    Token limits are deliberate safety controls. PG-19's full training split is
    about 1.97B tokens and should be sampled with an explicit budget first.
    """
    if min(train_tokens, validation_tokens, test_tokens) < 2:
        raise ValueError("each split token budget must be at least 2")

    dataset_name = dataset_name.lower().replace("-", "").replace("_", "")
    if dataset_name in {"wikitext103", "wiki103"}:
        dataset_name = "wikitext103"
        train_iter = lambda: _iter_wikitext103("train", cache_dir)
        validation_iter = lambda: _iter_wikitext103("validation", cache_dir)
        test_iter = lambda: _iter_wikitext103("test", cache_dir)
        source = {
            "dataset": "WikiText-103 raw",
            "upstream": "https://huggingface.co/datasets/Salesforce/wikitext",
            "project": "https://github.com/salesforce/wikitext",
            "license_note": (
                "The dataset card identifies the text as Wikipedia-derived and "
                "available under CC BY-SA. Follow the dataset card's attribution "
                "and share-alike terms for redistributed derivatives."
            ),
            "loader": "Hugging Face Datasets streaming",
            "revision": "main",
        }
    elif dataset_name in {"pg19", "projectgutenberg19"}:
        dataset_name = "pg19"
        train_iter = lambda: _iter_pg19("train", tfds_data_dir)
        validation_iter = lambda: _iter_pg19("validation", tfds_data_dir)
        test_iter = lambda: _iter_pg19("test", tfds_data_dir)
        source = {
            "dataset": "PG-19 Language Modeling Benchmark",
            "upstream": "https://github.com/google-deepmind/pg19",
            "loader_source": "https://github.com/tensorflow/datasets",
            "license_note": (
                "The official PG-19 repository identifies the benchmark with "
                "Apache-2.0 metadata and describes its books as Project Gutenberg "
                "works published before 1919. Check individual source-book status "
                "and the upstream terms before redistribution."
            ),
            "loader": "TensorFlow Datasets pg19 builder",
            "expected_train_tokens": 1_973_136_207,
            "expected_download_size_bytes": 11_740_397_875,
        }
    else:
        raise ValueError("dataset_name must be 'wikitext103' or 'pg19'")

    from agi import Brain

    brain = Brain()
    train_ids = encode_documents(train_iter(), brain.vocab, train_tokens)
    if len(train_ids) < 2:
        raise RuntimeError("upstream training split yielded fewer than 2 tokens")

    unknown_id = freeze_vocabulary(brain.vocab)
    validation_ids = encode_documents(
        validation_iter(), brain.vocab, validation_tokens
    )
    test_ids = encode_documents(test_iter(), brain.vocab, test_tokens)
    if len(validation_ids) < 2 or len(test_ids) < 2:
        raise RuntimeError("validation/test split yielded fewer than 2 tokens")

    metadata = {
        **source,
        "requested_token_limits": {
            "train": train_tokens,
            "validation": validation_tokens,
            "test": test_tokens,
        },
        "actual_tokens": {
            "train": len(train_ids),
            "validation": len(validation_ids),
            "test": len(test_ids),
        },
        "vocabulary_size_including_unk": len(brain.vocab.word_to_id) + 1,
        "unknown_token_id": unknown_id,
        "vocabulary_policy": "built on training split only; validation/test OOV -> UNK",
        "raw_data_committed_to_repo": False,
    }
    return EncodedCorpus(
        brain=brain,
        train_ids=train_ids,
        validation_ids=validation_ids,
        test_ids=test_ids,
        metadata=metadata,
    )
