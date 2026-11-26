"""Offline tests for corpus tokenization and vocabulary boundaries."""
import copy

from agi import Brain

from experiments.dansn_conditioned_gru_gates.corpus import (
    encode_documents,
    freeze_vocabulary,
)


def test_train_vocabulary_is_frozen_and_unknowns_map_to_unk():
    brain = Brain()
    train_ids = encode_documents(
        ["red fox runs", "the fox sleeps"], brain.vocab, max_tokens=20
    )
    assert len(train_ids) > 2

    unknown_id = freeze_vocabulary(brain.vocab)
    vocabulary_size = len(brain.vocab.word_to_id)

    validation_ids = encode_documents(
        ["purple fox runs beside a comet"], brain.vocab, max_tokens=20
    )

    assert len(brain.vocab.word_to_id) == vocabulary_size
    assert validation_ids[0] == unknown_id
    assert validation_ids[-1] == unknown_id
    assert brain.vocab.id_to_word[unknown_id] == "<unk>"


def test_encode_documents_obeys_exact_token_limit():
    brain = Brain()
    ids = encode_documents(
        ["red fox runs", "a blue bird flies", "another document"],
        brain.vocab,
        max_tokens=5,
    )
    assert len(ids) == 5


def test_empty_documents_are_ignored():
    brain = Brain()
    ids = encode_documents(["", "   ", "red fox"], brain.vocab)
    assert len(ids) == 2


def test_frozen_vocab_survives_deepcopy():
    brain = Brain()
    encode_documents(["red fox"], brain.vocab)
    unknown_id = freeze_vocabulary(brain.vocab)
    cloned = copy.deepcopy(brain.vocab)

    ids = encode_documents(["unknown fox"], cloned)
    assert ids[0] == unknown_id
    assert len(cloned.word_to_id) == len(brain.vocab.word_to_id)
