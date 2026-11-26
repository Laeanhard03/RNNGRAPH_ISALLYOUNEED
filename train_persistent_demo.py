"""Small synthetic demonstration of persistent DANSN-RNN online learning.

This is a wiring/demo script, not a benchmark or evidence of improved perplexity.
"""
import torch

from agi import Brain
from persistent_dansn_rnn import PersistentDANSNRNN, train_stream, evaluate_prequential


TRAIN_TEXT = """
a red fox lives near a blue lake.
the red fox watches the blue lake.
a blue bird flies over the lake.
the fox sees the blue bird.
a red fox lives near a blue lake.
the blue bird flies over the lake.
"""

HELD_OUT_TEXT = """
the red fox watches a blue bird.
a blue bird lives near the lake.
"""


def main():
    torch.manual_seed(7)
    brain = Brain()

    # Establish a fixed vocabulary first. Do not absorb the corpus into the
    # graph here: graph memory must grow only after each target is predicted.
    train_ids = brain.vocab.tokenize(TRAIN_TEXT)
    held_out_ids = brain.vocab.tokenize(HELD_OUT_TEXT)

    model = PersistentDANSNRNN(
        brain.engine,
        vocab_size=len(brain.vocab.word_to_id) + 1,
        hidden_size=48,
        window_size=8,
        graph_depth=3,
        message_rounds=1,
        max_active_nodes=64,
        max_memory_nodes=5_000,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)

    for epoch in range(1, 4):
        metrics = train_stream(model, train_ids, optimizer, chunk_size=8)
        print(
            f"epoch={epoch} prequential_train_ppl={metrics['perplexity']:.3f} "
            f"graph={model.graph_stats()}"
        )

    held_out = evaluate_prequential(model, held_out_ids, update_graph=True)
    print(
        f"held_out_prequential_ppl={held_out['perplexity']:.3f} "
        f"tokens={int(held_out['tokens'])}"
    )

    # Keep the original graph engine/vocabulary and neural memory checkpoint
    # separate; restore Brain first, then construct/load the same model config.
    brain.save("demo_vocab.bin", "demo_graph.bin")
    model.save_checkpoint("demo_persistent_memory.pt")


if __name__ == "__main__":
    main()
