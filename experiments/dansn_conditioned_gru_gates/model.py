"""DANSN-conditioned GRU: graph retrieval participates inside GRU gate equations.

This is an isolated experiment. The canonical agi.py graph engine and the
existing PersistentDANSNRNN implementation remain unchanged.

The DANSN readout conditions reset, update, and candidate-state preactivations.
It does not directly replace sigmoid gates with discrete graph operations.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import torch
from torch import Tensor, nn

from persistent_dansn_rnn import (
    PersistentDANSNRNN,
    RecurrentState,
    evaluate_prequential,
    train_stream,
)


CHECKPOINT_ARCHITECTURE = "dansn_conditioned_gru_gates_v1"


class DANSNConditionedGRUCell(nn.Module):
    """GRU cell whose reset/update/candidate computations use graph context.

    The reset and update gates remain learned, differentiable sigmoid gates.
    DANSN contributes a retrieved context vector through shared projections.

    Update convention:
        z = sigmoid(W_z x + U_z h + V_z c)
        r = sigmoid(W_r x + U_r h + V_r c)
        n = tanh(W_n x + U_n (r * h) + V_n c)
        h_new = (1 - z) * h + z * n

    Here z is the fraction of the candidate state to incorporate.
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        if input_size < 1 or hidden_size < 1:
            raise ValueError("input_size and hidden_size must be positive")

        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)

        # Input and retrieved context each contribute to all three pathways.
        self.input_projection = nn.Linear(input_size, 3 * hidden_size)
        self.context_projection = nn.Linear(
            hidden_size, 3 * hidden_size, bias=False
        )

        # Previous hidden state contributes separately to reset/update gates.
        self.hidden_gate_projection = nn.Linear(
            hidden_size, 2 * hidden_size, bias=False
        )
        # The candidate uses the reset-filtered prior hidden state.
        self.hidden_candidate_projection = nn.Linear(
            hidden_size, hidden_size, bias=False
        )

    def forward(
        self, inputs: Tensor, previous_hidden: Tensor, graph_context: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if inputs.shape[-1] != self.input_size:
            raise ValueError("input feature width does not match input_size")
        if previous_hidden.shape[-1] != self.hidden_size:
            raise ValueError("hidden feature width does not match hidden_size")
        if graph_context.shape[-1] != self.hidden_size:
            raise ValueError("graph context width does not match hidden_size")

        x_z, x_r, x_n = self.input_projection(inputs).chunk(3, dim=-1)
        c_z, c_r, c_n = self.context_projection(graph_context).chunk(3, dim=-1)
        h_z, h_r = self.hidden_gate_projection(previous_hidden).chunk(2, dim=-1)

        update_gate = torch.sigmoid(x_z + h_z + c_z)
        reset_gate = torch.sigmoid(x_r + h_r + c_r)
        candidate = torch.tanh(
            x_n
            + self.hidden_candidate_projection(reset_gate * previous_hidden)
            + c_n
        )
        hidden = (
            (1.0 - update_gate) * previous_hidden
            + update_gate * candidate
        )

        gate_trace = {
            "update_gate": update_gate,
            "reset_gate": reset_gate,
            "candidate": candidate,
        }
        return hidden, gate_trace


class DANSNConditionedGRU(PersistentDANSNRNN):
    """Persistent DANSN-RNN variant with graph-conditioned GRU gates.

    Reuses the existing graph path retrieval, persistent node-state writes,
    confidence/quality metadata, capacity control, and causal observe_target()
    operation. Replaces the standard GRUCell + post-GRU fusion gate with a
    GRU cell whose actual gate preactivations receive the DANSN readout.

    The target must still be scored before observe_target() reveals it to the
    graph. Use the inherited train_stream/evaluate_prequential utilities.
    """

    def __init__(self, engine, vocab_size: int, hidden_size: int = 64,
                 window_size: int = 12, graph_depth: int = 3,
                 message_rounds: int = 2, max_active_nodes: int = 128,
                 max_memory_nodes: int = 50_000,
                 structural_bias: float = 0.15) -> None:
        super().__init__(
            engine=engine,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            window_size=window_size,
            graph_depth=graph_depth,
            message_rounds=message_rounds,
            max_active_nodes=max_active_nodes,
            max_memory_nodes=max_memory_nodes,
            structural_bias=structural_bias,
        )

        # These belong to the earlier readout-fusion controller and are not
        # used by this architecture. Keep the shared graph reader and writer.
        del self.controller
        del self.fusion_gate
        del self.readout_projection

        self.gate_cell = DANSNConditionedGRUCell(hidden_size, hidden_size)
        # The inherited output head uses [recurrent_state ; graph_readout].

    def forward_step(
        self, token_id: int, history: Sequence[int],
        state: Optional[RecurrentState] = None,
    ):
        """Process one token and predict the next; does not write graph memory."""
        state = state or self.initial_state()
        token_id = int(token_id)
        self._validate_ids([token_id, *history])
        if not history or int(history[-1]) != token_id:
            raise ValueError("history must be non-empty and end with token_id")

        token = torch.tensor(token_id, dtype=torch.long, device=self.device)
        token_embedding = self.embedding(token)
        zero = token_embedding.new_zeros(self.hidden_size)
        previous_hidden = state.hidden if state.hidden is not None else zero

        graph_readout, matched_path, active_paths = self._read_graph(
            token_embedding, previous_hidden, history
        )
        hidden, gates = self.gate_cell(
            token_embedding, previous_hidden, graph_readout
        )
        logits = self.output(torch.cat((hidden, graph_readout), dim=-1))
        next_state = RecurrentState(hidden, graph_readout)

        trace = {
            "matched_path": matched_path,
            "active_paths": active_paths,
            "graph_readout": graph_readout,
            "update_gate": gates["update_gate"],
            "reset_gate": gates["reset_gate"],
            "candidate": gates["candidate"],
        }
        return logits, next_state, trace

    def save_checkpoint(self, path: str | Path) -> None:
        """Save model, architecture marker, and sparse persistent graph memory.

        Save the matching Brain graph/vocabulary separately, as in the parent
        PersistentDANSNRNN implementation.
        """
        torch.save({
            "format_version": 3,
            "architecture": CHECKPOINT_ARCHITECTURE,
            "config": {
                "hidden_size": self.hidden_size,
                "window_size": self.window_size,
                "graph_depth": self.graph_depth,
                "message_rounds": self.message_rounds,
                "max_active_nodes": self.max_active_nodes,
                "max_memory_nodes": self.max_memory_nodes,
                "structural_bias": self.structural_bias,
                "vocab_size": self.embedding.num_embeddings,
            },
            "model": self.state_dict(),
            "node_memory": {
                key: value.detach().cpu() for key, value in self.node_memory.items()
            },
            "node_memory_quality": self.node_memory_quality,
            "node_memory_last_access": self.node_memory_last_access,
            "node_memory_writes": self.node_memory_writes,
            "memory_clock": self._memory_clock,
        }, str(path))

    def load_checkpoint(self, path: str | Path) -> None:
        """Load only a checkpoint created by this gate-conditioned architecture."""
        try:
            checkpoint = torch.load(
                str(path), map_location=self.device, weights_only=True
            )
        except TypeError:
            checkpoint = torch.load(str(path), map_location=self.device)

        if checkpoint.get("architecture") != CHECKPOINT_ARCHITECTURE:
            raise ValueError("checkpoint is not a DANSN-conditioned GRU checkpoint")
        config = checkpoint["config"]
        if int(config["vocab_size"]) != self.embedding.num_embeddings:
            raise ValueError("checkpoint vocabulary size mismatch")
        if int(config["hidden_size"]) != self.hidden_size:
            raise ValueError("checkpoint hidden size mismatch")
        self.load_state_dict(checkpoint["model"])
        self._rebuild_path_index()

        self.node_memory = {
            tuple(int(token) for token in key): value.to(self.device)
            for key, value in checkpoint["node_memory"].items()
            if tuple(int(token) for token in key) in self._path_nodes
        }
        valid_keys = set(self.node_memory)
        self.node_memory_quality = {
            tuple(int(token) for token in key): float(value)
            for key, value in checkpoint.get("node_memory_quality", {}).items()
            if tuple(int(token) for token in key) in valid_keys
        }
        self.node_memory_last_access = {
            tuple(int(token) for token in key): int(value)
            for key, value in checkpoint.get("node_memory_last_access", {}).items()
            if tuple(int(token) for token in key) in valid_keys
        }
        self.node_memory_writes = {
            tuple(int(token) for token in key): int(value)
            for key, value in checkpoint.get("node_memory_writes", {}).items()
            if tuple(int(token) for token in key) in valid_keys
        }
        self._memory_clock = int(checkpoint.get("memory_clock", 0))
        for key in valid_keys:
            self.node_memory_last_access.setdefault(key, self._memory_clock)
            self.node_memory_quality.setdefault(key, 1.0)
            self.node_memory_writes.setdefault(key, 1)
        self._enforce_memory_budget()


__all__ = [
    "DANSNConditionedGRU",
    "DANSNConditionedGRUCell",
    "RecurrentState",
    "train_stream",
    "evaluate_prequential",
]
