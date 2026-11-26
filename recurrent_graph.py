"""Optional RNN-controlled recurrent graph-state block.

This module leaves the canonical GraphEngine untouched. It snapshots its trie
as a sparse directed graph, uses the GraphEngine's suffix lookup to select an
active neighborhood, and performs gated message passing over that neighborhood.

Requires PyTorch. The graph topology is a snapshot: call refresh_graph() after
changing the underlying GraphEngine.
"""

from dataclasses import dataclass, field
import math
from typing import Dict, Optional

import torch
from torch import nn


@dataclass
class RecurrentGraphState:
    """State carried between sequence steps for one sequence."""

    hidden: Optional[torch.Tensor] = None
    readout: Optional[torch.Tensor] = None
    node_states: Dict[int, torch.Tensor] = field(default_factory=dict)


class RecurrentGraphBlock(nn.Module):
    """A GRU controller coupled to sparse, context-selected graph dynamics.

    Args:
        engine: trained or partially trained canonical GraphEngine.
        vocab_size: number of token IDs including reserved ID zero.
        hidden_size: shared token, controller, and graph-state width.
        graph_depth: maximum number of trie edges expanded from the matched
            context node at each step.
        message_rounds: recurrent message-passing updates per sequence step.
        max_active_nodes: hard cap on active graph nodes per step.
        window_size: context window passed to the canonical suffix matcher.
        max_state_nodes: maximum number of persistent sequence-local node states.

    This is a research adapter, not a pretrained language model. It needs
    task-specific training before its output logits become useful.
    """

    def __init__(self, engine, vocab_size, hidden_size=64, graph_depth=3,
                 message_rounds=2, max_active_nodes=256, window_size=12,
                 max_state_nodes=2048):
        super().__init__()
        if vocab_size < 2:
            raise ValueError("vocab_size must include padding ID 0 and at least one token")
        if (hidden_size < 1 or graph_depth < 0 or message_rounds < 1
                or max_active_nodes < 1 or window_size < 1 or max_state_nodes < 1):
            raise ValueError("invalid recurrent graph dimensions or limits")

        self.hidden_size = hidden_size
        self.graph_depth = graph_depth
        self.message_rounds = message_rounds
        self.max_active_nodes = max_active_nodes
        self.window_size = window_size
        self.max_state_nodes = max_state_nodes
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.controller = nn.GRUCell(2 * hidden_size, hidden_size)
        self.message_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.node_update = nn.GRUCell(3 * hidden_size, hidden_size)
        self.query_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.readout_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gate = nn.Linear(2 * hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size)
        self.refresh_graph(engine)

    def refresh_graph(self, engine):
        """Rebuild the non-trainable topology snapshot from GraphEngine."""
        self._engine = engine
        self._nodes = []
        self._node_index = {}
        self._edges = []
        stack = [engine.root]
        while stack:
            node = stack.pop()
            if id(node) in self._node_index:
                continue
            index = len(self._nodes)
            self._node_index[id(node)] = index
            self._nodes.append(node)
            for child in node.synapses.values():
                stack.append(child)
        for src, node in enumerate(self._nodes):
            for child in node.synapses.values():
                dst = self._node_index[id(child)]
                self._edges.append((src, dst, float(child.frequency)))
        self._root_index = self._node_index[id(engine.root)]
        return self

    def initial_state(self):
        return RecurrentGraphState()

    def _active_subgraph(self, sequence):
        """Return BFS-ordered active node indices and incoming parent lists."""
        start, _matched = self._engine._find_valid_neuron(
            sequence, window_size=self.window_size
        )
        if start is None:
            return [], {}
        start_index = self._node_index.get(id(start))
        if start_index is None:
            return [], {}

        ordered = [start_index]
        depths = {start_index: 0}
        incoming = {start_index: []}
        queue = [start]
        cursor = 0
        while cursor < len(queue) and len(ordered) < self.max_active_nodes:
            node = queue[cursor]
            cursor += 1
            src = self._node_index[id(node)]
            depth = depths[src]
            if depth >= self.graph_depth:
                continue
            for child in node.synapses.values():
                dst = self._node_index[id(child)]
                if dst not in depths:
                    depths[dst] = depth + 1
                    incoming[dst] = []
                    ordered.append(dst)
                    queue.append(child)
                incoming.setdefault(dst, []).append(
                    (src, math.log1p(max(child.frequency, 0.0)))
                )
                if len(ordered) >= self.max_active_nodes:
                    break
        return ordered, incoming

    def forward_step(self, token_id, sequence, state=None):
        """Process one token; sequence includes token_id as its final element.

        Returns (vocabulary_logits, updated_state).
        """
        state = state or self.initial_state()
        device = self.embedding.weight.device
        token = torch.as_tensor(token_id, dtype=torch.long, device=device)
        token_embedding = self.embedding(token)
        zero = token_embedding.new_zeros(self.hidden_size)
        previous_hidden = state.hidden if state.hidden is not None else zero
        previous_readout = state.readout if state.readout is not None else zero
        controller_input = torch.cat((token_embedding, previous_readout), dim=-1)
        controller_hidden = self.controller(controller_input, previous_hidden)

        active, incoming = self._active_subgraph(sequence)
        node_states = dict(state.node_states)
        if active:
            current = {idx: node_states.get(idx, zero) for idx in active}
            for _ in range(self.message_rounds):
                updated = {}
                for idx in active:
                    messages = []
                    for parent_idx, log_strength in incoming.get(idx, []):
                        parent_state = current.get(
                            parent_idx, node_states.get(parent_idx, zero)
                        )
                        messages.append(
                            self.message_projection(parent_state) * log_strength
                        )
                    message = torch.stack(messages).mean(dim=0) if messages else zero
                    node_token = self._nodes[idx].chunk
                    node_embedding = self.embedding(
                        torch.as_tensor(node_token, dtype=torch.long, device=device)
                    )
                    node_input = torch.cat(
                        (node_embedding, controller_hidden, message), dim=-1
                    )
                    updated[idx] = self.node_update(node_input, current[idx])
                current = updated
            node_states.update(current)
            # Bound retained dynamic state; recently active nodes are kept.
            while len(node_states) > self.max_state_nodes:
                node_states.pop(next(iter(node_states)))
            states = torch.stack([current[idx] for idx in active], dim=0)
            query = self.query_projection(controller_hidden)
            keys = self.key_projection(states)
            scores = torch.mv(keys, query) / math.sqrt(self.hidden_size)
            weights = torch.softmax(scores, dim=0)
            graph_readout = torch.sum(states * weights.unsqueeze(-1), dim=0)
        else:
            graph_readout = zero

        gate = torch.sigmoid(
            self.gate(torch.cat((controller_hidden, graph_readout), dim=-1))
        )
        combined = self.norm(
            controller_hidden + gate * self.readout_projection(graph_readout)
        )
        logits = self.output(combined)
        return logits, RecurrentGraphState(
            hidden=combined, readout=graph_readout, node_states=node_states
        )

    def forward_sequence(self, input_ids, state=None):
        """Run a 1-D token-ID sequence; return [time, vocab] logits and state."""
        state = state or self.initial_state()
        prefix = []
        outputs = []
        for raw_token in input_ids:
            token_id = int(raw_token)
            prefix.append(token_id)
            logits, state = self.forward_step(token_id, prefix, state)
            outputs.append(logits)
        if outputs:
            return torch.stack(outputs, dim=0), state
        empty = self.embedding.weight.new_empty((0, self.output.out_features))
        return empty, state
