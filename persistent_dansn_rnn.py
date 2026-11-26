"""Experimental coupled RNN + DANSN with persistent graph-context memory.

The canonical agi.py engine is not modified. Graph paths are extended after
each revealed target; persistent vectors are keyed by stable token-path tuples.
Prediction always happens before the target is exposed to graph memory.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

PathKey = Tuple[int, ...]


@dataclass
class RecurrentState:
    hidden: Optional[Tensor] = None
    readout: Optional[Tensor] = None

    def detached(self) -> "RecurrentState":
        return RecurrentState(
            None if self.hidden is None else self.hidden.detach(),
            None if self.readout is None else self.readout.detach(),
        )


class PersistentDANSNRNN(nn.Module):
    """GRU whose every step reads and writes an existing DANSN graph.

    Build the vocabulary before construction. The first version uses a fixed
    vocabulary; map unseen tokens to UNK or rebuild the model/optimizer after
    an explicit vocabulary expansion.

    Save the graph/vocabulary with Brain.save() and this model plus sparse
    persistent node vectors with save_checkpoint(). Load only trusted graph
    pickle files.
    """

    def __init__(
        self,
        engine,
        vocab_size: int,
        hidden_size: int = 64,
        window_size: int = 12,
        graph_depth: int = 3,
        message_rounds: int = 2,
        max_active_nodes: int = 128,
        max_memory_nodes: int = 50_000,
        structural_bias: float = 0.15,
    ) -> None:
        super().__init__()
        if vocab_size < 2 or hidden_size < 1 or window_size < 2:
            raise ValueError("vocab_size >= 2, hidden_size >= 1, window_size >= 2 required")
        if graph_depth < 0 or message_rounds < 0 or max_active_nodes < 1 or max_memory_nodes < 1:
            raise ValueError("invalid graph limits")

        self.hidden_size = hidden_size
        self.window_size = window_size
        self.graph_depth = graph_depth
        self.message_rounds = message_rounds
        self.max_active_nodes = max_active_nodes
        self.max_memory_nodes = max_memory_nodes
        self.structural_bias = float(structural_bias)
        self.engine = engine

        d = hidden_size
        self.embedding = nn.Embedding(vocab_size, d, padding_idx=0)
        self.query_projection = nn.Linear(2 * d, d, bias=False)
        self.key_projection = nn.Linear(d, d, bias=False)
        self.value_projection = nn.Linear(d, d, bias=False)
        self.node_embedding_projection = nn.Linear(d, d, bias=False)
        self.message_projection = nn.Linear(d, d, bias=False)
        self.message_update = nn.GRUCell(3 * d, d)
        self.controller = nn.GRUCell(2 * d, d)
        self.readout_projection = nn.Linear(d, d, bias=False)
        self.fusion_gate = nn.Linear(2 * d, d)
        self.output = nn.Linear(2 * d, vocab_size)
        self.write_candidate = nn.Linear(3 * d, d)
        self.write_gate = nn.Linear(3 * d, d)
        self.memory_norm = nn.LayerNorm(d)

        self._path_nodes: Dict[PathKey, object] = {}
        self.node_memory: Dict[PathKey, Tensor] = {}
        # Non-differentiable quality/recency metadata keeps unreliable or stale
        # memory from dominating retrieval and makes capacity eviction useful.
        self.node_memory_quality: Dict[PathKey, float] = {}
        self.node_memory_last_access: Dict[PathKey, int] = {}
        self.node_memory_writes: Dict[PathKey, int] = {}
        self._memory_clock = 0
        self._rebuild_path_index()

    @property
    def device(self) -> torch.device:
        return self.embedding.weight.device

    def initial_state(self) -> RecurrentState:
        return RecurrentState()

    def _rebuild_path_index(self) -> None:
        self._path_nodes = {(): self.engine.root}
        stack = [((), self.engine.root)]
        while stack:
            path, node = stack.pop()
            for token_id, child in node.synapses.items():
                child_path = path + (int(token_id),)
                if child_path not in self._path_nodes:
                    self._path_nodes[child_path] = child
                    stack.append((child_path, child))

    def refresh_graph(self, engine=None) -> "PersistentDANSNRNN":
        if engine is not None:
            self.engine = engine
        self._rebuild_path_index()
        return self

    def _validate_ids(self, token_ids: Sequence[int]) -> None:
        size = self.embedding.num_embeddings
        for raw in token_ids:
            token_id = int(raw)
            if token_id <= 0 or token_id >= size:
                raise ValueError(
                    f"token ID {token_id} is outside [1, {size - 1}]. "
                    "Build the vocabulary first or map unseen items to UNK."
                )

    def _longest_context_path(self, history: Sequence[int]) -> Optional[PathKey]:
        """Choose the longest suffix that can actually continue or be read.

        A just-observed query prefix can create a longer terminal path (for
        example, [distractor, query_marker, key]) immediately before the model
        predicts the associated value. That path may also hold a vector written
        after observing the key, but it has no answer continuation and must not
        shadow a shorter association prefix ([query_marker, key]).
        Prefer the longest suffix with outgoing synapses. If none exists, fall
        back to the longest suffix with persistent memory, then any existing
        suffix. This favors a path that can supply candidate continuations while
        preserving memory-only reads when no continuation-bearing path exists.
        """
        max_context = max(self.window_size - 1, 1)
        first = max(0, len(history) - max_context)
        longest_existing: Optional[PathKey] = None
        longest_memory: Optional[PathKey] = None
        for start in range(first, len(history)):
            path = tuple(int(x) for x in history[start:])
            node = self._path_nodes.get(path)
            if node is None:
                continue
            if longest_existing is None:
                longest_existing = path
            if longest_memory is None and path in self.node_memory:
                longest_memory = path
            if node.synapses:
                return path
        return longest_memory or longest_existing

    def _active_subgraph(self, start_path: Optional[PathKey]):
        if start_path is None or start_path not in self._path_nodes:
            return [], {}
        ordered = [start_path]
        depths = {start_path: 0}
        incoming = {start_path: []}
        queue = [start_path]
        cursor = 0
        while cursor < len(queue) and len(ordered) < self.max_active_nodes:
            path = queue[cursor]
            cursor += 1
            node = self._path_nodes[path]
            depth = depths[path]
            if depth >= self.graph_depth:
                continue
            for token_id, child in node.synapses.items():
                child_path = path + (int(token_id),)
                self._path_nodes.setdefault(child_path, child)
                if child_path not in depths:
                    depths[child_path] = depth + 1
                    incoming[child_path] = []
                    ordered.append(child_path)
                    queue.append(child_path)
                strength = math.log1p(max(float(child.frequency), 0.0))
                incoming.setdefault(child_path, []).append((path, strength))
                if len(ordered) >= self.max_active_nodes:
                    break
        return ordered, incoming

    def _touch_memory(self, path: PathKey) -> None:
        """Record an access for LRU eviction without changing the memory vector."""
        if path not in self.node_memory:
            return
        self._memory_clock += 1
        self.node_memory_last_access[path] = self._memory_clock

    def _read_graph(self, token_embedding: Tensor, previous_hidden: Tensor,
                    history: Sequence[int]):
        zero = token_embedding.new_zeros(self.hidden_size)
        start_path = self._longest_context_path(history)
        active, incoming = self._active_subgraph(start_path)
        if not active:
            return zero, start_path, []

        # Retrieval updates only LRU metadata; stored vectors remain read-only.
        for path in active:
            self._touch_memory(path)

        query = self.query_projection(torch.cat((token_embedding, previous_hidden), dim=-1))
        current: Dict[PathKey, Tensor] = {}
        for path in active:
            node = self._path_nodes[path]
            node_id = torch.tensor(int(node.chunk), dtype=torch.long, device=self.device)
            base = self.embedding(node_id)
            persistent = self.node_memory.get(path, zero).to(device=self.device, dtype=base.dtype)
            current[path] = torch.tanh(self.node_embedding_projection(base) + persistent)

        # Transient message-passing activations; persistent memory is not mutated here.
        for _ in range(self.message_rounds):
            updated = {}
            for path in active:
                node = self._path_nodes[path]
                node_id = torch.tensor(int(node.chunk), dtype=torch.long, device=self.device)
                node_embedding = self.embedding(node_id)
                messages = [
                    self.message_projection(current[parent]) * strength
                    for parent, strength in incoming.get(path, [])
                    if parent in current and strength > 0
                ]
                message = torch.stack(messages).mean(dim=0) if messages else zero
                message_input = torch.cat((node_embedding, previous_hidden, message), dim=-1)
                updated[path] = self.message_update(message_input, current[path])
            current = updated

        states = torch.stack([current[path] for path in active], dim=0)
        keys = self.key_projection(states)
        values = self.value_projection(states)
        scores = torch.mv(keys, query) / math.sqrt(self.hidden_size)
        # Reliability is a small log-prior only for nodes that actually have
        # persistent memory. Structural graph nodes without a vector are not
        # penalized for having no quality metadata.
        quality_bias = [
            math.log(max(self.node_memory_quality.get(path, 1.0), 1e-4))
            if path in self.node_memory else 0.0
            for path in active
        ]
        scores = scores + scores.new_tensor(quality_bias)
        structural = []
        for path in active:
            freq = max(float(self._path_nodes[path].frequency), 0.0)
            depth = max(len(path) - len(start_path), 0) + 1
            structural.append(math.log1p(freq) / depth)
        scores = scores + scores.new_tensor(structural) * self.structural_bias
        weights = torch.softmax(scores, dim=0)
        readout = torch.sum(values * weights.unsqueeze(-1), dim=0)
        return readout, start_path, active

    def forward_step(self, token_id: int, history: Sequence[int],
                     state: Optional[RecurrentState] = None):
        """Predict next token. This call does not write to DANSN."""
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
        controller_input = torch.cat((token_embedding, graph_readout), dim=-1)
        hidden = self.controller(controller_input, previous_hidden)
        gate = torch.sigmoid(self.fusion_gate(torch.cat((hidden, graph_readout), dim=-1)))
        fused = self.memory_norm(hidden + gate * self.readout_projection(graph_readout))
        logits = self.output(torch.cat((fused, graph_readout), dim=-1))
        next_state = RecurrentState(fused, graph_readout)
        trace = {
            "matched_path": matched_path,
            "active_paths": active_paths,
            "graph_readout": graph_readout,
        }
        return logits, next_state, trace

    def _ensure_path(self, path: PathKey):
        node = self.engine.root
        prefix = ()
        for token_id in path:
            child = node.synapses.get(token_id)
            if child is None:
                child = node.__class__(int(token_id), 0)
                node.synapses[int(token_id)] = child
            prefix += (int(token_id),)
            self._path_nodes[prefix] = child
            node = child
        return node

    def _observe_path(self, path: PathKey):
        """Same root/path frequency increments as GraphEngine.observe_sequence."""
        self.engine.root.frequency += 1
        node = self.engine.root
        prefix = ()
        for token_id in path:
            token_id = int(token_id)
            child = node.synapses.get(token_id)
            if child is None:
                child = node.__class__(token_id, 1)
                node.synapses[token_id] = child
            else:
                child.frequency += 1
            prefix += (token_id,)
            self._path_nodes[prefix] = child
            node = child
        return node

    def observe_target(self, history: Sequence[int], target_id: int,
                       state: RecurrentState, confidence: float = 1.0) -> PathKey:
        """Reveal target, extend graph, and write memory to post-target context.

        Must be called only after target loss has been calculated. The state is
        keyed by the context after target reveal so the next recurrent step can
        retrieve it. No hidden state is broadcast to unrelated graph nodes.
        """
        if not history:
            raise ValueError("history must contain the current input token")
        target_id = int(target_id)
        self._validate_ids([*history, target_id])
        confidence = max(0.0, min(1.0, float(confidence)))

        window = tuple(int(x) for x in (*history, target_id)[-self.window_size:])
        self._observe_path(window)
        context_length = max(self.window_size - 1, 1)
        post_context = tuple(window[-context_length:])
        self._ensure_path(post_context)

        zero = self.embedding.weight.new_zeros(self.hidden_size)
        hidden = state.hidden if state.hidden is not None else zero
        readout = state.readout if state.readout is not None else zero
        target = torch.tensor(target_id, dtype=torch.long, device=self.device)
        write_input = torch.cat((hidden, self.embedding(target), readout), dim=-1)
        candidate = torch.tanh(self.write_candidate(write_input))
        gate = torch.sigmoid(self.write_gate(write_input)) * confidence
        previous = self.node_memory.get(post_context, zero).to(
            device=self.device, dtype=candidate.dtype
        )
        # Confidence zero means "do not write" exactly; avoid LayerNorm on the
        # blended state because it would change old memory even when gate=0.
        if confidence > 0.0:
            candidate = self.memory_norm(candidate)
            updated = previous + gate * (candidate - previous)
            self.node_memory[post_context] = updated
            old_quality = self.node_memory_quality.get(post_context, confidence)
            self.node_memory_quality[post_context] = (
                confidence if post_context not in self.node_memory_writes
                else 0.9 * old_quality + 0.1 * confidence
            )
            self.node_memory_writes[post_context] = self.node_memory_writes.get(post_context, 0) + 1
            self._touch_memory(post_context)
            self._enforce_memory_budget(protect=post_context)
        return post_context

    def _enforce_memory_budget(self, protect: Optional[PathKey] = None) -> None:
        """Evict least-recently-used memory; break ties by lower quality/usage."""
        while len(self.node_memory) > self.max_memory_nodes:
            candidates = [key for key in self.node_memory if key != protect]
            if not candidates:
                break
            victim = min(
                candidates,
                key=lambda key: (
                    self.node_memory_last_access.get(key, 0),
                    self.node_memory_quality.get(key, 0.0),
                    self.node_memory_writes.get(key, 0),
                ),
            )
            self.node_memory.pop(victim, None)
            self.node_memory_quality.pop(victim, None)
            self.node_memory_last_access.pop(victim, None)
            self.node_memory_writes.pop(victim, None)

    def detach_memory(self) -> None:
        self.node_memory = {key: value.detach() for key, value in self.node_memory.items()}

    def graph_stats(self) -> Dict[str, int]:
        stats = self.engine.stats()
        mean_quality = (
            sum(self.node_memory_quality.values()) / len(self.node_memory_quality)
            if self.node_memory_quality else 0.0
        )
        return {
            "nodes": int(stats.get("nodes", 0)),
            "root_words": int(stats.get("root_words", 0)),
            "memory_nodes": len(self.node_memory),
            "mean_memory_quality": mean_quality,
        }

    def save_checkpoint(self, path: str | Path) -> None:
        """Save model weights and sparse memory. Save Brain graph/vocab separately."""
        torch.save({
            "format_version": 2,
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
            "node_memory": {key: value.detach().cpu() for key, value in self.node_memory.items()},
            "node_memory_quality": self.node_memory_quality,
            "node_memory_last_access": self.node_memory_last_access,
            "node_memory_writes": self.node_memory_writes,
            "memory_clock": self._memory_clock,
        }, str(path))

    def load_checkpoint(self, path: str | Path) -> None:
        try:
            checkpoint = torch.load(str(path), map_location=self.device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(str(path), map_location=self.device)
        version = int(checkpoint.get("format_version", 1))
        if version not in (1, 2):
            raise ValueError("unsupported checkpoint format")
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
        if version == 1:
            self.node_memory_quality = {key: 1.0 for key in valid_keys}
            self.node_memory_writes = {key: 1 for key in valid_keys}
        self._memory_clock = int(checkpoint.get("memory_clock", 0))
        for key in valid_keys:
            self.node_memory_last_access.setdefault(key, self._memory_clock)
            self.node_memory_quality.setdefault(key, 1.0)
            self.node_memory_writes.setdefault(key, 1)
        self._enforce_memory_budget()


def train_stream(model: PersistentDANSNRNN, token_ids: Sequence[int],
                 optimizer: torch.optim.Optimizer, chunk_size: int = 16,
                 grad_clip: float = 1.0) -> Dict[str, float]:
    """Online causal training with graph writes every token and truncated BPTT."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    if len(token_ids) < 2:
        return {"mean_loss": 0.0, "perplexity": 1.0, "tokens": 0.0, "updates": 0.0}
    model._validate_ids(token_ids)

    state = model.initial_state()
    history = [int(token_ids[0])]
    chunk_losses: List[Tensor] = []
    observed_losses: List[float] = []
    updates = 0
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for index, raw_token in enumerate(token_ids[:-1]):
        token = int(raw_token)
        target = int(token_ids[index + 1])
        if history[-1] != token:
            raise RuntimeError("stream history lost causal alignment")
        logits, state, _trace = model.forward_step(token, history, state)
        target_tensor = torch.tensor([target], dtype=torch.long, device=model.device)
        loss = F.cross_entropy(logits.unsqueeze(0), target_tensor)
        chunk_losses.append(loss)
        observed_losses.append(float(loss.detach().cpu()))

        # The target enters graph topology/memory only after its loss is recorded.
        model.observe_target(history, target, state)
        history.append(target)

        if len(chunk_losses) >= chunk_size or index == len(token_ids) - 2:
            torch.stack(chunk_losses).mean().backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            state = state.detached()
            model.detach_memory()
            chunk_losses.clear()
            updates += 1

    mean_loss = sum(observed_losses) / len(observed_losses)
    return {
        "mean_loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 80.0)),
        "tokens": float(len(observed_losses)),
        "updates": float(updates),
    }


@torch.no_grad()
def evaluate_prequential(model: PersistentDANSNRNN, token_ids: Sequence[int],
                         update_graph: bool = True) -> Dict[str, float]:
    """Score each target before revealing it; optionally adapt graph, not weights."""
    if len(token_ids) < 2:
        return {"mean_loss": 0.0, "perplexity": 1.0, "tokens": 0.0}
    model._validate_ids(token_ids)
    state = model.initial_state()
    history = [int(token_ids[0])]
    losses = []
    was_training = model.training
    model.eval()

    for index, raw_token in enumerate(token_ids[:-1]):
        token = int(raw_token)
        target = int(token_ids[index + 1])
        if history[-1] != token:
            raise RuntimeError("stream history lost causal alignment")
        logits, state, _trace = model.forward_step(token, history, state)
        loss = F.cross_entropy(
            logits.unsqueeze(0),
            torch.tensor([target], dtype=torch.long, device=model.device),
        )
        losses.append(float(loss.cpu()))
        if update_graph:
            model.observe_target(history, target, state)
        history.append(target)

    model.train(was_training)
    mean_loss = sum(losses) / len(losses)
    return {
        "mean_loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 80.0)),
        "tokens": float(len(losses)),
    }
