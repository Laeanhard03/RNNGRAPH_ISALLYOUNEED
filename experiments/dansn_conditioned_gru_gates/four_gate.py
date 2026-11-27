"""Four-gate DANSN-GRU experiment.

The DANSN graph retrieval remains discrete for path selection. A learned
controller converts the retrieved readout plus graph-structure evidence into
four differentiable, vector-valued gates:
  1. update: blend old hidden state with candidate state;
  2. reset: filter the old hidden state used by the candidate;
  3. memory injection: control retrieved DANSN context entering the candidate;
  4. persistent write: control the post-score write into DANSN node memory.

The write gate is produced before seeing the target, then applied only after
the target loss has been computed. The canonical agi.py graph engine and the
existing recurrent-memory module are not modified.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch import Tensor, nn

from experiments.dansn_conditioned_gru_gates.model import (
    DANSNConditionedGRU,
)
from persistent_dansn_rnn import (
    RecurrentState,
    evaluate_prequential,
    train_stream,
)


FOUR_GATE_CHECKPOINT_ARCHITECTURE = "dansn_four_gate_gru_v8"
GRAPH_EVIDENCE_SIZE = 5


@dataclass
class FourGateState(RecurrentState):
    """Recurrent state carrying the pre-target persistent-write gate."""

    write_gate: Optional[Tensor] = None
    context_hidden: Optional[Tensor] = None
    # Snapshot of the exact pre-token state used to form the associative query.
    write_query_hidden: Optional[Tensor] = None
    write_query_context: Optional[Tensor] = None

    def detached(self) -> "FourGateState":
        return FourGateState(
            None if self.hidden is None else self.hidden.detach(),
            None if self.readout is None else self.readout.detach(),
            None if self.write_gate is None else self.write_gate.detach(),
            None if self.context_hidden is None else self.context_hidden.detach(),
            None if self.write_query_hidden is None else self.write_query_hidden.detach(),
            None if self.write_query_context is None else self.write_query_context.detach(),
        )


class DANSNFourGateCell(nn.Module):
    """GRU cell whose controller emits update, reset, memory, and write gates."""

    def __init__(self, hidden_size: int, evidence_size: int = GRAPH_EVIDENCE_SIZE):
        super().__init__()
        if hidden_size < 1 or evidence_size < 1:
            raise ValueError("hidden_size and evidence_size must be positive")

        self.hidden_size = int(hidden_size)
        self.evidence_size = int(evidence_size)
        # x_t, h_(t-1), retrieved memory m_t, and graph evidence phi_t.
        controller_input_size = 3 * hidden_size + evidence_size
        self.controller = nn.Linear(controller_input_size, 4 * hidden_size)
        self.candidate_input = nn.Linear(hidden_size, hidden_size)
        self.candidate_hidden = nn.Linear(hidden_size, hidden_size, bias=False)
        # Persistent DANSN vectors are a separate state source, not another
        # term buried inside the candidate activation.
        self.memory_projection = nn.Linear(hidden_size, hidden_size, bias=False)

        # Start conservatively: initially limit the memory injection and
        # persistent write paths while allowing update/reset to be learned.
        with torch.no_grad():
            # Start the memory path open enough to learn from immediately.
            # A strongly negative bias made the model learn to ignore retrieved
            # memory before it had a chance to shape the recurrent state.
            self.controller.bias[2 * hidden_size:3 * hidden_size].fill_(0.0)
            # Writes remain conservative until the controller learns when to store.
            self.controller.bias[3 * hidden_size:4 * hidden_size].fill_(-2.0)

    def forward(
        self,
        inputs: Tensor,
        previous_hidden: Tensor,
        graph_memory: Tensor,
        graph_evidence: Tensor,
        *,
        memory_available: bool | Tensor = True,
        memory_enabled: bool = True,
        write_enabled: bool = True,
        standard_gru_carry: bool = False,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        d = self.hidden_size
        if inputs.shape[-1] != d or previous_hidden.shape[-1] != d:
            raise ValueError("input and previous hidden widths must equal hidden_size")
        if graph_memory.shape[-1] != d:
            raise ValueError("graph_memory width must equal hidden_size")
        if graph_evidence.shape[-1] != self.evidence_size:
            raise ValueError("graph_evidence width does not match evidence_size")

        controller_input = torch.cat(
            (inputs, previous_hidden, graph_memory, graph_evidence), dim=-1
        )
        update_logits, reset_logits, memory_logits, write_logits = (
            self.controller(controller_input).chunk(4, dim=-1)
        )
        update_gate = torch.sigmoid(update_logits)
        reset_gate = torch.sigmoid(reset_logits)
        memory_gate = torch.sigmoid(memory_logits)
        write_gate = torch.sigmoid(write_logits)

        if not memory_enabled:
            memory_gate = torch.zeros_like(memory_gate)
        if not write_enabled:
            write_gate = torch.zeros_like(write_gate)

        # Reset is allowed to clear the carried recurrent state as well as
        # filtering the state used to form the candidate. This makes it possible
        # for graph evidence to suppress stale/distractor-driven hidden content.
        reset_hidden = reset_gate * previous_hidden
        candidate = torch.tanh(
            self.candidate_input(inputs)
            + self.candidate_hidden(reset_hidden)
        )
        # Standard GRU interpolation carries the unfiltered previous state;
        # reset filters only the recurrent input used by the candidate.
        recurrent_carry = previous_hidden if standard_gru_carry else reset_hidden
        recurrent_hidden = (
            (1.0 - update_gate) * recurrent_carry
            + update_gate * candidate
        )

        # Persistent DANSN memory remains a separate state source, but it
        # must not erase the recurrent cue/working context at a single step.
        # Keep at least half of the recurrent transition in the hidden state,
        # even when the learned memory gate saturates. The full controller
        # memory mix still drives the decoder skip and direct pointer route.
        memory_proposal = torch.tanh(self.memory_projection(graph_memory))
        if isinstance(memory_available, Tensor):
            availability = memory_available.to(
                device=inputs.device, dtype=inputs.dtype
            )
        else:
            availability = inputs.new_tensor(float(bool(memory_available)))
        memory_mix = memory_gate * availability
        hidden_memory_mix = 0.5 * memory_mix
        hidden = (
            (1.0 - hidden_memory_mix) * recurrent_hidden
            + hidden_memory_mix * memory_proposal
        )
        gates = {
            "update_gate": update_gate,
            "reset_gate": reset_gate,
            "memory_injection_gate": memory_gate,
            "effective_memory_mix": memory_mix,
            "hidden_memory_mix": hidden_memory_mix,
            "memory_available": availability,
            "persistent_write_gate": write_gate,
            "candidate": candidate,
            "memory_proposal": memory_proposal,
            "recurrent_hidden": recurrent_hidden,
            "recurrent_carry": recurrent_carry,
        }
        return hidden, gates


class DANSNFourGateGRU(DANSNConditionedGRU):
    """DANSN-conditioned GRU with a distinct persistent-write gate.

    Ablations preserve the same parameter shapes:
      - full
      - no_memory_injection
      - no_persistent_write
      - no_graph_evidence

    The write gate is state carried from prediction time to observe_target().
    It cannot inspect the next token before that token's prediction is scored.
    """

    ALLOWED_ABLATIONS = {
        "full",
        "no_memory_injection",
        "no_persistent_write",
        "no_graph_evidence",
        # Disable only the direct vocabulary pointer. Persistent vectors still
        # enter the recurrent hidden state and the decoder's memory skip path.
        "no_pointer",
    }

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
        gate_ablation: str = "full",
        write_threshold: float = 0.01,
        transition_variant: str = "legacy",
    ) -> None:
        # Construct the shared persistent DANSN graph reader and writer.
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
            gate_ablation="full",
        )
        if gate_ablation not in self.ALLOWED_ABLATIONS:
            raise ValueError(
                f"gate_ablation must be one of {sorted(self.ALLOWED_ABLATIONS)}"
            )
        if not 0.0 <= float(write_threshold) < 1.0:
            raise ValueError("write_threshold must be in [0, 1)")
        allowed_transition_variants = {
            "legacy", "standard_carry", "protected_context",
            "protected_context_standard_carry", "protected_context_write",
            "protected_context_write_standard_carry",
            "protected_context_query_aligned_write",
            "protected_context_query_aligned_write_standard_carry",
        }
        if transition_variant not in allowed_transition_variants:
            raise ValueError(
                f"transition_variant must be one of {sorted(allowed_transition_variants)}"
            )

        # The parent experiment's three-gate cell and scalar-independent
        # write MLP are replaced by this four-output controller.
        del self.gate_cell
        del self.write_gate
        self.four_gate_cell = DANSNFourGateCell(hidden_size)
        # Nonlinear pairwise selector. The query is conditioned on the
        # recurrent state; the candidate key contains its token embedding and
        # persistent path state. Explicit product features let the scorer learn
        # cue/candidate interactions instead of relying on a single dot product.
        self.pointer_scorer = nn.Sequential(
            nn.Linear(3 * hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )
        # The decoder sees the recurrent state, broad graph context, and a
        # separate gated persistent-memory channel. This prevents the final
        # prediction head from having to recover all associative information
        # only through the fused hidden vector.
        self.output = nn.Linear(3 * hidden_size, vocab_size)
        self.gate_ablation = gate_ablation
        self.write_threshold = float(write_threshold)
        self.transition_variant = transition_variant
        self.standard_gru_carry = transition_variant in {
            "standard_carry", "protected_context_standard_carry",
            "protected_context_write_standard_carry",
            "protected_context_query_aligned_write_standard_carry",
        }
        self.protected_context_enabled = transition_variant in {
            "protected_context", "protected_context_standard_carry",
            "protected_context_write", "protected_context_write_standard_carry",
            "protected_context_query_aligned_write",
            "protected_context_query_aligned_write_standard_carry",
        }
        self.context_write_enabled = transition_variant in {
            "protected_context_write", "protected_context_write_standard_carry",
            "protected_context_query_aligned_write",
            "protected_context_query_aligned_write_standard_carry",
        }
        self.query_aligned_write_enabled = transition_variant in {
            "protected_context_query_aligned_write",
            "protected_context_query_aligned_write_standard_carry",
        }
        # Expand the inherited writer to accept a context vector while
        # preserving the original hidden/target/readout projections.
        old_writer = self.write_candidate
        expanded_writer = nn.Linear(4 * hidden_size, hidden_size)
        with torch.no_grad():
            expanded_writer.weight.zero_()
            expanded_writer.weight[:, :hidden_size].copy_(old_writer.weight[:, :hidden_size])
            expanded_writer.weight[:, 2 * hidden_size:3 * hidden_size].copy_(
                old_writer.weight[:, hidden_size:2 * hidden_size]
            )
            expanded_writer.weight[:, 3 * hidden_size:4 * hidden_size].copy_(
                old_writer.weight[:, 2 * hidden_size:3 * hidden_size]
            )
            expanded_writer.bias.copy_(old_writer.bias)
        self.write_candidate = expanded_writer
        # Independent slow-changing state, separate from DANSN memory fusion.
        # Instantiated in every variant to keep parameter counts identical.
        self.context_input_projection = nn.Linear(hidden_size, hidden_size)
        self.context_hidden_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.context_retention_gate = nn.Linear(2 * hidden_size, hidden_size)
        with torch.no_grad():
            self.context_retention_gate.bias.fill_(3.0)

    def _graph_evidence(
        self,
        matched_path,
        active_paths: Sequence[tuple[int, ...]],
        reference: Tensor,
    ) -> Tensor:
        """Build bounded structural features from the selected graph neighborhood.

        Features: exact/suffix context match, active-neighborhood occupancy,
        normalized path frequency, persistent-memory coverage, and mean stored
        memory quality. These are evidence features, not learned gate values.
        """
        if not active_paths:
            values = [0.0] * GRAPH_EVIDENCE_SIZE
        else:
            count = len(active_paths)
            frequencies = [
                max(float(self._path_nodes[path].frequency), 0.0)
                for path in active_paths
            ]
            normalized_frequency = sum(
                math.log1p(value) for value in frequencies
            ) / count / math.log1p(100.0)
            normalized_frequency = min(max(normalized_frequency, 0.0), 1.0)

            memory_paths = [
                path for path in active_paths if path in self.node_memory
            ]
            memory_coverage = len(memory_paths) / count
            mean_quality = (
                sum(self.node_memory_quality.get(path, 1.0) for path in memory_paths)
                / len(memory_paths)
                if memory_paths else 0.0
            )
            occupancy = min(count / max(self.max_active_nodes, 1), 1.0)
            values = [
                float(matched_path is not None),
                occupancy,
                normalized_frequency,
                memory_coverage,
                min(max(float(mean_quality), 0.0), 1.0),
            ]

        return reference.new_tensor(values)

    def initial_state(self) -> FourGateState:
        return FourGateState()

    def _update_protected_context(
        self,
        token_embedding: Tensor,
        previous_context: Optional[Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Update an independent, high-retention recurrent context channel."""
        previous = torch.zeros_like(token_embedding) if previous_context is None else previous_context
        proposal = torch.tanh(
            self.context_input_projection(token_embedding)
            + self.context_hidden_projection(previous)
        )
        retention = torch.sigmoid(self.context_retention_gate(
            torch.cat((token_embedding, previous), dim=-1)
        ))
        if previous_context is None:
            # Do not attenuate the first cue by blending it with an empty state.
            updated = proposal
        else:
            updated = retention * previous + (1.0 - retention) * proposal
        return updated, retention

    def _read_persistent_memory(
        self,
        token_embedding: Tensor,
        previous_hidden: Tensor,
        matched_path,
        active_paths: Sequence[tuple[int, ...]],
    ) -> tuple[Tensor, list[tuple[int, ...]], Tensor, bool, Tensor, Tensor]:
        """Read persistent vectors and build a direct next-token pointer.

        The broad graph readout performs transient message passing over a
        neighborhood. This readout attends to persistent vectors on immediate
        outgoing continuations of the matched context. If no direct continuation
        has a stored vector, it falls back to memory-bearing active paths for
        vector state fusion, but does not emit a token pointer from that fallback.
        The pointer head is therefore grounded only in actual next-token edges.
        """
        zero = token_embedding.new_zeros(self.hidden_size)
        pointer = token_embedding.new_zeros(self.embedding.num_embeddings)
        if not active_paths:
            empty = token_embedding.new_zeros(0)
            empty_ids = torch.empty(0, dtype=torch.long, device=self.device)
            return zero, [], pointer, False, empty, empty_ids

        direct_candidates: list[tuple[int, ...]] = []
        if matched_path is not None:
            prefix_len = len(matched_path)
            direct_candidates = [
                path for path in active_paths
                if len(path) == prefix_len + 1
                and path[:prefix_len] == matched_path
                and path in self.node_memory
            ]
        candidates = direct_candidates or [
            path for path in active_paths if path in self.node_memory
        ]
        if not candidates:
            empty = token_embedding.new_zeros(0)
            empty_ids = torch.empty(0, dtype=torch.long, device=self.device)
            return zero, [], pointer, False, empty, empty_ids

        query = self.query_projection(
            torch.cat((token_embedding, previous_hidden), dim=-1)
        )
        keys: list[Tensor] = []
        values: list[Tensor] = []
        scores: list[Tensor] = []
        for path in candidates:
            stored = self.node_memory[path].to(
                device=self.device, dtype=token_embedding.dtype
            )
            node_id = torch.tensor(
                int(path[-1]), dtype=torch.long, device=self.device
            )
            node_embedding = self.embedding(node_id)
            node_features = torch.tanh(
                self.node_embedding_projection(node_embedding) + stored
            )
            keys.append(self.key_projection(node_features))
            values.append(self.value_projection(stored))

            node = self._path_nodes[path]
            frequency = max(float(node.frequency), 0.0)
            quality = max(float(self.node_memory_quality.get(path, 1.0)), 1e-4)
            prior = (
                math.log1p(frequency) * self.structural_bias
                + math.log(quality)
            )
            pair_features = torch.cat(
                (query, keys[-1], query * keys[-1]), dim=-1
            )
            learned_score = self.pointer_scorer(pair_features).squeeze(-1)
            scores.append(learned_score + query.new_tensor(prior))
            self._touch_memory(path)

        weights = torch.softmax(torch.stack(scores), dim=0)
        retrieved = torch.sum(
            torch.stack(values, dim=0) * weights.unsqueeze(-1), dim=0
        )

        pointer_available = bool(direct_candidates)
        if pointer_available:
            candidate_ids = torch.tensor(
                [int(path[-1]) for path in direct_candidates],
                dtype=torch.long,
                device=self.device,
            )
            pointer = pointer.scatter_add(0, candidate_ids, weights)
        else:
            candidate_ids = torch.empty(0, dtype=torch.long, device=self.device)
        return (
            retrieved, candidates, pointer, pointer_available,
            torch.stack(scores), candidate_ids,
        )

    def forward_step(
        self,
        token_id: int,
        history: Sequence[int],
        state: Optional[RecurrentState] = None,
    ):
        """Predict the next token and calculate all four gates; never writes."""
        state = state or self.initial_state()
        token_id = int(token_id)
        self._validate_ids([token_id, *history])
        if not history or int(history[-1]) != token_id:
            raise ValueError("history must be non-empty and end with token_id")

        token = torch.tensor(token_id, dtype=torch.long, device=self.device)
        token_embedding = self.embedding(token)
        zero = token_embedding.new_zeros(self.hidden_size)
        previous_hidden = state.hidden if state.hidden is not None else zero
        previous_context = getattr(state, "context_hidden", None)
        context_for_selection = (
            previous_context
            if self.protected_context_enabled and previous_context is not None
            else zero
        )
        # The selector sees the temporary hidden state plus the protected
        # contextual trace. The four-gate controller itself remains unchanged.
        selection_hidden = previous_hidden + context_for_selection

        graph_readout, matched_path, active_paths = self._read_graph(
            token_embedding, previous_hidden, history
        )
        (
            persistent_readout,
            persistent_paths,
            persistent_token_distribution,
            pointer_available,
            pointer_scores,
            pointer_candidate_ids,
        ) = self._read_persistent_memory(
            token_embedding, selection_hidden, matched_path, active_paths
        )
        evidence = self._graph_evidence(matched_path, active_paths, token_embedding)
        if self.gate_ablation == "no_graph_evidence":
            evidence = torch.zeros_like(evidence)

        memory_available = bool(persistent_paths)
        hidden, gates = self.four_gate_cell(
            token_embedding,
            previous_hidden,
            persistent_readout,
            evidence,
            memory_available=memory_available,
            memory_enabled=self.gate_ablation != "no_memory_injection",
            write_enabled=self.gate_ablation != "no_persistent_write",
            standard_gru_carry=self.standard_gru_carry,
        )
        memory_output_context = (
            gates["effective_memory_mix"] * persistent_readout
        )
        neural_logits = self.output(torch.cat(
            (hidden, graph_readout, memory_output_context), dim=-1
        ))
        neural_probabilities = torch.softmax(neural_logits, dim=-1)
        pointer_enabled = self.gate_ablation != "no_pointer"
        pointer_mix = (
            gates["effective_memory_mix"].mean()
            if pointer_available and pointer_enabled
            else neural_logits.new_zeros(())
        )
        mixed_probabilities = (
            (1.0 - pointer_mix) * neural_probabilities
            + pointer_mix * persistent_token_distribution
        )
        # The DANSN pointer is a causal next-edge distribution. It is available
        # only when the pre-target matched context has stored outgoing vectors.
        # The memory gate controls how much of this associative distribution is
        # blended with the GRU's learned vocabulary distribution.
        logits = torch.log(mixed_probabilities.clamp_min(1e-9))
        if self.protected_context_enabled:
            next_context, context_retention = self._update_protected_context(
                token_embedding, previous_context
            )
        else:
            next_context = None
            context_retention = torch.zeros_like(token_embedding)
        next_state = FourGateState(
            hidden=hidden,
            readout=graph_readout,
            write_gate=gates["persistent_write_gate"],
            context_hidden=next_context,
            # Candidate writes run after target reveal, but must snapshot the
            # exact pre-token state that was used for pointer scoring.
            write_query_hidden=previous_hidden,
            write_query_context=context_for_selection,
        )
        trace = {
            "matched_path": matched_path,
            "active_paths": active_paths,
            "graph_readout": graph_readout,
            "persistent_memory_readout": persistent_readout,
            "persistent_memory_paths": persistent_paths,
            "memory_output_context": memory_output_context,
            "neural_probabilities": neural_probabilities,
            "persistent_token_distribution": persistent_token_distribution,
            "pointer_scores": pointer_scores,
            "pointer_candidate_ids": pointer_candidate_ids,
            "pointer_query": self.query_projection(
                torch.cat((token_embedding, selection_hidden), dim=-1)
            ),
            "previous_hidden": previous_hidden,
            "selection_hidden": selection_hidden,
            "previous_context_hidden": (
                previous_context if previous_context is not None else zero
            ),
            "protected_context_hidden": (
                next_context if next_context is not None else zero
            ),
            "protected_context_retention_gate": context_retention,
            "protected_context_enabled": token_embedding.new_tensor(
                float(self.protected_context_enabled)
            ),
            "context_write_enabled": token_embedding.new_tensor(
                float(self.context_write_enabled)
            ),
            "effective_pointer_mix": pointer_mix,
            "pointer_available": pointer_available,
            "pointer_enabled": pointer_enabled,
            "graph_evidence": evidence,
            **gates,
        }
        return logits, next_state, trace

    def _observe_path_gated(self, path: tuple[int, ...], strength: float) -> None:
        """Add/reinforce observed graph edges by the scalar write-gate strength.

        GraphEngine's canonical observe_sequence() uses unit increments. This
        experiment uses fractional frequency increments so g_write can control
        structural reinforcement without modifying agi.py.
        """
        strength = min(max(float(strength), 0.0), 1.0)
        if strength <= 0.0:
            return
        self.engine.root.frequency += strength
        node = self.engine.root
        prefix: tuple[int, ...] = ()
        for raw_token in path:
            token_id = int(raw_token)
            child = node.synapses.get(token_id)
            if child is None:
                child = node.__class__(token_id, strength)
                node.synapses[token_id] = child
            else:
                child.frequency += strength
            prefix += (token_id,)
            self._path_nodes[prefix] = child
            node = child

    def observe_target(
        self,
        history: Sequence[int],
        target_id: int,
        state: RecurrentState,
        confidence: float = 1.0,
    ):
        """Score first, then let the saved write gate control graph and vector writes."""
        if not history:
            raise ValueError("history must contain the current input token")
        target_id = int(target_id)
        self._validate_ids([*history, target_id])
        confidence = min(max(float(confidence), 0.0), 1.0)

        window = tuple(int(x) for x in (*history, target_id)[-self.window_size:])
        context_length = max(self.window_size - 1, 1)
        post_context = tuple(window[-context_length:])

        write_gate = getattr(state, "write_gate", None)
        if write_gate is None:
            return post_context
        if write_gate.shape[-1] != self.hidden_size:
            raise ValueError("persistent write gate width does not match hidden_size")

        effective_gate = write_gate.to(
            device=self.device, dtype=self.embedding.weight.dtype
        ).clamp(0.0, 1.0) * confidence
        write_strength = float(effective_gate.mean().detach().cpu())
        if confidence <= 0.0 or write_strength <= self.write_threshold:
            return post_context

        # The same gate controls discrete edge reinforcement and the soft
        # persistent-vector blend. The edge weight is a non-differentiable
        # frequency statistic; vector blending remains differentiable.
        self._observe_path_gated(window, write_strength)
        self._ensure_path(post_context)

        zero = self.embedding.weight.new_zeros(self.hidden_size)
        hidden = state.hidden if state.hidden is not None else zero
        readout = state.readout if state.readout is not None else zero
        target = torch.tensor(target_id, dtype=torch.long, device=self.device)
        context_hidden = getattr(state, "context_hidden", None)
        if self.query_aligned_write_enabled:
            # The pointer was scored from the pre-token state. Store that exact
            # query-time hidden/context pair, not the post-key recurrent state.
            query_hidden = getattr(state, "write_query_hidden", None)
            query_context = getattr(state, "write_query_context", None)
            if query_hidden is not None:
                hidden = query_hidden
            context_hidden = query_context
        if context_hidden is None or not self.context_write_enabled:
            context_hidden = zero
        write_input = torch.cat(
            (hidden, context_hidden, self.embedding(target), readout), dim=-1
        )
        candidate = self.memory_norm(torch.tanh(self.write_candidate(write_input)))

        previous = self.node_memory.get(post_context, zero).to(
            device=self.device, dtype=candidate.dtype
        )
        updated = previous + effective_gate * (candidate - previous)
        self.node_memory[post_context] = updated

        effective_quality = confidence * float(effective_gate.mean().detach().cpu())
        old_quality = self.node_memory_quality.get(post_context, effective_quality)
        self.node_memory_quality[post_context] = (
            effective_quality if post_context not in self.node_memory_writes
            else 0.9 * old_quality + 0.1 * effective_quality
        )
        self.node_memory_writes[post_context] = (
            self.node_memory_writes.get(post_context, 0) + 1
        )
        self._touch_memory(post_context)
        self._enforce_memory_budget(protect=post_context)
        return post_context

    def save_checkpoint(self, path: str | Path) -> None:
        """Save four-gate parameters and sparse graph memory; save Brain separately."""
        torch.save({
            "format_version": 1,
            "architecture": FOUR_GATE_CHECKPOINT_ARCHITECTURE,
            "config": {
                "hidden_size": self.hidden_size,
                "window_size": self.window_size,
                "graph_depth": self.graph_depth,
                "message_rounds": self.message_rounds,
                "max_active_nodes": self.max_active_nodes,
                "max_memory_nodes": self.max_memory_nodes,
                "structural_bias": self.structural_bias,
                "vocab_size": self.embedding.num_embeddings,
                "gate_ablation": self.gate_ablation,
                "write_threshold": self.write_threshold,
                "transition_variant": self.transition_variant,
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
        """Load only checkpoints created by the four-gate architecture."""
        try:
            checkpoint = torch.load(
                str(path), map_location=self.device, weights_only=True
            )
        except TypeError:
            checkpoint = torch.load(str(path), map_location=self.device)

        if checkpoint.get("architecture") != FOUR_GATE_CHECKPOINT_ARCHITECTURE:
            raise ValueError("checkpoint is not a DANSN four-gate GRU checkpoint")
        config = checkpoint["config"]
        if int(config["vocab_size"]) != self.embedding.num_embeddings:
            raise ValueError("checkpoint vocabulary size mismatch")
        if int(config["hidden_size"]) != self.hidden_size:
            raise ValueError("checkpoint hidden size mismatch")
        if config.get("gate_ablation", "full") != self.gate_ablation:
            raise ValueError("checkpoint gate_ablation does not match this model")
        if float(config.get("write_threshold", self.write_threshold)) != self.write_threshold:
            raise ValueError("checkpoint write_threshold does not match this model")
        if config.get("transition_variant", "legacy") != self.transition_variant:
            raise ValueError("checkpoint transition_variant does not match this model")

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
    "DANSNFourGateGRU",
    "DANSNFourGateCell",
    "FourGateState",
    "train_stream",
    "evaluate_prequential",
]
