"""
feedback.py - graph-native error correction ("backprop without weight matrices").

Two small channels, both driven by the exact softmax cross-entropy gradient
(delta_logit_c = -lr * (p_c - 1[c == target])) through the engine's OWN scoring:

  COUNTS   Teacher-forced over a known-good answer, edit the synapse counts the
           engine already uses (base = 1 + ln count). If the right next word has
           no edge at the node the engine really lands on, GROW the edge.
           (Counts are capped: the attention radar reads them linearly.)

  GAIN     One tiny number per prompt word: how loudly that word may speak in
           the deep-attention radar (a learned IDF). Gradient flows back through
           the BFS bridges:  dL/dlog(g_t) = sum_c (p_c - y_c) * m_tc / (attn_c * T).
           Because gains belong to WORDS, not paths, a fix on one question
           carries over to every other question that uses those words.

Credit assignment: every teacher-forced step edits the node/words it really used.
"""

import collections
import math
import random

from agi import wrap_prompt

MIN_BASE = 0.1
MAX_FREQ = 1e4
MIN_GAIN, MAX_GAIN = 1e-3, 5.0


class SelfCorrector:
    def __init__(self, brain, window=12, temperature=0.8, lr_counts=0.5,
                 lr_gain=0.2, channels=("gain", "counts")):
        self.b, self.window, self.T = brain, window, temperature
        self.lr_counts, self.lr_gain, self.channels = lr_counts, lr_gain, channels

    def gold_ids(self, answer):
        """Answer tokens INCLUDING the terminal, so the engine also learns to stop."""
        v = self.b.vocab
        s = answer.strip()
        ids = v.tokenize(s if s.endswith((".", "?", "!")) else s + ".")
        out = []
        for i in ids:
            out.append(i)
            if i in v.stop_ids():
                break
        return out

    def answer(self, prompt, temperature=0.05, top_k=1, seed=0, length=30, **kw):
        v = self.b.vocab
        seq = v.tokenize(wrap_prompt(prompt))
        return tuple(self.b.engine.predict_sequence(seq, v, length, self.window, temperature,
                                                    top_k, rng=random.Random(seed), **kw))

    def matches(self, prompt, golds, **kw):
        return self.answer(prompt, **kw) in golds

    def _pass(self, prompt, gold, target_p):
        v, e = self.b.vocab, self.b.engine
        seq = v.tokenize(wrap_prompt(prompt))
        ign = v.attention_ignored_ids()
        topics = [t for t in seq if t not in ign]
        cur, worst, grown = list(seq), 1.0, 0
        for y in gold:
            d = e.next_distribution(cur, topics, v, self.window, self.T, top_k=0)
            probs = d.probabilities()
            py = probs.get(y, 0.0)
            worst = min(worst, py)
            if py < target_p:
                if "gain" in self.channels and d.candidates:
                    grad = collections.defaultdict(float)
                    for c in d.candidates:
                        coef = probs[c.chunk] - (1.0 if c.chunk == y else 0.0)
                        if abs(coef) < 1e-4:
                            continue
                        for t, m, _hop in c.bridges:
                            grad[t] += coef * m / (c.attn * self.T)
                    for t, gr in grad.items():
                        g = e.topic_gain.get(t, 1.0) * math.exp(-self.lr_gain * gr)
                        e.topic_gain[t] = min(MAX_GAIN, max(MIN_GAIN, g))
                if "counts" in self.channels:
                    node = d.valid_neuron
                    if node is None:
                        e.observe_sequence(cur[-(self.window - 1):] + [y]); grown += 1
                    elif y not in node.synapses:
                        e.observe_sequence(d.matched_context + [y]); grown += 1
                    else:
                        for chunk, p in probs.items():
                            child = node.synapses.get(chunk)
                            g = p - (1.0 if chunk == y else 0.0)
                            if child is None or abs(g) < 1e-3:
                                continue
                            old = 1.0 + math.log(child.frequency)
                            new = max(MIN_BASE, old * math.exp(-self.T * self.lr_counts * g))
                            new = min(new, max(old, 1.0 + math.log(MAX_FREQ)))
                            child.frequency = math.exp(new - 1.0)
            cur.append(y)
        return worst, grown

    def correct(self, prompt, answer, max_epochs=15, target_p=0.9):
        gold, grown_total = self.gold_ids(answer), 0
        for epoch in range(1, max_epochs + 1):
            worst, grown = self._pass(prompt, gold, target_p)
            grown_total += grown
            if worst >= target_p:
                break
        return {"epochs": epoch, "min_p": worst, "grown": grown_total}
