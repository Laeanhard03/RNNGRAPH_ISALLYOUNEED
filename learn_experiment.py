"""
learn_experiment.py - can the graph engine correct its own mistakes with a
backprop-style update, without any dense ML weights?

Idea: the engine already turns evidence into a softmax-like roulette wheel
(adjusted = final ** (1/T), normalised). Add ONE sparse learned parameter
family on top of it:

    w[(topic_token, candidate_token)]   # "how much does this prompt word
                                        #  vouch for this answer word"

    adjusted_c = final_c ** (1/T) * exp( sum_t  w[t, c] )

That is exactly softmax(logit + bias), so cross-entropy has an exact gradient:

    dLoss/dw[t, c] = p_c - 1[c == target]

and gradient descent on it is the whole "backprop". Counts, trie and BFS
radar are untouched; w only re-weighs edges the graph already has.
"""

import math
import random

from agi import Brain, GraphEngine, wrap_prompt


class LearnableEngine(GraphEngine):
    def __init__(self):
        super().__init__()
        self.w = {}  # (topic_token, candidate_chunk) -> float

    def next_distribution(self, seq, topic_tokens, vocab, window_size=12,
                          temperature=0.8, top_k=5, attention_gain=15.0,
                          attention_depth=6, fuzzy_min=0.15, _attn_cache=None):
        d = super().next_distribution(seq, topic_tokens, vocab, window_size,
                                      temperature, 0, attention_gain,
                                      attention_depth, fuzzy_min, _attn_cache)
        topics = set(topic_tokens)
        for c in d.candidates:
            c.adjusted *= math.exp(sum(self.w.get((t, c.chunk), 0.0) for t in topics))
        d.candidates.sort(key=lambda c: c.adjusted, reverse=True)
        if top_k and len(d.candidates) > top_k:
            d.hidden = len(d.candidates) - top_k
            d.candidates = d.candidates[:top_k]
        return d

    def feedback(self, vocab, prompt, answer, lr=0.15, window=12,
                 temperature=0.8, terminal="?", clip=6.0):
        """One teacher-forced gradient step over a (prompt, correct answer)
        pair. Returns (loss, unreachable_steps)."""
        seq = vocab.tokenize(wrap_prompt(prompt, terminal))
        target = vocab.tokenize(answer)
        ign = vocab.attention_ignored_ids()
        topics = [t for t in seq if t not in ign]
        cur, loss, unreachable = list(seq), 0.0, 0
        for y in target:
            d = self.next_distribution(cur, topics, vocab, window, temperature, top_k=0)
            probs = d.probabilities()
            if y in probs:
                loss += -math.log(max(probs[y], 1e-12))
                for t in set(topics):
                    for c, p in probs.items():
                        g = p - (1.0 if c == y else 0.0)
                        k = (t, c)
                        self.w[k] = max(-clip, min(clip, self.w.get(k, 0.0) - lr * g))
            else:
                unreachable += 1  # the graph has no edge for it: w can't help
            cur.append(y)
        return loss, unreachable


# ------------------------------------------------------------ experiment
HARD = {
    "australia": ("sydney", "canberra"), "turkey": ("istanbul", "ankara"),
    "brazil": ("rio", "brasilia"), "switzerland": ("zurich", "bern"),
    "morocco": ("casablanca", "rabat"), "nigeria": ("lagos", "abuja"),
}
CLEAN = {"spain": "madrid", "france": "paris", "italy": "rome", "germany": "berlin",
         "japan": "tokyo", "egypt": "cairo", "canada": "ottawa", "peru": "lima"}


def corpus():
    L = []
    for c, city in CLEAN.items():
        for _ in range(2):
            L.append(f"[user] What is the capital of {c}? [ai] The capital of {c} is {city}.")
        L.append(f"[user] Capital of {c}? [ai] It is {city}.")
    for c, (wrong, right) in HARD.items():
        for _ in range(3):
            L.append(f"[user] What is the capital of {c}? [ai] The capital of {c} is {wrong}.")
        L.append(f"[user] What is the capital of {c}? [ai] The capital of {c} is {right}.")
    for _ in range(20):
        L.append("[user] Hello. [ai] Hello, how can I help you today?")
    return "\n".join(L)


def accuracy(brain, table, prompt_fmt, runs=30):
    hits = 0
    for c, ans in table.items():
        ans = ans[1] if isinstance(ans, tuple) else ans
        hits += sum(ans in brain.reply(prompt_fmt.format(c), rng=random.Random(s), terminal="?")
                    for s in range(runs))
    return hits / (len(table) * runs)


def report(brain, tag):
    print(f"  {tag:<12} hard(seen phrasing) {accuracy(brain, HARD, 'What is the capital of {}?'):6.1%}"
          f" | hard(rephrased) {accuracy(brain, HARD, 'Tell me the capital of {}?'):6.1%}"
          f" | clean controls {accuracy(brain, CLEAN, 'What is the capital of {}?'):6.1%}")


if __name__ == "__main__":
    brain = Brain(engine=LearnableEngine())
    brain.absorb_text(corpus())
    print("Frequency alone (3 wrong vs 1 right per hard country):")
    report(brain, "before")
    train = [(f"What is the capital of {c}?", f"The capital of {c} is {right}.")
             for c, (_, right) in HARD.items()]
    for epoch in range(1, 9):
        total_loss, unreach = 0.0, 0
        for p, a in train:
            l, u = brain.engine.feedback(brain.vocab, p, a)
            total_loss += l
            unreach += u
        if epoch in (1, 2, 4, 8):
            print(f"  epoch {epoch}: loss {total_loss / len(train):.3f}, "
                  f"unreachable steps {unreach}, learned weights {len(brain.engine.w)}")
    print("After error-driven correction:")
    report(brain, "after")
    v = brain.vocab
    node = brain.engine.root.synapses[v.word_to_id["australia"]].synapses[v.word_to_id["is"]]
    print("\nAustralia 'is' -> sydney/canberra raw counts (never edited):",
          node.synapses[v.word_to_id["sydney"]].frequency, "/",
          node.synapses[v.word_to_id["canberra"]].frequency)
    print("\nUnreachable fact (never observed, so no edge exists):")
    l, u = brain.engine.feedback(v, "What is the capital of kenya?", "The capital of kenya is nairobi.")
    print(f"  feedback on kenya->nairobi: unreachable steps = {u} (weights cannot create edges)")
