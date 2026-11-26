"""
agi.py - Python port of the Deep Attention Neuron-Synapse Network.

Rebuilt from the Go sources in the Gemini chat (agi.go + the final
neurosymbolic.go library). Same architecture:

  1. Vocabulary / tokenizer      punctuation = real tokens, [user]/[ai] kept
  2. Neuron-synapse trie         sliding-window "ReadBook" learning
  3. Causal masking              longest-suffix backoff to a valid node
  4. Fuzzy generalization        Jaccard overlap of outgoing synapses
  5. Deep attention radar        BFS multi-hop co-occurrence gravity
  6. Autoregressive sampling     log-frequency base, temperature, top-k, roulette
  7. X-ray inspector             uses the SAME scoring path as generation

Design note: generation and inspection both call `next_distribution()`, so
`inspect` can never again disagree with what the engine actually does
(that was Bug 1 in the chat).
"""

import math
import os
import pickle
import random
import re
from collections import deque

# --------------------------------------------------------------------------
# 1. DATA ENCODING
# --------------------------------------------------------------------------
CLEAN_RE = re.compile(r"[^a-z0-9\s.,?!:;\[\]]+")
PUNCTUATION = [".", ",", "?", "!", ":", ";"]
CONTROL_TOKENS = ["[user]", "[ai]"]
TERMINALS = {".", "?", "!"}

# The Go source clipped this list ("... w == \"of\" || w == ..."), so this is a
# reasonable reconstruction: structural words that follow almost every noun.
STOPWORDS = {
    "the", "of", "and", "a", "an", "to", "in", "is", "it", "for", "on",
    "with", "that", "this", "as", "at", "by", "or", "are", "be",
}


class Vocabulary:
    def __init__(self, merge_terminals=False):
        # merge_terminals: treat "?" and "!" as "." so a prompt typed with a
        # different terminal still reaches the paths learned from the data.
        # (Cost: replies print "." where the data had "?"/"!".)
        self.merge_terminals = merge_terminals
        self.word_to_id = {}
        self.id_to_word = {}
        self.next_id = 1

    def tokenize(self, sentence):
        clean = sentence.lower()
        for p in PUNCTUATION:
            clean = clean.replace(p, f" {p} ")
        clean = clean.replace("-", " ")
        clean = CLEAN_RE.sub("", clean)
        seq = []
        merge = getattr(self, "merge_terminals", False)
        for word in clean.split():
            if merge and word in ("?", "!"):
                word = "."
            wid = self.word_to_id.get(word)
            if wid is None:
                wid = self.next_id
                self.word_to_id[word] = wid
                self.id_to_word[wid] = word
                self.next_id += 1
            seq.append(wid)
        return seq

    def decode(self, seq, smart=True):
        words = [self.id_to_word.get(i, "?") for i in seq]
        text = " ".join(words)
        if smart:  # "garden ." -> "garden."
            text = re.sub(r"\s+([.,?!:;])", r"\1", text)
        return text

    def ids_for(self, words):
        return {self.word_to_id[w] for w in words if w in self.word_to_id}

    def fuzzy_ignored_ids(self):
        """Punctuation, control tokens and stopwords: structural noise that
        must not count as 'shared meaning' (the spain == cupcakes bug)."""
        return self.ids_for(PUNCTUATION + CONTROL_TOKENS + sorted(STOPWORDS))

    def attention_ignored_ids(self):
        """Tokens that must not act as attention topics."""
        return self.ids_for(PUNCTUATION + CONTROL_TOKENS)

    def stop_ids(self):
        return self.ids_for(sorted(TERMINALS) + CONTROL_TOKENS)

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)


# --------------------------------------------------------------------------
# 2. GRAPH TOPOLOGY
# --------------------------------------------------------------------------
class Neuron:
    __slots__ = ("chunk", "frequency", "synapses")

    def __init__(self, chunk, frequency=0):
        self.chunk = chunk
        self.frequency = frequency
        self.synapses = {}


class Candidate:
    __slots__ = ("chunk", "base", "attn", "bridges", "final", "adjusted")

    def __init__(self, chunk, base):
        self.chunk = chunk
        self.base = base
        self.attn = 1.0
        self.bridges = []
        self.final = 0.0
        self.adjusted = 0.0


class Distribution:
    """Everything the engine computed for one next-token decision."""

    def __init__(self):
        self.valid_neuron = None
        self.matched_context = []
        self.twin = None
        self.similarity = 0.0
        self.candidates = []  # sorted by adjusted score, top-k applied
        self.hidden = 0       # how many were cut by top-k

    def probabilities(self):
        total = sum(c.adjusted for c in self.candidates)
        if total <= 0:
            return {}
        return {c.chunk: c.adjusted / total for c in self.candidates}


class GraphEngine:
    def __init__(self):
        self.root = Neuron(0)
        self.topic_gain = {}  # word id -> learned attention gain (default 1.0)

    # ---------------- learning ----------------
    def observe_sequence(self, seq):
        self.root.frequency += 1
        cur = self.root
        for chunk in seq:
            child = cur.synapses.get(chunk)
            if child is not None:
                child.frequency += 1
            else:
                child = Neuron(chunk, 1)
                cur.synapses[chunk] = child
            cur = child

    def read_book(self, full_seq, window_size=12):
        """Slide a window across the whole text, wiring every position."""
        if len(full_seq) < window_size:
            self.observe_sequence(full_seq)
            return
        for i in range(len(full_seq) - window_size + 1):
            self.observe_sequence(full_seq[i:i + window_size])
        for i in range(len(full_seq) - window_size + 1, len(full_seq)):
            self.observe_sequence(full_seq[i:])

    # ---------------- forgetting ----------------
    def prune_unused(self, min_frequency):
        self._prune(self.root, min_frequency)

    def _prune(self, neuron, min_frequency):
        for chunk, child in list(neuron.synapses.items()):
            if child.frequency < min_frequency:
                del neuron.synapses[chunk]
            else:
                self._prune(child, min_frequency)

    def sleep(self):
        """Fixed Sleep: forget weak transient paths only. The old version
        merged neighbouring vocab IDs (garden ~ sandwich) - never again."""
        self.prune_unused(2)

    # ---------------- fuzzy generalization ----------------
    @staticmethod
    def fuzzy_similarity(a, b, ignored):
        if a is None or b is None or not a.synapses or not b.synapses:
            return 0.0
        a_keys = a.synapses.keys() - ignored
        return GraphEngine._jaccard_with(a_keys, b, ignored)

    @staticmethod
    def _jaccard_with(a_keys, b, ignored):
        b_all = b.synapses.keys()
        inter = len(a_keys & b_all)  # a_keys already has ignored removed
        b_size = len(b_all) - len(b_all & ignored)
        union = len(a_keys) + b_size - inter
        return inter / union if union else 0.0

    def find_fuzzy_match(self, target, min_similarity, ignored):
        if target is None or not target.synapses:
            return None, 0.0
        a_keys = target.synapses.keys() - ignored
        best, best_score = None, 0.0
        for cand in self.root.synapses.values():
            if cand.chunk == target.chunk or not cand.synapses:
                continue
            score = self._jaccard_with(a_keys, cand, ignored)
            if score > best_score and score >= min_similarity:
                best, best_score = cand, score
        return best, best_score

    # ---------------- deep attention radar ----------------
    def deep_search(self, start, target_chunk, max_depth):
        """BFS radar: best (freq / depth) link to target_chunk reachable
        from `start` within max_depth hops. Returns (score, depth)."""
        score, depth = self.attention_map(start, max_depth).get(target_chunk, (0.0, 0))
        return score, depth

    def attention_map(self, start, max_depth):
        """One BFS that scores EVERY reachable chunk at once. Identical
        result to calling deep_search per target, but one pass."""
        best = {}
        if start is None:
            return best
        queue = deque([(start, 1)])
        visited = {start.chunk}
        while queue:
            node, depth = queue.popleft()
            decay = 1.0 / depth
            for chunk, link in node.synapses.items():
                score = link.frequency * decay
                prev = best.get(chunk)
                if prev is None or score > prev[0]:
                    best[chunk] = (score, depth)
                if depth < max_depth and chunk not in visited:
                    visited.add(chunk)
                    queue.append((link, depth + 1))
        return best

    # ---------------- prediction core ----------------
    def _find_valid_neuron(self, seq, window_size):
        max_ctx = max(window_size - 1, 1)
        for start in range(len(seq)):
            ctx = seq[start:]
            if len(ctx) > max_ctx:
                continue
            cur = self.root
            ok = True
            for chunk in ctx:
                nxt = cur.synapses.get(chunk)
                if nxt is None:
                    ok = False
                    break
                cur = nxt
            if ok and cur.synapses:
                return cur, ctx
        return None, []

    def next_distribution(self, seq, topic_tokens, vocab, window_size=12,
                          temperature=0.8, top_k=5, attention_gain=15.0,
                          attention_depth=6, fuzzy_min=0.15, _attn_cache=None):
        d = Distribution()
        d.valid_neuron, d.matched_context = self._find_valid_neuron(seq, window_size)
        if d.valid_neuron is None:
            return d

        base = {}
        for chunk, child in d.valid_neuron.synapses.items():
            base[chunk] = 1.0 + math.log(child.frequency)

        d.twin, d.similarity = self.find_fuzzy_match(
            d.valid_neuron, fuzzy_min, vocab.fuzzy_ignored_ids())
        if d.twin is not None:
            for chunk, child in d.twin.synapses.items():
                base[chunk] = base.get(chunk, 0.0) + \
                    (1.0 + math.log(child.frequency)) * d.similarity

        if _attn_cache is None:
            _attn_cache = {}
        maps = []
        for t in topic_tokens:
            if t not in _attn_cache:
                node = self.root.synapses.get(t)
                _attn_cache[t] = self.attention_map(node, attention_depth) if node else {}
            maps.append((t, _attn_cache[t]))

        cands = []
        for chunk, b in base.items():
            c = Candidate(chunk, b)
            for t, amap in maps:
                hit = amap.get(chunk)
                if hit and hit[0] > 0:
                    m = hit[0] * attention_gain * getattr(self, "topic_gain", {}).get(t, 1.0)
                    c.attn += m
                    c.bridges.append((t, m, hit[1]))
            c.final = c.base * c.attn
            c.adjusted = c.final ** (1.0 / temperature) \
                if temperature not in (1.0, 0) and temperature > 0 else c.final
            cands.append(c)

        cands.sort(key=lambda c: c.adjusted, reverse=True)
        if top_k and len(cands) > top_k:
            d.hidden = len(cands) - top_k
            cands = cands[:top_k]
        d.candidates = cands
        return d

    def predict_sequence(self, input_seq, vocab, length=30, window_size=12,
                         temperature=0.8, top_k=5, attention_gain=15.0,
                         attention_depth=6, rng=None, stop_ids=None, fuzzy_min=0.15):
        """Autoregressive generation. Returns the generated token IDs."""
        if not input_seq:
            return []
        rng = rng or random
        stop_ids = vocab.stop_ids() if stop_ids is None else stop_ids
        ignored_attn = vocab.attention_ignored_ids()
        topics = [t for t in input_seq if t not in ignored_attn]

        cur = list(input_seq)
        out = []
        cache = {}
        for _ in range(length):
            d = self.next_distribution(cur, topics, vocab, window_size, temperature,
                                       top_k, attention_gain, attention_depth,
                                       fuzzy_min, _attn_cache=cache)
            if not d.candidates:
                break
            total = sum(c.adjusted for c in d.candidates)
            target = rng.random() * total
            run, pick = 0.0, d.candidates[-1].chunk
            for c in d.candidates:
                run += c.adjusted
                if run >= target:
                    pick = c.chunk
                    break
            if pick in stop_ids:
                break
            out.append(pick)
            cur.append(pick)
        return out

    # ---------------- X-ray ----------------
    def inspect(self, seq, vocab, window_size=12, temperature=0.8, top_k=5,
                attention_gain=15.0, attention_depth=6):
        ignored_attn = vocab.attention_ignored_ids()
        topics = [t for t in seq if t not in ignored_attn]
        L = [f'[X-RAY ENGINE TRACE FOR PROMPT]: "{vocab.decode(seq, smart=False)}"',
             f"   -> Active Attention Tokens: {[vocab.id_to_word[t] for t in topics]}"]
        d = self.next_distribution(seq, topics, vocab, window_size, temperature,
                                   top_k=0, attention_gain=attention_gain,
                                   attention_depth=attention_depth)
        if d.valid_neuron is None:
            L += ["[STEP 1: CAUSAL MASKING]",
                  "   -> [DEAD END] No valid continuation found."]
            return "\n".join(L)
        L += ["[STEP 1: CAUSAL MASKING]",
              f'   -> Backed off to valid structure: "{vocab.decode(d.matched_context, smart=False)}"',
              f"   -> Node frequency: {d.valid_neuron.frequency}",
              f"   -> Legal next words: {len(d.valid_neuron.synapses)}"]
        if d.twin is not None:
            L += ["[STEP 1.5: FUZZY GENERALIZATION ACTIVATED]",
                  f"   -> [{vocab.id_to_word[d.valid_neuron.chunk]}] overlaps with "
                  f"[{vocab.id_to_word[d.twin.chunk]}] by {d.similarity * 100:.1f}%"]
        L.append("[STEP 2 & 3: SYNTACTIC POOL + DEEP ATTENTION GRAVITY]")
        for i, c in enumerate(d.candidates):
            if i >= 15:
                L.append(f"   ... and {len(d.candidates) - 15} more hidden")
                break
            br = ""
            if c.bridges:
                br = " | Bridges: " + ", ".join(
                    f"{vocab.id_to_word[t]}(+{m:.1f}@hop{h})" for t, m, h in c.bridges)
            L.append(f"   --> [{vocab.id_to_word[c.chunk]}] base {c.base:.2f} | "
                     f"attn x{c.attn:.1f} | final {c.final:.1f}{br}")
        top = d.candidates[:top_k] if top_k else d.candidates
        tot = sum(c.adjusted for c in top)
        L.append(f"[STEP 4: TEMPERATURE ({temperature:.2f}) & TOP-K ({top_k})]")
        for c in top:
            L.append(f"   --> [{vocab.id_to_word[c.chunk]}] {c.adjusted / tot * 100:.1f}%")
        return "\n".join(L)

    # ---------------- persistence ----------------
    def save(self, filename):
        import sys
        old = sys.getrecursionlimit()
        sys.setrecursionlimit(max(old, 10000))
        try:
            with open(filename, "wb") as f:
                pickle.dump({"root": self.root, "topic_gain": self.topic_gain}, f)
        finally:
            sys.setrecursionlimit(old)

    @staticmethod
    def load(filename):
        import sys
        old = sys.getrecursionlimit()
        sys.setrecursionlimit(max(old, 10000))
        try:
            with open(filename, "rb") as f:
                eng = GraphEngine()
                obj = pickle.load(f)
                if isinstance(obj, dict):
                    eng.root, eng.topic_gain = obj["root"], obj["topic_gain"]
                else:  # old format: root only
                    eng.root = obj
                return eng
        finally:
            sys.setrecursionlimit(old)

    def stats(self):
        nodes, edges = 0, 0
        stack = [self.root]
        while stack:
            n = stack.pop()
            for ch in n.synapses.values():
                nodes += 1
                stack.append(ch)
        return {"nodes": nodes, "root_words": len(self.root.synapses)}


# --------------------------------------------------------------------------
# Chat wrapper (what main.go did around GenerateText)
# --------------------------------------------------------------------------
def wrap_prompt(text, terminal="."):
    """Wrap a prompt in the control tokens. If the user typed no terminal
    punctuation one is appended (default "." like main.go). NOTE: the engine
    treats "." and "?" as different tokens, so pass terminal="?" for questions
    if your training data ends questions with "?"."""
    text = text.strip()
    if not text.endswith(tuple(TERMINALS)):
        text += terminal
    return f"[user] {text} [ai]"


def clean_text_data(raw):
    """Port of cleanTextData: one thought per line, hard terminal walls."""
    out = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t", 1)
        keep = parts[1].strip() if len(parts) == 2 else line
        if not keep:
            continue
        if not keep.endswith((".", "?", "!")):
            keep += "."
        out.append(keep)
    return " ".join(out) + " "


class Brain:
    """Engine + vocabulary bundled, with disk persistence."""

    def __init__(self, vocab=None, engine=None, merge_terminals=False):
        self.vocab = vocab or Vocabulary(merge_terminals)
        self.engine = engine or GraphEngine()

    def absorb_text(self, raw, window=12):
        self.engine.read_book(self.vocab.tokenize(clean_text_data(raw)), window)

    def absorb_file(self, path, window=12):
        with open(path, encoding="utf-8") as f:
            self.absorb_text(f.read(), window)

    def absorb_folder(self, folder, window=12):
        n = 0
        for name in sorted(os.listdir(folder)):
            if name.endswith(".txt"):
                self.absorb_file(os.path.join(folder, name), window)
                n += 1
        return n

    def reply(self, text, length=30, window=12, temperature=0.8, top_k=5, rng=None,
              terminal="."):
        seq = self.vocab.tokenize(wrap_prompt(text, terminal))
        out = self.engine.predict_sequence(seq, self.vocab, length, window,
                                           temperature, top_k, rng=rng)
        return self.vocab.decode(out)

    def inspect(self, text, **kw):
        return self.engine.inspect(self.vocab.tokenize(wrap_prompt(text)), self.vocab, **kw)

    def save(self, vocab_path="vocab.bin", net_path="network.bin"):
        self.vocab.save(vocab_path)
        self.engine.save(net_path)

    @staticmethod
    def load(vocab_path="vocab.bin", net_path="network.bin"):
        return Brain(Vocabulary.load(vocab_path), GraphEngine.load(net_path))
