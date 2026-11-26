"""test_agi.py - run with `python test_agi.py` (also pytest-compatible)."""

import os
import random
import tempfile
from collections import deque

from agi import Brain, GraphEngine, Vocabulary, wrap_prompt

CAPITALS = {
    "spain": "madrid", "france": "paris", "italy": "rome", "germany": "berlin",
    "japan": "tokyo", "egypt": "cairo", "canada": "ottawa", "peru": "lima",
    "norway": "oslo", "greece": "athens",
}


def build_corpus():
    lines = []
    for country, city in CAPITALS.items():
        lines.append(f"[user] What is the capital of {country}? [ai] The capital of {country} is {city}.")
        lines.append(f"[user] Capital of {country}? [ai] It is {city}.")
        lines.append(f"[user] Can you tell me the capital of {country}? [ai] Sure, the capital of {country} is {city}.")
    lines.append("[user] How long do I boil potatoes? [ai] The cooking time for boiled potatoes is twenty minutes.")
    lines.append("[user] How do I cook rice? [ai] The cooking time for white rice is fifteen minutes.")
    lines.append("[user] Best way to cook pasta? [ai] The cooking time for pasta is ten minutes.")
    for _ in range(3):
        lines.append("[user] How do I start a garden? [ai] Pick a sunny spot and prepare the soil.")
        lines.append("[user] How do I make a breakfast sandwich? [ai] Toast the bread and add egg and cheese.")
    for _ in range(25):
        lines.append("[user] Hello. [ai] Hello, how can I help you today?")
        lines.append("[user] Good morning. [ai] Good morning, how are you?")
    lines.append("[user] Do you like cupcakes? [ai] I like cupcakes and cake.")
    return "\n".join(lines)


def make_brain():
    b = Brain()
    b.absorb_text(build_corpus())
    return b


def test_tokenizer():
    v = Vocabulary()
    seq = v.tokenize("[user] Hello, world! [ai]")
    words = [v.id_to_word[i] for i in seq]
    assert words == ["[user]", "hello", ",", "world", "!", "[ai]"], words
    assert v.decode(seq) == "[user] hello, world! [ai]"


def test_observe_counts():
    e = GraphEngine()
    e.observe_sequence([1, 2, 3])
    e.observe_sequence([1, 2, 4])
    assert e.root.synapses[1].frequency == 2
    assert e.root.synapses[1].synapses[2].synapses[3].frequency == 1
    assert e.root.synapses[1].synapses[2].synapses[4].frequency == 1


def test_causal_masking_backoff():
    b = make_brain()
    seq = b.vocab.tokenize("zzzunknown the capital of spain is")
    node, ctx = b.engine._find_valid_neuron(seq, 12)
    assert node is not None and b.vocab.decode(ctx) == "the capital of spain is"
    node, ctx = b.engine._find_valid_neuron(b.vocab.tokenize("qqq"), 12)
    assert node is None


def test_fuzzy_no_punctuation_poisoning():
    b = make_brain()
    v, e = b.vocab, b.engine
    spain = e.root.synapses[v.word_to_id["spain"]]
    cupcakes = e.root.synapses[v.word_to_id["cupcakes"]]
    ignored = v.fuzzy_ignored_ids()
    filtered = e.fuzzy_similarity(spain, cupcakes, ignored)
    raw = e.fuzzy_similarity(spain, cupcakes, set())
    assert filtered < 0.15, filtered
    assert raw > filtered, (raw, filtered)


def test_capital_of_spain_not_potatoes():
    b = make_brain()
    hits, runs = 0, 40
    for seed in range(runs):
        out = b.reply("capital of spain?", rng=random.Random(seed))
        assert "potato" not in out and "cooking" not in out, out
        hits += "madrid" in out
    assert hits / runs >= 0.9, hits / runs


def test_all_capitals():
    b = make_brain()
    ok = 0
    for country, city in CAPITALS.items():
        outs = [b.reply(f"What is the capital of {country}?", rng=random.Random(s)) for s in range(10)]
        ok += sum(city in o for o in outs) >= 9
    assert ok == len(CAPITALS), ok


def test_unseen_phrasing_generalises_via_backoff():
    b = make_brain()
    out = b.reply("tell me the capital of japan?", rng=random.Random(1))
    assert "tokyo" in out, out


def test_inspect_matches_generation():
    b = make_brain()
    seq = b.vocab.tokenize(wrap_prompt("capital of spain?"))
    topics = [t for t in seq if t not in b.vocab.attention_ignored_ids()]
    d = b.engine.next_distribution(seq, topics, b.vocab, temperature=0.8, top_k=5)
    probs = d.probabilities()
    assert abs(sum(probs.values()) - 1) < 1e-9
    report = b.inspect("capital of spain?")
    assert "[X-RAY" in report and "CAUSAL MASKING" in report
    first = b.vocab.tokenize(b.reply("capital of spain?", rng=random.Random(3)))[0]
    assert first in probs


def test_sleep_does_not_merge_neighbouring_ids():
    b = make_brain()
    v, e = b.vocab, b.engine
    garden, sandwich = v.word_to_id["garden"], v.word_to_id["sandwich"]
    before = len(e.root.synapses[garden].synapses)
    e.sleep()
    g = e.root.synapses[garden]
    assert v.word_to_id["sandwich"] not in g.synapses
    assert len(g.synapses) <= before
    out = b.reply("how do I start a garden", rng=random.Random(0))
    assert "sandwich" not in out and "egg" not in out, out


def test_attention_map_equals_literal_bfs():
    b = make_brain()
    e = b.engine

    def literal(start, target, max_depth):
        q = deque([(start, 1)])
        seen = {start.chunk}
        best, bd = 0.0, 0
        while q:
            node, depth = q.popleft()
            link = node.synapses.get(target)
            if link is not None:
                s = link.frequency / depth
                if s > best:
                    best, bd = s, depth
            if depth < max_depth:
                for ch, child in node.synapses.items():
                    if ch not in seen:
                        seen.add(ch)
                        q.append((child, depth + 1))
        return best, bd

    start = e.root.synapses[b.vocab.word_to_id["capital"]]
    for target in list(b.vocab.id_to_word)[:60]:
        assert e.deep_search(start, target, 6) == literal(start, target, 6)


def test_save_load_roundtrip():
    b = make_brain()
    with tempfile.TemporaryDirectory() as d:
        vp, np_ = os.path.join(d, "v.bin"), os.path.join(d, "n.bin")
        b.save(vp, np_)
        c = Brain.load(vp, np_)
    out1 = b.reply("capital of peru?", rng=random.Random(5))
    out2 = c.reply("capital of peru?", rng=random.Random(5))
    assert out1 == out2 and "lima" in out1


def test_known_weakness_terminal_mismatch():
    b = make_brain()
    rate = lambda p: sum("madrid" in b.reply(p, rng=random.Random(s)) for s in range(40)) / 40
    assert rate("capital of spain?") >= 0.9
    assert rate("capital of spain") < 0.5
    r = sum("madrid" in b.reply("capital of spain", rng=random.Random(s), terminal="?")
            for s in range(40)) / 40
    assert r >= 0.9


def test_merge_terminals_fixes_mismatch():
    b = Brain(merge_terminals=True)
    b.absorb_text(build_corpus())
    rate = lambda p: sum("madrid" in b.reply(p, rng=random.Random(s)) for s in range(40)) / 40
    assert rate("capital of spain") >= 0.9
    assert rate("capital of spain?") >= 0.9
    assert rate("capital of spain!") >= 0.9


def test_prune():
    b = make_brain()
    before = b.engine.stats()["nodes"]
    b.engine.prune_unused(3)
    assert b.engine.stats()["nodes"] < before


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as ex:
            failed += 1
            print(f"FAIL  {name}: {ex}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
