"""feedback_experiment.py - real-data test of the graph-native feedback loop.
Usage: python feedback_experiment.py [max_questions]
Data: ChatterBot English corpus (data/chatterbot_pairs.json), 70% of questions get
feedback, 30% are held out (never corrected) to measure generalisation."""

import collections, json, random, sys, time
from agi import Brain
from feedback import SelfCorrector


def term(s):
    s = s.strip()
    return s if s.endswith((".", "?", "!")) else s + "."


def swap(q):
    q = term(q)
    return q[:-1] + ("." if q[-1] == "?" else "?")


def load(max_q=None):
    d = json.load(open("data/chatterbot_pairs.json"))
    ok = lambda q, a: "\n" not in a and "```" not in a and len(a.split()) <= 25 and len(q.split()) <= 25
    pairs = [(q, a) for q, a, _ in d["train"] if ok(q, a)]
    if max_q:
        qs = sorted({q for q, _ in pairs})
        keep = set(random.Random(0).sample(qs, min(max_q, len(qs))))
        pairs = [p for p in pairs if p[0] in keep]
    return pairs


def setup(pairs, merge=False):
    b = Brain(merge_terminals=merge)
    b.engine.read_book(b.vocab.tokenize("\n".join(f"[user] {term(q)} [ai] {term(a)}" for q, a in pairs) + "\n"), 12)
    stops = b.vocab.stop_ids()

    def first(a):
        out = []
        for i in b.vocab.tokenize(term(a)):
            if i in stops: break
            out.append(i)
        return tuple(out)

    golds, cnt = collections.defaultdict(set), collections.defaultdict(collections.Counter)
    for q, a in pairs:
        golds[q].add(first(a)); cnt[q][a] += 1
    preferred = {q: c.most_common(1)[0][0] for q, c in cnt.items()}
    return b, golds, preferred


def run(pairs, channels, verbose=False, merge=False):
    b, golds, preferred = setup(pairs, merge)
    sc = SelfCorrector(b, channels=channels) if channels else SelfCorrector(b)
    qs = sorted(golds); random.Random(1).shuffle(qs)
    cut = int(len(qs) * 0.7); fb, hold = qs[:cut], qs[cut:]

    def rec(questions, pert=False, **kw):
        return sum(sc.matches(swap(q) if pert else q, golds[q], **kw) for q in questions) / len(questions)

    def table():
        return (rec(fb), rec(hold), rec(fb, True), rec(hold, True))

    before = table()
    sample_q = hold[:5]
    s_before = {q: b.vocab.decode(list(sc.answer(q))) for q in sample_q}
    log = []
    for rnd in range(1, 5):
        wrong = 0
        for q in fb:
            for prompt in (q, swap(q)):
                if not sc.matches(prompt, golds[q]):
                    wrong += 1
                    sc.correct(prompt, preferred[q])
        log.append(wrong)
        if wrong == 0:
            break
    after = table()
    return b, sc, sample_q, s_before, before, after, log


def main():
    max_q = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else None
    merge = "merge" in sys.argv
    pairs = load(max_q)
    nq = len({q for q, _ in pairs})
    print(f"{len(pairs)} pairs | {nq} questions | 70% feedback / 30% held-out | merge_terminals={merge}")
    rows = []
    configs = [("gain",), ("gain", "counts")] if merge else [("counts",), ("gain",), ("gain", "counts")]
    for channels in configs:
        t = time.time()
        b, sc, sq, sb, before, after, log = run(pairs, channels, merge=merge)
        rows.append((channels, before, after, log, time.time() - t))
        print(f"\n[{'+'.join(channels)}] mistakes per round: {log} ({time.time()-t:.0f}s)")
        if channels == ("gain", "counts"):
            keep = (b, sc, sq, sb)
    print("\n                     exact:feedback  exact:held-out  swapped:feedback  swapped:held-out")
    print(f"  before (any)        {rows[0][1][0]:6.1%}          {rows[0][1][1]:6.1%}          {rows[0][1][2]:6.1%}            {rows[0][1][3]:6.1%}")
    for ch, _, a, _, _ in rows:
        print(f"  after {'+'.join(ch):<12}  {a[0]:6.1%}          {a[1]:6.1%}          {a[2]:6.1%}            {a[3]:6.1%}")
    b, sc, sq, sb = keep
    g, w = b.engine.topic_gain, b.vocab.id_to_word
    lo = sorted(g.items(), key=lambda kv: kv[1])[:10]; hi = sorted(g.items(), key=lambda kv: -kv[1])[:10]
    print("\nlearned gains, quietest:", ", ".join(f"{w[k]} x{v:.2f}" for k, v in lo))
    print("learned gains, loudest :", ", ".join(f"{w[k]} x{v:.2f}" for k, v in hi))
    print("\nHeld-out questions (never corrected), before -> after:")
    for q in sq:
        print(f"  Q: {q[:60]!r}\n     {sb[q]!r} -> {b.vocab.decode(list(sc.answer(q)))!r}")

    b0, golds, _ = setup(pairs, merge)
    sc0 = SelfCorrector(b0)
    allq = sorted(golds)
    r = lambda pert, **kw: sum(sc0.matches(swap(q) if pert else q, golds[q], **kw) for q in allq) / len(allq)
    print("\nAblations, untrained engine, ALL questions (exact / tail-swapped):")
    for name, kw in [("full engine", {}), ("attention off", dict(attention_gain=0.0)),
                     ("fuzzy twin off", dict(fuzzy_min=2.0)),
                     ("both off (pure trie)", dict(attention_gain=0.0, fuzzy_min=2.0))]:
        print(f"  {name:<22} {r(False, **kw):6.1%} / {r(True, **kw):6.1%}")


if __name__ == "__main__":
    main()
