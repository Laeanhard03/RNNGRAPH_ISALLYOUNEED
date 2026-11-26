"""build_dataset.py - rebuild data/chatterbot_pairs.json (works where Hugging Face is blocked).
Downloads the ChatterBot English corpus from GitHub, turns each conversation into
consecutive (question, answer) pairs, de-duplicates, and splits by question hash
(85% train / 15% test, so a question never lands in both).
Needs: pip install pyyaml"""

import glob, hashlib, io, json, os, tarfile, urllib.request

import yaml

URL = "https://codeload.github.com/gunthercox/chatterbot-corpus/tar.gz/refs/heads/master"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "chatterbot_pairs.json")


def main():
    print("Downloading ChatterBot corpus from GitHub...")
    blob = urllib.request.urlopen(URL, timeout=60).read()
    pairs = []
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        for m in sorted(tar.getmembers(), key=lambda m: m.name):
            if "/data/english/" in m.name and m.name.endswith(".yml"):
                topic = os.path.basename(m.name)[:-4]
                doc = yaml.safe_load(tar.extractfile(m).read().decode("utf-8"))
                for conv in doc.get("conversations", []):
                    conv = [str(x).strip() for x in conv if x is not None and str(x).strip()]
                    pairs += [(q, a, topic) for q, a in zip(conv, conv[1:])]
    seen, uniq = set(), []
    for p in pairs:
        if (p[0], p[1]) not in seen:
            seen.add((p[0], p[1])); uniq.append(p)
    train, test = [], []
    for q, a, t in uniq:
        h = int(hashlib.md5(q.lower().encode()).hexdigest(), 16) % 100
        (test if h < 15 else train).append((q, a, t))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"train": train, "test": test}, open(OUT, "w"))
    print(f"{len(pairs)} pairs, {len(uniq)} unique -> train {len(train)} / test {len(test)}  ({OUT})")


if __name__ == "__main__":
    main()
