"""hf_tokenized.py - builds hf_tokenized.txt with [user]/[ai] turn tags.
Needs internet + `pip install datasets` (run it on your own machine)."""

from datasets import load_dataset

print("Downloading dataset...")
ds = load_dataset("HuggingFaceTB/everyday-conversations-llama3.1-2k")

with open("hf_tokenized.txt", "w", encoding="utf-8") as f:
    for row in ds["train_sft"]:
        for msg in row["messages"]:
            content = msg["content"].strip()
            if not content.endswith((".", "?", "!")):
                content += "."
            tag = "[user]" if msg["role"] == "user" else "[ai]"
            f.write(f"{tag} {content} ")
        f.write("\n")

print("Extraction complete! Saved to hf_tokenized.txt")
