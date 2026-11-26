"""main.py - interactive CLI (port of main.go). Run: python main.py"""

import os
import signal
import sys

from agi import Brain

VOCAB, NET = "vocab.bin", "network.bin"


def load_or_blank():
    if os.path.exists(VOCAB) and os.path.exists(NET):
        print("--- PERSISTENT MEMORY LOADED FROM DISK ---")
        return Brain.load(VOCAB, NET)
    print("--- NO SAVED MEMORY FOUND. INITIALIZING BLANK SLATE ---")
    return Brain()


def main():
    brain = load_or_blank()

    def save_and_exit(*_):
        print("\n[CTRL+C DETECTED] Saving memory...")
        brain.save(VOCAB, NET)
        print("[MEMORY SAVED] Engine shut down successfully.")
        sys.exit(0)

    signal.signal(signal.SIGINT, save_and_exit)
    signal.signal(signal.SIGTERM, save_and_exit)

    print("=======================================")
    print(" DEEP ATTENTION NEURON-SYNAPSE NETWORK ")
    print("=======================================")
    print("Commands:")
    print("  <anything>           converse / generate")
    print("  inspect <phrase>     X-ray the math for the next word")
    print("  absorb_file <f.txt>  wire new knowledge")
    print("  absorb_folder <dir>  bulk-wire a folder of .txt files")
    print("  sleep                forget weak paths")
    print("  prune <n>            sever synapses with frequency < n")
    print("  stats                graph size")
    print("  exit                 save and quit")

    while True:
        try:
            line = input("\nPrompt > ").strip()
        except EOFError:
            break
        low = line.lower()
        if low == "exit":
            print("Saving persistent memory to disk...")
            brain.save(VOCAB, NET)
            print("Engine shut down successfully.")
            break
        elif low.startswith("absorb_file "):
            path = line[12:].strip()
            if not os.path.exists(path):
                print(f"Error: could not read file '{path}'.")
                continue
            print(f"--- CLEANING AND ABSORBING: {path} ---")
            brain.absorb_file(path)
            print("[KNOWLEDGE ABSORBED AND WIRED INTO MEMORY]")
        elif low.startswith("absorb_folder "):
            folder = line[14:].strip()
            if not os.path.isdir(folder):
                print(f"Error: could not read directory '{folder}'.")
                continue
            n = brain.absorb_folder(folder)
            print(f"[BATCH COMPLETE: {n} FILES WIRED INTO MEMORY]")
        elif low.startswith("inspect "):
            phrase = line[8:].strip()
            if phrase:
                print(brain.inspect(phrase))
        elif low == "sleep":
            print("[Zzz... SLEEP CYCLE: pruning weak transient thoughts ...]")
            brain.engine.sleep()
        elif low.startswith("prune "):
            try:
                brain.engine.prune_unused(int(line[6:].strip()))
            except ValueError:
                print("Invalid threshold. Use: prune <number>")
        elif low == "stats":
            print(brain.engine.stats(), "vocab:", len(brain.vocab.word_to_id))
        elif line:
            print("\n[AI GENERATING]:", brain.reply(line))


if __name__ == "__main__":
    main()
