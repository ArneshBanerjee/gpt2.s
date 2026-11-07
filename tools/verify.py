"""Check ./gpt2 against Hugging Face GPT-2 (PyTorch, fp32, greedy).

1. Tokenizer: prompt token ids must match exactly.
2. Model: greedy generated token ids must match exactly.
"""
import subprocess
import sys

import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

TOKENIZER_CASES = [
    "Hello world",
    "The meaning of life is",
    "I'm sure they'll say it's fine, we've done this.",
    "   leading spaces and   gaps   ",
    "line one\nline two\n\n  indented",
    "numbers 12345 and 3.14159, plus $100!!!",
    "tabs\tand\ttabs  \t mixed",
    "CamelCaseWords and snake_case_words",
    "'quoted' \"double\" (parens) [brackets] {braces}",
    "x",
    "  ",
    "a\n",
]
GEN_CASES = [
    "The meaning of life is",
    "Alan Turing theorized that computers would one day become",
    "In a shocking finding, scientists discovered a herd of unicorns",
    "def fibonacci(n):",
]
N_GEN = 40


def run_asm(prompt, n):
    out = subprocess.run(["./gpt2", "-i", "-t", "0", "-n", str(n), prompt],
                         capture_output=True, text=True, check=True).stdout
    lines = out.split("\n")
    return [int(t) for t in lines[0].split()], [int(t) for t in lines[1].split()]


def main():
    tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").eval()
    ok = True

    for text in TOKENIZER_CASES:
        want = tok.encode(text)
        got, _ = run_asm(text, 1)
        match = got == want
        ok &= match
        print(f"tok  {'PASS' if match else 'FAIL'}  {text!r}")
        if not match:
            print("     want", want, "\n     got ", got)

    for text in GEN_CASES:
        ids = tok.encode(text)
        with torch.no_grad():
            out = model.generate(torch.tensor([ids]), max_new_tokens=N_GEN, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        want = out[0, len(ids):].tolist()
        _, got = run_asm(text, N_GEN)
        same = next((i for i, (a, b) in enumerate(zip(want, got)) if a != b), len(want))
        match = got == want
        ok &= match
        print(f"gen  {'PASS' if match else 'FAIL'}  {same}/{N_GEN} tokens match  {text!r}")
        if not match:
            print("     want", tok.decode(want)[:200])
            print("     got ", tok.decode(got)[:200])

    print("ALL PASS" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
