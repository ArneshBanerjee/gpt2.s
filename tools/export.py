"""Convert GPT-2 small from Hugging Face into two flat files for the asm engine.

gpt2_124M.bin  fp32 weights, every matrix stored [out][in] (row = one output)
gpt2_tok.bin   BPE tables: byte->id, merge hash table, id->bytes
"""
import struct
import numpy as np
import json
from huggingface_hub import hf_hub_download
from transformers import GPT2LMHeadModel

HASH_BITS = 17
EMPTY = 0xFFFFFFFF


def export_weights(path):
    sd = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").state_dict()
    f32 = lambda t: t.detach().float().numpy().astype("<f4")
    parts = [f32(sd["transformer.wte.weight"]), f32(sd["transformer.wpe.weight"])]
    for i in range(12):
        p = f"transformer.h.{i}."
        # HF Conv1D weights are [in][out]; transpose so each output is a contiguous row.
        parts += [
            f32(sd[p + "ln_1.weight"]), f32(sd[p + "ln_1.bias"]),
            f32(sd[p + "attn.c_attn.weight"]).T, f32(sd[p + "attn.c_attn.bias"]),
            f32(sd[p + "attn.c_proj.weight"]).T, f32(sd[p + "attn.c_proj.bias"]),
            f32(sd[p + "ln_2.weight"]), f32(sd[p + "ln_2.bias"]),
            f32(sd[p + "mlp.c_fc.weight"]).T, f32(sd[p + "mlp.c_fc.bias"]),
            f32(sd[p + "mlp.c_proj.weight"]).T, f32(sd[p + "mlp.c_proj.bias"]),
        ]
    parts += [f32(sd["transformer.ln_f.weight"]), f32(sd["transformer.ln_f.bias"])]
    with open(path, "wb") as f:
        f.write(struct.pack("<4s7I", b"GPT2", 12, 12, 768, 1024, 50257, 0, 0) + bytes(32))
        for a in parts:
            f.write(np.ascontiguousarray(a).tobytes())
    print(path, sum(a.size for a in parts), "floats")


def bytes_to_unicode():
    # GPT-2's reversible byte -> printable unicode char mapping.
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def export_tokenizer(path):
    enc = json.load(open(hf_hub_download("openai-community/gpt2", "vocab.json"), encoding="utf-8"))
    lines = open(hf_hub_download("openai-community/gpt2", "merges.txt"), encoding="utf-8").read().split("\n")
    merges = [tuple(l.split(" ")) for l in lines[1:] if l]
    b2u = bytes_to_unicode()
    byte2id = [enc[b2u[b]] for b in range(256)]
    table = [(EMPTY, 0, 0)] * (1 << HASH_BITS)
    for rank, (a, b) in enumerate(merges):
        key = (enc[a] << 16) | enc[b]
        h = ((key * 0x9E3779B1) & 0xFFFFFFFF) >> (32 - HASH_BITS)
        while table[h][0] != EMPTY:
            h = (h + 1) & ((1 << HASH_BITS) - 1)
        table[h] = (key, rank, enc[a + b])
    n = len(enc)
    u2b = {c: b for b, c in b2u.items()}
    dec = {i: s for s, i in enc.items()}
    blobs = [bytes(u2b[c] for c in dec[i]) for i in range(n)]
    offs = np.cumsum([0] + [len(b) for b in blobs]).astype("<u4")
    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", b"TOK2", n, HASH_BITS) + bytes(4))
        f.write(struct.pack("<256I", *byte2id))
        f.write(b"".join(struct.pack("<3I", *e) for e in table))
        f.write(offs.tobytes())
        f.write(b"".join(blobs))
    print(path, n, "tokens", len(merges), "merges")


if __name__ == "__main__":
    export_weights("gpt2_124M.bin")
    export_tokenizer("gpt2_tok.bin")
