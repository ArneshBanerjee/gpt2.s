# gpt2.s

GPT-2 small (124M parameters) in hand-written AArch64 assembly for Apple Silicon.

The program calls no C code. It calls no libc functions. It does all I/O with raw
Darwin syscalls (`svc #0x80`). It reads time from the ARM hardware timer. It
contains its own `exp`, `tanh`, GELU, softmax, LayerNorm, NEON matrix code, BPE
tokenizer, and sampler.

There is no conversion step. The program reads the original Hugging Face files
(`model.safetensors`, `vocab.json`, `merges.txt`) as they are, and parses them
in assembly at startup.

```
$ ./gpt2 "Alan Turing theorized that computers would one day become"
Alan Turing theorized that computers would one day become "more like machines." ...

[gpt2.s] 10 prompt + 60 generated tokens, 100.7 tok/s]
```

## Numbers (Apple M1, one core)

| | Speed | Note |
|---|---|---|
| GPT-2 in CSS | about 0.065 tok/s (15.3 s/token) | from the author's blog post |
| PyTorch, CPU | 40 tok/s | `transformers` `generate`, same machine |
| **gpt2.s** | **90 to 100 tok/s** | one thread, fp32 |

- Machine code: 6,324 bytes. Source: 1,834 lines of assembly.
- Startup (parse all three files and prepare the weights): about 0.13 s.
- The output is exact. Greedy output matches PyTorch token for token (see below).

## Build and run

You need macOS on Apple Silicon and the Xcode command line tools.

```bash
make weights
```

```bash
make
```

```bash
./gpt2 -n 100 -t 0.8 "The meaning of life is"
```

`make weights` downloads three files from the
[Hugging Face GPT-2 repository](https://huggingface.co/openai-community/gpt2)
with `curl` (about 550 MB): `model.safetensors`, `vocab.json` and `merges.txt`.
Nothing changes them. The program reads them from the current folder.

Options:

- `-n N` sets the number of tokens to generate. The default is 64.
- `-t T` sets the sampling temperature. `0` means greedy. The default is 0.8.
- `-i` prints token ids instead of text. Line 1 is the prompt. Line 2 is the output.

## Verify

```bash
uv run python tools/verify.py
```

This script needs `uv`. It is the only Python in the project, and it only
checks the result. It compares `./gpt2` with Hugging Face GPT-2 in PyTorch
(fp32, greedy):

- 12 tokenizer cases (contractions, runs of spaces, newlines, tabs, digits, punctuation). The token ids must match exactly.
- 4 prompts with 40 generated tokens each. The token ids must match exactly.

The current result is `ALL PASS`.

## How it works

| File | Contents |
|---|---|
| `src/main.s` | Argument parsing, `open`/`lseek`/`mmap` of the three files, sampling (xorshift64*), timing (`cntvct_el0`), output. |
| `src/load.s` | Startup. Parses the safetensors JSON header to find each tensor. Parses `vocab.json` (with its `\uXXXX` escapes) and `merges.txt` (UTF-8), and builds the tokenizer hash tables. |
| `src/tok.s` | GPT-2 pre-tokenizer rules and byte-level BPE. Merges are found in an open-addressing hash table. |
| `src/model.s` | The forward pass for one token: embeddings, 12 blocks, 12-head attention with a KV cache, final LayerNorm, tied output head. |
| `src/math.s` | NEON matrix-vector product, LayerNorm, softmax, GELU, and `expf` (range reduction and a degree-6 polynomial). |

`model.safetensors` is mapped into memory with `mmap`. The checkpoint stores the
48 block matrices as `[input][output]`. The matrix kernel is fastest on
`[output][input]`, where each output is one dot product over one row read in
order. So at startup the program copies those matrices once, transposed, into
memory that it maps for itself (340 MB). All other buffers, including the 75 MB
KV cache, are zero-fill sections in the binary, so there is no allocator.

## Honest limits

- **libSystem is linked but not used.** macOS on Apple Silicon kills static
  binaries at launch (SIGKILL), so the binary must name `libSystem.dylib` for
  the loader. `nm -u gpt2` is empty: the program imports zero symbols.
- **More memory than the weights alone.** The transposed copy adds 340 MB to
  the 550 MB file mapping.
- **Non-ASCII text is approximate in the tokenizer.** Bytes of `0x80` and higher
  are treated as letters. ASCII text matches the GPT-2 tokenizer exactly.
- **One thread, fp32.** The program is limited by memory bandwidth (it reads
  about 500 MB of weights for each token).
- **The context is 1024 tokens**, the same as GPT-2.
