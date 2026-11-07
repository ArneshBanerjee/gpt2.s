# gpt2.s

GPT-2 small (124M parameters) in hand-written AArch64 assembly for Apple Silicon.

The program calls no C code. It calls no libc functions. It does all I/O with raw
Darwin syscalls (`svc #0x80`). It reads time from the ARM hardware timer. It
contains its own `exp`, `tanh`, GELU, softmax, LayerNorm, NEON matrix code, BPE
tokenizer, and sampler.

```
$ ./gpt2 "Alan Turing theorized that computers would one day become"
Alan Turing theorized that computers would one day become "more like machines." ...

[gpt2.s] 10 prompt + 60 generated tokens, 100.7 tok/s]
```

## Benchmarks

| | Speed | Note |
|---|---|---|
| PyTorch, CPU | 40 tok/s | `transformers` `generate`, same machine |
| **gpt2.s** | **103 tok/s** | one thread, fp32 |

- Machine code: 4,244 bytes. Source: 1,222 lines of assembly.
- The output is exact. Greedy output matches PyTorch token for token (see below).

## Build and run

You need macOS on Apple Silicon, the Xcode command line tools, and `uv`.

```bash
make weights
```

```bash
make
```

```bash
./gpt2 -n 100 -t 0.8 "The meaning of life is"
```

`make weights` downloads GPT-2 from Hugging Face (about 500 MB) and writes
`gpt2_124M.bin` and `gpt2_tok.bin`. The program loads these two files from the
current folder.

Options:

- `-n N` sets the number of tokens to generate. The default is 64.
- `-t T` sets the sampling temperature. `0` means greedy. The default is 0.8.
- `-i` prints token ids instead of text. Line 1 is the prompt. Line 2 is the output.

## Verify

```bash
uv run python tools/verify.py
```

This script compares `./gpt2` with Hugging Face GPT-2 in PyTorch (fp32, greedy):

- 12 tokenizer cases (contractions, runs of spaces, newlines, tabs, digits, punctuation). The token ids must match exactly.
- 4 prompts with 40 generated tokens each. The token ids must match exactly.

The current result is `ALL PASS`.

## Inner Workings

| File | Contents |
|---|---|
| `src/main.s` | Argument parsing, `open`/`lseek`/`mmap` of the weight file, sampling (xorshift64*), timing (`cntvct_el0`), output. |
| `src/tok.s` | GPT-2 pre-tokenizer rules and byte-level BPE. Merges are found in an open-addressing hash table. |
| `src/model.s` | The forward pass for one token: embeddings, 12 blocks, 12-head attention with a KV cache, final LayerNorm, tied output head. |
| `src/math.s` | NEON matrix-vector product, LayerNorm, softmax, GELU, and `expf` (range reduction and a degree-6 polynomial). |
| `tools/export.py` | Converts the weights to one flat fp32 file. Each matrix is stored as `[out][in]`, so each output is one contiguous dot product. |

The weights are mapped into memory with `mmap`, so the program does no copy and
has no allocator. All buffers, including the 75 MB KV cache, are zero-fill
sections in the binary.

## Limitations

- **libSystem is linked but not used.** macOS on Apple Silicon kills static
  binaries at launch (SIGKILL), so the binary must name `libSystem.dylib` for
  the loader. `nm -u gpt2` is empty: the program imports zero symbols.
- **Python runs offline only.** It converts the weights and the tokenizer
  tables. The CSS project does the same. Python is not used at run time.
- **Non-ASCII text is approximate in the tokenizer.** Bytes of `0x80` and higher
  are treated as letters. ASCII text matches the GPT-2 tokenizer exactly.
- **One thread, fp32.** The program is limited by memory bandwidth (it reads
  500 MB of weights for each token).
- **The context is 1024 tokens**, the same as GPT-2.

## License

MIT. See [LICENSE](LICENSE).
