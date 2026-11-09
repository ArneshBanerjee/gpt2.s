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

## Benchmarks

| | Speed | Note |
|---|---|---|
| PyTorch, CPU | 40 tok/s | `transformers` `generate`, same machine |
| **gpt2.s** | **90 to 100 tok/s** | one thread, fp32 |

- Machine code: 9,368 bytes. Source: 2,692 lines of assembly.
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
- `-j N` sets how many processes share each matrix product and the attention
  heads. The default is 1; `N` is capped at the number of cores. The output is
  the same for every `N`. Whether it helps depends on the chip: on an M1 one
  core already uses the memory bandwidth (no gain), on an M6 `-j 8` is about
  1.5x faster.
- `-h` stores the weight matrices (the 48 block matrices and the output head)
  as fp16, which halves the memory traffic. Embeddings, biases, LayerNorm
  weights, products and sums stay fp32. The output is no longer exact: greedy
  text can take a different word where two are nearly tied. With half the bytes,
  more processes help on an M1 too: `-h -j 4` was up to 1.2x faster than `-h`.
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

## Inner Workings

| File | Contents |
|---|---|
| `src/main.s` | Argument parsing, `open`/`lseek`/`mmap` of the three files, sampling (xorshift64*), timing (`cntvct_el0`), output. |
| `src/load.s` | Startup. Parses the safetensors JSON header to find each tensor. Parses `vocab.json` (with its `\uXXXX` escapes) and `merges.txt` (UTF-8), and builds the tokenizer hash tables. |
| `src/tok.s` | GPT-2 pre-tokenizer rules and byte-level BPE. Merges are found in an open-addressing hash table. |
| `src/model.s` | The forward pass for one token: embeddings, 12 blocks, 12-head attention with a KV cache, final LayerNorm, tied output head. |
| `src/math.s` | NEON matrix-vector product, LayerNorm, softmax, GELU, and `expf` (range reduction and a degree-6 polynomial). |
| `src/par.s` | Worker processes (`-j`): `fork`, the shared-memory section, and the job protocol that splits matrix rows and attention heads between processes. |

`model.safetensors` is mapped into memory with `mmap`. The checkpoint stores the
48 block matrices as `[input][output]`. At startup the program copies them once
into tiles of 64 outputs, `[output/64][input][64]`, in memory that it maps for
itself (340 MB). The matrix kernel keeps a tile's 64 sums in registers and
reads the tile from start to end. One load of `x` brings 4 inputs, which serve
all 64 outputs, so the weights are the only stream from memory. The output head
uses `wte` the same way, so it gets a tiled copy too (154 MB, padded to 50,304
rows); the embedding lookup still reads `wte` from the file.

All other buffers, including the 75 MB KV cache, are zero-fill sections in the
binary, so there is no allocator. The KV cache is stored by head,
`[layer][head][position][64]`, so attention reads each head's keys and values
in order.

Threads would need libpthread (a new Darwin thread starts in its entry point),
so `-j` uses processes. At startup, before anything is written to them, the
activations and the KV cache (one section, `__DATA,__shm`) are remapped as
shared memory. After loading, the program forks its workers. The weights stay
shared copy-on-write because nobody writes them. Each matrix product is one
job: every process claims one tile (64 rows) at a time with compare-and-swap and runs the
same `matvec` on them, so a row gives the same bits whichever process computes
it. Attention is split the same way, one head per claim.

## Limitations

- **libSystem is linked but not used.** macOS on Apple Silicon kills static
  binaries at launch (SIGKILL), so the binary must name `libSystem.dylib` for
  the loader. `nm -u gpt2` is empty: the program imports zero symbols.
- **More memory than the weights alone.** The tiled copies add 340 MB (the
  block matrices) and 154 MB (the output head) to the 550 MB file mapping.
- **Non-ASCII text is approximate in the tokenizer.** Bytes of `0x80` and higher
  are treated as letters. ASCII text matches the GPT-2 tokenizer exactly.
- **fp32, limited by memory bandwidth.** It reads about 500 MB of weights for
  each token. More processes (`-j`) help only where one core can't use all the
  memory bandwidth: on an M1 they don't (`-j 7` was 9% slower than one core),
  on an M6 about 1.4x with 2 processes and 1.5x with 8.
- **`-h` uses about 200 MB less memory after startup.** The tiled copies are
  built in fp32, converted to fp16 in place, and their second halves are given
  back, so the peak during startup is the same as without `-h`.
- **Processes spin while they wait.** During generation every process keeps a
  core busy. A waiting process spins for a moment, then gives up its core, and
  an idle worker sleeps 0.2 ms at a time. Workers exit when the program ends or
  is killed (they check that the parent is still alive). If a worker dies, the
  program stops with an error instead of waiting for its rows.
- **The context is 1024 tokens**, the same as GPT-2.

## License

MIT. See [LICENSE](LICENSE).
