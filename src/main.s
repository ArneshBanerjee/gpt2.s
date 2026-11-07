// gpt2.s: GPT-2 small (124M) inference in AArch64 assembly.
// No libc calls: all I/O is raw Darwin syscalls (svc #0x80), timing reads
// the ARM generic timer directly, and exp/tanh/sampling are written here.
//
// usage: gpt2 [-n tokens] [-t temperature] [-i] "prompt"
//   -n  tokens to generate (default 64)
//   -t  sampling temperature, 0 = greedy (default 0.8)
//   -i  print token ids instead of text (line 1: prompt, line 2: output)
.include "macros.inc"

.text
.p2align 2

.globl _main
_main:
    mov     x19, x0                  // argc
    mov     x20, x1                  // argv
    mov     x21, #64                 // tokens to generate
    FCONST  s8, 0x3f4ccccd           // temperature 0.8
    mov     x22, #0                  // id mode
    mov     x23, #0                  // prompt
    mov     x24, #1
.Largs:
    cmp     x24, x19
    b.ge    .Largs_done
    ldr     x0, [x20, x24, lsl #3]
    ldrb    w1, [x0]
    cmp     w1, #'-'
    b.ne    .Lprompt
    ldrb    w1, [x0, #1]
    ldrb    w2, [x0, #2]
    cbnz    w2, .Lprompt
    cmp     w1, #'i'
    b.eq    .Lflag_i
    add     x24, x24, #1
    cmp     x24, x19
    b.ge    .Lusage
    ldr     x0, [x20, x24, lsl #3]
    cmp     w1, #'n'
    b.eq    .Lflag_n
    cmp     w1, #'t'
    b.eq    .Lflag_t
    b       .Lusage
.Lflag_i:
    mov     x22, #1
    b       .Lnext
.Lflag_n:
    bl      parse_uint
    mov     x21, x0
    b       .Lnext
.Lflag_t:
    bl      parse_float
    fmov    s8, s0
    b       .Lnext
.Lprompt:
    mov     x23, x0
.Lnext:
    add     x24, x24, #1
    b       .Largs
.Largs_done:
    cbz     x23, .Lusage
    cbz     x21, .Lusage

    // the original Hugging Face files, read as they are
    ADDR    x0, path_model
    bl      map_file
    bl      _load_model
    ADDR    x0, path_vocab
    bl      map_file
    mov     x25, x0
    mov     x26, x1
    ADDR    x0, path_merges
    bl      map_file
    mov     x2, x0
    mov     x3, x1
    mov     x0, x25
    mov     x1, x26
    bl      _build_tokenizer

    mrs     x0, cntvct_el0           // seed the RNG from the hardware timer
    orr     x0, x0, #1
    ADDR    x1, rng_state
    str     x0, [x1]

    mov     x0, x23
    bl      _encode
    mov     x25, x0                  // prompt token count
    cbnz    x25, 1f
    ADDR    x1, _tokens              // empty prompt: start from <|endoftext|>
    MOV32   w2, EOT
    str     w2, [x1]
    mov     x25, #1
1:
    // echo the prompt (as text, or as ids)
    cbz     x22, 3f
    mov     x26, #0
2:  ADDR    x1, _tokens
    ldr     w1, [x1, x26, lsl #2]
    mov     x0, #1
    bl      print_uint
    bl      print_space
    add     x26, x26, #1
    cmp     x26, x25
    b.lt    2b
    ADDR    x1, s_newline
    mov     x0, #1
    bl      print_str
    b       4f
3:  mov     x1, x23
    mov     x0, #1
    bl      print_str

    // run the prompt through the model
4:  mov     x26, #0
5:  ADDR    x1, _tokens
    ldr     w0, [x1, x26, lsl #2]
    mov     w1, w26
    bl      _forward
    add     x26, x26, #1
    cmp     x26, x25
    b.lt    5b

    // generate
    mrs     x27, cntvct_el0
    mov     x28, #0                  // generated count
.Lgen:
    bl      sample
    mov     w24, w0
    cbz     x22, 6f
    mov     x1, x24
    mov     x0, #1
    bl      print_uint
    bl      print_space
    b       7f
6:  mov     w0, w24
    bl      _print_token
7:  add     x28, x28, #1
    cmp     x28, x21
    b.ge    .Lgen_done
    cmp     x26, #N_CTX
    b.ge    .Lgen_done
    mov     w0, w24
    mov     w1, w26
    bl      _forward
    add     x26, x26, #1
    b       .Lgen
.Lgen_done:
    mrs     x0, cntvct_el0
    sub     x27, x0, x27             // elapsed ticks

    // stats on stderr: [P prompt + G generated tokens, X.Y tok/s]
    ADDR    x1, s_stats0
    mov     x0, #2
    bl      print_str
    mov     x1, x25
    mov     x0, #2
    bl      print_uint
    ADDR    x1, s_stats1
    mov     x0, #2
    bl      print_str
    mov     x1, x28
    mov     x0, #2
    bl      print_uint
    ADDR    x1, s_stats2
    mov     x0, #2
    bl      print_str
    mrs     x0, cntfrq_el0
    mul     x0, x0, x28
    mov     x1, #10
    mul     x0, x0, x1
    cmp     x27, #0
    csinc   x27, x27, xzr, ne
    udiv    x19, x0, x27             // tokens/s * 10
    mov     x1, #10
    udiv    x1, x19, x1
    mov     x0, #2
    bl      print_uint
    ADDR    x1, s_dot
    mov     x0, #2
    bl      print_str
    mov     x1, #10
    udiv    x2, x19, x1
    msub    x1, x2, x1, x19
    mov     x0, #2
    bl      print_uint
    ADDR    x1, s_stats3
    mov     x0, #2
    bl      print_str
    mov     x0, #0
    SYSCALL SYS_EXIT

.Lusage:
    ADDR    x1, s_usage
    mov     x0, #2
    bl      print_str
    mov     x0, #2
    SYSCALL SYS_EXIT

// sample() -> w0 = next token from _logits
sample:
    stp     x29, x30, [sp, #-32]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    ADDR    x19, _logits
    MOV32   x20, N_VOCAB
    fcmp    s8, #0.0
    b.gt    2f
    // greedy: argmax
    ldr     s0, [x19]
    mov     x0, #0
    mov     x2, #1
1:  ldr     s1, [x19, x2, lsl #2]
    fcmp    s1, s0
    fcsel   s0, s1, s0, gt
    csel    x0, x2, x0, gt
    add     x2, x2, #1
    cmp     x2, x20
    b.lt    1b
    b       5f
    // temperature: p = softmax(logits / T), then draw from p
2:  fmov    s0, #1.0
    fdiv    s0, s0, s8
    mov     x2, #0
3:  ldr     s1, [x19, x2, lsl #2]
    fmul    s1, s1, s0
    str     s1, [x19, x2, lsl #2]
    add     x2, x2, #1
    cmp     x2, x20
    b.lt    3b
    mov     x0, x19
    mov     x1, x20
    bl      _softmax
    // xorshift64*
    ADDR    x3, rng_state
    ldr     x0, [x3]
    eor     x0, x0, x0, lsr #12
    eor     x0, x0, x0, lsl #25
    eor     x0, x0, x0, lsr #27
    str     x0, [x3]
    movz    x1, #0xDD1D
    movk    x1, #0x4F6C, lsl #16
    movk    x1, #0xF491, lsl #32
    movk    x1, #0x2545, lsl #48
    mul     x0, x0, x1
    lsr     x0, x0, #40              // 24 random bits
    ucvtf   s0, w0
    FCONST  s1, 0x33800000           // 2^-24
    fmul    s0, s0, s1               // u in [0, 1)
    movi    v1.16b, #0
    mov     x0, #0
    sub     x3, x20, #1
4:  ldr     s2, [x19, x0, lsl #2]
    fadd    s1, s1, s2
    fcmp    s1, s0
    b.gt    5f
    add     x0, x0, #1
    cmp     x0, x3
    b.lt    4b
5:  ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #32
    ret

// map_file(x0 = path) -> x0 = read-only mapping, x1 = size
map_file:
    stp     x29, x30, [sp, #-48]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    str     x21, [sp, #32]
    mov     x19, x0
    mov     x1, #0                   // O_RDONLY
    mov     x2, #0
    SYSCALL SYS_OPEN
    b.cs    1f
    mov     x21, x0                  // fd
    mov     x1, #0
    mov     x2, #2                   // SEEK_END -> file size
    SYSCALL SYS_LSEEK
    b.cs    1f
    cbz     x0, 1f                   // empty file
    mov     x20, x0                  // length
    mov     x1, x0
    mov     x0, #0
    mov     x2, #1                   // PROT_READ
    mov     x3, #2                   // MAP_PRIVATE
    mov     x4, x21
    mov     x5, #0
    SYSCALL SYS_MMAP
    b.cs    1f
    mov     x1, x20
    ldr     x21, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #48
    ret
1:  ADDR    x1, s_nofile
    mov     x0, #2
    bl      print_str
    mov     x1, x19
    mov     x0, #2
    bl      print_str
    ADDR    x1, s_newline
    mov     x0, #2
    bl      print_str
    mov     x0, #1
    SYSCALL SYS_EXIT

// print_str(x0 = fd, x1 = NUL-terminated string)
print_str:
    mov     x2, #0
1:  ldrb    w3, [x1, x2]
    cbz     w3, 2f
    add     x2, x2, #1
    b       1b
2:  SYSCALL SYS_WRITE
    ret

print_space:
    ADDR    x1, s_space
    mov     x0, #1
    b       print_str

// print_uint(x0 = fd, x1 = value)
print_uint:
    sub     sp, sp, #32
    add     x2, sp, #32              // write digits backwards
    mov     x4, #10
1:  udiv    x3, x1, x4
    msub    x5, x3, x4, x1
    add     w5, w5, #'0'
    strb    w5, [x2, #-1]!
    mov     x1, x3
    cbnz    x1, 1b
    mov     x1, x2
    add     x2, sp, #32
    sub     x2, x2, x1
    SYSCALL SYS_WRITE
    add     sp, sp, #32
    ret

// parse_uint(x0 = string) -> x0
parse_uint:
    mov     x1, x0
    mov     x0, #0
    mov     x3, #10
1:  ldrb    w2, [x1], #1
    sub     w2, w2, #'0'
    cmp     w2, #9
    b.hi    2f
    madd    x0, x0, x3, x2
    b       1b
2:  ret

// parse_float(x0 = string like "0.75") -> s0
parse_float:
    movi    v0.16b, #0
    fmov    s1, #10.0
    fmov    s2, #1.0                 // place value after the point
    mov     w4, #0                   // seen '.'
1:  ldrb    w2, [x0], #1
    cmp     w2, #'.'
    b.ne    2f
    mov     w4, #1
    b       1b
2:  sub     w2, w2, #'0'
    cmp     w2, #9
    b.hi    4f
    ucvtf   s3, w2
    cbnz    w4, 3f
    fmadd   s0, s0, s1, s3           // integer part: v = v*10 + d
    b       1b
3:  fdiv    s2, s2, s1               // fraction: v += d * 10^-k
    fmadd   s0, s3, s2, s0
    b       1b
4:  ret

.section __TEXT,__cstring,cstring_literals
path_model:   .asciz "model.safetensors"
path_vocab:   .asciz "vocab.json"
path_merges:  .asciz "merges.txt"
s_nofile:     .asciz "gpt2: cannot load "
s_newline:    .asciz "\n"
s_space:      .asciz " "
s_dot:        .asciz "."
s_stats0:     .asciz "\n\n[gpt2.s] "
s_stats1:     .asciz " prompt + "
s_stats2:     .asciz " generated tokens, "
s_stats3:     .asciz " tok/s]\n"
s_usage:      .asciz "usage: gpt2 [-n tokens] [-t temperature] [-i] \"prompt\"\n"

.data
.p2align 3
.globl _g_tok
_g_tok:       .quad 0
rng_state:    .quad 0

.globl _tokens
.zerofill __DATA,__bss,_tokens,N_CTX*4,4
