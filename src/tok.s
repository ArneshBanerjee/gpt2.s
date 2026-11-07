// GPT-2 byte-level BPE tokenizer: encode text -> token ids, print token id.
//
// Pre-tokenizer emulates GPT-2's regex for ASCII:
//   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
// Bytes >= 0x80 are treated as letters (an approximation for UTF-8 text).
// BPE: repeatedly merge the adjacent pair with the lowest merge rank,
// found through an open-addressing hash table built by tools/export.py.
.include "macros.inc"

.equ C_OTHER, 0
.equ C_LETTER, 1
.equ C_DIGIT, 2
.equ C_SPACE, 3
.equ MAX_PROMPT, 16384

.text
.p2align 2

// w0 = byte -> w0 = class. Leaf. Clobbers w1.
cls:
    cmp     w0, #128
    b.hs    1f
    sub     w1, w0, #'0'
    cmp     w1, #9
    b.ls    2f
    orr     w1, w0, #0x20
    sub     w1, w1, #'a'
    cmp     w1, #25
    b.ls    1f
    cmp     w0, #' '
    b.eq    3f
    sub     w1, w0, #9               // \t \n \v \f \r
    cmp     w1, #4
    b.ls    3f
    mov     w0, #C_OTHER
    ret
1:  mov     w0, #C_LETTER
    ret
2:  mov     w0, #C_DIGIT
    ret
3:  mov     w0, #C_SPACE
    ret

// encode(x0 = NUL-terminated text) -> x0 = token count, ids in _tokens
.globl _encode
_encode:
    stp     x29, x30, [sp, #-96]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #64]
    stp     x27, x28, [sp, #80]

    mov     x19, x0                  // text
    mov     x20, #0                  // n = strlen(text)
0:  ldrb    w1, [x19, x20]
    cbz     w1, 0f
    add     x20, x20, #1
    b       0b
0:  mov     x1, #MAX_PROMPT
    cmp     x20, x1
    csel    x20, x20, x1, lo
    ADDR    x0, _g_tok
    ldr     x23, [x0]                // tokenizer file base
    ADDR    x21, _tokens             // output ids
    ADDR    x27, work                // BPE work buffer
    add     x28, x23, #T_TABLE       // merge hash table
    MOV32   w15, 0x9E3779B1          // hash multiplier
    mov     x22, #0                  // token count
    mov     x24, #0                  // i = chunk start

.Lchunk:
    cmp     x24, x20
    b.ge    .Ldone
    // contractions: 's 't 'm 'd 're 've 'll
    ldrb    w0, [x19, x24]
    cmp     w0, #'\''
    b.ne    .Lword
    add     x1, x24, #1
    cmp     x1, x20
    b.ge    .Lword
    ldrb    w2, [x19, x1]
    cmp     w2, #'s'
    b.eq    .Lc2
    cmp     w2, #'t'
    b.eq    .Lc2
    cmp     w2, #'m'
    b.eq    .Lc2
    cmp     w2, #'d'
    b.eq    .Lc2
    add     x3, x24, #2
    cmp     x3, x20
    b.ge    .Lword
    ldrb    w3, [x19, x3]
    cmp     w2, #'r'
    b.ne    1f
    cmp     w3, #'e'
    b.eq    .Lc3
1:  cmp     w2, #'v'
    b.ne    2f
    cmp     w3, #'e'
    b.eq    .Lc3
2:  cmp     w2, #'l'
    b.ne    .Lword
    cmp     w3, #'l'
    b.eq    .Lc3
    b       .Lword
.Lc2:
    add     x25, x24, #2
    b       .Lbpe
.Lc3:
    add     x25, x24, #3
    b       .Lbpe

.Lword:
    // optional leading space, then a run of one class (letters/digits/other)
    mov     x25, x24
    ldrb    w0, [x19, x24]
    cmp     w0, #' '
    b.ne    1f
    add     x1, x24, #1
    cmp     x1, x20
    b.ge    1f
    mov     x25, x1
1:  ldrb    w0, [x19, x25]
    bl      cls
    mov     w26, w0
    cmp     w26, #C_SPACE
    b.eq    .Lspace
2:  add     x25, x25, #1
    cmp     x25, x20
    b.ge    .Lbpe
    ldrb    w0, [x19, x25]
    bl      cls
    cmp     w0, w26
    b.eq    2b
    b       .Lbpe

.Lspace:
    // \s+(?!\S) then \s+ : a run followed by text gives up its last char
    mov     x25, x24
3:  add     x25, x25, #1
    cmp     x25, x20
    b.ge    .Lbpe
    ldrb    w0, [x19, x25]
    bl      cls
    cmp     w0, #C_SPACE
    b.eq    3b
    sub     x1, x25, x24
    cmp     x1, #2
    b.lt    .Lbpe
    sub     x25, x25, #1

.Lbpe:
    // chunk is text[x24 .. x25). Map each byte to its base token.
    add     x2, x23, #T_BYTE2ID
    mov     x3, #0                   // L = number of symbols
    mov     x4, x24
4:  ldrb    w0, [x19, x4]
    ldr     w0, [x2, x0, lsl #2]
    str     w0, [x27, x3, lsl #2]
    add     x3, x3, #1
    add     x4, x4, #1
    cmp     x4, x25
    b.lt    4b

.Lmerge:
    cmp     x3, #2
    b.lt    .Lemit
    mov     w4, #-1                  // best rank
    mov     x5, #-1                  // best index
    mov     x6, #0                   // k
5:  ldr     w7, [x27, x6, lsl #2]
    add     x8, x6, #1
    ldr     w8, [x27, x8, lsl #2]
    orr     w7, w8, w7, lsl #16      // key = (a << 16) | b
    mul     w8, w7, w15
    lsr     w8, w8, #(32 - T_HBITS)
6:  add     x10, x8, x8, lsl #1      // slot * 12
    add     x10, x28, x10, lsl #2
    ldr     w11, [x10]
    cmp     w11, w7
    b.eq    7f
    cmn     w11, #1                  // empty slot: pair is not a merge
    b.eq    8f
    add     w8, w8, #1
    and     w8, w8, #((1 << T_HBITS) - 1)
    b       6b
7:  ldr     w11, [x10, #4]           // rank
    cmp     w11, w4
    b.hs    8f
    mov     w4, w11
    mov     x5, x6
    ldr     w12, [x10, #8]           // merged token id
8:  add     x6, x6, #1
    sub     x7, x3, #1
    cmp     x6, x7
    b.lt    5b
    tbnz    x5, #63, .Lemit          // no merge left
    str     w12, [x27, x5, lsl #2]
    add     x6, x5, #1               // shift w[k+2..] left by one
9:  add     x7, x6, #1
    cmp     x7, x3
    b.ge    10f
    ldr     w8, [x27, x7, lsl #2]
    str     w8, [x27, x6, lsl #2]
    mov     x6, x7
    b       9b
10: sub     x3, x3, #1
    b       .Lmerge

.Lemit:
    mov     x4, #0
11: cmp     x22, #N_CTX
    b.ge    .Ldone
    ldr     w0, [x27, x4, lsl #2]
    str     w0, [x21, x22, lsl #2]
    add     x22, x22, #1
    add     x4, x4, #1
    cmp     x4, x3
    b.lt    11b
    mov     x24, x25
    b       .Lchunk

.Ldone:
    mov     x0, x22
    ldp     x27, x28, [sp, #80]
    ldp     x25, x26, [sp, #64]
    ldp     x23, x24, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #96
    ret

// print_token(w0 = id): write the token's bytes to stdout
.globl _print_token
_print_token:
    ADDR    x1, _g_tok
    ldr     x1, [x1]
    MOV32   x2, T_OFFS
    add     x2, x1, x2
    add     x2, x2, w0, uxtw #2
    ldp     w3, w4, [x2]             // start, end offsets
    MOV32   x5, T_BLOB
    add     x1, x1, x5
    add     x1, x1, x3
    sub     x2, x4, x3
    mov     x0, #1
    SYSCALL SYS_WRITE
    ret

.zerofill __DATA,__bss,work,MAX_PROMPT*4,4
