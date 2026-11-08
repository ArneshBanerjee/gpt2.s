// Startup: read Hugging Face's original files, with no conversion step.
//
// load_model:      parse the JSON header of model.safetensors and fill
//                  _tensors with a pointer to each tensor we use.
// build_tokenizer: parse vocab.json and merges.txt and build the tables
//                  that src/tok.s reads (byte -> id, merge hash table,
//                  id -> bytes).
.include "macros.inc"

.equ BLOB_MAX,  1 << 20               // bytes of all token strings (about 320 KB used)
.equ S_HBITS,   17                    // string -> id hash table: 131,072 slots
.equ EMPTY,     0xFFFFFFFF
// all 48 block matrices: 12 x (768x2304 + 768x768 + 768x3072 + 3072x768) floats
.equ TILED_BYTES, N_LAYER * (3 + 1 + 4 + 4) * N_EMBD * N_EMBD * 4

// GPT-2's byte-level alphabet maps every byte to a printable code point.
// Code points below 256 are the byte itself. Code points 256 + n are the
// unprintable bytes in order: 0..32, then 127..160, then 173.
.macro CP2BYTE r
    cmp     \r, #256
    b.lo    .Lcp_done\@
    sub     \r, \r, #256
    cmp     \r, #33
    b.lo    .Lcp_done\@
    cmp     \r, #67
    b.eq    .Lcp_173\@
    add     \r, \r, #(127 - 33)
    b       .Lcp_done\@
.Lcp_173\@:
    mov     \r, #173
.Lcp_done\@:
.endm

.text
.p2align 2

// ---------------------------------------------------------------------------
// load_model(x0 = mapped model.safetensors, x1 = file size)
// File layout: u64 header length, JSON header, then raw tensor data.
.globl _load_model
_load_model:
    stp     x29, x30, [sp, #-80]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #64]
    ldr     x2, [x0]                 // header length
    add     x3, x2, #8
    cmp     x3, x1
    b.hs    .Lmodel_bad
    add     x19, x0, #8              // cursor in the JSON header
    add     x20, x19, x2             // end of header = start of tensor data
    mov     x21, #0                  // tensors found
    ADDR    x22, _tensors

.Lnext_key:
    cmp     x19, x20                 // find the next key: "name"
    b.hs    .Lheader_done
    ldrb    w0, [x19], #1
    cmp     w0, #'"'
    b.ne    .Lnext_key
    mov     x23, x19
1:  ldrb    w0, [x19], #1
    cmp     w0, #'"'
    b.ne    1b
    sub     x1, x19, x23
    sub     x1, x1, #1
    mov     x0, x23
    bl      tensor_slot
    mov     x25, x0                  // slot, or -1 for a tensor we do not use
    mov     x26, #-1                 // byte offset of the data
2:  ldrb    w0, [x19], #1            // walk the value object up to its '}'
    cmp     w0, #'}'
    b.eq    4f
    cmp     w0, #'['                 // only "data_offsets":[ is preceded by ts":
    b.ne    2b
    ldurb   w1, [x19, #-5]
    cmp     w1, #'t'
    b.ne    2b
    ldurb   w1, [x19, #-4]
    cmp     w1, #'s'
    b.ne    2b
    ldurb   w1, [x19, #-3]
    cmp     w1, #'"'
    b.ne    2b
    mov     x26, #0
    mov     x1, #10
3:  ldrb    w0, [x19]                // first number = start offset
    sub     w0, w0, #'0'
    cmp     w0, #9
    b.hi    2b
    madd    x26, x26, x1, x0
    add     x19, x19, #1
    b       3b
4:  tbnz    x25, #63, .Lnext_key
    tbnz    x26, #63, .Lmodel_bad
    add     x0, x20, x26
    str     x0, [x22, x25, lsl #3]
    add     x21, x21, #1
    b       .Lnext_key

.Lheader_done:
    cmp     x21, #N_TENSORS
    b.ne    .Lmodel_bad

    // The checkpoint stores the block matrices as [input][output]. Copy
    // them once into tiles of 64 outputs, [output/64][input][64], in memory
    // that we map for ourselves: then _matvec reads each matrix in order, and
    // one load of x serves 64 outputs.
    mov     x0, #0
    MOV32   x1, TILED_BYTES
    mov     x2, #3                   // PROT_READ | PROT_WRITE
    mov     x3, #0x1002              // MAP_ANON | MAP_PRIVATE
    mov     x4, #-1
    mov     x5, #0
    SYSCALL SYS_MMAP
    b.cs    .Lno_memory
    mov     x23, x0                  // next free byte in the new memory
    add     x24, x22, #32            // this block's 12 pointers
    mov     x25, #N_LAYER
5:  mov     x26, #0                  // 4 matrices per block
6:  ADDR    x0, block_mats
    add     x0, x0, x26, lsl #3
    ldrh    w19, [x0]                // index in the block's 12 pointers
    ldrh    w20, [x0, #2]            // inputs
    ldrh    w21, [x0, #4]            // outputs
    mov     x0, x23
    ldr     x1, [x24, x19, lsl #3]
    mov     x2, x20
    mov     x3, x21
    bl      tile_matrix
    str     x23, [x24, x19, lsl #3]
    mul     x0, x20, x21
    add     x23, x23, x0, lsl #2
    add     x26, x26, #1
    cmp     x26, #4
    b.lo    6b
    add     x24, x24, #96
    subs    x25, x25, #1
    b.ne    5b

    // The output head reads wte (tied weights) as a matrix: tile it too, with
    // its rows padded to whole tiles. The embedding lookup keeps reading wte.
    mov     x0, #0
    MOV32   x1, N_VOCAB_PAD*ROW
    mov     x2, #3                   // PROT_READ | PROT_WRITE
    mov     x3, #0x1002              // MAP_ANON | MAP_PRIVATE
    mov     x4, #-1
    mov     x5, #0
    SYSCALL SYS_MMAP
    b.cs    .Lno_memory
    mov     x23, x0
    ldr     x1, [x22]
    MOV32   x2, N_VOCAB
    bl      tile_rows
    ADDR    x0, _head_w
    str     x23, [x0]

    ldp     x25, x26, [sp, #64]
    ldp     x23, x24, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #80
    ret
.Lno_memory:
    ADDR    x1, s_no_memory
    b       fail
.Lmodel_bad:
    ADDR    x1, s_bad_model
    b       fail

// tile_matrix(x0 = dst, x1 = src[nin][nout], x2 = nin, x3 = nout): copy into
// tiles of 64 outputs, dst[nout/64][nin][64]. nout must be a multiple of 64.
// Leaf.
tile_matrix:
    lsl     x4, x3, #2               // bytes in a source row
    mov     x6, #0                   // byte offset of this tile in a source row
1:  add     x7, x1, x6
    mov     x9, x2
2:  mov     x8, x7
    ld1     {v0.4s, v1.4s, v2.4s, v3.4s}, [x8], #64
    ld1     {v4.4s, v5.4s, v6.4s, v7.4s}, [x8], #64
    ld1     {v16.4s, v17.4s, v18.4s, v19.4s}, [x8], #64
    ld1     {v20.4s, v21.4s, v22.4s, v23.4s}, [x8]
    st1     {v0.4s, v1.4s, v2.4s, v3.4s}, [x0], #64
    st1     {v4.4s, v5.4s, v6.4s, v7.4s}, [x0], #64
    st1     {v16.4s, v17.4s, v18.4s, v19.4s}, [x0], #64
    st1     {v20.4s, v21.4s, v22.4s, v23.4s}, [x0], #64
    add     x7, x7, x4
    subs    x9, x9, #1
    b.ne    2b
    add     x6, x6, #256
    cmp     x6, x4
    b.lo    1b
    ret

// tile_rows(x0 = dst, x1 = src[n][768], x2 = n): the [n][768] rows of src
// into tiles of 64 rows, dst[n/64][768][64]. Rows past n stay zero. Leaf.
tile_rows:
    mov     x9, #0                   // row
    MOV32   x12, TILE_BYTES
    mov     x14, #ROW
1:  cmp     x9, x2
    b.hs    3f
    lsr     x10, x9, #6
    madd    x13, x10, x12, x0
    and     x11, x9, #63
    add     x13, x13, x11, lsl #2    // dst: tile, then this row's lane
    madd    x15, x9, x14, x1         // src row
    mov     x7, #N_EMBD
2:  ldr     s0, [x15], #4
    str     s0, [x13]
    add     x13, x13, #256           // next input: 64 floats on
    subs    x7, x7, #1
    b.ne    2b
    add     x9, x9, #1
    b       1b
3:  ret

// tensor_slot(x0 = name, x1 = length) -> x0 = slot in _tensors, or -1
tensor_slot:
    stp     x29, x30, [sp, #-32]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    mov     x19, x0
    mov     x20, x1
    ADDR    x2, top_names
    bl      find_name
    tbz     x0, #63, 9f
    cmp     x20, #4                  // "h.<layer>.<name>"
    b.lo    8f
    ldrb    w3, [x19]
    cmp     w3, #'h'
    b.ne    8f
    ldrb    w3, [x19, #1]
    cmp     w3, #'.'
    b.ne    8f
    mov     x4, #2
    mov     x5, #0
    mov     x6, #10
1:  cmp     x4, x20
    b.hs    8f
    ldrb    w3, [x19, x4]
    add     x4, x4, #1
    cmp     w3, #'.'
    b.eq    2f
    sub     w3, w3, #'0'
    cmp     w3, #9
    b.hi    8f
    madd    x5, x5, x6, x3
    b       1b
2:  cmp     x5, #N_LAYER
    b.hs    8f
    add     x0, x19, x4              // the part after "h.<layer>."
    sub     x1, x20, x4
    mov     x6, #12
    madd    x19, x5, x6, xzr
    add     x19, x19, #4             // first slot of this layer
    ADDR    x2, layer_names
    bl      find_name
    tbnz    x0, #63, 9f              // e.g. attn.bias, the causal mask: not used
    add     x0, x0, x19
    b       9f
8:  mov     x0, #-1
9:  ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #32
    ret

// find_name(x0 = text, x1 = length, x2 = list of NUL-terminated names,
// ended by an empty name) -> x0 = index in the list, or -1
find_name:
    stp     x29, x30, [sp, #-48]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    mov     x19, x0
    mov     x20, x1
    mov     x21, x2
    mov     x22, #0
1:  ldrb    w3, [x21]
    cbz     w3, 4f
    mov     x3, #0                   // compare text[0..len) with the name
2:  ldrb    w4, [x21, x3]
    cmp     x3, x20
    b.eq    3f
    cbz     w4, 5f
    ldrb    w5, [x19, x3]
    cmp     w4, w5
    b.ne    5f
    add     x3, x3, #1
    b       2b
3:  cbnz    w4, 5f                   // text ended first
    mov     x0, x22
    b       6f
5:  ldrb    w3, [x21], #1            // skip to the next name
    cbnz    w3, 5b
    add     x22, x22, #1
    b       1b
4:  mov     x0, #-1
6:  ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #48
    ret

// ---------------------------------------------------------------------------
// build_tokenizer(x0 = vocab.json, x1 = its size, x2 = merges.txt, x3 = its size)
.globl _build_tokenizer
_build_tokenizer:
    stp     x29, x30, [sp, #-96]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #64]
    stp     x27, x28, [sp, #80]
    mov     x19, x0
    add     x20, x0, x1
    mov     x27, x2
    add     x28, x2, x3

    ADDR    x21, tokdata
    ADDR    x0, _g_tok
    str     x21, [x0]
    MOV32   x24, T_OFFS
    add     x24, x21, x24            // id -> start offset
    MOV32   x25, T_BLOB
    add     x25, x21, x25            // all token bytes
    ADDR    x26, strmap

    // both hash tables start empty (all ones)
    movi    v0.16b, #0xff
    add     x0, x21, #T_TABLE
    MOV32   x1, (1 << T_HBITS)*12
1:  st1     {v0.16b}, [x0], #16
    subs    x1, x1, #16
    b.ne    1b
    mov     x0, x26
    mov     x1, #((1 << S_HBITS)*4)
2:  st1     {v0.16b}, [x0], #16
    subs    x1, x1, #16
    b.ne    2b

    // vocab.json: {"token": id, ...}. Ids come in order 0, 1, 2, ...
    mov     x22, #0                  // bytes written to the blob
    mov     x23, #0                  // next id
.Lv_key:
    cmp     x19, x20
    b.hs    .Lv_done
    ldrb    w0, [x19], #1
    cmp     w0, #'"'
    b.ne    .Lv_key
    str     w22, [x24, x23, lsl #2]
    mov     x3, x22                  // start of this token in the blob
.Lv_char:
    ldrb    w0, [x19], #1
    cmp     w0, #'"'
    b.eq    .Lv_end
    cmp     w0, #'\\'
    b.ne    .Lv_cp
    ldrb    w0, [x19], #1            // \" and \\ stand for themselves
    cmp     w0, #'u'
    b.ne    .Lv_cp
    mov     w0, #0                   // \uXXXX
    mov     x4, #4
3:  ldrb    w1, [x19], #1
    sub     w2, w1, #'0'
    cmp     w2, #9
    b.ls    4f
    orr     w1, w1, #0x20
    sub     w2, w1, #('a' - 10)
4:  add     w0, w2, w0, lsl #4
    subs    x4, x4, #1
    b.ne    3b
.Lv_cp:
    CP2BYTE w0
    mov     x1, #BLOB_MAX
    cmp     x22, x1
    b.hs    .Ltok_bad
    strb    w0, [x25, x22]
    add     x22, x22, #1
    b       .Lv_char
.Lv_end:
    mov     x4, #0                   // the id after the colon
5:  ldrb    w0, [x19]
    sub     w0, w0, #'0'
    cmp     w0, #9
    b.ls    6f
    add     x19, x19, #1
    cmp     x19, x20
    b.lo    5b
    b       .Ltok_bad
6:  mov     x1, #10
7:  ldrb    w0, [x19]
    sub     w0, w0, #'0'
    cmp     w0, #9
    b.hi    8f
    madd    x4, x4, x1, x0
    add     x19, x19, #1
    b       7b
8:  cmp     x4, x23
    b.ne    .Ltok_bad
    add     x0, x25, x3              // add bytes -> id to the string table
    sub     x1, x22, x3
    mov     x2, #0
    mov     x3, #0
    bl      strhash
9:  ldr     w1, [x26, x0, lsl #2]
    cmn     w1, #1
    b.eq    10f
    add     w0, w0, #1
    and     w0, w0, #((1 << S_HBITS) - 1)
    b       9b
10: str     w23, [x26, x0, lsl #2]
    add     x23, x23, #1
    b       .Lv_key
.Lv_done:
    MOV32   x0, N_VOCAB
    cmp     x23, x0
    b.ne    .Ltok_bad
    str     w22, [x24, x23, lsl #2]  // end of the last token

    // byte -> id, for the first step of BPE
    mov     x23, #0
11: ADDR    x0, buf_a
    strb    w23, [x0]
    mov     x1, #1
    mov     x2, #0
    mov     x3, #0
    bl      find_token
    tbnz    x0, #63, .Ltok_bad
    add     x1, x21, #T_BYTE2ID
    str     w0, [x1, x23, lsl #2]
    add     x23, x23, #1
    cmp     x23, #256
    b.lo    11b

    // merges.txt: a header line, then "a b" per line, in rank order
    mov     x19, x27
12: cmp     x19, x28                 // skip the header line
    b.hs    .Ltok_bad
    ldrb    w0, [x19], #1
    cmp     w0, #'\n'
    b.ne    12b
    mov     x23, #0                  // rank
    add     x20, x21, #T_TABLE       // merge hash table
.Lm_line:
    cmp     x19, x28
    b.hs    .Lm_done
    ldrb    w0, [x19]
    cmp     w0, #'\n'
    b.ne    13f
    add     x19, x19, #1             // empty line
    b       .Lm_line
13: mov     x0, x19
    mov     x1, x28
    ADDR    x2, buf_a
    bl      decode_token
    mov     x19, x0
    mov     x26, x1                  // length of a
    add     x19, x19, #1             // the space
    mov     x0, x19
    mov     x1, x28
    ADDR    x2, buf_b
    bl      decode_token
    mov     x19, x0
    mov     x27, x1                  // length of b
    ADDR    x0, buf_a                // id of a
    mov     x1, x26
    mov     x2, #0
    mov     x3, #0
    bl      find_token
    tbnz    x0, #63, .Ltok_bad
    mov     x22, x0
    ADDR    x0, buf_b                // id of b
    mov     x1, x27
    mov     x2, #0
    mov     x3, #0
    bl      find_token
    tbnz    x0, #63, .Ltok_bad
    orr     x22, x0, x22, lsl #16    // key = (a << 16) | b
    ADDR    x0, buf_a                // id of a+b
    mov     x1, x26
    ADDR    x2, buf_b
    mov     x3, x27
    bl      find_token
    tbnz    x0, #63, .Ltok_bad
    MOV32   w1, 0x9E3779B1           // same hash as the lookup in tok.s
    mul     w2, w22, w1
    lsr     w2, w2, #(32 - T_HBITS)
14: add     x3, x2, x2, lsl #1
    add     x3, x20, x3, lsl #2
    ldr     w4, [x3]
    cmn     w4, #1
    b.eq    15f
    add     w2, w2, #1
    and     w2, w2, #((1 << T_HBITS) - 1)
    b       14b
15: str     w22, [x3]
    str     w23, [x3, #4]
    str     w0, [x3, #8]
    add     x23, x23, #1
    b       .Lm_line
.Lm_done:
    cbz     x23, .Ltok_bad
    ldp     x27, x28, [sp, #80]
    ldp     x25, x26, [sp, #64]
    ldp     x23, x24, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #96
    ret
.Ltok_bad:
    ADDR    x1, s_bad_tok
    b       fail

// decode_token(x0 = UTF-8 text, x1 = end, x2 = output) -> x0 = stop position,
// x1 = bytes written. Stops at a space, a newline or the end. Leaf.
decode_token:
    mov     x3, #0
1:  cmp     x0, x1
    b.hs    3f
    ldrb    w4, [x0]
    cmp     w4, #' '
    b.eq    3f
    cmp     w4, #'\n'
    b.eq    3f
    add     x0, x0, #1
    cmp     w4, #0x80
    b.lo    2f
    and     w4, w4, #0x1f            // 2-byte sequence: 110xxxxx 10xxxxxx
    ldrb    w5, [x0], #1
    and     w5, w5, #0x3f
    orr     w4, w5, w4, lsl #6
2:  CP2BYTE w4
    strb    w4, [x2, x3]
    add     x3, x3, #1
    cmp     x3, #255
    b.lo    1b
3:  mov     x1, x3
    ret

// strhash(x0 = p, x1 = len, x2 = q, x3 = len2) -> w0 = slot for bytes p ++ q.
// FNV-1a, then a multiply to spread the bits. Leaf.
strhash:
    MOV32   w9, 0x811C9DC5
    MOV32   w10, 0x01000193
    mov     x4, #0
1:  cmp     x4, x1
    b.hs    2f
    ldrb    w5, [x0, x4]
    eor     w9, w9, w5
    mul     w9, w9, w10
    add     x4, x4, #1
    b       1b
2:  mov     x4, #0
3:  cmp     x4, x3
    b.hs    4f
    ldrb    w5, [x2, x4]
    eor     w9, w9, w5
    mul     w9, w9, w10
    add     x4, x4, #1
    b       3b
4:  MOV32   w10, 0x9E3779B1
    mul     w0, w9, w10
    lsr     w0, w0, #(32 - S_HBITS)
    ret

// find_token(x0 = p, x1 = len, x2 = q, x3 = len2) -> x0 = id of the token
// with bytes p ++ q, or -1.
find_token:
    stp     x29, x30, [sp, #-48]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    mov     x19, x0
    mov     x20, x1
    mov     x21, x2
    mov     x22, x3
    bl      strhash
    ADDR    x6, strmap
    ADDR    x7, tokdata
    MOV32   x8, T_OFFS
    add     x8, x7, x8
    MOV32   x9, T_BLOB
    add     x9, x7, x9
1:  ldr     w10, [x6, x0, lsl #2]
    cmn     w10, #1
    b.eq    5f
    add     x11, x8, x10, lsl #2     // candidate's bytes and length
    ldp     w12, w13, [x11]
    sub     w13, w13, w12
    add     x14, x20, x22
    cmp     x13, x14
    b.ne    4f
    add     x12, x9, x12
    mov     x13, #0
2:  cmp     x13, x20                 // compare the first part
    b.hs    3f
    ldrb    w14, [x19, x13]
    ldrb    w15, [x12, x13]
    cmp     w14, w15
    b.ne    4f
    add     x13, x13, #1
    b       2b
3:  add     x12, x12, x20            // then the second part
    mov     x13, #0
31: cmp     x13, x22
    b.hs    6f
    ldrb    w14, [x21, x13]
    ldrb    w15, [x12, x13]
    cmp     w14, w15
    b.ne    4f
    add     x13, x13, #1
    b       31b
4:  add     w0, w0, #1
    and     w0, w0, #((1 << S_HBITS) - 1)
    b       1b
5:  mov     x10, #-1
6:  mov     x0, x10
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #48
    ret

// fail(x1 = message): print it to stderr and exit 1
fail:
    mov     x2, #0
1:  ldrb    w3, [x1, x2]
    cbz     w3, 2f
    add     x2, x2, #1
    b       1b
2:  mov     x0, #2
    SYSCALL SYS_WRITE
    mov     x0, #1
    SYSCALL SYS_EXIT

.section __TEXT,__const
.p2align 3
// The 4 matrices of a block: pointer index, inputs, outputs (and padding).
block_mats:
    .hword 2, N_EMBD, 3*N_EMBD, 0    // attn.c_attn
    .hword 4, N_EMBD, N_EMBD, 0      // attn.c_proj
    .hword 8, N_EMBD, 4*N_EMBD, 0    // mlp.c_fc
    .hword 10, 4*N_EMBD, N_EMBD, 0   // mlp.c_proj
top_names:
    .asciz "wte.weight"
    .asciz "wpe.weight"
    .asciz "ln_f.weight"
    .asciz "ln_f.bias"
    .byte 0
layer_names:
    .asciz "ln_1.weight"
    .asciz "ln_1.bias"
    .asciz "attn.c_attn.weight"
    .asciz "attn.c_attn.bias"
    .asciz "attn.c_proj.weight"
    .asciz "attn.c_proj.bias"
    .asciz "ln_2.weight"
    .asciz "ln_2.bias"
    .asciz "mlp.c_fc.weight"
    .asciz "mlp.c_fc.bias"
    .asciz "mlp.c_proj.weight"
    .asciz "mlp.c_proj.bias"
    .byte 0
s_bad_model: .asciz "gpt2: model.safetensors is not the GPT-2 small checkpoint\n"
s_no_memory: .asciz "gpt2: not enough memory for the tiled weights\n"
s_bad_tok:   .asciz "gpt2: cannot read vocab.json or merges.txt\n"

.globl _tensors, _head_w
.zerofill __DATA,__bss,_tensors,N_TENSORS*8,3
.zerofill __DATA,__bss,_head_w,8,3          // wte in tiles, for the output head
.zerofill __DATA,__bss,tokdata,T_BLOB + BLOB_MAX,4
.zerofill __DATA,__bss,strmap,(1 << S_HBITS)*4,4
.zerofill __DATA,__bss,buf_a,256,4
.zerofill __DATA,__bss,buf_b,256,4
