// GPT-2 forward pass for one token, with a KV cache.
.include "macros.inc"

.text
.p2align 2

// forward(w0 = token, w1 = position). Leaves next-token scores in _logits.
.globl _forward
_forward:
    stp     x29, x30, [sp, #-64]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    mov     w23, w0
    mov     w22, w1
    ADDR    x0, _g_weights
    ldr     x19, [x0]
    add     x19, x19, #W_HEADER      // x19 = wte

    // x = wte[token] + wpe[pos]
    mov     x2, #ROW
    madd    x0, x23, x2, x19
    MOV32   x3, WPE_OFF
    add     x3, x19, x3
    madd    x1, x22, x2, x3
    ADDR    x4, act_x
    mov     x5, #N_EMBD
1:  ld1     {v0.4s, v1.4s}, [x0], #32
    ld1     {v2.4s, v3.4s}, [x1], #32
    fadd    v0.4s, v0.4s, v2.4s
    fadd    v1.4s, v1.4s, v3.4s
    st1     {v0.4s, v1.4s}, [x4], #32
    subs    x5, x5, #8
    b.ne    1b

    MOV32   x20, LAYER0_OFF
    add     x20, x19, x20            // x20 = current block weights
    mov     x21, #0                  // layer index
.Llayer:
    // h = ln_1(x)
    ADDR    x0, act_h
    ADDR    x1, act_x
    add     x2, x20, #L_LN1_W
    add     x3, x20, #L_LN1_B
    bl      _layernorm
    // qkv = h @ W_qkv + b
    ADDR    x0, act_qkv
    ADDR    x1, act_h
    MOV32   x2, L_QKV_W
    add     x2, x20, x2
    MOV32   x3, L_QKV_B
    add     x3, x20, x3
    mov     x4, #(3*N_EMBD)
    mov     x5, #N_EMBD
    bl      _matvec
    // append k, v to this layer's cache at row `pos`
    lsl     x24, x21, #10            // layer * N_CTX
    mov     x2, #ROW
    mul     x24, x24, x2             // byte offset of this layer's cache
    ADDR    x0, kcache
    add     x0, x0, x24
    madd    x0, x22, x2, x0
    ADDR    x1, vcache
    add     x1, x1, x24
    madd    x1, x22, x2, x1
    ADDR    x3, act_qkv
    add     x3, x3, #ROW             // k part
    add     x4, x3, #ROW             // v part
    mov     x5, #N_EMBD
2:  ld1     {v0.4s, v1.4s, v2.4s, v3.4s}, [x3], #64
    st1     {v0.4s, v1.4s, v2.4s, v3.4s}, [x0], #64
    ld1     {v4.4s, v5.4s, v6.4s, v7.4s}, [x4], #64
    st1     {v4.4s, v5.4s, v6.4s, v7.4s}, [x1], #64
    subs    x5, x5, #16
    b.ne    2b
    // att = causal self-attention over positions 0..pos
    ADDR    x0, kcache
    add     x0, x0, x24
    ADDR    x1, vcache
    add     x1, x1, x24
    mov     x2, x22
    bl      attention
    // x += att @ W_proj + b
    ADDR    x0, act_tmp
    ADDR    x1, act_att
    MOV32   x2, L_PROJ_W
    add     x2, x20, x2
    MOV32   x3, L_PROJ_B
    add     x3, x20, x3
    mov     x4, #N_EMBD
    mov     x5, #N_EMBD
    bl      _matvec
    ADDR    x0, act_x
    ADDR    x1, act_tmp
    bl      _vec_add
    // h = ln_2(x)
    ADDR    x0, act_h
    ADDR    x1, act_x
    MOV32   x2, L_LN2_W
    add     x2, x20, x2
    MOV32   x3, L_LN2_B
    add     x3, x20, x3
    bl      _layernorm
    // x += gelu(h @ W_fc + b) @ W_proj2 + b
    ADDR    x0, act_fc
    ADDR    x1, act_h
    MOV32   x2, L_FC_W
    add     x2, x20, x2
    MOV32   x3, L_FC_B
    add     x3, x20, x3
    mov     x4, #(4*N_EMBD)
    mov     x5, #N_EMBD
    bl      _matvec
    ADDR    x0, act_fc
    mov     x1, #(4*N_EMBD)
    bl      _gelu
    ADDR    x0, act_tmp
    ADDR    x1, act_fc
    MOV32   x2, L_FCP_W
    add     x2, x20, x2
    MOV32   x3, L_FCP_B
    add     x3, x20, x3
    mov     x4, #N_EMBD
    mov     x5, #(4*N_EMBD)
    bl      _matvec
    ADDR    x0, act_x
    ADDR    x1, act_tmp
    bl      _vec_add

    MOV32   x0, L_SIZE
    add     x20, x20, x0
    add     x21, x21, #1
    cmp     x21, #N_LAYER
    b.lt    .Llayer

    // x20 now points at ln_f. logits = ln_f(x) @ wte^T (tied weights)
    ADDR    x0, act_h
    ADDR    x1, act_x
    mov     x2, x20
    add     x3, x20, #ROW
    bl      _layernorm
    ADDR    x0, _logits
    ADDR    x1, act_h
    mov     x2, x19
    mov     x3, #0
    MOV32   x4, N_VOCAB
    mov     x5, #N_EMBD
    bl      _matvec

    ldp     x23, x24, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #64
    ret

// attention(x0 = layer K cache, x1 = layer V cache, x2 = pos) -> act_att
// 12 heads of 64 dims. Reads q from act_qkv.
attention:
    stp     x29, x30, [sp, #-64]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    mov     x19, x0
    mov     x20, x1
    add     x21, x2, #1              // T = pos + 1
    mov     x22, #0                  // head
.Lhead:
    lsl     x23, x22, #8             // head * 64 floats * 4 bytes
    // q for this head -> v16..v31
    ADDR    x0, act_qkv
    add     x0, x0, x23
    ld1     {v16.4s, v17.4s, v18.4s, v19.4s}, [x0], #64
    ld1     {v20.4s, v21.4s, v22.4s, v23.4s}, [x0], #64
    ld1     {v24.4s, v25.4s, v26.4s, v27.4s}, [x0], #64
    ld1     {v28.4s, v29.4s, v30.4s, v31.4s}, [x0], #64
    // scores[t] = q . k[t] / 8
    ADDR    x24, act_scores
    add     x4, x19, x23
    mov     x6, #0
    mov     x8, #ROW
    fmov    s2, #0.125               // 1/sqrt(64)
1:  madd    x7, x6, x8, x4
    ld1     {v4.4s, v5.4s, v6.4s, v7.4s}, [x7], #64
    fmul    v0.4s, v4.4s, v16.4s
    fmul    v1.4s, v5.4s, v17.4s
    fmla    v0.4s, v6.4s, v18.4s
    fmla    v1.4s, v7.4s, v19.4s
    ld1     {v4.4s, v5.4s, v6.4s, v7.4s}, [x7], #64
    fmla    v0.4s, v4.4s, v20.4s
    fmla    v1.4s, v5.4s, v21.4s
    fmla    v0.4s, v6.4s, v22.4s
    fmla    v1.4s, v7.4s, v23.4s
    ld1     {v4.4s, v5.4s, v6.4s, v7.4s}, [x7], #64
    fmla    v0.4s, v4.4s, v24.4s
    fmla    v1.4s, v5.4s, v25.4s
    fmla    v0.4s, v6.4s, v26.4s
    fmla    v1.4s, v7.4s, v27.4s
    ld1     {v4.4s, v5.4s, v6.4s, v7.4s}, [x7], #64
    fmla    v0.4s, v4.4s, v28.4s
    fmla    v1.4s, v5.4s, v29.4s
    fmla    v0.4s, v6.4s, v30.4s
    fmla    v1.4s, v7.4s, v31.4s
    fadd    v0.4s, v0.4s, v1.4s
    faddp   v0.4s, v0.4s, v0.4s
    faddp   s0, v0.2s
    fmul    s0, s0, s2
    str     s0, [x24, x6, lsl #2]
    add     x6, x6, #1
    cmp     x6, x21
    b.lt    1b
    mov     x0, x24
    mov     x1, x21
    bl      _softmax
    // out = sum_t p[t] * v[t]
    movi    v16.16b, #0
    movi    v17.16b, #0
    movi    v18.16b, #0
    movi    v19.16b, #0
    movi    v20.16b, #0
    movi    v21.16b, #0
    movi    v22.16b, #0
    movi    v23.16b, #0
    movi    v24.16b, #0
    movi    v25.16b, #0
    movi    v26.16b, #0
    movi    v27.16b, #0
    movi    v28.16b, #0
    movi    v29.16b, #0
    movi    v30.16b, #0
    movi    v31.16b, #0
    add     x4, x20, x23
    mov     x6, #0
    mov     x8, #ROW
2:  ldr     s0, [x24, x6, lsl #2]
    madd    x7, x6, x8, x4
    ld1     {v4.4s, v5.4s, v6.4s, v7.4s}, [x7], #64
    fmla    v16.4s, v4.4s, v0.s[0]
    fmla    v17.4s, v5.4s, v0.s[0]
    fmla    v18.4s, v6.4s, v0.s[0]
    fmla    v19.4s, v7.4s, v0.s[0]
    ld1     {v4.4s, v5.4s, v6.4s, v7.4s}, [x7], #64
    fmla    v20.4s, v4.4s, v0.s[0]
    fmla    v21.4s, v5.4s, v0.s[0]
    fmla    v22.4s, v6.4s, v0.s[0]
    fmla    v23.4s, v7.4s, v0.s[0]
    ld1     {v4.4s, v5.4s, v6.4s, v7.4s}, [x7], #64
    fmla    v24.4s, v4.4s, v0.s[0]
    fmla    v25.4s, v5.4s, v0.s[0]
    fmla    v26.4s, v6.4s, v0.s[0]
    fmla    v27.4s, v7.4s, v0.s[0]
    ld1     {v4.4s, v5.4s, v6.4s, v7.4s}, [x7], #64
    fmla    v28.4s, v4.4s, v0.s[0]
    fmla    v29.4s, v5.4s, v0.s[0]
    fmla    v30.4s, v6.4s, v0.s[0]
    fmla    v31.4s, v7.4s, v0.s[0]
    add     x6, x6, #1
    cmp     x6, x21
    b.lt    2b
    ADDR    x0, act_att
    add     x0, x0, x23
    st1     {v16.4s, v17.4s, v18.4s, v19.4s}, [x0], #64
    st1     {v20.4s, v21.4s, v22.4s, v23.4s}, [x0], #64
    st1     {v24.4s, v25.4s, v26.4s, v27.4s}, [x0], #64
    st1     {v28.4s, v29.4s, v30.4s, v31.4s}, [x0], #64
    add     x22, x22, #1
    cmp     x22, #N_HEAD
    b.lt    .Lhead
    ldp     x23, x24, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #64
    ret

// Activations and KV cache (zero-filled, no allocator needed).
.zerofill __DATA,__bss,act_x,N_EMBD*4,6
.zerofill __DATA,__bss,act_h,N_EMBD*4,6
.zerofill __DATA,__bss,act_qkv,3*N_EMBD*4,6
.zerofill __DATA,__bss,act_att,N_EMBD*4,6
.zerofill __DATA,__bss,act_tmp,N_EMBD*4,6
.zerofill __DATA,__bss,act_fc,4*N_EMBD*4,6
.zerofill __DATA,__bss,act_scores,N_CTX*4,6
.zerofill __DATA,__bss,kcache,N_LAYER*N_CTX*ROW,14
.zerofill __DATA,__bss,vcache,N_LAYER*N_CTX*ROW,14
.globl _logits
.zerofill __DATA,__bss,_logits,(N_VOCAB+15)*4,6
