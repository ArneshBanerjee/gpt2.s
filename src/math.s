// Math kernels: matvec, layernorm, softmax, GELU, expf, vec_add.
// No libm. exp() is range reduction + a degree-6 polynomial.
.include "macros.inc"

.text
.p2align 2

// matvec(x0 out, x1 x, x2 W, x3 bias or 0, x4 nout, x5 nin)
// out[i] = sum_j W[i][j] * x[j] + bias[i], with W stored in tiles of 64 outputs,
// [nout/64][nin][64] (src/load.s builds them). The 64 sums of a tile stay in
// v0-v15. One load of x brings 4 inputs, used by element, so the weights are
// the only stream read from memory, in order. nout must be a multiple of 64,
// nin a multiple of 4.
.globl _matvec
_matvec:
    ADDR    x9, _use_sme             // -s: the SME2 kernels in sme.s
    ldr     x9, [x9]
    ADDR    x10, _half_mode
    ldr     x10, [x10]
    cbz     x9, .Lmv_neon
    cbz     x10, .Lmv_sme32
    b       _matvec_sme_h
.Lmv_sme32:
    b       _matvec_sme
.Lmv_neon:
    cbnz    x10, _matvec_h
    stp     d8, d9, [sp, #-64]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]
1:  mov     x6, x1
    mov     x7, x5
    movi    v0.16b, #0
    movi    v1.16b, #0
    movi    v2.16b, #0
    movi    v3.16b, #0
    movi    v4.16b, #0
    movi    v5.16b, #0
    movi    v6.16b, #0
    movi    v7.16b, #0
    movi    v8.16b, #0
    movi    v9.16b, #0
    movi    v10.16b, #0
    movi    v11.16b, #0
    movi    v12.16b, #0
    movi    v13.16b, #0
    movi    v14.16b, #0
    movi    v15.16b, #0
2:  ld1     {v16.4s}, [x6], #16      // x[i..i+3]
    ld1     {v24.4s, v25.4s, v26.4s, v27.4s}, [x2], #64
    ld1     {v28.4s, v29.4s, v30.4s, v31.4s}, [x2], #64
    fmla    v0.4s, v24.4s, v16.s[0]
    fmla    v1.4s, v25.4s, v16.s[0]
    fmla    v2.4s, v26.4s, v16.s[0]
    fmla    v3.4s, v27.4s, v16.s[0]
    fmla    v4.4s, v28.4s, v16.s[0]
    fmla    v5.4s, v29.4s, v16.s[0]
    fmla    v6.4s, v30.4s, v16.s[0]
    fmla    v7.4s, v31.4s, v16.s[0]
    ld1     {v24.4s, v25.4s, v26.4s, v27.4s}, [x2], #64
    ld1     {v28.4s, v29.4s, v30.4s, v31.4s}, [x2], #64
    fmla    v8.4s, v24.4s, v16.s[0]
    fmla    v9.4s, v25.4s, v16.s[0]
    fmla    v10.4s, v26.4s, v16.s[0]
    fmla    v11.4s, v27.4s, v16.s[0]
    fmla    v12.4s, v28.4s, v16.s[0]
    fmla    v13.4s, v29.4s, v16.s[0]
    fmla    v14.4s, v30.4s, v16.s[0]
    fmla    v15.4s, v31.4s, v16.s[0]
    ld1     {v24.4s, v25.4s, v26.4s, v27.4s}, [x2], #64
    ld1     {v28.4s, v29.4s, v30.4s, v31.4s}, [x2], #64
    fmla    v0.4s, v24.4s, v16.s[1]
    fmla    v1.4s, v25.4s, v16.s[1]
    fmla    v2.4s, v26.4s, v16.s[1]
    fmla    v3.4s, v27.4s, v16.s[1]
    fmla    v4.4s, v28.4s, v16.s[1]
    fmla    v5.4s, v29.4s, v16.s[1]
    fmla    v6.4s, v30.4s, v16.s[1]
    fmla    v7.4s, v31.4s, v16.s[1]
    ld1     {v24.4s, v25.4s, v26.4s, v27.4s}, [x2], #64
    ld1     {v28.4s, v29.4s, v30.4s, v31.4s}, [x2], #64
    fmla    v8.4s, v24.4s, v16.s[1]
    fmla    v9.4s, v25.4s, v16.s[1]
    fmla    v10.4s, v26.4s, v16.s[1]
    fmla    v11.4s, v27.4s, v16.s[1]
    fmla    v12.4s, v28.4s, v16.s[1]
    fmla    v13.4s, v29.4s, v16.s[1]
    fmla    v14.4s, v30.4s, v16.s[1]
    fmla    v15.4s, v31.4s, v16.s[1]
    ld1     {v24.4s, v25.4s, v26.4s, v27.4s}, [x2], #64
    ld1     {v28.4s, v29.4s, v30.4s, v31.4s}, [x2], #64
    fmla    v0.4s, v24.4s, v16.s[2]
    fmla    v1.4s, v25.4s, v16.s[2]
    fmla    v2.4s, v26.4s, v16.s[2]
    fmla    v3.4s, v27.4s, v16.s[2]
    fmla    v4.4s, v28.4s, v16.s[2]
    fmla    v5.4s, v29.4s, v16.s[2]
    fmla    v6.4s, v30.4s, v16.s[2]
    fmla    v7.4s, v31.4s, v16.s[2]
    ld1     {v24.4s, v25.4s, v26.4s, v27.4s}, [x2], #64
    ld1     {v28.4s, v29.4s, v30.4s, v31.4s}, [x2], #64
    fmla    v8.4s, v24.4s, v16.s[2]
    fmla    v9.4s, v25.4s, v16.s[2]
    fmla    v10.4s, v26.4s, v16.s[2]
    fmla    v11.4s, v27.4s, v16.s[2]
    fmla    v12.4s, v28.4s, v16.s[2]
    fmla    v13.4s, v29.4s, v16.s[2]
    fmla    v14.4s, v30.4s, v16.s[2]
    fmla    v15.4s, v31.4s, v16.s[2]
    ld1     {v24.4s, v25.4s, v26.4s, v27.4s}, [x2], #64
    ld1     {v28.4s, v29.4s, v30.4s, v31.4s}, [x2], #64
    fmla    v0.4s, v24.4s, v16.s[3]
    fmla    v1.4s, v25.4s, v16.s[3]
    fmla    v2.4s, v26.4s, v16.s[3]
    fmla    v3.4s, v27.4s, v16.s[3]
    fmla    v4.4s, v28.4s, v16.s[3]
    fmla    v5.4s, v29.4s, v16.s[3]
    fmla    v6.4s, v30.4s, v16.s[3]
    fmla    v7.4s, v31.4s, v16.s[3]
    ld1     {v24.4s, v25.4s, v26.4s, v27.4s}, [x2], #64
    ld1     {v28.4s, v29.4s, v30.4s, v31.4s}, [x2], #64
    fmla    v8.4s, v24.4s, v16.s[3]
    fmla    v9.4s, v25.4s, v16.s[3]
    fmla    v10.4s, v26.4s, v16.s[3]
    fmla    v11.4s, v27.4s, v16.s[3]
    fmla    v12.4s, v28.4s, v16.s[3]
    fmla    v13.4s, v29.4s, v16.s[3]
    fmla    v14.4s, v30.4s, v16.s[3]
    fmla    v15.4s, v31.4s, v16.s[3]
    subs    x7, x7, #4
    b.ne    2b
    cbz     x3, 3f
    ld1     {v16.4s, v17.4s, v18.4s, v19.4s}, [x3], #64
    ld1     {v20.4s, v21.4s, v22.4s, v23.4s}, [x3], #64
    ld1     {v24.4s, v25.4s, v26.4s, v27.4s}, [x3], #64
    ld1     {v28.4s, v29.4s, v30.4s, v31.4s}, [x3], #64
    fadd    v0.4s, v0.4s, v16.4s
    fadd    v1.4s, v1.4s, v17.4s
    fadd    v2.4s, v2.4s, v18.4s
    fadd    v3.4s, v3.4s, v19.4s
    fadd    v4.4s, v4.4s, v20.4s
    fadd    v5.4s, v5.4s, v21.4s
    fadd    v6.4s, v6.4s, v22.4s
    fadd    v7.4s, v7.4s, v23.4s
    fadd    v8.4s, v8.4s, v24.4s
    fadd    v9.4s, v9.4s, v25.4s
    fadd    v10.4s, v10.4s, v26.4s
    fadd    v11.4s, v11.4s, v27.4s
    fadd    v12.4s, v12.4s, v28.4s
    fadd    v13.4s, v13.4s, v29.4s
    fadd    v14.4s, v14.4s, v30.4s
    fadd    v15.4s, v15.4s, v31.4s
3:  st1     {v0.4s, v1.4s, v2.4s, v3.4s}, [x0], #64
    st1     {v4.4s, v5.4s, v6.4s, v7.4s}, [x0], #64
    st1     {v8.4s, v9.4s, v10.4s, v11.4s}, [x0], #64
    st1     {v12.4s, v13.4s, v14.4s, v15.4s}, [x0], #64
    subs    x4, x4, #64
    b.ne    1b
    ldp     d10, d11, [sp, #16]
    ldp     d12, d13, [sp, #32]
    ldp     d14, d15, [sp, #48]
    ldp     d8, d9, [sp], #64
    ret

// matvec_h: as _matvec, but the tiles hold fp16 weights (-h). Each group of
// 32 weights is widened to fp32 as it is loaded; x, the products and the sums
// stay fp32, in the same order as _matvec.
.globl _matvec_h
_matvec_h:
    stp     d8, d9, [sp, #-64]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]
1:  mov     x6, x1
    mov     x7, x5
    movi    v0.16b, #0
    movi    v1.16b, #0
    movi    v2.16b, #0
    movi    v3.16b, #0
    movi    v4.16b, #0
    movi    v5.16b, #0
    movi    v6.16b, #0
    movi    v7.16b, #0
    movi    v8.16b, #0
    movi    v9.16b, #0
    movi    v10.16b, #0
    movi    v11.16b, #0
    movi    v12.16b, #0
    movi    v13.16b, #0
    movi    v14.16b, #0
    movi    v15.16b, #0
2:  ld1     {v16.4s}, [x6], #16      // x[i..i+3]
    ld1     {v24.8h, v25.8h, v26.8h, v27.8h}, [x2], #64
    fcvtl   v17.4s, v24.4h
    fcvtl2  v18.4s, v24.8h
    fcvtl   v19.4s, v25.4h
    fcvtl2  v20.4s, v25.8h
    fcvtl   v21.4s, v26.4h
    fcvtl2  v22.4s, v26.8h
    fcvtl   v23.4s, v27.4h
    fcvtl2  v28.4s, v27.8h
    fmla    v0.4s, v17.4s, v16.s[0]
    fmla    v1.4s, v18.4s, v16.s[0]
    fmla    v2.4s, v19.4s, v16.s[0]
    fmla    v3.4s, v20.4s, v16.s[0]
    fmla    v4.4s, v21.4s, v16.s[0]
    fmla    v5.4s, v22.4s, v16.s[0]
    fmla    v6.4s, v23.4s, v16.s[0]
    fmla    v7.4s, v28.4s, v16.s[0]
    ld1     {v24.8h, v25.8h, v26.8h, v27.8h}, [x2], #64
    fcvtl   v17.4s, v24.4h
    fcvtl2  v18.4s, v24.8h
    fcvtl   v19.4s, v25.4h
    fcvtl2  v20.4s, v25.8h
    fcvtl   v21.4s, v26.4h
    fcvtl2  v22.4s, v26.8h
    fcvtl   v23.4s, v27.4h
    fcvtl2  v28.4s, v27.8h
    fmla    v8.4s, v17.4s, v16.s[0]
    fmla    v9.4s, v18.4s, v16.s[0]
    fmla    v10.4s, v19.4s, v16.s[0]
    fmla    v11.4s, v20.4s, v16.s[0]
    fmla    v12.4s, v21.4s, v16.s[0]
    fmla    v13.4s, v22.4s, v16.s[0]
    fmla    v14.4s, v23.4s, v16.s[0]
    fmla    v15.4s, v28.4s, v16.s[0]
    ld1     {v24.8h, v25.8h, v26.8h, v27.8h}, [x2], #64
    fcvtl   v17.4s, v24.4h
    fcvtl2  v18.4s, v24.8h
    fcvtl   v19.4s, v25.4h
    fcvtl2  v20.4s, v25.8h
    fcvtl   v21.4s, v26.4h
    fcvtl2  v22.4s, v26.8h
    fcvtl   v23.4s, v27.4h
    fcvtl2  v28.4s, v27.8h
    fmla    v0.4s, v17.4s, v16.s[1]
    fmla    v1.4s, v18.4s, v16.s[1]
    fmla    v2.4s, v19.4s, v16.s[1]
    fmla    v3.4s, v20.4s, v16.s[1]
    fmla    v4.4s, v21.4s, v16.s[1]
    fmla    v5.4s, v22.4s, v16.s[1]
    fmla    v6.4s, v23.4s, v16.s[1]
    fmla    v7.4s, v28.4s, v16.s[1]
    ld1     {v24.8h, v25.8h, v26.8h, v27.8h}, [x2], #64
    fcvtl   v17.4s, v24.4h
    fcvtl2  v18.4s, v24.8h
    fcvtl   v19.4s, v25.4h
    fcvtl2  v20.4s, v25.8h
    fcvtl   v21.4s, v26.4h
    fcvtl2  v22.4s, v26.8h
    fcvtl   v23.4s, v27.4h
    fcvtl2  v28.4s, v27.8h
    fmla    v8.4s, v17.4s, v16.s[1]
    fmla    v9.4s, v18.4s, v16.s[1]
    fmla    v10.4s, v19.4s, v16.s[1]
    fmla    v11.4s, v20.4s, v16.s[1]
    fmla    v12.4s, v21.4s, v16.s[1]
    fmla    v13.4s, v22.4s, v16.s[1]
    fmla    v14.4s, v23.4s, v16.s[1]
    fmla    v15.4s, v28.4s, v16.s[1]
    ld1     {v24.8h, v25.8h, v26.8h, v27.8h}, [x2], #64
    fcvtl   v17.4s, v24.4h
    fcvtl2  v18.4s, v24.8h
    fcvtl   v19.4s, v25.4h
    fcvtl2  v20.4s, v25.8h
    fcvtl   v21.4s, v26.4h
    fcvtl2  v22.4s, v26.8h
    fcvtl   v23.4s, v27.4h
    fcvtl2  v28.4s, v27.8h
    fmla    v0.4s, v17.4s, v16.s[2]
    fmla    v1.4s, v18.4s, v16.s[2]
    fmla    v2.4s, v19.4s, v16.s[2]
    fmla    v3.4s, v20.4s, v16.s[2]
    fmla    v4.4s, v21.4s, v16.s[2]
    fmla    v5.4s, v22.4s, v16.s[2]
    fmla    v6.4s, v23.4s, v16.s[2]
    fmla    v7.4s, v28.4s, v16.s[2]
    ld1     {v24.8h, v25.8h, v26.8h, v27.8h}, [x2], #64
    fcvtl   v17.4s, v24.4h
    fcvtl2  v18.4s, v24.8h
    fcvtl   v19.4s, v25.4h
    fcvtl2  v20.4s, v25.8h
    fcvtl   v21.4s, v26.4h
    fcvtl2  v22.4s, v26.8h
    fcvtl   v23.4s, v27.4h
    fcvtl2  v28.4s, v27.8h
    fmla    v8.4s, v17.4s, v16.s[2]
    fmla    v9.4s, v18.4s, v16.s[2]
    fmla    v10.4s, v19.4s, v16.s[2]
    fmla    v11.4s, v20.4s, v16.s[2]
    fmla    v12.4s, v21.4s, v16.s[2]
    fmla    v13.4s, v22.4s, v16.s[2]
    fmla    v14.4s, v23.4s, v16.s[2]
    fmla    v15.4s, v28.4s, v16.s[2]
    ld1     {v24.8h, v25.8h, v26.8h, v27.8h}, [x2], #64
    fcvtl   v17.4s, v24.4h
    fcvtl2  v18.4s, v24.8h
    fcvtl   v19.4s, v25.4h
    fcvtl2  v20.4s, v25.8h
    fcvtl   v21.4s, v26.4h
    fcvtl2  v22.4s, v26.8h
    fcvtl   v23.4s, v27.4h
    fcvtl2  v28.4s, v27.8h
    fmla    v0.4s, v17.4s, v16.s[3]
    fmla    v1.4s, v18.4s, v16.s[3]
    fmla    v2.4s, v19.4s, v16.s[3]
    fmla    v3.4s, v20.4s, v16.s[3]
    fmla    v4.4s, v21.4s, v16.s[3]
    fmla    v5.4s, v22.4s, v16.s[3]
    fmla    v6.4s, v23.4s, v16.s[3]
    fmla    v7.4s, v28.4s, v16.s[3]
    ld1     {v24.8h, v25.8h, v26.8h, v27.8h}, [x2], #64
    fcvtl   v17.4s, v24.4h
    fcvtl2  v18.4s, v24.8h
    fcvtl   v19.4s, v25.4h
    fcvtl2  v20.4s, v25.8h
    fcvtl   v21.4s, v26.4h
    fcvtl2  v22.4s, v26.8h
    fcvtl   v23.4s, v27.4h
    fcvtl2  v28.4s, v27.8h
    fmla    v8.4s, v17.4s, v16.s[3]
    fmla    v9.4s, v18.4s, v16.s[3]
    fmla    v10.4s, v19.4s, v16.s[3]
    fmla    v11.4s, v20.4s, v16.s[3]
    fmla    v12.4s, v21.4s, v16.s[3]
    fmla    v13.4s, v22.4s, v16.s[3]
    fmla    v14.4s, v23.4s, v16.s[3]
    fmla    v15.4s, v28.4s, v16.s[3]
    subs    x7, x7, #4
    b.ne    2b
    cbz     x3, 3f
    ld1     {v16.4s, v17.4s, v18.4s, v19.4s}, [x3], #64
    ld1     {v20.4s, v21.4s, v22.4s, v23.4s}, [x3], #64
    ld1     {v24.4s, v25.4s, v26.4s, v27.4s}, [x3], #64
    ld1     {v28.4s, v29.4s, v30.4s, v31.4s}, [x3], #64
    fadd    v0.4s, v0.4s, v16.4s
    fadd    v1.4s, v1.4s, v17.4s
    fadd    v2.4s, v2.4s, v18.4s
    fadd    v3.4s, v3.4s, v19.4s
    fadd    v4.4s, v4.4s, v20.4s
    fadd    v5.4s, v5.4s, v21.4s
    fadd    v6.4s, v6.4s, v22.4s
    fadd    v7.4s, v7.4s, v23.4s
    fadd    v8.4s, v8.4s, v24.4s
    fadd    v9.4s, v9.4s, v25.4s
    fadd    v10.4s, v10.4s, v26.4s
    fadd    v11.4s, v11.4s, v27.4s
    fadd    v12.4s, v12.4s, v28.4s
    fadd    v13.4s, v13.4s, v29.4s
    fadd    v14.4s, v14.4s, v30.4s
    fadd    v15.4s, v15.4s, v31.4s
3:  st1     {v0.4s, v1.4s, v2.4s, v3.4s}, [x0], #64
    st1     {v4.4s, v5.4s, v6.4s, v7.4s}, [x0], #64
    st1     {v8.4s, v9.4s, v10.4s, v11.4s}, [x0], #64
    st1     {v12.4s, v13.4s, v14.4s, v15.4s}, [x0], #64
    subs    x4, x4, #64
    b.ne    1b
    ldp     d10, d11, [sp, #16]
    ldp     d12, d13, [sp, #32]
    ldp     d14, d15, [sp, #48]
    ldp     d8, d9, [sp], #64
    ret

// to_half(x0 dst, x1 src, x2 n): fp32 -> fp16, round to nearest. n a multiple
// of 8. dst may be at or before src (each write lands on data already read).
.globl _to_half
_to_half:
1:  ld1     {v0.4s, v1.4s}, [x1], #32
    fcvtn   v0.4h, v0.4s
    fcvtn2  v0.8h, v1.4s
    st1     {v0.8h}, [x0], #16
    subs    x2, x2, #8
    b.ne    1b
    ret

// layernorm(x0 out, x1 x, x2 gamma, x3 beta), n = 768, eps = 1e-5
.globl _layernorm
_layernorm:
    movi    v0.16b, #0
    movi    v1.16b, #0
    mov     x4, x1
    mov     x5, #N_EMBD
1:  ld1     {v2.4s, v3.4s}, [x4], #32
    fadd    v0.4s, v0.4s, v2.4s
    fadd    v1.4s, v1.4s, v3.4s
    subs    x5, x5, #8
    b.ne    1b
    fadd    v0.4s, v0.4s, v1.4s
    faddp   v0.4s, v0.4s, v0.4s
    faddp   s0, v0.2s
    FCONST  s5, 0x44400000          // 768.0
    fdiv    s0, s0, s5              // mean
    dup     v16.4s, v0.s[0]
    movi    v0.16b, #0
    movi    v1.16b, #0
    mov     x4, x1
    mov     x5, #N_EMBD
2:  ld1     {v2.4s, v3.4s}, [x4], #32
    fsub    v2.4s, v2.4s, v16.4s
    fsub    v3.4s, v3.4s, v16.4s
    fmla    v0.4s, v2.4s, v2.4s
    fmla    v1.4s, v3.4s, v3.4s
    subs    x5, x5, #8
    b.ne    2b
    fadd    v0.4s, v0.4s, v1.4s
    faddp   v0.4s, v0.4s, v0.4s
    faddp   s0, v0.2s
    fdiv    s0, s0, s5              // variance
    FCONST  s1, 0x3727c5ac          // 1e-5
    fadd    s0, s0, s1
    fsqrt   s0, s0
    fmov    s1, #1.0
    fdiv    s0, s1, s0              // 1/sqrt(var+eps)
    dup     v17.4s, v0.s[0]
    mov     x5, #N_EMBD
3:  ld1     {v2.4s}, [x1], #16
    ld1     {v3.4s}, [x2], #16
    ld1     {v4.4s}, [x3], #16
    fsub    v2.4s, v2.4s, v16.4s
    fmul    v2.4s, v2.4s, v17.4s
    fmla    v4.4s, v2.4s, v3.4s
    st1     {v4.4s}, [x0], #16
    subs    x5, x5, #4
    b.ne    3b
    ret

// s0 = exp(s0). Leaf. Clobbers s1, s2, w9, w10.
.globl _expf
_expf:
    FCONST  s1, 0x42b00000          // 88.0
    fminnm  s0, s0, s1
    FCONST  s1, 0xc2ae0000          // -87.0
    fmaxnm  s0, s0, s1
    FCONST  s1, 0x3fb8aa3b          // log2(e)
    fmul    s1, s0, s1
    frintn  s1, s1                  // k = round(x / ln2)
    fcvtzs  w10, s1
    FCONST  s2, 0x3f317200          // ln2 high part
    fmsub   s0, s1, s2, s0          // r = x - k*ln2_hi
    FCONST  s2, 0x35bfbe8e          // ln2 low part
    fmsub   s0, s1, s2, s0          //   - k*ln2_lo, |r| <= ln2/2
    FCONST  s1, 0x3ab60b61          // 1/720
    FCONST  s2, 0x3c088889          // 1/120
    fmadd   s1, s1, s0, s2
    FCONST  s2, 0x3d2aaaab          // 1/24
    fmadd   s1, s1, s0, s2
    FCONST  s2, 0x3e2aaaab          // 1/6
    fmadd   s1, s1, s0, s2
    fmov    s2, #0.5
    fmadd   s1, s1, s0, s2
    fmov    s2, #1.0
    fmadd   s1, s1, s0, s2
    fmadd   s1, s1, s0, s2          // p(r) ~= e^r
    add     w10, w10, #127
    lsl     w10, w10, #23
    fmov    s2, w10                 // 2^k built from raw exponent bits
    fmul    s0, s1, s2
    ret

// softmax(x0 x, x1 n), in place.
.globl _softmax
_softmax:
    stp     x29, x30, [sp, #-48]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     d8, d9, [sp, #32]
    mov     x19, x0
    mov     x20, x1
    ldr     s8, [x0]                // max
    mov     x2, #1
1:  cmp     x2, x20
    b.ge    2f
    ldr     s0, [x19, x2, lsl #2]
    fmax    s8, s8, s0
    add     x2, x2, #1
    b       1b
2:  movi    v9.16b, #0              // sum
    mov     x2, #0
3:  ldr     s0, [x19, x2, lsl #2]
    fsub    s0, s0, s8
    bl      _expf
    str     s0, [x19, x2, lsl #2]
    fadd    s9, s9, s0
    add     x2, x2, #1
    cmp     x2, x20
    b.lt    3b
    fmov    s0, #1.0
    fdiv    s9, s0, s9
    mov     x2, #0
4:  ldr     s0, [x19, x2, lsl #2]
    fmul    s0, s0, s9
    str     s0, [x19, x2, lsl #2]
    add     x2, x2, #1
    cmp     x2, x20
    b.lt    4b
    ldp     d8, d9, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #48
    ret

// gelu(x0 x, x1 n), in place. GPT-2's tanh form:
// 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3))), tanh(z) = 1 - 2/(e^2z + 1)
.globl _gelu
_gelu:
    stp     x29, x30, [sp, #-48]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    str     d8, [sp, #32]
    mov     x19, x0
    mov     x20, x1
1:  ldr     s8, [x19]
    fmul    s0, s8, s8
    fmul    s0, s0, s8
    FCONST  s1, 0x3d372713          // 0.044715
    fmadd   s0, s0, s1, s8
    FCONST  s1, 0x3f4c422a          // sqrt(2/pi)
    fmul    s0, s0, s1
    fadd    s0, s0, s0
    bl      _expf
    fmov    s1, #1.0
    fadd    s0, s0, s1
    fmov    s2, #2.0
    fdiv    s0, s2, s0
    fsub    s0, s1, s0              // tanh
    fadd    s0, s0, s1
    fmul    s0, s0, s8
    fmov    s1, #0.5
    fmul    s0, s0, s1
    str     s0, [x19], #4
    subs    x20, x20, #1
    b.ne    1b
    ldr     d8, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #48
    ret

// vec_add(x0 dst, x1 src), dst += src, n = 768
.globl _vec_add
_vec_add:
    mov     x2, #N_EMBD
1:  ld1     {v0.4s, v1.4s}, [x0]
    ld1     {v2.4s, v3.4s}, [x1], #32
    fadd    v0.4s, v0.4s, v2.4s
    fadd    v1.4s, v1.4s, v3.4s
    st1     {v0.4s, v1.4s}, [x0], #32
    subs    x2, x2, #8
    b.ne    1b
    ret
