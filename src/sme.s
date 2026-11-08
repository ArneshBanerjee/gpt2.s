// SME2 matrix-vector kernel (Arm's Scalable Matrix Extension 2, documented in
// the Arm architecture reference manual and the ACLE).
//
// Same contract and tile layout as _matvec in math.s: W in tiles of 64 outputs,
// [nout/64][nin][64]. The 64 sums of a tile live in the ZA array (one group of
// 4 vectors of 16 floats). For each input, one 4-vector load brings its 64
// weights and one FMLA (multiple vectors, indexed) adds weight * x[i] into
// all 64 sums. Every sum is accumulated in the same order as _matvec, with
// fused multiply-adds, so the results are bit-identical for finite values (in
// our tests on an M6, a NaN came out as the default NaN, while NEON kept its
// payload).
//
// Written for a 512-bit streaming vector length (4 vectors = 64 floats); main.s
// uses it only when rdsvl reports 64 bytes. On the M6, FPSR read 0x0800009f
// after smstart/smstop; nothing in this program reads FPSR. This is a private
// routine of this program: it does not follow the AAPCS64 SME rules for callers
// with live ZA (TPIDR2_EL0 lazy saves), which plain C callers would need.
//
// In a microbenchmark on an M6, one thread using SME2 read memory at about
// 141 GB/s, against about 87 GB/s for one thread using NEON.
.include "macros.inc"
.arch_extension sme2
.arch_extension sme-f16f16           // multi-vector fcvt fp16 -> fp32 (-h)

.text
.p2align 2

// matvec_sme(x0 out, x1 x, x2 W tiles, x3 bias or 0, x4 nout, x5 nin)
// nout a multiple of 64, nin a multiple of 4.
.globl _matvec_sme
_matvec_sme:
    stp     d8, d9, [sp, #-64]!      // smstart and smstop clear the vector
    stp     d10, d11, [sp, #16]      // registers; d8-d15 are callee-saved
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]
    smstart                          // streaming mode and ZA on
    ptrue   p0.s
    ptrue   pn8.s
    mov     w8, #0                   // ZA slice base
1:  zero    {za}
    mov     x6, x1
    mov     x7, x5
2:  ld1rqw  {z4.s}, p0/z, [x6]       // x[i..i+3], repeated in every 128-bit block
    add     x6, x6, #16
    ld1w    {z0.s-z3.s}, pn8/z, [x2]                 // input i: 64 weights
    fmla    za.s[w8, 0, vgx4], {z0.s-z3.s}, z4.s[0]
    ld1w    {z16.s-z19.s}, pn8/z, [x2, #4, mul vl]   // input i+1
    fmla    za.s[w8, 0, vgx4], {z16.s-z19.s}, z4.s[1]
    ld1w    {z20.s-z23.s}, pn8/z, [x2, #8, mul vl]   // input i+2
    fmla    za.s[w8, 0, vgx4], {z20.s-z23.s}, z4.s[2]
    ld1w    {z24.s-z27.s}, pn8/z, [x2, #12, mul vl]  // input i+3
    fmla    za.s[w8, 0, vgx4], {z24.s-z27.s}, z4.s[3]
    addvl   x2, x2, #16
    subs    x7, x7, #4
    b.ne    2b
    mova    {z0.s-z3.s}, za.s[w8, 0, vgx4]           // the 64 sums
    cbz     x3, 3f
    ld1w    {z16.s-z19.s}, pn8/z, [x3]
    fadd    z0.s, z0.s, z16.s
    fadd    z1.s, z1.s, z17.s
    fadd    z2.s, z2.s, z18.s
    fadd    z3.s, z3.s, z19.s
    add     x3, x3, #256
3:  st1w    {z0.s-z3.s}, pn8, [x0]
    add     x0, x0, #256
    subs    x4, x4, #64
    b.ne    1b
    smstop
    ldp     d10, d11, [sp, #16]
    ldp     d12, d13, [sp, #32]
    ldp     d14, d15, [sp, #48]
    ldp     d8, d9, [sp], #64
    ret

// matvec_sme_h: as _matvec_sme, but the tiles hold fp16 weights (-h). Per
// input, one 2-vector load brings its 64 fp16 weights and two multi-vector
// FCVTs widen them to 4 fp32 vectors; x, the products and the sums stay fp32
// in the same order as _matvec_h, so the results are bit-identical to it.
// The FCVTs need FEAT_SME_F16F16; main.s checks for it before using this.
.globl _matvec_sme_h
_matvec_sme_h:
    stp     d8, d9, [sp, #-64]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]
    smstart
    ptrue   p0.s
    ptrue   pn8.h                    // fp16 loads
    ptrue   pn9.s                    // fp32 bias loads and stores
    mov     w8, #0
1:  zero    {za}
    mov     x6, x1
    mov     x7, x5
2:  ld1rqw  {z4.s}, p0/z, [x6]       // x[i..i+3], repeated in every 128-bit block
    add     x6, x6, #16
    ld1h    {z0.h-z1.h}, pn8/z, [x2]                 // input i: 64 fp16 weights
    fcvt    {z16.s-z17.s}, z0.h
    fcvt    {z18.s-z19.s}, z1.h
    fmla    za.s[w8, 0, vgx4], {z16.s-z19.s}, z4.s[0]
    ld1h    {z2.h-z3.h}, pn8/z, [x2, #2, mul vl]     // input i+1
    fcvt    {z20.s-z21.s}, z2.h
    fcvt    {z22.s-z23.s}, z3.h
    fmla    za.s[w8, 0, vgx4], {z20.s-z23.s}, z4.s[1]
    ld1h    {z0.h-z1.h}, pn8/z, [x2, #4, mul vl]     // input i+2
    fcvt    {z24.s-z25.s}, z0.h
    fcvt    {z26.s-z27.s}, z1.h
    fmla    za.s[w8, 0, vgx4], {z24.s-z27.s}, z4.s[2]
    ld1h    {z2.h-z3.h}, pn8/z, [x2, #6, mul vl]     // input i+3
    fcvt    {z28.s-z29.s}, z2.h
    fcvt    {z30.s-z31.s}, z3.h
    fmla    za.s[w8, 0, vgx4], {z28.s-z31.s}, z4.s[3]
    addvl   x2, x2, #8
    subs    x7, x7, #4
    b.ne    2b
    mova    {z0.s-z3.s}, za.s[w8, 0, vgx4]
    cbz     x3, 3f
    ld1w    {z16.s-z19.s}, pn9/z, [x3]
    fadd    z0.s, z0.s, z16.s
    fadd    z1.s, z1.s, z17.s
    fadd    z2.s, z2.s, z18.s
    fadd    z3.s, z3.s, z19.s
    add     x3, x3, #256
3:  st1w    {z0.s-z3.s}, pn9, [x0]
    add     x0, x0, #256
    subs    x4, x4, #64
    b.ne    1b
    smstop
    ldp     d10, d11, [sp, #16]
    ldp     d12, d13, [sp, #32]
    ldp     d14, d15, [sp, #48]
    ldp     d8, d9, [sp], #64
    ret
