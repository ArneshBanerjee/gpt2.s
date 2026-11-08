// Shared macros and constants. AArch64, Darwin (Mach-O) syntax.

.macro ADDR reg, sym
    adrp    \reg, \sym@PAGE
    add     \reg, \reg, \sym@PAGEOFF
.endm

// Load a 32-bit constant (works for w and x registers).
.macro MOV32 reg, val
    movz    \reg, #((\val) & 0xFFFF)
    movk    \reg, #(((\val) >> 16) & 0xFFFF), lsl #16
.endm

// Load a float constant from its IEEE-754 bit pattern. Clobbers w9.
.macro FCONST sreg, bits
    MOV32   w9, \bits
    fmov    \sreg, w9
.endm

// Raw Darwin syscall: number in x16, trap with svc #0x80. Carry set = error.
.macro SYSCALL num
    mov     x16, #\num
    svc     #0x80
.endm

.equ SYS_EXIT,   1
.equ SYS_WRITE,  4
.equ SYS_OPEN,   5
.equ SYS_MMAP,   197
.equ SYS_LSEEK,  199

// GPT-2 small
.equ N_VOCAB,  50257
.equ N_VOCAB_PAD, 50304             // rounded up to whole tiles of 64 rows
.equ N_CTX,    1024
.equ N_EMBD,   768
.equ TILE_BYTES, N_EMBD*64*4        // one tile of 64 outputs x 768 inputs
.equ N_LAYER,  12
.equ N_HEAD,   12
.equ EOT,      50256
.equ ROW,      N_EMBD*4            // bytes in one 768-float row

// _tensors (src/load.s): pointers into model.safetensors.
//   0 wte, 1 wpe, 2 ln_f.weight, 3 ln_f.bias, then 12 per block:
//   ln_1 w/b, attn.c_attn w/b, attn.c_proj w/b, ln_2 w/b, mlp.c_fc w/b, mlp.c_proj w/b
.equ N_TENSORS, 4 + 12*N_LAYER

// Tokenizer tables, built at startup by src/load.s.
.equ T_BYTE2ID, 16
.equ T_TABLE,   T_BYTE2ID + 256*4
.equ T_HBITS,   17
.equ T_OFFS,    T_TABLE + (1 << T_HBITS)*12
.equ T_BLOB,    T_OFFS + (N_VOCAB+1)*4
