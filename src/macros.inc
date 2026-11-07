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
.equ N_CTX,    1024
.equ N_EMBD,   768
.equ N_LAYER,  12
.equ N_HEAD,   12
.equ EOT,      50256
.equ ROW,      N_EMBD*4            // bytes in one 768-float row

// Weight file: 64-byte header, then fp32 tensors (see tools/export.py).
.equ W_HEADER,   64
.equ WPE_OFF,    N_VOCAB*ROW
.equ LAYER0_OFF, (N_VOCAB+N_CTX)*ROW
// Byte offsets inside one transformer block.
.equ L_LN1_W,  0
.equ L_LN1_B,  768*4
.equ L_QKV_W,  1536*4
.equ L_QKV_B,  1771008*4
.equ L_PROJ_W, 1773312*4
.equ L_PROJ_B, 2363136*4
.equ L_LN2_W,  2363904*4
.equ L_LN2_B,  2364672*4
.equ L_FC_W,   2365440*4
.equ L_FC_B,   4724736*4
.equ L_FCP_W,  4727808*4
.equ L_FCP_B,  7087104*4
.equ L_SIZE,   7087872*4

// Tokenizer file layout (see tools/export.py).
.equ T_BYTE2ID, 16
.equ T_TABLE,   T_BYTE2ID + 256*4
.equ T_HBITS,   17
.equ T_OFFS,    T_TABLE + (1 << T_HBITS)*12
.equ T_BLOB,    T_OFFS + (N_VOCAB+1)*4
