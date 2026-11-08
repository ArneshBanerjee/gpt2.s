// Parallel matvec across processes, still with no libc.
//
// Threads are out: a new Darwin thread starts in libpthread's entry point.
// So after loading, main forks worker processes with the raw fork syscall.
// The weights are read-only and stay shared copy-on-write. The activations
// and the job block live in one section (__DATA,__shm) that main remaps
// MAP_SHARED at startup, before anything is written to it.
//
// A job is one matvec. Every process (main included) claims chunks of CHUNK
// rows with compare-and-swap and runs the ordinary _matvec on them. A row is
// computed by the same code whoever claims it, so the output is bit-identical
// to the single-process run.
.include "macros.inc"

.equ SYS_GETPID,  20
.equ SYS_GETPPID, 39
.equ SYS_FORK,    2
.equ SYS_WAIT4,   7
.equ SYS_SELECT,  93
.equ MACH_SWTCH_PRI, -59             // Mach trap: give up the core for a moment
.equ SPINS,       (1 << 14)          // then yield; workers sleep after 4x more
.equ CHUNK,       64                 // one tile of 64 outputs (src/load.s)

// job block: one 128-byte line of fields, then two counters on their own lines
.equ J_SEQ,   0      // even = stable; main makes it odd while it writes fields
.equ J_OUT,   8
.equ J_X,     16
.equ J_W,     24
.equ J_B,     32
.equ J_NOUT,  40
.equ J_NIN,   48
.equ J_QUIT,  56
.equ J_NEXT,  128    // (seq << 32) | next row to claim
.equ J_DONE,  256    // rows finished

.text
.p2align 2

// shm_init(): remap the __shm section as shared memory. Call first thing.
.globl _shm_init
_shm_init:
    ADDR    x0, section$start$__DATA$__shm
    ADDR    x1, section$end$__DATA$__shm
    sub     x1, x1, x0
    mov     x2, #0x3fff
    bic     x1, x1, x2               // whole 16 KB pages, rounded down: the
                                     // section ends in 16 KB of padding
    mov     x2, #3                   // PROT_READ | PROT_WRITE
    mov     x3, #0x1011              // MAP_ANON | MAP_FIXED | MAP_SHARED
    mov     x4, #-1
    mov     x5, #0
    SYSCALL SYS_MMAP
    b.cc    1f
    ADDR    x0, shm_failed           // private memory: no workers then
    mov     x1, #1
    str     x1, [x0]
1:  ret

// start_workers(x0 = count): fork that many workers. Returns in main only.
.globl _start_workers
_start_workers:
    stp     x29, x30, [sp, #-32]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    mov     x19, x0
    ADDR    x1, shm_failed
    ldr     x1, [x1]
    cbnz    x1, 2f
    ADDR    x1, nworkers
    str     x0, [x1]
    SYSCALL SYS_GETPID
    mov     x20, x0                  // our pid: workers exit if it goes away
1:  cbz     x19, 2f
    SYSCALL SYS_FORK
    b.cs    3f                       // fork failed: run with fewer workers
    cbnz    x1, worker               // x1 = 1 in the child
    sub     x19, x19, #1
    b       1b
3:  ADDR    x1, nworkers
    ldr     x0, [x1]
    sub     x0, x0, x19
    str     x0, [x1]
2:  ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #32
    ret

// stop_workers(): tell the workers to exit.
.globl _stop_workers
_stop_workers:
    ADDR    x0, _job
    mov     x1, #1
    stlr    x1, [x0]                 // never mind the seqlock: any change wakes them
    add     x0, x0, #J_QUIT
    str     x1, [x0]
    ret

// worker: never returns. x20 = parent pid.
// Waiting: spin, then give up the core (swtch_pri), then sleep 0.2 ms at a
// time, checking that main is still alive.
worker:
    ADDR    x19, _job
    mov     x27, #0                  // last job seen
    mov     x28, #0                  // spins since the last job
.Lw_wait:
    ldar    x0, [x19]
    ldr     x1, [x19, #J_QUIT]
    cbnz    x1, .Lw_exit
    tbnz    x0, #0, .Lw_spin         // main is writing the fields
    cmp     x0, x27
    b.eq    .Lw_spin
    ldp     x21, x22, [x19, #J_OUT]
    ldp     x23, x24, [x19, #J_W]
    ldp     x25, x26, [x19, #J_NOUT]
    dmb     ishld
    ldr     x1, [x19]
    cmp     x1, x0
    b.ne    .Lw_wait                 // torn read: try again
    mov     x27, x0
    bl      work
    mov     x28, #0
    b       .Lw_wait
.Lw_spin:
    add     x28, x28, #1
    cmp     x28, #SPINS
    b.hs    1f
    yield
    b       .Lw_wait
1:  cmp     x28, #(4 * SPINS)
    b.hs    2f
    mov     x0, #0
    SYSCALL MACH_SWTCH_PRI
    b       .Lw_wait
2:  SYSCALL SYS_GETPPID              // idle: is main still there?
    cmp     x0, x20
    b.ne    .Lw_exit
    bl      nap
    b       .Lw_wait
.Lw_exit:
    mov     x0, #0
    SYSCALL SYS_EXIT

// work(x0 = seq) with x19 = job, x21..x26 = out, x, W, bias, nout, nin.
// Claims chunks of the job tagged seq until none are left. nin = 0 marks an
// attention job: x21 = K cache, x22 = V cache, x23 = T, one head per claim.
work:
    stp     x29, x30, [sp, #-32]!
    mov     x29, sp
    stp     x27, x28, [sp, #16]
    mov     w27, w0                  // tag (low 32 bits of seq)
    lsl     x28, x26, #2             // row bytes
.Lk_claim:
    add     x9, x19, #J_NEXT
    ldr     x10, [x9]
    lsr     x11, x10, #32
    cmp     w11, w27
    b.ne    .Lk_done                 // a newer job: this one is finished
    mov     w12, w10                 // row
    cmp     x12, x25
    b.ge    .Lk_done
    mov     x15, #CHUNK
    cmp     x28, #0
    csinc   x15, x15, xzr, ne        // attention: one head per claim
    add     x13, x10, x15
    mov     x14, x10
    casal   x14, x13, [x9]
    cmp     x14, x10
    b.ne    .Lk_claim
    cbz     x28, .Lk_att
    sub     x4, x25, x12
    mov     x5, #CHUNK
    cmp     x4, x5
    csel    x4, x4, x5, lt           // rows in this chunk
    stp     x4, x12, [sp, #-16]!
    add     x0, x21, x12, lsl #2
    mov     x1, x22
    madd    x2, x12, x28, x23
    add     x3, x24, x12, lsl #2
    cmp     x24, #0
    csel    x3, xzr, x3, eq
    mov     x5, x26
    bl      _matvec
    ldp     x4, x12, [sp], #16
    add     x9, x19, #J_DONE
    staddl  x4, [x9]
    b       .Lk_claim
.Lk_att:
    mov     x0, x21
    mov     x1, x22
    mov     x2, x23
    mov     x3, x12
    bl      _attention_head
    mov     x4, #1
    add     x9, x19, #J_DONE
    staddl  x4, [x9]
    b       .Lk_claim
.Lk_done:
    ldp     x27, x28, [sp, #16]
    ldp     x29, x30, [sp], #32
    ret

// pattention(x0 = layer K cache, x1 = layer V cache, x2 = pos): the 12 heads,
// one per claim. Marked as an attention job by nin = 0.
.globl _pattention
_pattention:
    add     x2, x2, #1               // T
    mov     x3, #0
    mov     x4, #N_HEAD
    mov     x5, #0
    ADDR    x9, nworkers
    ldr     x9, [x9]
    cbnz    x9, pjob
    stp     x29, x30, [sp, #-48]!    // no workers: run the heads here
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    mov     x19, x0
    mov     x20, x1
    mov     x21, x2
    mov     x22, #0
1:  mov     x0, x19
    mov     x1, x20
    mov     x2, x21
    mov     x3, x22
    bl      _attention_head
    add     x22, x22, #1
    cmp     x22, #N_HEAD
    b.lt    1b
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #48
    ret

// pmatvec: same arguments as _matvec, split across the workers.
.globl _pmatvec
_pmatvec:
    ADDR    x9, nworkers
    ldr     x9, [x9]
    cbnz    x9, pjob
    b       _matvec
// pjob: publish the job in x0..x5, work on it here too, wait for the rest.
pjob:    stp     x29, x30, [sp, #-64]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #-16]!
    ADDR    x19, _job
    ldr     x20, [x19]
    add     x20, x20, #1
    str     x20, [x19]               // odd: fields changing
    dmb     ish
    stp     x0, x1, [x19, #J_OUT]
    stp     x2, x3, [x19, #J_W]
    stp     x4, x5, [x19, #J_NOUT]
    add     x20, x20, #1             // the new even seq
    lsl     x9, x20, #32
    str     x9, [x19, #J_NEXT]
    str     xzr, [x19, #J_DONE]
    stlr    x20, [x19]               // publish
    mov     x21, x0
    mov     x22, x1
    mov     x23, x2
    mov     x24, x3
    mov     x25, x4
    mov     x26, x5
    mov     x0, x20
    bl      work
    add     x24, x19, #J_DONE        // wait for the others
    mov     x26, #0                  // spins
1:  ldar    x10, [x24]
    cmp     x10, x25
    b.eq    2f
    add     x26, x26, #1
    cmp     x26, #SPINS
    b.hs    3f
    yield
    b       1b
3:  cmp     x26, #(4 * SPINS)
    b.hs    4f
    mov     x0, #0                   // a worker is slow (descheduled?): yield
    SYSCALL MACH_SWTCH_PRI
    tst     x26, #(SPINS - 1)
    b.ne    1b
    bl      _check_workers           // now and then: has one died?
    b       1b
4:  bl      _check_workers           // still waiting (a stopped worker?):
    bl      nap                      // check, then sleep 0.2 ms
    b       1b
2:  ldp     x25, x26, [sp], #16
    ldp     x23, x24, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #64
    ret

// nap(): sleep 0.2 ms (select with no descriptors).
nap:
    sub     sp, sp, #16
    str     xzr, [sp]                // tv_sec
    mov     x0, #200
    str     x0, [sp, #8]             // tv_usec
    mov     x0, #0
    mov     x1, #0
    mov     x2, #0
    mov     x3, #0
    mov     x4, sp
    SYSCALL SYS_SELECT
    add     sp, sp, #16
    ret

// check_workers(): if a worker has exited, stop: its rows may never arrive.
// Called while main waits for a job, and once per generated token.
.globl _check_workers
_check_workers:
    sub     sp, sp, #16
    mov     x0, #-1                  // any child
    mov     x1, sp                   // status
    mov     x2, #1                   // WNOHANG
    mov     x3, #0
    SYSCALL SYS_WAIT4
    add     sp, sp, #16
    b.cs    1f
    cmp     x0, #0
    b.gt    2f
1:  ret
2:  bl      _stop_workers
    mov     x0, #2
    ADDR    x1, s_dead
    mov     x2, #S_DEAD_LEN
    SYSCALL SYS_WRITE
    mov     x0, #1
    SYSCALL SYS_EXIT

.section __TEXT,__const
s_dead:       .ascii "\ngpt2: a worker process died\n"
.equ S_DEAD_LEN, 29

.data
.p2align 3
nworkers:     .quad 0
shm_failed:   .quad 0

// Shared section. _job first (page-aligned), the padding last, so whole pages
// cover every buffer and nothing else.
.globl _job, act_x, act_h, act_qkv, act_att, act_tmp, act_fc, _logits
.globl act_scores, kcache, vcache
.zerofill __DATA,__shm,_job,384,14
.zerofill __DATA,__shm,act_x,N_EMBD*4,7
.zerofill __DATA,__shm,act_h,N_EMBD*4,7
.zerofill __DATA,__shm,act_qkv,3*N_EMBD*4,7
.zerofill __DATA,__shm,act_att,N_EMBD*4,7
.zerofill __DATA,__shm,act_tmp,N_EMBD*4,7
.zerofill __DATA,__shm,act_fc,4*N_EMBD*4,7
.zerofill __DATA,__shm,_logits,N_VOCAB_PAD*4,7
.zerofill __DATA,__shm,act_scores,N_HEAD*N_CTX*4,7
.zerofill __DATA,__shm,kcache,N_LAYER*N_CTX*ROW,14
.zerofill __DATA,__shm,vcache,N_LAYER*N_CTX*ROW,14
.zerofill __DATA,__shm,shm_pad,16384,7
