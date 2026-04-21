/* T3 anchor-test — native FP64 throughput micro-benchmark.
 *
 * Replaces the pure-Python probe that was proportionally bounded by
 * CPython interpreter overhead (~0.001 GFLOPS) and therefore always
 * reported a ratio << 1 vs the 2007 Harpertown 9 GFLOPS peak baseline,
 * unconditionally triggering the HARDWARE_MISMATCH hard-fail.
 *
 * Kernel: tight FMA/MAC loop on L2-resident double arrays. Sized so the
 * working set (two 64 KB arrays = 128 KB) fits comfortably in L2 across
 * common desktop and server CPUs since ~2010, so the measurement is
 * compute-bound rather than memory-bound. Compiled with -O3 -march=native
 * so the compiler auto-vectorises (AVX2 / AVX-512 / NEON), yielding a
 * near-peak single-core FP64 throughput estimate that is directly
 * comparable to the Harpertown 9 GFLOPS peak constant.
 *
 * Emits a single scalar (GFLOPS, FP64) to stdout. Exit code 0 on success.
 *
 * Build (done by hardware_normalization.py on first use):
 *   cc -O3 -march=native -ffast-math -o hw_probe hardware_flops_probe.c -lm
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* 8192 doubles per array = 64 KB; two arrays = 128 KB (L2-resident on any
 * CPU since ~2010). Chosen to be a pure compute loop — no cache misses. */
#define N 8192

/* Rep count tuned so wall-time is ~0.3-1.0 sec on a 2010-era CPU; modern
 * silicon finishes in ~50-200 ms. Short enough that thermal throttling
 * stays off, long enough that clock_gettime() granularity is irrelevant. */
#define REPS 60000

/* Per-element flop count inside the inner kernel body:
 *   y[i] = y[i] * alpha + x[i]   →   1 MUL + 1 ADD = 2 flops */
#define FLOPS_PER_ITER 2.0

static double wall_time(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double) ts.tv_sec + 1.0e-9 * (double) ts.tv_nsec;
}

/* Deterministic 64-bit LCG → [0, 1) double. Matches the Python probe's
 * seed (0xDEADBEEF) so cross-probe runs stay comparable across calls. */
static double rng_next(unsigned long long* state) {
  *state = *state * 6364136223846793005ULL + 1442695040888963407ULL;
  return (double) (*state >> 11) / (double) (1ULL << 53);
}

/* Main kernel: two-stream DAXPY-MAC to expose ILP.
 * Attribute pragmas hint the compiler to vectorise aggressively. */
static double daxpy_peak(double* restrict x,
                         double* restrict y,
                         double alpha,
                         double beta,
                         int reps) {
  for (int r = 0; r < reps; ++r) {
    for (int i = 0; i < N; ++i) {
      y[i] = y[i] * alpha + x[i];
    }
    for (int i = 0; i < N; ++i) {
      x[i] = x[i] * beta + y[i];
    }
  }
  double s = 0.0;
  for (int i = 0; i < N; ++i)
    s += y[i];
  return s;
}

int main(void) {
  double* x = (double*) aligned_alloc(64, N * sizeof(double));
  double* y = (double*) aligned_alloc(64, N * sizeof(double));
  if (!x || !y) {
    fprintf(stderr, "hardware_flops_probe: alloc failed\n");
    free(x);
    free(y);
    return 1;
  }

  unsigned long long state = 0xDEADBEEFULL;
  for (int i = 0; i < N; ++i) {
    x[i] = rng_next(&state);
    y[i] = rng_next(&state);
  }

  /* Warm-up: evicts cold caches, settles DVFS on modern CPUs. */
  (void) daxpy_peak(x, y, 1.0000001, 0.9999999, REPS / 10);

  double t0 = wall_time();
  double sink = daxpy_peak(x, y, 1.0000001, 0.9999999, REPS);
  double t1 = wall_time();

  double total_time = t1 - t0;
  /* Two kernel passes per rep × N elements × FLOPS_PER_ITER flops. */
  double total_flops = 2.0 * (double) REPS * (double) N * FLOPS_PER_ITER;
  double gflops = total_flops / total_time / 1.0e9;

  /* Emit sink to stderr so -O3 can't dead-code-eliminate the kernel. */
  if (sink == 4.242e42)
    fprintf(stderr, "%g", sink);

  printf("%.6f\n", gflops);

  free(x);
  free(y);
  return 0;
}
