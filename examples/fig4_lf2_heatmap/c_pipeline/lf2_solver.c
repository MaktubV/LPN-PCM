/* LF2 Algorithm Solver — C implementation for speed.
 *
 * Solves LPN instances using the LF2 (Levieil-Fouque 2) algorithm.
 * Reads a binary problem file, writes 0/1 results per problem.
 *
 * Parameters (hardcoded to match experiment):
 *   K=16, A=2, B=8, K_PRIME=8, ROTATIONS=2
 *
 * Compile: gcc -O3 -march=native -std=c11 -o lf2_solver.exe lf2_solver.c
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Constants ─────────────────────────────────────────────────────── */

#define K            16
#define A_PARAM       2
#define B_PARAM       8
#define K_PRIME       8
#define ROTATIONS     2

#define MAGIC        0x4C463232u  /* "LF22" */

#define MAX_FCL            5000
#define MAX_REDUCED_SAMPLES (1u << 20)  /* 1M safety cap */

/* ── Data structures ────────────────────────────────────────────────── */

/* One (equation, parity) pair */
typedef struct {
    uint16_t eq;
    uint8_t  parity;
} Sample;

/* Dynamic array of samples */
typedef struct {
    Sample  *data;
    uint32_t size;
    uint32_t capacity;
} SampleArray;

/* Group data for LF2 reduce */
typedef struct {
    uint32_t count;
    uint32_t offset;   /* offset into flat storage */
} Group;

/* ── Allocator ──────────────────────────────────────────────────────── */

static SampleArray *sa_create(uint32_t capacity) {
    SampleArray *sa = malloc(sizeof(SampleArray));
    if (!sa) { fprintf(stderr, "malloc failed\n"); exit(1); }
    sa->data = malloc((size_t)capacity * sizeof(Sample));
    if (!sa->data) { fprintf(stderr, "malloc failed\n"); exit(1); }
    sa->size = 0;
    sa->capacity = capacity;
    return sa;
}

static void sa_free(SampleArray *sa) {
    if (sa) {
        free(sa->data);
        free(sa);
    }
}

/* ── Bit utilities ──────────────────────────────────────────────────── */

static inline int popcount16(uint16_t x) {
    return __builtin_popcount(x);
}

/* Right-rotate x by shift bits within K bits */
static inline uint16_t rotate_right(uint16_t x, int shift) {
    shift &= (K - 1);
    if (shift == 0) return x;
    return ((x >> shift) | (x << (K - shift))) & ((1u << K) - 1);
}

/* Left-rotate x by shift bits within K bits */
static inline uint16_t rotate_left(uint16_t x, int shift) {
    shift &= (K - 1);
    if (shift == 0) return x;
    return ((x << shift) | (x >> (K - shift))) & ((1u << K) - 1);
}

/* ── LF2 Reduce Round ───────────────────────────────────────────────── */

/* Group samples by bits [bit_start, bit_end), XOR all unordered pairs.
 * Returns a newly allocated SampleArray. Caller must sa_free() it. */
static SampleArray *lf2_reduce_round(const Sample *samples, uint32_t n,
                                     int bit_start, int bit_end) {
    int      n_bits  = bit_end - bit_start;          /* = 8 */
    uint32_t n_groups = 1u << n_bits;                 /* = 256 */
    uint16_t mask = ((1u << n_bits) - 1) << bit_start;

    /* ── Pass 1: count elements per group ── */
    Group *groups = calloc(n_groups, sizeof(Group));
    if (!groups) { fprintf(stderr, "calloc failed\n"); exit(1); }

    for (uint32_t i = 0; i < n; i++) {
        uint32_t key = (samples[i].eq & mask) >> bit_start;
        groups[key].count++;
    }

    /* ── Compute offsets and total pairs ── */
    uint64_t total_pairs = 0;
    uint32_t offset = 0;
    for (uint32_t g = 0; g < n_groups; g++) {
        groups[g].offset = offset;
        offset += groups[g].count;
        uint64_t c = groups[g].count;
        total_pairs += c * (c - 1) / 2;
    }

    /* Safety cap */
    if (total_pairs > MAX_REDUCED_SAMPLES) {
        total_pairs = MAX_REDUCED_SAMPLES;
    }

    SampleArray *result = sa_create((uint32_t)total_pairs);

    /* ── Pass 2: fill flat storage ── */
    Sample *flat = malloc(n * sizeof(Sample));
    if (!flat) { fprintf(stderr, "malloc failed\n"); exit(1); }

    /* Reset offsets to use as write cursors */
    uint32_t *cursors = malloc(n_groups * sizeof(uint32_t));
    if (!cursors) { fprintf(stderr, "malloc failed\n"); exit(1); }
    for (uint32_t g = 0; g < n_groups; g++) {
        cursors[g] = groups[g].offset;
    }

    for (uint32_t i = 0; i < n; i++) {
        uint32_t key = (samples[i].eq & mask) >> bit_start;
        flat[cursors[key]++] = samples[i];
    }

    free(cursors);

    /* ── Pairing: XOR all unordered pairs within each group ── */
    for (uint32_t g = 0; g < n_groups; g++) {
        uint32_t cnt  = groups[g].count;
        uint32_t base = groups[g].offset;
        if (cnt < 2) continue;

        for (uint32_t i = 0; i < cnt && result->size < (uint32_t)total_pairs; i++) {
            uint16_t eq_i = flat[base + i].eq;
            uint8_t  p_i  = flat[base + i].parity;
            for (uint32_t j = i + 1; j < cnt && result->size < (uint32_t)total_pairs; j++) {
                result->data[result->size].eq     = eq_i ^ flat[base + j].eq;
                result->data[result->size].parity = p_i  ^ flat[base + j].parity;
                result->size++;
            }
        }
    }

    free(flat);
    free(groups);
    return result;
}

/* ── Exhaustive Search ──────────────────────────────────────────────── */

/* Search all 2^k candidates. k must be ≤ 12 for reasonable runtime.
 * Returns the candidate with the highest score (most matching parities). */
static uint16_t exhaustive_search(const Sample *samples, uint32_t n, int k) {
    uint32_t num_candidates = 1u << k;
    uint16_t best_candidate = 0;
    int      best_score     = -1;

    /* Pre-extract low k bits of each equation for fast inner loop */
    uint16_t mask = (1u << k) - 1;
    uint16_t *eq_low = malloc(n * sizeof(uint16_t));
    if (!eq_low) { fprintf(stderr, "malloc failed\n"); exit(1); }
    for (uint32_t i = 0; i < n; i++) {
        eq_low[i] = samples[i].eq & mask;
    }

    for (uint32_t cand = 0; cand < num_candidates; cand++) {
        int score = 0;
        for (uint32_t i = 0; i < n; i++) {
            int parity = popcount16(eq_low[i] & (uint16_t)cand) & 1;
            if (parity == samples[i].parity) {
                score++;
            }
        }
        if (score > best_score) {
            best_score = score;
            best_candidate = (uint16_t)cand;
        }
    }

    free(eq_low);
    return best_candidate;
}

/* ── LF2 Solve Rotated ──────────────────────────────────────────────── */

/* Recover a segment of the secret at a given rotation.
 * Returns a packed uint32: bits [15:0] = recovered_original,
 *                           bits [31:16] = recovered_mask */
static uint32_t lf2_solve_rotated(const uint16_t *A, const uint8_t *b_list,
                                  uint32_t n, int rotation) {
    /* Build rotated samples */
    Sample *rotated = malloc(n * sizeof(Sample));
    if (!rotated) { fprintf(stderr, "malloc failed\n"); exit(1); }
    for (uint32_t i = 0; i < n; i++) {
        rotated[i].eq     = rotate_right(A[i], rotation);
        rotated[i].parity = b_list[i];
    }

    /* Perform one round of LF2 reduce (a=2 -> exactly 1 round) */
    Sample  *cur_eq   = rotated;
    uint32_t cur_n    = n;
    int      cur_k    = K;

    /* a-1 = 1 round */
    {
        int bit_start = cur_k - B_PARAM;  /* 16 - 8 = 8 */
        int bit_end   = cur_k;            /* 16 */

        SampleArray *reduced = lf2_reduce_round(cur_eq, cur_n, bit_start, bit_end);

        /* Now use the reduced samples */
        cur_eq = reduced->data;
        cur_n  = reduced->size;
        cur_k -= B_PARAM;  /* now 8 */

        /* Exhaustive search */
        uint16_t recovered_rotated = exhaustive_search(cur_eq, cur_n, cur_k);

        /* Rotate back to original coordinates */
        uint16_t recovered_original = rotate_left(recovered_rotated, rotation);

        /* Compute the mask of bits recovered */
        uint16_t recovered_mask =
            (((1u << cur_k) - 1) << rotation) & ((1u << K) - 1);
        if (rotation + cur_k > K) {
            recovered_mask |= ((1u << ((rotation + cur_k) % K)) - 1);
        }

        sa_free(reduced);
        free(rotated);

        return ((uint32_t)recovered_mask << 16) | recovered_original;
    }
}

/* ── Solve Remaining Bits ───────────────────────────────────────────── */

/* Substitute known bits and exhaustively search the remaining ones. */
static uint32_t solve_remaining_bits(const uint16_t *A, const uint8_t *b_list,
                                     uint32_t n,
                                     uint16_t recovered_secret,
                                     uint16_t recovered_mask,
                                     uint16_t unrecovered_mask) {
    int remaining_bits = popcount16(unrecovered_mask);

    /* Build bit mapping: original bit position → compressed position */
    int bit_mapping[K];
    int current_bit = 0;
    for (int i = 0; i < K; i++) {
        if ((unrecovered_mask >> i) & 1) {
            bit_mapping[i] = current_bit++;
        } else {
            bit_mapping[i] = -1;
        }
    }

    /* Build new samples with known bits substituted out */
    Sample *mapped_samples = malloc(n * sizeof(Sample));
    if (!mapped_samples) { fprintf(stderr, "malloc failed\n"); exit(1); }

    for (uint32_t idx = 0; idx < n; idx++) {
        uint16_t eq = A[idx];

        /* Compute parity contributed by recovered bits */
        uint16_t recovered_part = eq & recovered_mask;
        int recovered_parity = popcount16(recovered_part & recovered_secret) & 1;

        /* New parity: original minus recovered contribution */
        int new_parity = (b_list[idx] - recovered_parity) & 1;

        /* Map unrecovered bits to compressed positions */
        uint16_t unrecovered_part = eq & unrecovered_mask;
        uint16_t mapped_eq = 0;
        for (int i = 0; i < K; i++) {
            if ((unrecovered_mask >> i) & 1 && (unrecovered_part >> i) & 1) {
                mapped_eq |= (1u << bit_mapping[i]);
            }
        }

        mapped_samples[idx].eq     = mapped_eq;
        mapped_samples[idx].parity = (uint8_t)new_parity;
    }

    /* Exhaustive search on remaining bits */
    uint16_t mapped_secret = exhaustive_search(mapped_samples, n, remaining_bits);

    /* Map back to original bit positions */
    uint16_t full_secret = recovered_secret;
    for (int i = 0; i < K; i++) {
        if (bit_mapping[i] >= 0 && ((mapped_secret >> bit_mapping[i]) & 1)) {
            full_secret |= (1u << i);
        }
    }

    free(mapped_samples);

    uint16_t full_mask = recovered_mask | unrecovered_mask;
    return ((uint32_t)full_mask << 16) | full_secret;
}

/* ── Top-level Recovery ─────────────────────────────────────────────── */

/* Returns 1 if the secret was fully recovered, 0 otherwise. */
static int recover_full_secret_lf2(const uint16_t *A, const uint8_t *b_list,
                                   uint32_t n, uint16_t true_secret) {
    uint16_t recovered_secret = 0;
    uint32_t recovered_mask   = 0;

    int rotations[2] = {0, 8};

    for (int r = 0; r < ROTATIONS; r++) {
        int rotation = rotations[r];

        uint32_t packed = lf2_solve_rotated(A, b_list, n, rotation);
        uint16_t segment = (uint16_t)(packed & 0xFFFF);
        uint16_t mask    = (uint16_t)(packed >> 16);

        /* Merge new bits */
        uint16_t new_bits_mask = mask & ~recovered_mask;
        recovered_secret |= (segment & new_bits_mask);
        recovered_mask   |= mask;

        if (popcount16(recovered_mask) == K) break;
    }

    /* If any bits still unrecovered, solve them */
    uint16_t unrecovered_mask = ((1u << K) - 1) & ~recovered_mask;
    if (unrecovered_mask) {
        uint32_t packed = solve_remaining_bits(A, b_list, n,
                                                recovered_secret,
                                                recovered_mask,
                                                unrecovered_mask);
        recovered_secret = (uint16_t)(packed & 0xFFFF);
        recovered_mask   = (uint16_t)(packed >> 16);
    }

    return (recovered_secret == true_secret) ? 1 : 0;
}

/* ── Main ───────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: lf2_solver.exe <input.bin> <output.bin>\n");
        return 1;
    }

    const char *input_path  = argv[1];
    const char *output_path = argv[2];

    FILE *fin = fopen(input_path, "rb");
    if (!fin) {
        fprintf(stderr, "Cannot open input: %s\n", input_path);
        return 1;
    }

    /* ── Read header ── */
    uint32_t magic, num_problems, fcl_count, err_count;
    if (fread(&magic, 4, 1, fin) != 1 || magic != MAGIC) {
        fprintf(stderr, "Bad magic number: 0x%08X (expected 0x%08X)\n", magic, MAGIC);
        fclose(fin);
        return 1;
    }
    fread(&num_problems, 4, 1, fin);
    fread(&fcl_count, 4, 1, fin);
    fread(&err_count, 4, 1, fin);

    /* Skip FCL list and err list (we don't need them) */
    fseek(fin, fcl_count * 4 + err_count * 4, SEEK_CUR);

    printf("Problems: %u, FCL grid: %u, err grid: %u\n",
           num_problems, fcl_count, err_count);
    fflush(stdout);

    /* ── Open output ── */
    FILE *fout = fopen(output_path, "wb");
    if (!fout) {
        fprintf(stderr, "Cannot open output: %s\n", output_path);
        fclose(fin);
        return 1;
    }

    /* Buffers */
    uint16_t *A_buf = malloc(MAX_FCL * sizeof(uint16_t));
    uint8_t  *b_buf = malloc(MAX_FCL * sizeof(uint8_t));
    if (!A_buf || !b_buf) {
        fprintf(stderr, "malloc failed\n");
        fclose(fin); fclose(fout);
        return 1;
    }

    int total_success = 0;

    /* ── Process each problem: solve + write result immediately ── */
    for (uint32_t p = 0; p < num_problems; p++) {
        uint32_t num_samples, tg_solu, reserved;
        if (fread(&num_samples, 4, 1, fin) != 1 ||
            fread(&tg_solu, 4, 1, fin) != 1 ||
            fread(&reserved, 4, 1, fin) != 1) {
            fprintf(stderr, "Error reading problem %u header\n", p);
            break;
        }

        if (num_samples > MAX_FCL) {
            fprintf(stderr, "FCL=%u exceeds MAX_FCL=%u at problem %u\n",
                    num_samples, MAX_FCL, p);
            fclose(fin); fclose(fout);
            return 1;
        }

        /* Read packed samples and unpack */
        for (uint32_t i = 0; i < num_samples; i++) {
            uint32_t packed;
            if (fread(&packed, 4, 1, fin) != 1) {
                fprintf(stderr, "Error reading sample %u of problem %u\n", i, p);
                break;
            }
            A_buf[i] = (uint16_t)(packed >> 1);
            b_buf[i] = (uint8_t)(packed & 1);
        }

        uint8_t result = (uint8_t)recover_full_secret_lf2(
            A_buf, b_buf, num_samples, (uint16_t)tg_solu);
        total_success += result;

        /* Write result immediately so progress is visible */
        fwrite(&result, 1, 1, fout);
        fflush(fout);

        if ((p + 1) % 100 == 0) {
            printf("Progress: %u / %u  (success so far: %d)\n",
                   p + 1, num_problems, total_success);
            fflush(stdout);
        }
    }

    /* ── Summary ── */
    printf("Done. Success: %d / %u (%.1f%%)\n",
           total_success, num_problems, 100.0 * total_success / num_problems);

    /* Cleanup */
    free(A_buf);
    free(b_buf);
    fclose(fin);
    fclose(fout);
    return 0;
}
