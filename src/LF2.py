"""LF2 algorithm — heuristic improvement based on Levieil & Fouque (2006).

LF2 differs from BKW/LF1 in that it XORs all unordered pairs of samples
within each partition group, rather than discarding all but a single anchor.
This produces more reduced samples, preserving more bias information, and is
particularly useful when the original sample budget is limited.
"""

import random
from collections import defaultdict
from typing import List, Tuple


def lf2_reduce_round(samples: List[Tuple[int, int]], bit_start: int, bit_end: int) -> List[Tuple[int, int]]:
    """LF2 single reduction round: group by the target block, XOR all unordered pairs within each group.

    Unlike BKW/LF1, this does not discard the anchor sample; instead it computes
    XOR for every pair (i<j) and retains all results, producing C(len(group), 2)
    new samples per group.

    Args:
        samples: list of (vector, parity) tuples
        bit_start: start index of the block to eliminate (lower bit)
        bit_end: end index of the block (exclusive)

    Returns:
        list of reduced samples
    """
    mask = ((1 << (bit_end - bit_start)) - 1) << bit_start

    groups = defaultdict(list)
    for eq, parity in samples:
        key = (eq & mask) >> bit_start
        groups[key].append((eq, parity))

    new_samples = []
    for group in groups.values():
        n = len(group)
        # Compute XOR for all unordered pairs (i < j)
        for i in range(n):
            eq_i, parity_i = group[i]
            for j in range(i + 1, n):
                eq_j, parity_j = group[j]
                new_eq = eq_i ^ eq_j
                new_parity = parity_i ^ parity_j
                new_samples.append((new_eq, new_parity))

    return new_samples


def exhaustive_search(samples: List[Tuple[int, int]], k: int, timeout: int = 10) -> int:
    """Exhaustive search over reduced samples to recover a k-bit partial secret."""
    import time

    if k > 25:
        print(f"Warning: Exhaustive search over {2**k} possibilities may be slow for k={k}")

    equations, parities = zip(*samples) if samples else ([], [])

    best_secret = 0
    best_score = -1

    start_time = time.time()

    for candidate in range(1 << k):
        if time.time() - start_time > timeout:
            print(f"Exhaustive search timed out after {timeout} seconds")
            break

        score = 0
        for eq, parity in zip(equations, parities):
            candidate_parity = bin(eq & candidate).count('1') % 2
            if candidate_parity == parity:
                score += 1

        if score > best_score:
            best_score = score
            best_secret = candidate

    return best_secret


def lf2_solve_rotated_exhaustive(A: List[int], b_list: List[int], k: int, a: int, b: int,
                                 rotation: int = 0, timeout: int = 10) -> Tuple[int, int]:
    """LF2 with rotation + exhaustive search to recover a partial secret.

    Unlike the BKW version, reduction uses lf2_reduce_round.
    """
    # Rotate all samples
    rotated_A = []
    for eq in A:
        rotated_eq = ((eq >> rotation) | ((eq & ((1 << rotation) - 1)) << (k - rotation))) & ((1 << k) - 1)
        rotated_A.append(rotated_eq)

    samples = [(eq, parity) for eq, parity in zip(rotated_A, b_list)]
    current_k = k

    # Perform a-1 rounds of LF2 reduction
    for i in range(1, a):
        bit_start = current_k - b
        bit_end = current_k
        samples = lf2_reduce_round(samples, bit_start, bit_end)
        current_k -= b
        # Optional: truncate if too many samples (LF2 heuristics usually do not truncate, but this guards against memory explosion)
        # if len(samples) > 2**20:  # cap maximum sample count
        #     samples = samples[:2**20]

    recovered_rotated = exhaustive_search(samples, current_k, timeout)

    # Rotate back to original coordinates
    recovered_original = ((recovered_rotated << rotation) & ((1 << k) - 1)) | (recovered_rotated >> (k - rotation))

    recovered_mask = (((1 << current_k) - 1) << rotation) & ((1 << k) - 1)
    if rotation + current_k > k:
        recovered_mask |= ((1 << ((rotation + current_k) % k)) - 1)

    return recovered_original, recovered_mask


def solve_remaining_bits(A: List[int], b_list: List[int], k: int,
                         recovered_secret: int, recovered_mask: int,
                         unrecovered_mask: int, timeout: int = 10) -> Tuple[int, int]:
    """Substitute known bits and exhaustively search over the remaining unknown bits."""
    new_A = []
    new_b_list = []

    for eq, parity in zip(A, b_list):
        recovered_part = eq & recovered_mask
        unrecovered_part = eq & unrecovered_mask
        recovered_parity = bin(recovered_part & recovered_secret).count('1') % 2
        new_parity = (parity - recovered_parity) % 2
        new_A.append(unrecovered_part)
        new_b_list.append(new_parity)

    remaining_bits = bin(unrecovered_mask).count('1')

    bit_mapping = {}
    current_bit = 0
    for i in range(k):
        if (unrecovered_mask >> i) & 1:
            bit_mapping[i] = current_bit
            current_bit += 1

    mapped_A = []
    for eq in new_A:
        mapped_eq = 0
        for i in range(k):
            if (unrecovered_mask >> i) & 1 and (eq >> i) & 1:
                mapped_eq |= (1 << bit_mapping[i])
        mapped_A.append(mapped_eq)

    mapped_samples = list(zip(mapped_A, new_b_list))
    mapped_secret = exhaustive_search(mapped_samples, remaining_bits, timeout)

    full_secret = recovered_secret
    for original_pos, mapped_pos in bit_mapping.items():
        if (mapped_secret >> mapped_pos) & 1:
            full_secret |= (1 << original_pos)

    full_mask = recovered_mask | unrecovered_mask

    return full_secret, full_mask


def recover_full_secret_lf2(A: List[int], b_list: List[int], k: int, a: int, b: int,
                            true_secret: int, timeout: int = 10) -> Tuple[int, bool]:
    """Recover the full secret vector using the LF2 algorithm.

    Parameters are identical to recover_full_secret_exhaustive; only the reduction
    step uses lf2_reduce_round internally.

    Returns:
        (recovered_secret, success)
    """
    k_prime = k - (a - 1) * b

    if k_prime >= k:
        recovered_secret, mask = lf2_solve_rotated_exhaustive(A, b_list, k, a, b, 0, timeout)
        success = (recovered_secret & mask) == (true_secret & mask)
        return recovered_secret & mask, success

    recovered_secret = 0
    recovered_mask = 0
    rotations_needed = (k + k_prime - 1) // k_prime

    for i in range(rotations_needed):
        rotation = i * k_prime
        segment, mask = lf2_solve_rotated_exhaustive(A, b_list, k, a, b, rotation, timeout)

        new_bits_mask = mask & ~recovered_mask
        recovered_secret |= (segment & new_bits_mask)
        recovered_mask |= mask

        recovered_count = bin(recovered_mask).count('1')

        if recovered_count == k:
            break

    unrecovered_mask = ((1 << k) - 1) & ~recovered_mask
    if unrecovered_mask:
        recovered_secret, recovered_mask = solve_remaining_bits(
            A, b_list, k, recovered_secret, recovered_mask,
            unrecovered_mask, timeout
        )

    success = recovered_secret == true_secret
    return recovered_secret, success

