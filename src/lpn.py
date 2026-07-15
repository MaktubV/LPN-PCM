"""LPN problem generation and utility functions."""

import random


def get_bit(num: int, n: int) -> int:
    """Return the value of the n-th bit of integer num (0-indexed from LSB)."""
    return (num >> n) & 1


def random_int_with_hamming_weight_range(max_bits: int, min_hw: int, max_hw: int) -> int:
    """Generate a random integer with Hamming weight in [min_hw, max_hw]."""
    if max_hw <= 0:
        return 0
    if min_hw > max_hw or min_hw > max_bits:
        raise ValueError("invalid hamming-weight range")
    min_hw = max(0, min_hw)
    max_hw = min(max_hw, max_bits)
    num_ones = random.randint(min_hw, max_hw)
    positions = random.sample(range(max_bits), num_ones)
    result = 0
    for pos in positions:
        result |= 1 << pos
    return result


def gen_problem(n: int, m: int, err: float, min_terms: int = 1, max_terms: int = None):
    """Generate an LPN problem instance.

    Args:
        n: secret vector dimension
        m: number of samples
        err: noise rate tau
        min_terms: minimum Hamming weight of coefficient vectors
        max_terms: maximum Hamming weight of coefficient vectors (None = unbounded)

    Returns:
        (A, b_list, true_secret): coefficient list, noisy label list, true secret vector
    """
    fc_list = []
    counts = 0
    while counts < m:
        if max_terms is None:
            _equa = random.randint(1, (2**n) - 1)
        else:
            _equa = random_int_with_hamming_weight_range(n, min_terms, max_terms)
        fc_list.append(_equa)
        counts += 1

    solu = random.randint(0, (2**n) - 1)
    A = fc_list
    right_list = []
    for _coef in A:
        parity = bin(_coef & solu).count('1') % 2
        right_list.append(parity)

    num_to_change = int(len(A) * err)
    change_index = random.sample(range(len(A)), num_to_change)
    for _index in change_index:
        right_list[_index] = 1 - right_list[_index]

    return A, right_list, solu


def calu_sim(solu: int, tg_solu: int, n: int) -> float:
    """Compute the similarity between a candidate solution and the target secret (0~1)."""
    similarity = n - bin(solu ^ tg_solu).count('1')
    return similarity / n