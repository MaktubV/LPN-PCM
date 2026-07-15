"""Ising encoding — Hamiltonian construction corresponding to Eq. (5) in the paper.

Uses PyQUBO's product(1-2x) for exact XOR-to-spin-product mapping.
For w=2 problems, no auxiliary variables are needed; for w>2, PyQUBO automatically introduces them."""

from functools import reduce
import operator
import multiprocessing as mp
from pyqubo import Binary


def _build_qubo(args):
    """Build QUBO in a subprocess, returning a serializable dict"""
    n, m, A, b_list = args
    H = 0
    user_names = set()
    for idx in range(m):
        a = A[idx]
        b = b_list[idx]
        a_str = format(a, f"0{n}b")
        bits_list = []
        for jdx in range(n):
            s = a_str[jdx]
            if s == '1':
                bits_list.append(Binary(f"{n-1-jdx}"))
                user_names.add(f"{n-1-jdx}")
        product = reduce(operator.mul, (1 - 2*x for x in bits_list))
        expr = (1 - product) / 2
        if b == 1:
            expr = -expr
        H = H + expr

    model = H.compile()
    qubo, offset = model.to_qubo()
    all_names = set(model.variables)
    aux_names = all_names - user_names
    # print(f"Auxiliary variables: {len(aux_names)}")
    return qubo


def set_H_qubo(n: int, m: int, A: list, b_list: list, workers: int = 1) -> dict:
    """Build QUBO matrix in a subprocess to avoid memory leaks.

    Args:
        n: number of original variables
        m: number of samples
        A: coefficient list
        b_list: label list
        workers: number of parallel processes

    Returns:
        qubo: {(var_name, var_name): coeff} dict
    """
    with mp.Pool(processes=1, maxtasksperchild=1) as pool:
        qubo = pool.apply(_build_qubo, ((n, m, A, b_list),))
    return qubo


def _build_qubo_with_aux(args):
    """Build QUBO in a subprocess, returning (qubo, aux_nums)."""
    n, m, A, b_list = args
    H = 0
    user_names = set()
    for idx in range(m):
        a = A[idx]
        b = b_list[idx]
        a_str = format(a, f"0{n}b")
        bits_list = []
        for jdx in range(n):
            s = a_str[jdx]
            if s == '1':
                bits_list.append(Binary(f"{n-1-jdx}"))
                user_names.add(f"{n-1-jdx}")
        product = reduce(operator.mul, (1 - 2*x for x in bits_list))
        expr = (1 - product) / 2
        if b == 1:
            expr = -expr
        H = H + expr

    model = H.compile()
    qubo, offset = model.to_qubo()
    all_names = set(model.variables)
    aux_names = all_names - user_names
    return qubo, len(aux_names)


def set_H_qubo_with_aux(n: int, m: int, A: list, b_list: list):
    """Build QUBO matrix and return auxiliary variable count (subprocess to avoid memory leaks).

    Args:
        n: number of original variables
        m: number of samples
        A: coefficient list
        b_list: label list

    Returns:
        qubo: {(var_name, var_name): coeff} dict
        aux_nums: number of auxiliary variables
    """
    with mp.Pool(processes=1, maxtasksperchild=1) as pool:
        qubo, aux_nums = pool.apply(_build_qubo_with_aux, ((n, m, A, b_list),))
    return qubo, aux_nums