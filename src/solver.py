"""Solver wrapper — Simulated Annealing (SA) based QUBO solver."""

import dimod
from dwave.samplers import SimulatedAnnealingSampler


def annealing_solve_Q(n: int, Q: dict, num_reads: int = 1000) -> dict:
    """Solve a QUBO problem using classical simulated annealing.

    Args:
        n: number of variables (only the first n bits are extracted)
        Q: QUBO dict {(i,j): coeff}
        num_reads: number of annealing reads

    Returns:
        res_dict: {solution bitstring: occurrence count}
    """
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    sampler = SimulatedAnnealingSampler()

    response = sampler.sample(
        bqm,
        num_reads=num_reads,
        beta_range=(0.1, 1.0),
        num_sweeps=1000,
    )

    res_dict = {}
    for sample, energy in response.data(['sample', 'energy']):
        solu_int = 0
        for i in range(n):
            if sample.get(f"{i}", 0) == 1:
                solu_int |= (1 << i)
        solu_str = format(solu_int, f"0{n}b")
        if res_dict.get(solu_str, 0) == 0:
            res_dict[solu_str] = 1
        else:
            res_dict[solu_str] += 1

    return res_dict