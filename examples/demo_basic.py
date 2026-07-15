"""Demonstration of the PCM pipeline: LPN → Ising encoding → SA solving.

This script walks through the core Physical Computing Mapping (PCM) workflow:
  1. Generate a w=2 LPN problem
  2. Encode it as an Ising Hamiltonian (QUBO)
  3. Solve via simulated annealing
  4. Verify the result
"""
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.lpn import gen_problem, calu_sim
from src.ising import set_H_qubo
from src.solver import annealing_solve_Q


def main():
    # Parameters
    n, m, tau = 16, 500, 0.25

    print(f"PCM Pipeline Demo: n={n}, m={m}, tau={tau}")
    print("-" * 50)

    # 1. Generate a w=2 LPN problem
    A, b_list, secret = gen_problem(n, m, tau, min_terms=1, max_terms=2)

    # 2. Encode as Ising Hamiltonian (QUBO)
    t0 = time.time()
    Q = set_H_qubo(n, m, A, b_list)
    encode_time = time.time() - t0

    # 3. Solve via simulated annealing
    res = annealing_solve_Q(n, Q, num_reads=1000)
    best_key, _ = max(res.items(), key=lambda item: item[1])
    solution = int(best_key, 2)
    solve_time = time.time() - t0 - encode_time

    # 4. Verify
    sim = calu_sim(solution, secret, n)
    success = solution == secret

    print(f"QUBO terms:   {len(Q)}")
    print(f"Encode time:  {encode_time:.2f}s")
    print(f"Solve time:   {solve_time:.2f}s")
    print(f"Recovered:    {'YES' if success else 'NO'}")
    print(f"Similarity:   {sim:.3f}")
    print(f"Secret:       0b{secret:0{n}b}")
    print(f"Solution:     0b{solution:0{n}b}")


if __name__ == '__main__':
    main()