"""Fig 3(d-f): SA vs BKW (LF1) success rate heatmaps on pre-generated w=2 LPN problems.

Loads pre-generated problems with pre-computed QUBO matrices from pkl,
then solves each with SA and LF1 in parallel via joblib.
"""
import sys, os, pickle
import numpy as np
from collections import defaultdict
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.solver import annealing_solve_Q
from src.bkw import recover_full_secret_exhaustive

n = 16
m_list = [100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
err_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
a = 2
b_bkw = 8
n_jobs = 10

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, 'data', 'problems_heatmap_with_qubo.pkl')
RES_DIR = os.path.join(SCRIPT_DIR, 'res')
os.makedirs(RES_DIR, exist_ok=True)


def process_one(prob):
    """Process one problem instance, returning (m, err, sa_success, lf1_success)."""
    m = prob['m']
    err = prob['err']
    A = prob['A']
    b_list = prob['b_list']
    tg_solu = prob['tg_solu']
    Q = prob['Q']

    # SA solver
    res_dict = annealing_solve_Q(n, Q, num_reads=1000)
    max_key, _ = max(res_dict.items(), key=lambda item: item[1])
    sa_ok = 1 if int(max_key, 2) == tg_solu else 0

    # BKW (LF1) solver
    _, success = recover_full_secret_exhaustive(A, b_list, n, a, b_bkw, tg_solu, timeout=10)
    lf1_ok = 1 if success else 0

    return m, err, sa_ok, lf1_ok


def run():
    with open(DATA_FILE, 'rb') as f:
        problems = pickle.load(f)

    with tqdm_joblib(desc="Testing", total=len(problems)):
        raw_results = Parallel(n_jobs=n_jobs)(
            delayed(process_one)(prob) for prob in problems
        )

    # Aggregate by (m, err)
    stats = defaultdict(lambda: [0, 0, 0])  # [sa_success, lf1_success, count]
    for m, err, sa_s, lf1_s in raw_results:
        s = stats[(m, err)]
        s[0] += sa_s
        s[1] += lf1_s
        s[2] += 1

    sa_heatmap = np.zeros((len(m_list), len(err_list)))
    bkw_heatmap = np.zeros((len(m_list), len(err_list)))

    for i, m in enumerate(m_list):
        for j, err in enumerate(err_list):
            sa_s, bkw_s, count = stats[(m, err)]
            if count > 0:
                sa_heatmap[i, j] = sa_s / count
                bkw_heatmap[i, j] = bkw_s / count
            print(f"m={m},tau={err}: SA={sa_heatmap[i,j]:.3f}, BKW={bkw_heatmap[i,j]:.3f}")

    np.savez(os.path.join(RES_DIR, 'heatmap_results.npz'),
             m_list=m_list, err_list=err_list,
             sa_heatmap=sa_heatmap, bkw_heatmap=bkw_heatmap)
    print(f"Results saved to {os.path.join(RES_DIR, 'heatmap_results.npz')}")


if __name__ == '__main__':
    run()