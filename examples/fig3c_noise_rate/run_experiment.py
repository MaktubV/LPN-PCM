"""Fig 3(c): Impact of noise rate tau on SA and LF1 success rates."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pickle
import numpy as np
from collections import defaultdict
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib

from src.lpn import calu_sim
from src.bkw import recover_full_secret_exhaustive
from src.LF2 import recover_full_secret_lf2
from src.solver import annealing_solve_Q

n = 16
m = 1000
err_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
a = 2
b_bkw = 8
num_jobs = 12

# Resolve paths relative to this script so the script runs from any CWD
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, 'data', 'problems_by_err_with_qubo_0.4.pkl')
RES_DIR = os.path.join(SCRIPT_DIR, 'res')
os.makedirs(RES_DIR, exist_ok=True)


def process_one(prob):
    """Process one problem instance, returning (err, sa_success, bkw_success, lf2_success, sa_sim, bkw_sim, lf2_sim)."""
    err = prob['err']
    A = prob['A']
    b_list = prob['b_list']
    tg_solu = prob['tg_solu']
    Q = prob['Q']

    # SA solver
    res_dict = annealing_solve_Q(n, Q, num_reads=1000)
    max_key, _ = max(res_dict.items(), key=lambda item: item[1])
    direct_solu = int(max_key, 2)
    sa_sim = calu_sim(direct_solu, tg_solu, n)
    sa_success = 1 if direct_solu == tg_solu else 0

    # BKW (LF1) solver
    bkw_solu, success = recover_full_secret_exhaustive(A, b_list, n, a, b_bkw, tg_solu, timeout=100)
    bkw_sim = calu_sim(bkw_solu, tg_solu, n)
    bkw_success = 1 if success else 0

    # LF2 BKW solver
    lf2_solu, lf2_success = recover_full_secret_lf2(A, b_list, n, a, b_bkw, tg_solu, timeout=100)
    lf2_sim = calu_sim(lf2_solu, tg_solu, n)
    lf2_success = 1 if lf2_success else 0

    return err, sa_success, bkw_success, lf2_success, sa_sim, bkw_sim, lf2_sim


def run():
    with open(DATA_FILE, 'rb') as f:
        problems = pickle.load(f)

    with tqdm_joblib(desc="Testing", total=len(problems)):
        results = Parallel(n_jobs=num_jobs)(
            delayed(process_one)(prob) for prob in problems
        )

    # Aggregate by err
    final = defaultdict(lambda: [0, 0, 0, 0.0, 0.0, 0.0, 0])
    for err, sa_s, bkw_s, lf2_s, sa_sim, bkw_sim, lf2_sim in results:
        stats = final[err]
        stats[0] += sa_s
        stats[1] += bkw_s
        stats[2] += lf2_s
        stats[3] += sa_sim
        stats[4] += bkw_sim
        stats[5] += lf2_sim
        stats[6] += 1

    sa_success, bkw_success = [], []
    lf2_success = []
    sa_sim, bkw_sim, lf2_sim = [], [], []

    for err in err_list:
        r = final[err]
        count = r[6]
        sa_success.append(r[0] / count if count > 0 else 0)
        bkw_success.append(r[1] / count if count > 0 else 0)
        lf2_success.append(r[2] / count if count > 0 else 0)
        sa_sim.append(r[3] / count if count > 0 else 0)
        bkw_sim.append(r[4] / count if count > 0 else 0)
        lf2_sim.append(r[5] / count if count > 0 else 0)
        print(f"tau={err}: SA={sa_success[-1]:.3f}, BKW={bkw_success[-1]:.3f}, LF2={lf2_success[-1]:.3f}")

    # Save summary
    import json
    summary = {
        'err_list': err_list,
        'sa_success': sa_success, 'bkw_success': bkw_success, 'lf2_success': lf2_success,
        'sa_sim': sa_sim, 'bkw_sim': bkw_sim, 'lf2_sim': lf2_sim,
    }
    with open(os.path.join(RES_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {os.path.join(RES_DIR, 'summary.json')}")


if __name__ == '__main__':
    run()