"""Import C solver results and combine with SA heatmap to create final npz.

Reads:  ../res/lf2_results.bin (C solver output)
        ../../fig3def_heatmap/res_published/heatmap_results.npz (SA heatmap)
Writes: ../res/heatmap_results.npz
"""
import numpy as np, os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG4_DIR = os.path.dirname(SCRIPT_DIR)  # fig4_lf2_heatmap/
LF2_BIN = os.path.join(FIG4_DIR, 'res', 'lf2_results.bin')
FIG3DEF_NPZ = os.path.join(FIG4_DIR, '..', 'fig3def_heatmap', 'res_published', 'heatmap_results.npz')
OUT_NPZ = os.path.join(FIG4_DIR, 'res', 'heatmap_results.npz')
os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)

m_list = [100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
ERR_LIST = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
LOOPS = 100


def main():
    results = np.fromfile(LF2_BIN, dtype=np.uint8)
    n_expected = len(m_list) * len(ERR_LIST) * LOOPS
    if len(results) != n_expected:
        print(f"WARNING: got {len(results)} results, expected {n_expected}")

    lf2_heatmap = np.zeros((len(m_list), len(ERR_LIST)))
    idx = 0
    for i in range(len(m_list)):
        for j in range(len(ERR_LIST)):
            chunk = results[idx:idx + LOOPS]
            lf2_heatmap[i, j] = np.mean(chunk)
            idx += LOOPS
            print(f"m={m_list[i]:4d}, tau={ERR_LIST[j]:.2f}: LF2={lf2_heatmap[i,j]:.3f}")

    sa_data = np.load(FIG3DEF_NPZ)
    sa_heatmap = sa_data['sa_heatmap']
    print(f"\nLoaded SA heatmap from {FIG3DEF_NPZ}")

    np.savez(OUT_NPZ, m_list=np.array(m_list), err_list=np.array(ERR_LIST),
             sa_heatmap=sa_heatmap, lf2_heatmap=lf2_heatmap)
    print(f"Saved to {OUT_NPZ}")


if __name__ == '__main__':
    main()