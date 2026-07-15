"""Export problems from pickle to binary format for C LF2 solver.

Reads problems from fig3def's shared pkl, writes binary for C solver.
Input:  ../../fig3def_heatmap/data/problems_heatmap_with_qubo.pkl
Output: ../data/problems_input.bin
"""
import struct, pickle, os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = SCRIPT_DIR                                                    # c_pipeline/
FIG4_DIR = os.path.dirname(PIPELINE_DIR)                                     # fig4_lf2_heatmap/
FIG3DEF_PKL = os.path.join(FIG4_DIR, '..', 'fig3def_heatmap', 'data', 'problems_heatmap_with_qubo.pkl')
OUTPUT_BIN = os.path.join(FIG4_DIR, 'data', 'problems_input.bin')
os.makedirs(os.path.dirname(OUTPUT_BIN), exist_ok=True)

m_list = [100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
ERR_LIST = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
MAGIC = 0x4C463232  # "LF22"


def main():
    with open(FIG3DEF_PKL, 'rb') as f:
        problems = pickle.load(f)

    num_problems = len(problems)
    print(f"Loaded {num_problems} problems from {FIG3DEF_PKL}")

    with open(OUTPUT_BIN, 'wb') as f:
        # Header
        f.write(struct.pack('<I', MAGIC))
        f.write(struct.pack('<I', num_problems))
        f.write(struct.pack('<I', len(m_list)))
        f.write(struct.pack('<I', len(ERR_LIST)))

        for v in m_list:
            f.write(struct.pack('<I', v))
        for v in ERR_LIST:
            f.write(struct.pack('<f', v))

        # Problem records
        for prob in problems:
            m = prob['m']
            tg_solu = prob['tg_solu']
            A = prob['A']
            b_list = prob['b_list']

            f.write(struct.pack('<I', m))
            f.write(struct.pack('<I', tg_solu))
            f.write(struct.pack('<I', 0))  # reserved

            for a, b in zip(A, b_list):
                packed = (a << 1) | b
                f.write(struct.pack('<I', packed))

    file_size = os.path.getsize(OUTPUT_BIN)
    print(f"Written {file_size:,} bytes to {OUTPUT_BIN}")


if __name__ == '__main__':
    main()