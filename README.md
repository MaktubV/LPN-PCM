# LPN-PCM: Physical Computing Mapping for Learning Parity with Noise

Code for applying physical computing mapping method to solve the LPN problem.

## Overview

This repository provides a Physical Computing Mapping (PCM) framework for solving the Learning Parity with Noise (LPN) problem. The core idea is to reformulate LPN as a combinatorial optimization task and encode it into an Ising Hamiltonian, which can then be solved by Ising solvers.

The experiments in this repository use classical simulated annealing (SA) as the Ising solver. For CIM hardware experiments (Fig. 2 in the paper), please submit the QUBO matrix via the QBoson cloud platform at https://cloud.qboson.com/.

## Structure

```
├── src/                          Core library
│   ├── lpn.py                    LPN problem generation and utility functions
│   ├── ising.py                  Ising / QUBO encoding (PCM, Eq. 5)
│   ├── bkw.py                    BKW algorithm (LF1)
│   ├── LF2.py                    LF2 algorithm (pairwise XOR per partition)
│   └── solver.py                 Simulated annealing solver (D-Wave Neal)
├── examples/                     Reproducible experiments
│   ├── demo_basic.py             End-to-end PCM pipeline demonstration
│   ├── fig3b_sample_size/        Success rate vs sample count m (Fig 3b)
│   ├── fig3c_noise_rate/         Success rate vs noise rate tau (Fig 3c)
│   ├── fig3def_heatmap/          SA vs LF1 heatmap (Fig 3d-f)
│   └── fig4_lf2_heatmap/         SA vs LF2 heatmap (Fig 4)
├── data/                         Supplementary Data1.xlsx
├── requirements.txt
└── LICENSE
```

## Quick Start

```bash
pip install -r requirements.txt

# Run the end-to-end demo
python examples/demo_basic.py

# Run a plot (pre-generated data is included)
cd examples/fig3b_sample_size
python plot.py
```

## Experiment-to-Figure Mapping

| Directory                      | Paper Figure | Key Scripts                             | Description                                                            |
| ------------------------------ | ------------ | --------------------------------------- | ---------------------------------------------------------------------- |
| `examples/demo_basic.py`     | —           | `demo_basic.py`                       | Minimal end-to-end PCM pipeline: generate → encode → solve → verify |
| `examples/fig3b_sample_size` | Fig 3b       | `run_experiment.py`, `plot.py`      | SA + LF1 success rate vs sample count m (n=16, tau=0.25)               |
| `examples/fig3c_noise_rate`  | Fig 3c       | `run_experiment.py`, `plot.py`      | SA + LF1 success rate vs noise rate tau (n=16, m=1000)                 |
| `examples/fig3def_heatmap`   | Fig 3d-f     | `run_heatmap.py`, `plot_heatmap.py` | SA vs LF1 success rate heatmaps over full (m, tau) grid                |
| `examples/fig4_lf2_heatmap`  | Fig 4        | `run_experiment.py`, `plot.py`      | SA vs LF2 success rate difference heatmap over full (m, tau) grid      |

Each experiment directory contains:

- `run_*.py` — runs the experiment and saves results to `res/` (excluded from version control)
- `plot*.py` — reads published result data from `res_published/` and generates figures in `fig/`
- `data/` — pre-generated problem instances (.pkl)
- `res_published/` — result data used in the paper (.npz or summary.json)

## Requirements

Python >= 3.10. Full dependency list in `requirements.txt`.

## Data Availability

The source data for Figures 2, 3, and 4 is provided in `data/Supplement_Data1.xlsx`.
Pre-generated problem instances for each experiment are stored in their respective `data/` subdirectories under `examples/`.

## License

GNU General Public License v3.0. See `LICENSE` for details.
