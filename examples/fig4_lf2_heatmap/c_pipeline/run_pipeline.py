"""One-click pipeline: export problems -> compile C solver -> run -> import results.

Usage: python run_pipeline.py
"""
import subprocess, os, sys, time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG4_DIR = os.path.dirname(SCRIPT_DIR)  # fig4_lf2_heatmap/


def run_step(step_name, cmd, cwd=None):
    print(f"\n{'='*60}")
    print(f"[{step_name}]")
    print(f"  Command: {' '.join(str(c) for c in cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=cwd or SCRIPT_DIR)
    if result.returncode != 0:
        print(f"ERROR: '{step_name}' failed with code {result.returncode}")
        sys.exit(result.returncode)


def main():
    t0 = time.time()

    # Step 1: Export problems
    run_step("Step 1/4: Export problems",
             [sys.executable, 'export_problems.py'])

    # Step 2: Compile C solver
    run_step("Step 2/4: Compile C solver", ['make'])

    # Step 3: Run C solver
    t_run = time.time()
    c_solver_exe = os.path.join(SCRIPT_DIR, 'lf2_solver.exe')
    problems_bin = os.path.join(FIG4_DIR, 'data', 'problems_input.bin')
    results_bin = os.path.join(FIG4_DIR, 'res', 'lf2_results.bin')
    os.makedirs(os.path.dirname(results_bin), exist_ok=True)
    run_step("Step 3/4: Run C solver",
             [c_solver_exe, problems_bin, results_bin])
    t_run_elapsed = time.time() - t_run

    # Step 4: Import results
    run_step("Step 4/4: Import results",
             [sys.executable, 'import_results.py'])

    t_total = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Pipeline complete!")
    print(f"  C solver time: {t_run_elapsed:.1f}s")
    print(f"  Total time:    {t_total:.1f}s")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()