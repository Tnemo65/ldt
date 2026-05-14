#!/usr/bin/env python3
"""
Subprocess-based benchmark runner.
Runs each agent as a separate subprocess to avoid multiprocessing.Pool deadlocks.
Worker counts are hardcoded here (Table A: 2, Table B: 2, Ablation: 1, BAR: 1).
Data is loaded from cached fold bundles in results/v3/features/fold_data/.
"""
import subprocess
import sys
import time
import json
from pathlib import Path
from datetime import datetime

OUT_DIR = Path('c:/proj/ldt/results/v3')
OUT_DIR.mkdir(parents=True, exist_ok=True)

def run_phase(name, script_path, args=None):
    t0 = time.perf_counter()
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)
    print(f'\n{"="*70}')
    print(f'PHASE: {name}')
    print(f'CMD:   {" ".join(cmd)}')
    print(f'{"="*70}')
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.perf_counter() - t0
    print(f'\n[PHASE {name}] done in {elapsed:.1f}s ({elapsed/60:.1f} min) — exit={result.returncode}')
    return elapsed, result.returncode

def main():
    t_all = time.perf_counter()
    print('=' * 70)
    print('BENCHMARK v3.3 — Subprocess Multi-Agent Runner')
    print('=' * 70)

    base = Path(__file__).parent

    phases = [
        ('table_a_batch',   base / 'table_a_agent.py',   ['--n-workers', '2']),
        ('gpu_lstm',        base / 'gpu_agent.py',         []),
        ('table_b_streaming', base / 'table_b_agent.py',  ['--n-workers', '2']),
        ('ablation',        base / 'ablation_agent.py',    ['--n-workers', '1']),
        ('bar_score',       base / 'bar_score_agent.py',   ['--n-workers', '1']),
        ('stats_batch',     base / 'stats_agent.py',       ['--table', 'batch']),
        ('stats_streaming', base / 'stats_agent.py',      ['--table', 'streaming']),
        ('plotting',        base / 'plotting_agent.py',    []),
    ]

    phase_times = {}
    for name, script, args in phases:
        elapsed, rc = run_phase(name, script, args)
        phase_times[name] = elapsed
        if rc != 0:
            print(f'\n!!! PHASE {name} FAILED with exit code {rc}. Continuing...')

    total_time = time.perf_counter() - t_all

    # Save environment
    import torch
    gpu_ok = torch.cuda.is_available()
    env = {
        'version': '3.3', 'timestamp': datetime.now().isoformat(),
        'python': sys.version,
        'cpu_cores': 18, 'gpu_available': gpu_ok,
        'gpu_device': torch.cuda.get_device_name(0) if gpu_ok else None,
        'total_minutes': round(total_time / 60, 1),
        'phase_times': {k: round(v, 1) for k, v in phase_times.items()},
        'workers': {'table_a': 2, 'table_b': 2, 'ablation': 1, 'bar_score': 1},
    }
    with open(OUT_DIR / 'environment.json', 'w') as f:
        json.dump(env, f, indent=2)

    print('\n' + '=' * 70)
    print(f'BENCHMARK v3.3 COMPLETE in {total_time/60:.1f} min')
    for name, secs in phase_times.items():
        print(f'  {name:25s}: {secs/60:.1f} min')
    print('=' * 70)

if __name__ == '__main__':
    main()
