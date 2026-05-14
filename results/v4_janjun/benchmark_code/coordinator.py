"""
BenchmarkCoordinator — orchestrates all agents, manages state, checkpoints.
"""
from __future__ import annotations

import json
import time
import sys
from pathlib import Path
from datetime import datetime
from multiprocessing import cpu_count

import numpy as np
import pandas as pd


class BenchmarkCoordinator:
    """
    Coordinates the full v3 benchmark pipeline across CPU + GPU agents.
    Manages:
      - Phase ordering and dependencies
      - Checkpoint / resume
      - Resource allocation hints
      - Final report generation
    """

    PHASES = [
        'data_prep',
        'table_a_batch',
        'gpu_lstm',
        'table_b_streaming',
        'ablation',
        'bar_score',
        'stats_batch',
        'stats_streaming',
        'plotting',
    ]

    def __init__(self, base_dir: Path | None = None):
        if base_dir is None:
            base_dir = Path(__file__).parent.parent
        self.base_dir = Path(base_dir)
        self.out_dir = self.base_dir / 'results' / 'v3'
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.out_dir / 'coordinator_state.json'
        self._state = self._load_state()
        self._n_cpu = cpu_count()

    def _load_state(self) -> dict:
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            'completed_phases': [],
            'started_at': None,
            'last_updated': datetime.now().isoformat(),
        }

    def _save_state(self):
        self._state['last_updated'] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self._state, f, indent=2)

    def is_phase_done(self, phase: str) -> bool:
        return phase in self._state.get('completed_phases', [])

    def mark_phase_done(self, phase: str):
        if phase not in self._state['completed_phases']:
            self._state['completed_phases'].append(phase)
        self._save_state()

    def run(self, phases: list[str] | None = None,
            force: bool = False) -> dict:
        """
        Run specified phases (or all if None).
        If force=False and a phase is already done, skip it (resume).
        Returns a summary dict.
        """
        if phases is None:
            phases = self.PHASES

        if not self._state.get('started_at'):
            self._state['started_at'] = datetime.now().isoformat()
            self._save_state()

        print('=' * 72)
        print('BENCHMARK v3.3 — MULTI-AGENT COORDINATOR')
        print(f'Started: {self._state["started_at"]}')
        print(f'Phases:  {", ".join(phases)}')
        print(f'CPU:     {self._n_cpu} logical cores')
        try:
            import torch
            gpu = torch.cuda.is_available()
            gpu_name = torch.cuda.get_device_name(0) if gpu else 'N/A'
        except Exception:
            gpu, gpu_name = False, 'N/A'
        print(f'GPU:     {"YES — " + gpu_name if gpu else "NO"}')
        print('=' * 72)

        total_t0 = time.perf_counter()
        phase_times = {}

        # ── Data Prep ──────────────────────────────────────────────
        if 'data_prep' in phases and (force or not self.is_phase_done('data_prep')):
            from .data_agent import DataAgent
            t0 = time.perf_counter()
            agent = DataAgent(self.base_dir, self.out_dir)
            agent.run()
            phase_times['data_prep'] = time.perf_counter() - t0
            self.mark_phase_done('data_prep')
            print(f'\n[COORD] Phase data_prep done ({phase_times["data_prep"]:.1f}s)\n')

        # ── Table A (CPU batch algorithms) ─────────────────────────
        if 'table_a_batch' in phases and (force or not self.is_phase_done('table_a_batch')):
            from .table_a_agent import TableAAgent
            t0 = time.perf_counter()
            agent = TableAAgent(self.base_dir, self.out_dir)
            agent.run(n_workers=2)
            phase_times['table_a_batch'] = time.perf_counter() - t0
            self.mark_phase_done('table_a_batch')
            print(f'\n[COORD] Phase table_a_batch done ({phase_times["table_a_batch"]:.1f}s)\n')

        # ── GPU LSTM-AE ────────────────────────────────────────────
        if 'gpu_lstm' in phases and (force or not self.is_phase_done('gpu_lstm')):
            from .gpu_agent import GPUAgent
            t0 = time.perf_counter()
            agent = GPUAgent(self.base_dir, self.out_dir)
            agent.run()
            phase_times['gpu_lstm'] = time.perf_counter() - t0
            self.mark_phase_done('gpu_lstm')
            print(f'\n[COORD] Phase gpu_lstm done ({phase_times["gpu_lstm"]:.1f}s)\n')

        # ── Table B (Streaming CPU) ─────────────────────────────────
        if 'table_b_streaming' in phases and (force or not self.is_phase_done('table_b_streaming')):
            from .table_b_agent import TableBAgent
            t0 = time.perf_counter()
            agent = TableBAgent(self.base_dir, self.out_dir)
            agent.run(n_workers=2)
            phase_times['table_b_streaming'] = time.perf_counter() - t0
            self.mark_phase_done('table_b_streaming')
            print(f'\n[COORD] Phase table_b_streaming done ({phase_times["table_b_streaming"]:.1f}s)\n')

        # ── Ablation Study ──────────────────────────────────────────
        if 'ablation' in phases and (force or not self.is_phase_done('ablation')):
            from .ablation_agent import AblationAgent
            t0 = time.perf_counter()
            agent = AblationAgent(self.base_dir, self.out_dir)
            agent.run(n_workers=1)
            phase_times['ablation'] = time.perf_counter() - t0
            self.mark_phase_done('ablation')
            print(f'\n[COORD] Phase ablation done ({phase_times["ablation"]:.1f}s)\n')

        # ── BAR Score ───────────────────────────────────────────────
        if 'bar_score' in phases and (force or not self.is_phase_done('bar_score')):
            from .bar_score_agent import BarScoreAgent
            t0 = time.perf_counter()
            agent = BarScoreAgent(self.base_dir, self.out_dir)
            agent.run(n_workers=2)
            phase_times['bar_score'] = time.perf_counter() - t0
            self.mark_phase_done('bar_score')
            print(f'\n[COORD] Phase bar_score done ({phase_times["bar_score"]:.1f}s)\n')

        # ── Statistical Analysis ───────────────────────────────────
        if 'stats_batch' in phases and (force or not self.is_phase_done('stats_batch')):
            from .stats_agent import StatsAgent
            t0 = time.perf_counter()
            agent = StatsAgent(self.base_dir, self.out_dir)
            agent.run(table='batch')
            phase_times['stats_batch'] = time.perf_counter() - t0
            self.mark_phase_done('stats_batch')
            print(f'\n[COORD] Phase stats_batch done ({phase_times["stats_batch"]:.1f}s)\n')

        if 'stats_streaming' in phases and (force or not self.is_phase_done('stats_streaming')):
            from .stats_agent import StatsAgent
            t0 = time.perf_counter()
            agent = StatsAgent(self.base_dir, self.out_dir)
            agent.run(table='streaming')
            phase_times['stats_streaming'] = time.perf_counter() - t0
            self.mark_phase_done('stats_streaming')
            print(f'\n[COORD] Phase stats_streaming done ({phase_times["stats_streaming"]:.1f}s)\n')

        # ── Plotting ───────────────────────────────────────────────
        if 'plotting' in phases and (force or not self.is_phase_done('plotting')):
            from .plotting_agent import PlottingAgent
            t0 = time.perf_counter()
            agent = PlottingAgent(self.base_dir, self.out_dir)
            agent.run()
            phase_times['plotting'] = time.perf_counter() - t0
            self.mark_phase_done('plotting')
            print(f'\n[COORD] Phase plotting done ({phase_times["plotting"]:.1f}s)\n')

        total_time = time.perf_counter() - total_t0

        # ── Combine & Finalize ──────────────────────────────────────
        self._finalize(phase_times, total_time)

        summary = {
            'phases_run': list(phase_times.keys()),
            'phase_times': {k: f'{v:.1f}s' for k, v in phase_times.items()},
            'total_time': f'{total_time / 60:.1f} min',
            'total_time_s': total_time,
        }
        self._save_summary(summary)
        return summary

    def _finalize(self, phase_times: dict, total_time: float):
        """Merge per-phase CSVs into final output files."""
        batch_path = self.out_dir / 'table_a_results.csv'
        gpu_path   = self.out_dir / 'gpu_lstm_results.csv'
        stream_path= self.out_dir / 'table_b_results.csv'
        abl_path   = self.out_dir / 'ablation_results.csv'
        bar_path   = self.out_dir / 'bar_score_results.csv'

        frames = []
        for path in [batch_path, gpu_path]:
            if path.exists():
                frames.append(pd.read_csv(path))
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            combined.to_csv(self.out_dir / 'benchmark_results_batch.csv', index=False)

        if stream_path.exists():
            df = pd.read_csv(stream_path)
            df.to_csv(self.out_dir / 'benchmark_results_streaming.csv', index=False)

        # Environment info
        try:
            import torch
            gpu_ok = torch.cuda.is_available()
            gpu_name = torch.cuda.get_device_name(0) if gpu_ok else None
        except Exception:
            gpu_ok, gpu_name = False, None

        env = {
            'version': '3.3',
            'timestamp': datetime.now().isoformat(),
            'python': sys.version,
            'cpu_cores': self._n_cpu,
            'gpu_available': gpu_ok,
            'gpu_device': gpu_name,
            'phases': list(phase_times.keys()),
            'phase_times': {k: round(v, 1) for k, v in phase_times.items()},
            'total_minutes': round(total_time / 60, 1),
        }
        with open(self.out_dir / 'environment.json', 'w') as f:
            json.dump(env, f, indent=2)

        print('\n' + '=' * 72)
        print('BENCHMARK v3.3 — COMPLETE')
        print(f'Total time: {total_time / 60:.1f} min')
        for phase, secs in phase_times.items():
            print(f'  {phase:25s}: {secs / 60:.1f} min')
        print('=' * 72)

    def _save_summary(self, summary: dict):
        with open(self.out_dir / 'run_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

    def print_status(self):
        print('\n── Coordinator State ──────────────────────────────────')
        print(f'Started:   {self._state.get("started_at", "never")}')
        print(f'Completed: {self._state.get("completed_phases", [])}')
        print(f'Last:      {self._state.get("last_updated", "never")}')
        print('────────────────────────────────────────────────────────\n')

    def reset(self):
        """Clear all state and results for a fresh run."""
        self._state = {'completed_phases': [], 'started_at': datetime.now().isoformat()}
        self._save_state()
        for f in self.out_dir.glob('*.csv'):
            f.unlink()
        for f in self.out_dir.glob('*.json'):
            if f.name != 'environment.json':
                f.unlink()
        print('Coordinator state reset. All results cleared.')
