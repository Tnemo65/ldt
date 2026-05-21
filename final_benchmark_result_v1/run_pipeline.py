#!/usr/bin/env python3
"""
Orchestrator for the complete MemStream benchmark pipeline.
Run all (or individual) experiment stages and save results to this directory.

Usage:
    python run_pipeline.py --all          # Run everything
    python run_pipeline.py --stage1       # Stage 1: coarse grid search
    python run_pipeline.py --stage2       # Stage 2: fine-tune training params
    python run_pipeline.py --stage3       # Stage 3: architecture variation
    python run_pipeline.py --ablation     # Ablation study
    python run_pipeline.py --comparison   # Compare with baselines
    python run_pipeline.py --context      # ContextBeta ablation
    python run_pipeline.py --drift        # Concept drift injection
    python run_pipeline.py --fulldata     # Full data training
    python run_pipeline.py --aggregate    # Aggregate + chart (after all)
"""
from __future__ import annotations

import os
import sys
import shutil
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                   stream=sys.stderr)
LOGGER = logging.getLogger("pipeline")

BASE_DIR = Path(__file__).parent.resolve()
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

PYTHON = sys.executable

DATA_DIR = r"C:\proj\ldt\GOOD_DATA"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTHONLEGACYWINDOWSSTDIO"] = "1"
os.environ["ARROW_DISABLE_MMAP"] = "1"


def _fix_results_dir(result_json_path: str, target_dir: Path):
    with open(result_json_path, "r") as f:
        content = f.read()
    if "c:\\\\proj\\\\ldt\\\\HP_benchmark_v5" in content:
        content = content.replace("c:\\\\proj\\\\ldt\\\\HP_benchmark_v5", str(BASE_DIR).replace("\\", "\\\\"))
    elif "c:\\proj\\ldt\\HP_benchmark_v5" in content:
        content = content.replace("c:\\proj\\ldt\\HP_benchmark_v5", str(BASE_DIR).replace("\\", "\\\\"))
    with open(result_json_path, "w") as f:
        f.write(content)


def run_stage1(max_train: int = 200000, max_val: int = 100000):
    LOGGER.info("=== STAGE 1: Coarse Grid Search ===")
    os.chdir(SRC_DIR)
    result_dir = BASE_DIR / "results" / "grid_search" / "stage1"
    result_dir.mkdir(parents=True, exist_ok=True)

    cmd = (
        f'"{PYTHON}" grid_search.py '
        f'--max-train {max_train} '
        f'--max-val {max_val} '
        f'--result-dir "{result_dir}" '
        f'--data-dir "{DATA_DIR}" '
        f'--stages 1'
    )
    LOGGER.info("Running: %s", cmd)
    os.system(cmd)


def run_stage2(max_train: int = 200000, max_val: int = 100000):
    LOGGER.info("=== STAGE 2: Fine-tune Training Params ===")
    os.chdir(SRC_DIR / "experiments")
    result_dir = BASE_DIR / "results" / "grid_search" / "stage2"
    result_dir.mkdir(parents=True, exist_ok=True)

    cmd = (
        f'"{PYTHON}" grid_search_s2.py '
        f'--max-train {max_train} '
        f'--max-val {max_val} '
        f'--result-dir "{result_dir}" '
        f'--data-dir "{DATA_DIR}"'
    )
    LOGGER.info("Running: %s", cmd)
    os.system(cmd)


def run_stage3(max_train: int = 200000, max_val: int = 100000):
    LOGGER.info("=== STAGE 3: Architecture Variation ===")
    os.chdir(SRC_DIR / "experiments")
    result_dir = BASE_DIR / "results" / "grid_search" / "stage3"
    result_dir.mkdir(parents=True, exist_ok=True)

    cmd = (
        f'"{PYTHON}" grid_search_s3.py '
        f'--max-train {max_train} '
        f'--max-val {max_val} '
        f'--result-dir "{result_dir}" '
        f'--data-dir "{DATA_DIR}"'
    )
    LOGGER.info("Running: %s", cmd)
    os.system(cmd)


def run_ablation(max_train: int = 200000, max_val: int = 100000, n_runs: int = 3):
    LOGGER.info("=== ABLATION STUDY ===")
    os.chdir(SRC_DIR)
    result_dir = BASE_DIR / "results" / "ablation"
    result_dir.mkdir(parents=True, exist_ok=True)

    cmd = (
        f'"{PYTHON}" ablation.py '
        f'--max-train {max_train} '
        f'--max-val {max_val} '
        f'--result-dir "{result_dir}" '
        f'--data-dir "{DATA_DIR}" '
        f'--n-runs {n_runs}'
    )
    LOGGER.info("Running: %s", cmd)
    os.system(cmd)


def run_comparison(max_train: int = 200000, max_val: int = 100000):
    LOGGER.info("=== COMPARISON WITH BASELINES ===")
    os.chdir(SRC_DIR)
    result_dir = BASE_DIR / "results" / "comparison"
    result_dir.mkdir(parents=True, exist_ok=True)

    cmd = (
        f'"{PYTHON}" comparison.py '
        f'--max-train {max_train} '
        f'--max-val {max_val} '
        f'--result-dir "{result_dir}" '
        f'--data-dir "{DATA_DIR}"'
    )
    LOGGER.info("Running: %s", cmd)
    os.system(cmd)


def run_context(max_train: int = 200000, max_val: int = 100000):
    LOGGER.info("=== CONTEXTBETA ABLATION ===")
    os.chdir(SRC_DIR / "experiments")
    result_path = BASE_DIR / "results" / "context_ablation_results.json"

    cmd = (
        f'"{PYTHON}" context_stratified.py '
        f'--max-train {max_train} '
        f'--max-val {max_val} '
        f'--data-dir "{DATA_DIR}" '
        f'--output "{result_path}"'
    )
    LOGGER.info("Running: %s", cmd)
    os.system(cmd)


def run_drift(max_train: int = 200000, epochs: int = 5000):
    LOGGER.info("=== CONCEPT DRIFT INJECTION ===")
    os.chdir(SRC_DIR / "experiments")
    result_path = BASE_DIR / "results" / "concept_drift_results.json"

    cmd = (
        f'"{PYTHON}" concept_drift.py '
        f'--max-train {max_train} '
        f'--epochs {epochs} '
        f'--data-dir "{DATA_DIR}" '
        f'--output "{result_path}"'
    )
    LOGGER.info("Running: %s", cmd)
    os.system(cmd)


def run_fulldata(epochs: int = 5000):
    LOGGER.info("=== FULL DATA TRAINING ===")
    os.chdir(SRC_DIR / "experiments")
    result_path = BASE_DIR / "results" / "full_data_results.json"

    cmd = (
        f'"{PYTHON}" full_data.py '
        f'--epochs {epochs} '
        f'--data-dir "{DATA_DIR}" '
        f'--output "{result_path}"'
    )
    LOGGER.info("Running: %s", cmd)
    os.system(cmd)


def run_aggregate():
    LOGGER.info("=== AGGREGATE RESULTS + CHARTS ===")
    os.chdir(SRC_DIR / "experiments")
    chart_dir = BASE_DIR / "results" / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)

    cmd = (
        f'"{PYTHON}" aggregate_results.py '
        f'--chart-dir "{chart_dir}"'
    )
    LOGGER.info("Running: %s", cmd)
    os.system(cmd)


def main():
    parser = argparse.ArgumentParser(description="MemStream Benchmark Pipeline")
    parser.add_argument("--all", action="store_true", help="Run all stages in sequence")
    parser.add_argument("--stage1", action="store_true", help="Stage 1: coarse grid search")
    parser.add_argument("--stage2", action="store_true", help="Stage 2: fine-tune training params")
    parser.add_argument("--stage3", action="store_true", help="Stage 3: architecture variation")
    parser.add_argument("--ablation", action="store_true", help="Ablation study")
    parser.add_argument("--comparison", action="store_true", help="Compare with baselines")
    parser.add_argument("--context", action="store_true", help="ContextBeta ablation")
    parser.add_argument("--drift", action="store_true", help="Concept drift injection")
    parser.add_argument("--fulldata", action="store_true", help="Full data training")
    parser.add_argument("--aggregate", action="store_true", help="Aggregate results + chart")
    parser.add_argument("--max-train", type=int, default=200000, help="Max training samples (default: 200000)")
    parser.add_argument("--max-val", type=int, default=100000, help="Max validation samples (default: 100000)")
    parser.add_argument("--n-runs", type=int, default=3, help="Number of runs for ablation (default: 3)")
    args = parser.parse_args()

    if args.all:
        run_stage1(args.max_train, args.max_val)
        run_stage2(args.max_train, args.max_val)
        run_stage3(args.max_train, args.max_val)
        run_ablation(args.max_train, args.max_val, args.n_runs)
        run_comparison(args.max_train, args.max_val)
        run_context(args.max_train, args.max_val)
        run_drift(args.max_train)
        run_fulldata()
        run_aggregate()
        LOGGER.info("=== ALL DONE ===")
        return

    stages = []
    if args.stage1:
        run_stage1(args.max_train, args.max_val)
        stages.append("stage1")
    if args.stage2:
        run_stage2(args.max_train, args.max_val)
        stages.append("stage2")
    if args.stage3:
        run_stage3(args.max_train, args.max_val)
        stages.append("stage3")
    if args.ablation:
        run_ablation(args.max_train, args.max_val, args.n_runs)
        stages.append("ablation")
    if args.comparison:
        run_comparison(args.max_train, args.max_val)
        stages.append("comparison")
    if args.context:
        run_context(args.max_train, args.max_val)
        stages.append("context")
    if args.drift:
        run_drift(args.max_train)
        stages.append("drift")
    if args.fulldata:
        run_fulldata()
        stages.append("fulldata")
    if args.aggregate:
        run_aggregate()
        stages.append("aggregate")

    if not stages:
        parser.print_help()


if __name__ == "__main__":
    main()
