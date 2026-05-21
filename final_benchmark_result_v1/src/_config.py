"""
Benchmark configuration — paths are relative to this project root.
Set DATA_DIR to point to your parquet data directory if different from ./data/
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
SRC_DIR = PROJECT_ROOT / "src"
RESULTS_DIR = PROJECT_ROOT / "results"
CHARTS_DIR = RESULTS_DIR / "charts"

GRID_SEARCH_STAGE1_DIR = RESULTS_DIR / "grid_search" / "stage1"
GRID_SEARCH_STAGE2_DIR = RESULTS_DIR / "grid_search" / "stage2"
GRID_SEARCH_STAGE3_DIR = RESULTS_DIR / "grid_search" / "stage3"

TRAIN_PATH = DATA_DIR / "train_clean.parquet"
VALID_PATH = DATA_DIR / "valid_polluted.parquet"
TEST_PATH = DATA_DIR / "test_polluted.parquet"
GT_MASK_PATH = DATA_DIR / "valid" / "ground_truth_mask.npy"

STAGE1_SUMMARY = GRID_SEARCH_STAGE1_DIR / "summary.json"
