#!/usr/bin/env python3
"""
================================================================================
STEP 1: GRID SEARCH — Find Best Hyperparameters (File 1/3)
================================================================================

Tim bo hyperparameter tot nhat cho MemStream voi bai toan NYC Taxi anomaly detection.

TAI SAO CHIA NHIEU GIA TRI:
  - memory_len: NYC taxi co rat nhieu neighborhood/context. Memory lon giu duoc
    nhieu trend, nhung qua lon thi cham va co drift. Can tim sweet spot.
  - k: So nguoi lang gieng. Nho hon thi nhay cam hon voi anomaly cu the,
    lon hon thi on dinh hon nhung co the miss nuanced anomalies.
  - epochs: Train nhieu epoch giu AE tot hon nhung ton thoi gian.
  - lr: Learning rate anh huong toi huong loi va thoi gian hoi tu.
  - noise_std: Noise lon lam AE kho phuc hoi, nhung co the cham hon voi
    anomaly. Paper dung 0.001 nhung co the can tang cho NYC.
  - beta: Nguong de update memory. Cao = update nhieu = adaptation nhanh,
    nhung co the bi poisoning.
  - gamma: KNN weighting. Paper gamma=0 tot nhat, nhung voi streaming
    co the can gamma>0 de tu phuc hoi memory poisoning.
  - out_dim: D = d/2, d, 2d, 5d (paper Table 6d)

STAGES:
  Stage 1 (Coarse): 3 * 3 * 1 * 1 = 9 configs, 5000 epochs
    - memory_len: [256, 512, 1024]
    - k: [3, 5, 10]
    - gamma: [0.0]
    - beta: [0.001] (fixed for eval mode)
    - Fixed: epochs=5000, lr=0.01, noise=0.001, out_dim=68, adam_betas=(0.9, 0.999)

  Stage 2 (Fine): top-15 from Stage 1, 5000 epochs
    - epochs: [2000, 5000, 10000]
    - lr: [0.005, 0.01, 0.02]
    - noise: [0.0005, 0.001, 0.005]

  Stage 3 (Architecture): top-10 from Stage 2, 5000 epochs
    - out_dim: [17, 34, 68, 136, 170]  # paper Table 6(d): d/2, d, 2d, 4d, 5d
    - memory_len: [512, 1024, 2048]
    - k: [3, 5, 7]
"""
from __future__ import annotations

import os
import sys
import json
import time
import shutil
import io
import pickle
import hashlib
import hmac
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["PYTHONLEGACYWINDOWSSTDIO"] = "1"
os.environ["ARROW_DISABLE_MMAP"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
LOGGER = logging.getLogger("grid_search")

PYTHON = r"C:\Users\Administrator\Desktop\AI ComfyUI\system\python\python.exe"


# ---------------------------------------------------------------------------
# Import core (inline de tranh circular import)
# ---------------------------------------------------------------------------
# sys.path.insert handled by Path(__file__) in src/ directory
sys.path.insert(0, str(Path(__file__).parent))
from benchmark_core import (
    MemStreamPipeline, extract_features_from_parquet,
    evaluate_scores, find_best_threshold, DEVICE, LOGGER
)


# ---------------------------------------------------------------------------
# Stage 1: Coarse Grid Search
# ---------------------------------------------------------------------------
def stage1_coarse(
    X_train: np.ndarray,
    hours_train: np.ndarray,
    dows_train: np.ndarray,
    rcs_train: np.ndarray,
    nb_train: np.ndarray,
    X_val: np.ndarray,
    hours_val: np.ndarray,
    dows_val: np.ndarray,
    rcs_val: np.ndarray,
    nb_val: np.ndarray,
    gt_mask_val: np.ndarray,
    output_dir: str,
    epochs: int = 5000,
    skip_existing: bool = True,
    beta_pct: int = 90,
) -> List[Dict]:
    """
    Stage 1: Coarse search tren cac tham so quan trong nhat.

    Grid: 3 * 3 * 5 * 5 = 225 configs (paper: epochs=5000, lr=0.01, adam_betas=(0.9, 0.999))
    """
    results_dir = os.path.join(output_dir, "stage1")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)

    GRID = {
        "memory_len": [256, 512, 1024],  # 3 values
        "k": [3, 5, 10],                 # 3 values
        "gamma": [0.0],                   # 1 value
        "beta": [0.001],                  # 1 value (fixed for eval mode)
    }
    # Total: 3 * 3 * 1 * 1 = 9 configs

    configs = _grid_configs(GRID)
    LOGGER.info(
        "Stage 1: %d configs | memory=%s | k=%s | gamma=%s | beta=%s | epochs=%d",
        len(configs), GRID["memory_len"], GRID["k"], GRID["gamma"], GRID["beta"], epochs
    )

    all_results = []
    for i, cfg in enumerate(configs):
        config_id = _make_id(cfg)
        result_path = os.path.join(results_dir, f"{config_id}.json")

        if skip_existing and os.path.exists(result_path):
            with open(result_path) as f:
                result = json.load(f)
            LOGGER.info("[%d/%d] SKIP %s — AUC-PR=%.4f",
                       i + 1, len(configs), config_id, result.get("auc_pr", 0))
            all_results.append(result)
            continue

        LOGGER.info("[%d/%d] %s", i + 1, len(configs), config_id)
        t0 = time.time()

        try:
            pipeline = MemStreamPipeline(
                d=38, out_dim=38,
                memory_len=cfg["memory_len"],
                k=cfg["k"],
                gamma=cfg["gamma"],
                beta=cfg["beta"],
                noise_std=0.001,
                lr=0.01,
                epochs=epochs,
                batch_size=1024,
                seed=42,
                cb_warmup=min(4096, cfg["memory_len"] * 4),
                verbose=False,
                adam_betas=(0.9, 0.999),  # paper Section 5
            )
            pipeline.train(
                X_train, hours_train, dows_train, rcs_train, nb_train,
                X_warmup=X_train[:cfg["memory_len"]],
                hours_warmup=hours_train[:cfg["memory_len"]],
                dows_warmup=dows_train[:cfg["memory_len"]],
                rcs_warmup=rcs_train[:cfg["memory_len"]],
                nb_warmup=nb_train[:cfg["memory_len"]],
            )

            # Score validation (streaming per-point) — FIX: enable memory updates
            # With update_memory=True, the model adapts to validation distribution
            # This is the KEY fix for low AUC-PR (from 0.09 to expected 0.4-0.8)
            adj_scores, metrics = pipeline.score_stream(
                X_val, hours_val, dows_val, rcs_val, nb_val,
                gt_mask=gt_mask_val,
                update_memory=True,  # FIX: enable streaming adaptation
            )
            best_t, best_f1 = find_best_threshold(adj_scores, gt_mask_val)
            metrics_opt = evaluate_scores(adj_scores, gt_mask_val, threshold=best_t)

            result = {
                "config_id": config_id,
                "stage": 1,
                "memory_len": cfg["memory_len"],
                "k": cfg["k"],
                "gamma": cfg["gamma"],
                "beta": cfg["beta"],
                "noise_std": 0.001,
                "lr": 0.01,
                "out_dim": 68,
                "epochs": epochs,
                "adam_betas": [0.9, 0.999],
                "auc_roc": float(metrics_opt["auc_roc"]),
                "auc_pr": float(metrics_opt["auc_pr"]),
                "f1": float(metrics_opt["f1"]),
                "precision": float(metrics_opt["precision"]),
                "recall": float(metrics_opt["recall"]),
                "fpr": float(metrics_opt["fpr"]),
                "acc": float(metrics_opt["acc"]),
                "best_threshold": float(best_t),
                "score_normal_mean": float(metrics_opt["score_normal_mean"]),
                "score_anomaly_mean": float(metrics_opt["score_anomaly_mean"]),
                "separation_ratio": float(metrics_opt["separation_ratio"]),
                "train_time_s": float(time.time() - t0),
            }

            with open(result_path, "w") as f:
                json.dump(result, f, indent=2)

            _save_checkpoint(pipeline, output_dir, config_id, result)

            all_results.append(result)
            LOGGER.info(
                "  -> AUC-PR=%.4f, AUC-ROC=%.4f, F1=%.4f, sep=%.2fx [%.1fs]",
                result["auc_pr"], result["auc_roc"], result["f1"],
                result["separation_ratio"], time.time() - t0
            )

        except Exception as e:
            LOGGER.error("  [ERROR] %s: %s", config_id, e)
            import traceback
            traceback.print_exc()
            result = {
                "config_id": config_id, "stage": 1,
                "error": str(e), "memory_len": cfg["memory_len"],
                "k": cfg["k"], "gamma": cfg["gamma"], "beta": cfg["beta"],
                "auc_pr": 0.0, "auc_roc": 0.0, "f1": 0.0,
                "train_time_s": float(time.time() - t0),
            }
            all_results.append(result)

    all_results.sort(key=lambda x: x.get("auc_pr", 0), reverse=True)
    _save_stage_summary(results_dir, all_results, 1)
    return all_results


# ---------------------------------------------------------------------------
# Stage 2: Fine-tune top configs
# ---------------------------------------------------------------------------
def stage2_fine(
    X_train: np.ndarray,
    hours_train: np.ndarray,
    dows_train: np.ndarray,
    rcs_train: np.ndarray,
    nb_train: np.ndarray,
    X_val: np.ndarray,
    hours_val: np.ndarray,
    dows_val: np.ndarray,
    rcs_val: np.ndarray,
    nb_val: np.ndarray,
    gt_mask_val: np.ndarray,
    top_results: List[Dict],
    output_dir: str,
    n_top: int = 3,  # was 15
    epochs: int = 5000,
    skip_existing: bool = True,
) -> List[Dict]:
    """
    Stage 2: Fine-tune top configs tu Stage 1.

    Moi top config se test them:
      - epochs: [2000, 5000, 10000]  # paper: 5000 epochs
      - lr: [0.005, 0.01, 0.02]
      - noise: [0.0005, 0.001, 0.005]
    """
    results_dir = os.path.join(output_dir, "stage2")
    os.makedirs(results_dir, exist_ok=True)

    # Chi lay top-N
    top = top_results[:n_top]
    LOGGER.info("Stage 2: Fine-tuning top %d configs", len(top))

    fine_grid = {
        "epochs": [500, 1000],   # 2 values (was 3)
        "lr": [0.01, 0.005],    # 2 values (was 3)
        "noise": [0.001],        # 1 value (was 3)
    }
    # Total: 3 * 2 * 2 * 1 = 12 configs

    all_results = []
    for base in top:
        if base.get("auc_pr", 0) <= 0.001:
            continue

        # Generate fine configs
        for ep in fine_grid["epochs"]:
            for lr in fine_grid["lr"]:
                for noise in fine_grid["noise"]:
                    cfg_id = (
                        f"M{base['memory_len']}_k{base['k']}_g{int(base['gamma']*100)}_"
                        f"b{str(base['beta']).replace('.','p')}_e{ep}_lr{int(lr*1000)}_n{noise}"
                    )
                    result_path = os.path.join(results_dir, f"{cfg_id}.json")

                    if skip_existing and os.path.exists(result_path):
                        with open(result_path) as f:
                            result = json.load(f)
                        LOGGER.info("  SKIP %s — AUC-PR=%.4f", cfg_id, result.get("auc_pr", 0))
                        all_results.append(result)
                        continue

                    LOGGER.info("  %s", cfg_id)
                    t0 = time.time()

                    try:
                        pipeline = MemStreamPipeline(
                            d=38, out_dim=38,
                            memory_len=base["memory_len"],
                            k=base["k"],
                            gamma=base["gamma"],
                            beta=base["beta"],
                            noise_std=noise,
                            lr=lr,
                            epochs=ep,
                            batch_size=1024,
                            seed=42,
                            cb_warmup=min(4096, base["memory_len"] * 4),
                            verbose=False,
                            adam_betas=(0.9, 0.999),  # paper Section 5
                        )
                        pipeline.train(
                            X_train, hours_train, dows_train, rcs_train, nb_train,
                            X_warmup=X_train[:base["memory_len"]],
                            hours_warmup=hours_train[:base["memory_len"]],
                            dows_warmup=dows_train[:base["memory_len"]],
                            rcs_warmup=rcs_train[:base["memory_len"]],
                            nb_warmup=nb_train[:base["memory_len"]],
                        )

                        adj_scores, metrics = pipeline.score_stream(
                            X_val, hours_val, dows_val, rcs_val, nb_val,
                            gt_mask=gt_mask_val,
                            update_memory=True,  # FIX: enable streaming adaptation
                        )
                        best_t, best_f1 = find_best_threshold(adj_scores, gt_mask_val)
                        metrics_opt = evaluate_scores(adj_scores, gt_mask_val, threshold=best_t)

                        result = {
                            "config_id": cfg_id,
                            "stage": 2,
                            "base_id": base["config_id"],
                            "memory_len": base["memory_len"],
                            "k": base["k"],
                            "gamma": base["gamma"],
                            "beta": base["beta"],
                            "noise_std": noise,
                            "lr": lr,
                            "out_dim": 68,
                            "epochs": ep,
                            "adam_betas": [0.9, 0.999],
                            "auc_roc": float(metrics_opt["auc_roc"]),
                            "auc_pr": float(metrics_opt["auc_pr"]),
                            "f1": float(metrics_opt["f1"]),
                            "precision": float(metrics_opt["precision"]),
                            "recall": float(metrics_opt["recall"]),
                            "fpr": float(metrics_opt["fpr"]),
                            "acc": float(metrics_opt["acc"]),
                            "best_threshold": float(best_t),
                            "score_normal_mean": float(metrics_opt["score_normal_mean"]),
                            "score_anomaly_mean": float(metrics_opt["score_anomaly_mean"]),
                            "separation_ratio": float(metrics_opt["separation_ratio"]),
                            "train_time_s": float(time.time() - t0),
                        }

                        with open(result_path, "w") as f:
                            json.dump(result, f, indent=2)

                        _save_checkpoint(pipeline, output_dir, cfg_id, result)

                        all_results.append(result)
                        LOGGER.info(
                            "    -> AUC-PR=%.4f, AUC-ROC=%.4f, F1=%.4f [%.1fs]",
                            result["auc_pr"], result["auc_roc"], result["f1"], time.time() - t0
                        )

                    except Exception as e:
                        LOGGER.error("    [ERROR] %s: %s", cfg_id, e)
                        result = {
                            "config_id": cfg_id, "stage": 2,
                            "error": str(e), "memory_len": base["memory_len"],
                            "k": base["k"], "auc_pr": 0.0,
                            "train_time_s": float(time.time() - t0),
                        }
                        all_results.append(result)

    all_results.sort(key=lambda x: x.get("auc_pr", 0), reverse=True)
    _save_stage_summary(results_dir, all_results, 2)
    return all_results


# ---------------------------------------------------------------------------
# Stage 3: Architecture variation
# ---------------------------------------------------------------------------
def stage3_architecture(
    X_train: np.ndarray,
    hours_train: np.ndarray,
    dows_train: np.ndarray,
    rcs_train: np.ndarray,
    nb_train: np.ndarray,
    X_val: np.ndarray,
    hours_val: np.ndarray,
    dows_val: np.ndarray,
    rcs_val: np.ndarray,
    nb_val: np.ndarray,
    gt_mask_val: np.ndarray,
    top_results: List[Dict],
    output_dir: str,
    n_top: int = 3,  # was 10
    epochs: int = 5000,
    skip_existing: bool = True,
) -> List[Dict]:
    """
    Stage 3: Test khac nhau ve architecture.

    Dua tren paper Table 6d: D = d/2, d, 2d, 4d, 5d
      - out_dim = 17: D = d/2
      - out_dim = 34: D = d
      - out_dim = 68: D = 2d (recommended)
      - out_dim = 136: D = 4d
      - out_dim = 170: D = 5d

    Cong them:
      - memory_len: [512, 1024, 2048]
      - k: [3, 5, 7]
    """
    results_dir = os.path.join(output_dir, "stage3")
    os.makedirs(results_dir, exist_ok=True)

    top = top_results[:n_top]
    LOGGER.info("Stage 3: Architecture variation, top %d configs", len(top))

    arch_grid = {
        "out_dim": [38, 76],    # d, 2d for d=38
        "memory_len": [1024],   # 1 value (best from S2)
        "k": [3],              # 1 value (best from S2)
    }
    # Total: 2 * 1 * 1 = 2 configs (was 12)

    all_results = []
    for best in top:
        if best.get("auc_pr", 0) <= 0.001:
            continue

        for od in arch_grid["out_dim"]:
            for ml in arch_grid["memory_len"]:
                for k in arch_grid["k"]:
                    cfg_id = (
                        f"M{ml}_k{k}_g{int(best['gamma']*100)}_b{str(best['beta']).replace('.','p')}_"
                        f"e{best['epochs']}_lr{int(best['lr']*1000)}_n{best['noise_std']}_od{od}"
                    )
                    result_path = os.path.join(results_dir, f"{cfg_id}.json")

                    if skip_existing and os.path.exists(result_path):
                        with open(result_path) as f:
                            result = json.load(f)
                        LOGGER.info("  SKIP %s — AUC-PR=%.4f", cfg_id, result.get("auc_pr", 0))
                        all_results.append(result)
                        continue

                    LOGGER.info("  %s", cfg_id)
                    t0 = time.time()

                    try:
                        pipeline = MemStreamPipeline(
                            d=38, out_dim=od,
                            memory_len=ml,
                            k=k,
                            gamma=best["gamma"],
                            beta=best["beta"],
                            noise_std=best["noise_std"],
                            lr=best["lr"],
                            epochs=epochs,
                            batch_size=1024,
                            seed=42,
                            cb_warmup=min(4096, ml * 4),
                            verbose=False,
                            adam_betas=(0.9, 0.999),  # paper Section 5
                        )
                        pipeline.train(
                            X_train, hours_train, dows_train, rcs_train, nb_train,
                            X_warmup=X_train[:ml],
                            hours_warmup=hours_train[:ml],
                            dows_warmup=dows_train[:ml],
                            rcs_warmup=rcs_train[:ml],
                            nb_warmup=nb_train[:ml],
                        )

                        adj_scores, metrics = pipeline.score_stream(
                            X_val, hours_val, dows_val, rcs_val, nb_val,
                            gt_mask=gt_mask_val,
                            update_memory=True,  # FIX: enable streaming adaptation
                        )
                        best_t, best_f1 = find_best_threshold(adj_scores, gt_mask_val)
                        metrics_opt = evaluate_scores(adj_scores, gt_mask_val, threshold=best_t)

                        result = {
                            "config_id": cfg_id,
                            "stage": 3,
                            "memory_len": ml,
                            "k": k,
                            "gamma": best["gamma"],
                            "beta": best["beta"],
                            "noise_std": best["noise_std"],
                            "lr": best["lr"],
                            "out_dim": od,
                            "epochs": epochs,
                            "adam_betas": [0.9, 0.999],
                            "auc_roc": float(metrics_opt["auc_roc"]),
                            "auc_pr": float(metrics_opt["auc_pr"]),
                            "f1": float(metrics_opt["f1"]),
                            "precision": float(metrics_opt["precision"]),
                            "recall": float(metrics_opt["recall"]),
                            "fpr": float(metrics_opt["fpr"]),
                            "acc": float(metrics_opt["acc"]),
                            "best_threshold": float(best_t),
                            "score_normal_mean": float(metrics_opt["score_normal_mean"]),
                            "score_anomaly_mean": float(metrics_opt["score_anomaly_mean"]),
                            "separation_ratio": float(metrics_opt["separation_ratio"]),
                            "train_time_s": float(time.time() - t0),
                        }

                        with open(result_path, "w") as f:
                            json.dump(result, f, indent=2)

                        _save_checkpoint(pipeline, output_dir, cfg_id, result)

                        all_results.append(result)
                        LOGGER.info(
                            "    -> AUC-PR=%.4f, AUC-ROC=%.4f, F1=%.4f [%.1fs]",
                            result["auc_pr"], result["auc_roc"], result["f1"], time.time() - t0
                        )

                    except Exception as e:
                        LOGGER.error("    [ERROR] %s: %s", cfg_id, e)
                        result = {
                            "config_id": cfg_id, "stage": 3,
                            "error": str(e), "out_dim": od, "memory_len": ml, "k": k,
                            "auc_pr": 0.0, "train_time_s": float(time.time() - t0),
                        }
                        all_results.append(result)

    all_results.sort(key=lambda x: x.get("auc_pr", 0), reverse=True)
    _save_stage_summary(results_dir, all_results, 3)
    return all_results


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _grid_configs(grid: Dict) -> List[Dict]:
    import itertools
    keys = list(grid.keys())
    vals = list(grid.values())
    configs = []
    for combo in itertools.product(*vals):
        cfg = dict(zip(keys, combo))
        configs.append(cfg)
    return configs


def _make_id(cfg: Dict) -> str:
    beta_str = str(cfg.get("beta", 0.5)).replace(".", "p")
    gamma_str = str(int(cfg.get("gamma", 0.0) * 100))
    return (f"M{cfg['memory_len']}_k{cfg['k']}_g{gamma_str}_b{beta_str}")


def _save_checkpoint(pipeline, output_dir: str, config_id: str, result: Dict) -> None:
    """Save model checkpoint."""
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{config_id}.pt")
    ckpt_state = {
        "ae": {
            "input_dim": 34,
            "out_dim": result.get("out_dim", 68),
            "encoder.weight": pipeline.ae.encoder.weight.detach().cpu().numpy(),
            "encoder.bias": pipeline.ae.encoder.bias.detach().cpu().numpy(),
            "decoder.weight": pipeline.ae.decoder.weight.detach().cpu().numpy(),
            "decoder.bias": pipeline.ae.decoder.bias.detach().cpu().numpy(),
        } if pipeline.ae is not None else {},
        "mean": pipeline.mean,
        "std": pipeline.std,
        "mem_M": pipeline.memory.active() if pipeline.memory else np.array([]),
        "mem_cnt": pipeline.memory.cnt if pipeline.memory else 0,
        "cb_T": pipeline.cb.T if pipeline.cb else None,
        "k": result["k"],
        "gamma": result["gamma"],
        "memory_len": result["memory_len"],
        "config_id": config_id,
    }
    buf = io.BytesIO()
    torch.save(ckpt_state, buf, pickle_module=pickle)
    data = buf.getvalue()
    with open(ckpt_path, "wb") as f:
        f.write(data)
    sig = hmac.new(
        b"benchmark-v4-signing-key-32ch!", data, hashlib.sha256
    ).hexdigest()
    with open(ckpt_path + ".hmac", "w") as f:
        f.write(sig)


def _save_stage_summary(results_dir: str, results: List[Dict], stage: int) -> None:
    """Save sorted results + best."""
    results.sort(key=lambda x: x.get("auc_pr", 0), reverse=True)
    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump({"stage": stage, "results": results, "best": results[0] if results else {}}, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Grid Search — MemStream v4")
    parser.add_argument("--base", default=str(Path(__file__).parent.parent),
                        help="Base directory")
    parser.add_argument("--epochs-s1", type=int, default=5000,
                        help="Stage 1 epochs (paper: 5000)")
    parser.add_argument("--epochs-s2", type=int, default=5000,
                        help="Stage 2 epochs (paper: 5000)")
    parser.add_argument("--epochs-s3", type=int, default=5000,
                        help="Stage 3 epochs (paper: 5000)")
    parser.add_argument("--max-train", type=int, default=500000,
                        help="Max training samples (0 = full dataset)")
    parser.add_argument("--max-val", type=int, default=500000,
                        help="Max validation samples (0 = full dataset)")
    parser.add_argument("--beta-pct", type=int, default=90,
                        help="Percentile for ContextBeta threshold (paper: 90-95)")
    parser.add_argument("--top-s2", type=int, default=15,
                        help="Top configs for Stage 2")
    parser.add_argument("--top-s3", type=int, default=10,
                        help="Top configs for Stage 3")
    parser.add_argument("--no-skip", action="store_true",
                        help="Retrain all")
    parser.add_argument("--stage", type=int, default=0,
                        help="Run only stage N (0=all)")
    args = parser.parse_args()

    base = Path(args.base)
    data_dir = base / "data"
    train_path = str(data_dir / "train_clean.parquet")
    valid_path = str(data_dir / "valid_polluted.parquet")
    output_dir = str(base / "results" / "grid_search")

    LOGGER.info("=" * 60)
    LOGGER.info("MemStream v4 — Grid Search")
    LOGGER.info("=" * 60)
    LOGGER.info("Stage 1: %d epochs", args.epochs_s1)
    LOGGER.info("Stage 2: %d epochs, top %d", args.epochs_s2, args.top_s2)
    LOGGER.info("Stage 3: %d epochs, top %d", args.epochs_s3, args.top_s3)
    LOGGER.info("Output: %s", output_dir)
    LOGGER.info("=" * 60)

    # Load data
    LOGGER.info("Loading training data...")
    X_train, hours_train, dows_train, rcs_train, nb_train = extract_features_from_parquet(
        train_path, max_rows=args.max_train  # was 500_000
    )
    LOGGER.info("Training: %d rows", len(X_train))

    LOGGER.info("Loading validation data...")
    X_val, hours_val, dows_val, rcs_val, nb_val = extract_features_from_parquet(
        valid_path, max_rows=500_000
    )
    gt_mask = np.load(str(data_dir / "valid" / "ground_truth_mask.npy"))
    gt_mask = gt_mask[-len(X_val):]
    LOGGER.info("Validation: %d rows (GT: %d anomalies, %.2f%%)",
               len(X_val), gt_mask.sum(), gt_mask.mean() * 100)

    skip = not args.no_skip
    t_total = time.time()

    # ---- Stage 1 ----
    if args.stage in [0, 1]:
        LOGGER.info("")
        LOGGER.info("=" * 40)
        LOGGER.info("STAGE 1: Coarse Grid Search")
        LOGGER.info("=" * 40)
        s1_results = stage1_coarse(
            X_train, hours_train, dows_train, rcs_train, nb_train,
            X_val, hours_val, dows_val, rcs_val, nb_val, gt_mask,
            output_dir, epochs=args.epochs_s1, skip_existing=skip
        )
        s1_best = s1_results[0] if s1_results else {}
        LOGGER.info("Stage 1 best: %s — AUC-PR=%.4f",
                   s1_best.get("config_id", "N/A"), s1_best.get("auc_pr", 0))
    else:
        s1_path = os.path.join(output_dir, "stage1", "summary.json")
        with open(s1_path) as f:
            s1_data = json.load(f)
        s1_results = s1_data["results"]
        s1_best = s1_results[0] if s1_results else {}

    # ---- Stage 2 ----
    if args.stage in [0, 2]:
        LOGGER.info("")
        LOGGER.info("=" * 40)
        LOGGER.info("STAGE 2: Fine-tune Top Configs")
        LOGGER.info("=" * 40)
        s2_results = stage2_fine(
            X_train, hours_train, dows_train, rcs_train, nb_train,
            X_val, hours_val, dows_val, rcs_val, nb_val, gt_mask,
            s1_results, output_dir, n_top=args.top_s2,
            epochs=args.epochs_s2, skip_existing=skip
        )
        s2_best = s2_results[0] if s2_results else {}
        LOGGER.info("Stage 2 best: %s — AUC-PR=%.4f",
                   s2_best.get("config_id", "N/A"), s2_best.get("auc_pr", 0))
    else:
        s2_path = os.path.join(output_dir, "stage2", "summary.json")
        if os.path.exists(s2_path):
            with open(s2_path) as f:
                s2_data = json.load(f)
            s2_results = s2_data["results"]
            s2_best = s2_results[0] if s2_results else {}
        else:
            s2_results, s2_best = [], {}

    # ---- Stage 3 ----
    if args.stage in [0, 3]:
        LOGGER.info("")
        LOGGER.info("=" * 40)
        LOGGER.info("STAGE 3: Architecture Variation")
        LOGGER.info("=" * 40)
        combined = s2_results + s1_results
        combined.sort(key=lambda x: x.get("auc_pr", 0), reverse=True)
        s3_results = stage3_architecture(
            X_train, hours_train, dows_train, rcs_train, nb_train,
            X_val, hours_val, dows_val, rcs_val, nb_val, gt_mask,
            combined, output_dir, n_top=args.top_s3,
            epochs=args.epochs_s3, skip_existing=skip
        )
        s3_best = s3_results[0] if s3_results else {}
        LOGGER.info("Stage 3 best: %s — AUC-PR=%.4f",
                   s3_best.get("config_id", "N/A"), s3_best.get("auc_pr", 0))
    else:
        s3_path = os.path.join(output_dir, "stage3", "summary.json")
        if os.path.exists(s3_path):
            with open(s3_path) as f:
                s3_data = json.load(f)
            s3_results = s3_data["results"]
            s3_best = s3_results[0] if s3_results else {}
        else:
            s3_results, s3_best = [], {}

    # ---- Final: Best across all stages ----
    all_stage_results = []
    for stage_dir in ["stage1", "stage2", "stage3"]:
        summary_path = os.path.join(output_dir, stage_dir, "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                data = json.load(f)
            all_stage_results.extend(data.get("results", []))

    all_stage_results.sort(key=lambda x: x.get("auc_pr", 0), reverse=True)
    best = all_stage_results[0] if all_stage_results else {}

    # Save final results
    final_output = {
        "all_results": all_stage_results,
        "best_config": best,
        "stages_summary": {
            "stage1_best": s1_best.get("config_id", "N/A") if s1_best else "N/A",
            "stage2_best": s2_best.get("config_id", "N/A") if s2_best else "N/A",
            "stage3_best": s3_best.get("config_id", "N/A") if s3_best else "N/A",
            "stage1_auc_pr": s1_best.get("auc_pr", 0) if s1_best else 0,
            "stage2_auc_pr": s2_best.get("auc_pr", 0) if s2_best else 0,
            "stage3_auc_pr": s3_best.get("auc_pr", 0) if s3_best else 0,
        },
        "total_time_s": float(time.time() - t_total),
        "epochs": {"s1": args.epochs_s1, "s2": args.epochs_s2, "s3": args.epochs_s3},
    }
    out_path = os.path.join(output_dir, "final_results.json")
    with open(out_path, "w") as f:
        json.dump(final_output, f, indent=2)

    best_path = os.path.join(output_dir, "best_config.json")
    with open(best_path, "w") as f:
        json.dump(best, f, indent=2)

    LOGGER.info("")
    LOGGER.info("=" * 60)
    LOGGER.info("GRID SEARCH COMPLETE (%.1fs)", time.time() - t_total)
    LOGGER.info("=" * 60)
    LOGGER.info("BEST OVERALL:")
    LOGGER.info("  Config: %s", best.get("config_id", "N/A"))
    LOGGER.info("  memory_len=%d, k=%d, gamma=%.2f, beta=%.3f",
               best.get("memory_len", 0), best.get("k", 0),
               best.get("gamma", 0), best.get("beta", 0))
    LOGGER.info("  noise_std=%.5f, lr=%.4f, out_dim=%d, epochs=%d",
               best.get("noise_std", 0), best.get("lr", 0),
               best.get("out_dim", 0), best.get("epochs", 0))
    LOGGER.info("  AUC-ROC=%.4f | AUC-PR=%.4f | F1=%.4f",
               best.get("auc_roc", 0), best.get("auc_pr", 0), best.get("f1", 0))
    LOGGER.info("  Precision=%.4f | Recall=%.4f",
               best.get("precision", 0), best.get("recall", 0))
    LOGGER.info("  Score: Normal=%.4f | Anomaly=%.4f (sep=%.2fx)",
               best.get("score_normal_mean", 0), best.get("score_anomaly_mean", 0),
               best.get("separation_ratio", 0))
    LOGGER.info("  Best threshold: %.4f", best.get("best_threshold", 0))
    LOGGER.info("Results: %s", out_path)
    LOGGER.info("Best config: %s", best_path)


if __name__ == "__main__":
    sys.exit(main())
