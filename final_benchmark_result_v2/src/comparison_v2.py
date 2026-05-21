#!/usr/bin/env python3
"""
Comparison v2: All models evaluated on valid_polluted (same as grid search).

Key changes from comparison.py:
  - Uses valid_polluted.parquet for evaluation (not test_polluted)
  - Uses valid/ground_truth_mask.npy (not test/)
  - MemStream: update_memory=True (streaming mode), 5000 epochs (paper standard)
  - Uses the best config from grid search (M2048_k10_od76_n0.001_e1000_lr10)

This ensures fair comparison: grid search found best config on valid_polluted,
and all methods are evaluated on the same split.
"""
from __future__ import annotations

import os, sys, json, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["PYTHONLEGACYWINDOWSSTDIO"] = "1"
os.environ["ARROW_DISABLE_MMAP"] = "1"

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stderr)
LOGGER = logging.getLogger("comparison_v2")

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.benchmark_core import (
    MemStreamPipeline, extract_features_from_parquet,
    evaluate_scores, find_best_threshold,
    plot_score_distribution, plot_score_timeseries,
    plot_detection_timeseries, plot_training_loss,
    compute_auc_roc, compute_auc_pr,
)

_MAX_OFFLINE_TRAIN = 100_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_batch_size() -> int:
    try:
        import psutil
        available_gb = psutil.virtual_memory().available / (1024 ** 3)
        safe = int(available_gb * 0.5)
        return max(64, min(safe, 2048))
    except Exception:
        return 256


# ---------------------------------------------------------------------------
# IsolationForest
# ---------------------------------------------------------------------------
def run_isolation_forest(X_train, X_val, gt_mask, n_estimators=200, max_samples=256, seed=42):
    from sklearn.ensemble import IsolationForest
    t0 = time.time()
    clf = IsolationForest(n_estimators=n_estimators, max_samples=max_samples,
                          contamination=0.08, random_state=seed, n_jobs=-1)
    clf.fit(X_train)
    scores = -clf.score_samples(X_val)
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    best_t, best_f1 = find_best_threshold(scores, gt_mask)
    metrics = evaluate_scores(scores, gt_mask, threshold=best_t)
    elapsed = time.time() - t0
    LOGGER.info("  [IsolationForest] AUC-PR=%.4f, AUC-ROC=%.4f, F1=%.4f [%.1fs]",
               metrics["auc_pr"], metrics["auc_roc"], metrics["f1"], elapsed)
    return {
        "method": "IsolationForest", "category": "offline",
        "auc_roc": metrics["auc_roc"], "auc_pr": metrics["auc_pr"],
        "f1": metrics["f1"], "precision": metrics["precision"],
        "recall": metrics["recall"], "fpr": metrics["fpr"], "acc": metrics["acc"],
        "best_threshold": float(best_t),
        "score_normal_mean": float(metrics["score_normal_mean"]),
        "score_anomaly_mean": float(metrics["score_anomaly_mean"]),
        "separation_ratio": float(metrics["separation_ratio"]),
        "time_s": elapsed,
        "params": {"n_estimators": n_estimators, "max_samples": max_samples},
        "_raw_scores": scores.tolist(),
    }


# ---------------------------------------------------------------------------
# Normal Autoencoder
# ---------------------------------------------------------------------------
def run_normal_ae(X_train, X_val, gt_mask, hidden_dims=[64, 32], epochs=100,
                  lr=0.001, batch_size=256, seed=42):
    try:
        import torch, torch.nn as nn, torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        return {"method": "Normal Autoencoder", "error": "PyTorch not installed"}

    t0 = time.time()
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = X_train.shape[1]

    n_train = len(X_train)
    if n_train > _MAX_OFFLINE_TRAIN:
        LOGGER.info("    [NormalAE] Subsampling train from %d to %d", n_train, _MAX_OFFLINE_TRAIN)
        idx = np.random.RandomState(seed).choice(n_train, _MAX_OFFLINE_TRAIN, replace=False)
        X_tr_sub = X_train[idx]
    else:
        X_tr_sub = X_train

    layers, in_dim = [], d
    for h_dim in hidden_dims:
        layers.extend([nn.Linear(in_dim, h_dim), nn.ReLU()]); in_dim = h_dim
    encoder = nn.Sequential(*layers)
    rev_dims = list(reversed(hidden_dims))
    decoder_layers = []
    for i, h_dim in enumerate(rev_dims):
        decoder_layers.append(nn.Linear(h_dim, rev_dims[i + 1] if i + 1 < len(rev_dims) else d))
        if i + 1 < len(rev_dims):
            decoder_layers.append(nn.ReLU())
    decoder = nn.Sequential(*decoder_layers)

    class AE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder; self.decoder = decoder
        def forward(self, x):
            return self.decoder(self.encoder(x))

    model = AE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_tr_t = torch.FloatTensor(X_tr_sub).to(device)
    X_va_t = torch.FloatTensor(X_val).to(device)
    train_loader = DataLoader(TensorDataset(X_tr_t), batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            xb = batch[0]
            optimizer.zero_grad()
            loss = criterion(model(xb), xb)
            loss.backward(); optimizer.step()
        if (epoch + 1) % 20 == 0:
            LOGGER.info("    [NormalAE] Epoch %d/%d", epoch + 1, epochs)

    model.eval()
    with torch.no_grad():
        recon = model(X_va_t)
        scores = torch.mean((recon - X_va_t) ** 2, dim=1).cpu().numpy()

    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    best_t, best_f1 = find_best_threshold(scores, gt_mask)
    metrics = evaluate_scores(scores, gt_mask, threshold=best_t)
    elapsed = time.time() - t0
    LOGGER.info("  [NormalAE] AUC-PR=%.4f, AUC-ROC=%.4f, F1=%.4f [%.1fs]",
               metrics["auc_pr"], metrics["auc_roc"], metrics["f1"], elapsed)
    return {
        "method": "Normal Autoencoder", "category": "offline",
        "auc_roc": metrics["auc_roc"], "auc_pr": metrics["auc_pr"],
        "f1": metrics["f1"], "precision": metrics["precision"],
        "recall": metrics["recall"], "fpr": metrics["fpr"], "acc": metrics["acc"],
        "best_threshold": float(best_t),
        "score_normal_mean": float(metrics["score_normal_mean"]),
        "score_anomaly_mean": float(metrics["score_anomaly_mean"]),
        "separation_ratio": float(metrics["separation_ratio"]),
        "time_s": elapsed,
        "params": {"hidden_dims": hidden_dims, "epochs": epochs, "lr": lr},
        "_raw_scores": scores.tolist(),
    }


# ---------------------------------------------------------------------------
# One-Class SVM
# ---------------------------------------------------------------------------
def run_ocsvm(X_train, X_val, gt_mask, kernel="rbf", nu=0.1, gamma="scale"):
    from sklearn.svm import OneClassSVM
    t0 = time.time()
    max_train = min(len(X_train), _MAX_OFFLINE_TRAIN)
    idx = np.random.RandomState(42).choice(len(X_train), max_train, replace=False)
    X_tr_sub = X_train[idx]
    clf = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
    clf.fit(X_tr_sub)
    raw_scores = clf.decision_function(X_val)
    scores = -(raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-8)
    best_t, best_f1 = find_best_threshold(scores, gt_mask)
    metrics = evaluate_scores(scores, gt_mask, threshold=best_t)
    elapsed = time.time() - t0
    LOGGER.info("  [OCSVM] AUC-PR=%.4f, AUC-ROC=%.4f, F1=%.4f [%.1fs]",
               metrics["auc_pr"], metrics["auc_roc"], metrics["f1"], elapsed)
    return {
        "method": "One-Class SVM", "category": "offline",
        "auc_roc": metrics["auc_roc"], "auc_pr": metrics["auc_pr"],
        "f1": metrics["f1"], "precision": metrics["precision"],
        "recall": metrics["recall"], "fpr": metrics["fpr"], "acc": metrics["acc"],
        "best_threshold": float(best_t),
        "score_normal_mean": float(metrics["score_normal_mean"]),
        "score_anomaly_mean": float(metrics["score_anomaly_mean"]),
        "separation_ratio": float(metrics["separation_ratio"]),
        "time_s": elapsed,
        "params": {"kernel": kernel, "nu": nu, "gamma": gamma},
        "_raw_scores": scores.tolist(),
    }


# ---------------------------------------------------------------------------
# MemStream v5 (streaming mode, 5000 epochs)
# ---------------------------------------------------------------------------
def run_memstream(X_train, hours_train, dows_train, rcs_train, nb_train,
                  X_val, hours_val, dows_val, rcs_val, nb_val,
                  gt_mask, best_config, epochs=5000, update_memory=True,
                  seed=42, batch_size=None):
    t0 = time.time()
    _bs = _safe_batch_size() if batch_size is None else batch_size
    try:
        pipeline = MemStreamPipeline(
            d=best_config.get("d", 38),
            out_dim=best_config.get("out_dim", 76),
            memory_len=best_config["memory_len"],
            k=best_config["k"],
            gamma=best_config.get("gamma", 0.0),
            beta=best_config.get("beta", 0.5),
            noise_std=best_config.get("noise_std", 0.001),
            lr=best_config.get("lr", 0.01),
            epochs=epochs,
            batch_size=_bs,
            seed=seed,
            cb_warmup=min(4096, best_config["memory_len"] * 4),
            verbose=False,
            adam_betas=tuple(best_config.get("adam_betas", [0.9, 0.999])),
        )
        pipeline.train(
            X_train, hours_train, dows_train, rcs_train, nb_train,
            X_warmup=X_train[:best_config["memory_len"]],
            hours_warmup=hours_train[:best_config["memory_len"]],
            dows_warmup=dows_train[:best_config["memory_len"]],
            rcs_warmup=rcs_train[:best_config["memory_len"]],
            nb_warmup=nb_train[:best_config["memory_len"]],
        )

        adj_scores, ms_metrics = pipeline.score_stream(
            X_val, hours_val, dows_val, rcs_val, nb_val,
            gt_mask=gt_mask,
            update_memory=update_memory,
        )
        best_t, best_f1 = find_best_threshold(adj_scores, gt_mask)
        metrics = evaluate_scores(adj_scores, gt_mask, threshold=best_t)
        elapsed = time.time() - t0

        LOGGER.info("  [MemStream v5] AUC-PR=%.4f, AUC-ROC=%.4f, F1=%.4f [%.1fs]",
                   metrics["auc_pr"], metrics["auc_roc"], metrics["f1"], elapsed)
        LOGGER.info("    Separation: normal=%.2f, anomaly=%.2f (%.2fx)",
                   metrics["score_normal_mean"], metrics["score_anomaly_mean"],
                   metrics["separation_ratio"])
        LOGGER.info("    Memory updates: %d", ms_metrics.get("n_memory_updates", 0))

        detected_mask = (adj_scores >= best_t).astype(np.int32)
        return {
            "method": "MemStream v5", "category": "streaming",
            "auc_roc": metrics["auc_roc"], "auc_pr": metrics["auc_pr"],
            "f1": metrics["f1"], "precision": metrics["precision"],
            "recall": metrics["recall"], "fpr": metrics["fpr"], "acc": metrics["acc"],
            "best_threshold": float(best_t),
            "score_normal_mean": float(metrics["score_normal_mean"]),
            "score_anomaly_mean": float(metrics["score_anomaly_mean"]),
            "separation_ratio": float(metrics["separation_ratio"]),
            "time_s": elapsed,
            "n_memory_updates": ms_metrics.get("n_memory_updates", 0),
            "params": {
                "memory_len": best_config["memory_len"],
                "k": best_config["k"],
                "gamma": best_config.get("gamma", 0.0),
                "beta": best_config.get("beta", 0.5),
                "noise_std": best_config.get("noise_std", 0.001),
                "lr": best_config.get("lr", 0.01),
                "out_dim": best_config.get("out_dim", 76),
                "epochs": epochs,
                "update_memory": update_memory,
                "batch_size": _bs,
            },
            "_raw_scores": adj_scores.tolist(),
            "_detected_mask": detected_mask.tolist(),
            "_gt_mask": gt_mask.tolist(),
            "_epoch_losses": [float(x) for x in pipeline.epoch_losses],
        }
    except Exception as e:
        LOGGER.error("  [MemStream v5] Error: %s", e)
        import traceback; traceback.print_exc()
        return {"method": "MemStream v5", "error": str(e), "_raw_scores": []}


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def _plot_comparison_bars(all_results, output_dir):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    methods = [k for k, v in all_results.items() if "error" not in v]
    if not methods:
        return
    methods.sort(key=lambda m: all_results[m].get("auc_pr", 0), reverse=True)
    metrics = ["auc_pr", "auc_roc", "f1", "precision", "recall"]
    labels = ["AUC-PR", "AUC-ROC", "F1", "Precision", "Recall"]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]
    n_m = len(methods)
    bar_width = 0.15
    x = np.arange(n_m)
    fig, ax = plt.subplots(figsize=(max(10, n_m * 1.8), 7))
    for i, (m, lbl, c) in enumerate(zip(metrics, labels, colors)):
        vals = [all_results[m_].get(m, 0) for m_ in methods]
        bars = ax.bar(x + i * bar_width, vals, bar_width, label=lbl, color=c, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)
    method_labels = [f"{all_results[m_].get('method', m_)}\n({all_results[m_].get('category','?')[:3]})" for m_ in methods]
    ax.set_xticks(x + bar_width * 2)
    ax.set_xticklabels(method_labels, fontsize=9)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Anomaly Detection Methods — Comparison v2 (valid_polluted)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "v2_comparison_bars.png"), dpi=150, bbox_inches="tight")
    LOGGER.info("  v2_comparison_bars.png saved")
    plt.close(fig)


def _plot_runtime_bar(all_results, output_dir):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except Exception:
        return
    methods = [k for k, v in all_results.items() if "error" not in v]
    if not methods:
        return
    methods.sort(key=lambda m: all_results[m].get("auc_pr", 0), reverse=True)
    names = [all_results[m].get("method", m) for m in methods]
    times = [all_results[m].get("time_s", 0) for m in methods]
    cat_colors = {"streaming": "#2196F3", "offline": "#FF9800"}
    colors = [cat_colors.get(all_results[m].get("category", ""), "#9E9E9E") for m in methods]
    fig, ax = plt.subplots(figsize=(max(8, len(methods) * 1.5), 6))
    bars = ax.bar(names, times, color=colors, alpha=0.85, edgecolor="white", linewidth=1)
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(times) * 0.01,
                f"{t:.1f}s", ha="center", va="bottom", fontsize=9)
    ax.legend(handles=[Patch(facecolor="#2196F3", label="Streaming"),
                        Patch(facecolor="#FF9800", label="Offline")], fontsize=10)
    ax.set_ylabel("Runtime (seconds)", fontsize=12)
    ax.set_title("Runtime Comparison v2", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "v2_runtime.png"), dpi=150, bbox_inches="tight")
    LOGGER.info("  v2_runtime.png saved")
    plt.close(fig)


def _generate_memstream_viz(pipeline, X_val, hours_val, dows_val, rcs_val, nb_val,
                            gt_mask, best_config, adj_scores, output_dir, epochs):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    best_t, _ = find_best_threshold(adj_scores, gt_mask)
    ms_metrics = evaluate_scores(adj_scores, gt_mask, best_t)

    # Loss curve
    if pipeline.epoch_losses:
        plot_training_loss(
            pipeline.epoch_losses,
            save_path=os.path.join(output_dir, "v2_memstream_loss.png"),
            title=f"MemStream v5 — AE Training Loss | mem={best_config['memory_len']}, k={best_config['k']}, epochs={epochs}",
        )

    # Score distribution
    plot_score_distribution(
        adj_scores, gt_mask, best_t,
        save_path=os.path.join(output_dir, "v2_score_dist.png"),
        title=f"MemStream v5 — Score Distribution | AUC-PR={ms_metrics['auc_pr']:.4f}",
    )

    # Timeseries
    plot_score_timeseries(
        adj_scores, gt_mask, best_t,
        save_path=os.path.join(output_dir, "v2_timeseries.png"),
        title=f"MemStream v5 — Anomaly Scores | AUC-PR={ms_metrics['auc_pr']:.4f}, AUC-ROC={ms_metrics['auc_roc']:.4f}",
        max_display=50000,
    )

    # Detection timeline
    plot_detection_timeseries(
        adj_scores, gt_mask, best_t,
        save_path=os.path.join(output_dir, "v2_detection_timeline.png"),
        title=f"MemStream v5 — Detection Timeline | TP={ms_metrics['tp']}, FP={ms_metrics['fp']}, FN={ms_metrics['fn']}",
        max_display=30000,
    )
    LOGGER.info("  MemStream v2 visualizations saved")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=str(Path(__file__).parent.parent))
    parser.add_argument("--epochs", type=int, default=5000, help="MemStream epochs (paper: 5000)")
    parser.add_argument("--methods", default="all", help="Comma-separated: IF,MemStream,NormalAE,OCSVM,all")
    parser.add_argument("--update-memory", action="store_true", default=True,
                       help="Enable streaming memory update for MemStream (default: True)")
    parser.add_argument("--no-update-memory", dest="update_memory", action="store_false",
                       help="Disable streaming memory update for MemStream")
    parser.add_argument("--max-train", type=int, default=500000)
    parser.add_argument("--max-val", type=int, default=500000)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    base = Path(args.base)
    data_dir = base / "data"
    output_dir = str(base / "results" / "comparison_v2")
    os.makedirs(output_dir, exist_ok=True)

    # Load best config
    grid_path = base / "results" / "grid_search" / "final_results.json"
    if grid_path.exists():
        with open(grid_path) as f:
            grid_data = json.load(f)
        best_config = grid_data.get("best_config", {})
        LOGGER.info("Loaded best config: %s", best_config.get("config_id", "N/A"))
    else:
        LOGGER.warning("No grid results found, using defaults")
        best_config = {
            "memory_len": 2048, "k": 10, "gamma": 0.0, "beta": 0.001,
            "noise_std": 0.001, "lr": 0.01, "out_dim": 76, "epochs": 5000,
            "adam_betas": [0.9, 0.999],
        }

    # Load data — USE valid_polluted (consistent with grid search)
    train_path = str(data_dir / "train_clean.parquet")
    valid_path = str(data_dir / "valid_polluted.parquet")

    LOGGER.info("Loading training data (train_clean.parquet)...")
    X_train, hours_train, dows_train, rcs_train, nb_train = extract_features_from_parquet(
        train_path, max_rows=args.max_train if args.max_train > 0 else None
    )
    LOGGER.info("Loading validation data (valid_polluted.parquet)...")
    X_val, hours_val, dows_val, rcs_val, nb_val = extract_features_from_parquet(
        valid_path, max_rows=args.max_val if args.max_val > 0 else None
    )
    gt_mask = np.load(str(data_dir / "valid" / "ground_truth_mask.npy"))
    gt_mask = gt_mask[-len(X_val):]
    LOGGER.info("Train: %d rows, Val: %d rows (GT: %d anomalies, %.2f%%)",
               len(X_train), len(X_val), gt_mask.sum(), gt_mask.mean() * 100)

    # Methods
    methods_arg = args.methods
    if methods_arg.lower() == "all":
        run_methods = ["IF", "MemStream", "NormalAE", "OCSVM"]
    else:
        run_methods = [m.strip() for m in methods_arg.split(",")]

    LOGGER.info("")
    LOGGER.info("=" * 60)
    LOGGER.info("COMPARISON v2 — eval on valid_polluted")
    LOGGER.info("=" * 60)
    LOGGER.info("Methods: %s", run_methods)
    LOGGER.info("MemStream: mem=%d, k=%d, gamma=%.2f, beta=%.4f, out_dim=%d, epochs=%d",
               best_config["memory_len"], best_config["k"],
               best_config.get("gamma", 0), best_config.get("beta", 0.5),
               best_config.get("out_dim", 76), args.epochs)
    LOGGER.info("MemStream update_memory=%s", args.update_memory)
    LOGGER.info("=" * 60)

    all_results = {}
    pipeline_ref = {"pipeline": None}

    for method in run_methods:
        LOGGER.info("")
        LOGGER.info("=== Running %s ===", method)
        seed = 42
        try:
            if method == "MemStream":
                result = run_memstream(
                    X_train, hours_train, dows_train, rcs_train, nb_train,
                    X_val, hours_val, dows_val, rcs_val, nb_val,
                    gt_mask, best_config, epochs=args.epochs,
                    update_memory=args.update_memory, seed=seed, batch_size=args.batch_size,
                )
            elif method == "IF":
                result = run_isolation_forest(X_train, X_val, gt_mask, seed=seed)
            elif method == "NormalAE":
                result = run_normal_ae(X_train, X_val, gt_mask, seed=seed)
            elif method == "OCSVM":
                result = run_ocsvm(X_train, X_val, gt_mask)
            else:
                LOGGER.warning("Unknown method: %s", method)
                continue
            if "error" not in result:
                all_results[method] = result
            else:
                all_results[method] = {"method": method, "error": result.get("error", "unknown")}
        except Exception as e:
            LOGGER.error("  %s failed: %s", method, e)
            all_results[method] = {"method": method, "error": str(e)}

    # Summary
    LOGGER.info("")
    LOGGER.info("=" * 90)
    LOGGER.info("COMPARISON v2 RESULTS (sorted by AUC-PR)")
    LOGGER.info("=" * 90)
    sortable = [(k, v.get("auc_pr", 0)) for k, v in all_results.items() if "error" not in v]
    sortable.sort(key=lambda x: x[1], reverse=True)
    LOGGER.info(f"{'Rank':<5} {'Method':<25} {'Category':<10} {'AUC-PR':>8} {'AUC-ROC':>8} {'F1':>8} {'Prec':>8} {'Rec':>8} {'Time':>10}")
    LOGGER.info("-" * 90)
    for rank, (method, _) in enumerate(sortable, 1):
        r = all_results[method]
        LOGGER.info(f"{rank:<5} {r.get('method',method):<25} {r.get('category',''):<10} "
                   f"{r['auc_pr']:>8.4f} {r['auc_roc']:>8.4f} {r['f1']:>8.4f} "
                   f"{r['precision']:>8.4f} {r['recall']:>8.4f} {r['time_s']:>9.1f}s")

    # Save
    clean_results = {}
    for k, v in all_results.items():
        clean = {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
        clean_results[k] = clean

    output = {
        "results": clean_results,
        "ranking": [k for k, _ in sortable],
        "best_config": best_config,
        "methods_run": list(all_results.keys()),
        "epochs": args.epochs,
        "update_memory": args.update_memory,
        "eval_dataset": "valid_polluted",
        "train_dataset": "train_clean",
    }
    out_path = os.path.join(output_dir, "comparison_v2_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    LOGGER.info("")
    LOGGER.info("Results saved: %s", out_path)

    # Charts
    LOGGER.info("Generating charts...")
    _plot_comparison_bars(all_results, output_dir)
    _plot_runtime_bar(all_results, output_dir)

    # MemStream viz
    if "MemStream" in all_results and "error" not in all_results.get("MemStream", {}):
        try:
            pipeline = MemStreamPipeline(
                d=best_config.get("d", 38),
                out_dim=best_config.get("out_dim", 76),
                memory_len=best_config["memory_len"],
                k=best_config["k"],
                gamma=best_config.get("gamma", 0.0),
                beta=best_config.get("beta", 0.5),
                noise_std=best_config.get("noise_std", 0.001),
                lr=best_config.get("lr", 0.01),
                epochs=args.epochs,
                batch_size=args.batch_size or _safe_batch_size(),
                seed=42,
                cb_warmup=min(4096, best_config["memory_len"] * 4),
                verbose=False,
                adam_betas=tuple(best_config.get("adam_betas", [0.9, 0.999])),
            )
            pipeline.train(
                X_train, hours_train, dows_train, rcs_train, nb_train,
                X_warmup=X_train[:best_config["memory_len"]],
                hours_warmup=hours_train[:best_config["memory_len"]],
                dows_warmup=dows_train[:best_config["memory_len"]],
                rcs_warmup=rcs_train[:best_config["memory_len"]],
                nb_warmup=nb_train[:best_config["memory_len"]],
            )
            adj_scores, _ = pipeline.score_stream(
                X_val, hours_val, dows_val, rcs_val, nb_val,
                gt_mask=gt_mask, update_memory=args.update_memory,
            )
            _generate_memstream_viz(
                pipeline, X_val, hours_val, dows_val, rcs_val, nb_val,
                gt_mask, best_config, adj_scores, output_dir, args.epochs,
            )
        except Exception as e:
            LOGGER.warning("MemStream viz failed: %s", e)

    LOGGER.info("All done. Charts in: %s", output_dir)


if __name__ == "__main__":
    main()
