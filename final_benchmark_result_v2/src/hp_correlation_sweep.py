#!/usr/bin/env python3
"""
Part 4: HP Correlation Sweeps.
Runs 4 sweeps on top of the best config from Part 1 grid search:
  4A: memory_len  ∈ [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
  4B: out_dim     ∈ [19, 38, 76, 152]
  4C: beta        ∈ [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
  4D: gamma       ∈ [0, 0.25, 0.5, 1.0]

Each sweep saves results JSON, a chart, and a log file.
"""
from __future__ import annotations

import os, sys, json, time, argparse, logging
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTHONLEGACYWINDOWSSTDIO"] = "1"
os.environ["ARROW_DISABLE_MMAP"] = "1"

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent))

from src._config import (
    TRAIN_PATH, VALID_PATH, GT_MASK_PATH,
    RESULTS_DIR,
)
from src.benchmark_core import (
    extract_features_from_parquet, MemStreamPipeline,
)

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
LOGGER = logging.getLogger("hp_sweep")


# ---------------------------------------------------------------------------
def find_best_threshold(scores, gt_mask, n_steps=200):
    best_f1, best_t = 0.0, float(scores.mean())
    for t in np.linspace(scores.min(), scores.max(), n_steps):
        pred = scores >= t
        tp = (pred & gt_mask).sum()
        fp = (pred & ~gt_mask).sum()
        fn = (~pred & gt_mask).sum()
        prec = tp / (tp + fp + 1e-10)
        rec = tp / (tp + fn + 1e-10)
        f1 = 2 * prec * rec / (prec + rec + 1e-10)
        if f1 > best_f1:
            best_f1, best_t = float(t), float(f1)
    return best_t, best_f1


def evaluate_scores(scores, gt_mask, threshold):
    from sklearn.metrics import roc_auc_score, average_precision_score
    pred = scores >= threshold
    tp = int((pred & gt_mask).sum())
    fp = int((pred & ~gt_mask).sum())
    tn = int((~pred & ~gt_mask).sum())
    fn = int((~pred & gt_mask).sum())
    prec = tp / (tp + fp + 1e-10)
    rec = tp / (tp + fn + 1e-10)
    return {
        "auc_roc": float(roc_auc_score(gt_mask, scores)),
        "auc_pr": float(average_precision_score(gt_mask, scores)),
        "f1": float(2 * prec * rec / (prec + rec + 1e-10)),
        "precision": float(prec), "recall": float(rec),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "score_normal_mean": float(scores[~gt_mask].mean()),
        "score_anomaly_mean": float(scores[gt_mask].mean()),
        "separation_ratio": float(scores[gt_mask].mean() / (scores[~gt_mask].mean() + 1e-10)),
        "fpr": float(fp / (fp + tn + 1e-10)),
        "acc": float((tp + tn) / len(gt_mask)),
    }


def run_config(X_train, hours_train, dows_train, rcs_train, nb_train,
               X_val, hours_val, dows_val, rcs_val, nb_val, gt_mask,
               memory_len, k, gamma, beta, noise_std, lr, out_dim, epochs,
               seed=42):
    pipeline = MemStreamPipeline(
        d=38, out_dim=out_dim,
        memory_len=memory_len, k=k, gamma=gamma, beta=beta,
        noise_std=noise_std, lr=lr, epochs=epochs,
        batch_size=1024, seed=seed,
        cb_warmup=min(4096, memory_len * 4), verbose=False,
    )
    pipeline.train(
        X_train, hours_train, dows_train, rcs_train, nb_train,
        X_warmup=X_train[:memory_len],
        hours_warmup=hours_train[:memory_len],
        dows_warmup=dows_train[:memory_len],
        rcs_warmup=rcs_train[:memory_len],
        nb_warmup=nb_train[:memory_len],
    )
    adj_scores, _ = pipeline.score_stream(
        X_val, hours_val, dows_val, rcs_val, nb_val,
        gt_mask=gt_mask, update_memory=True,
    )
    best_t, best_f1 = find_best_threshold(adj_scores, gt_mask)
    metrics = evaluate_scores(adj_scores, gt_mask, best_t)
    return metrics, best_t, adj_scores


# ---------------------------------------------------------------------------
def load_best_config():
    path = RESULTS_DIR / "grid_search" / "best_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Best config not found: {path}")
    with open(path) as f:
        cfg = json.load(f)
    LOGGER.info("Loaded best config: %s", cfg.get("config_id"))
    return cfg


def run_sweep(name, param_name, param_values, base_config, X_train, hours_train,
              dows_train, rcs_train, nb_train, X_val, hours_val, dows_val,
              rcs_val, nb_val, gt_mask, out_dir):
    LOGGER.info("=" * 60)
    LOGGER.info("SWEEP %s: %s ∈ %s", name, param_name, param_values)
    LOGGER.info("=" * 60)

    results = []
    for val in param_values:
        cfg = dict(base_config)
        cfg[param_name] = val
        cfg_id = f"{name}_{param_name}{val}"
        result_path = out_dir / f"{cfg_id}.json"

        LOGGER.info("[%s] %s", name, cfg_id)
        t0 = time.time()
        try:
            metrics, best_t, scores = run_config(
                X_train, hours_train, dows_train, rcs_train, nb_train,
                X_val, hours_val, dows_val, rcs_val, nb_val, gt_mask,
                memory_len=cfg["memory_len"],
                k=cfg["k"],
                gamma=cfg["gamma"],
                beta=cfg["beta"],
                noise_std=cfg["noise_std"],
                lr=cfg["lr"],
                out_dim=cfg["out_dim"],
                epochs=cfg["epochs"],
            )
            elapsed = time.time() - t0
            result = {
                "sweep": name, "param": param_name, "value": val,
                "config_id": cfg_id,
                "auc_roc": metrics["auc_roc"],
                "auc_pr": metrics["auc_pr"],
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "fpr": metrics["fpr"],
                "acc": metrics["acc"],
                "best_threshold": best_t,
                "score_normal_mean": metrics["score_normal_mean"],
                "score_anomaly_mean": metrics["score_anomaly_mean"],
                "separation_ratio": metrics["separation_ratio"],
                "train_time_s": elapsed,
                "status": "MATCHED",
            }
            np.save(out_dir / f"{cfg_id}_scores.npy", scores)
        except Exception as e:
            elapsed = time.time() - t0
            LOGGER.error("  ERROR %s: %s", cfg_id, e)
            result = {
                "sweep": name, "param": param_name, "value": val,
                "config_id": cfg_id,
                "auc_pr": 0.0, "auc_roc": 0.0, "f1": 0.0,
                "precision": 0.0, "recall": 0.0, "train_time_s": elapsed,
                "status": "NOMATCH",
                "error": str(e),
            }

        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        results.append(result)
        LOGGER.info("  -> AUC-PR=%.4f, AUC-ROC=%.4f, F1=%.4f, sep=%.2fx [%.1fs]",
                   result["auc_pr"], result["auc_roc"], result["f1"],
                   result["separation_ratio"], result["train_time_s"])

    # Save sweep summary
    sweep_path = out_dir / f"{name}_results.json"
    with open(sweep_path, "w") as f:
        json.dump({"sweep": name, "param": param_name, "results": results}, f, indent=2)

    # Generate chart
    try:
        _plot_sweep(name, param_name, param_values, results, out_dir)
    except Exception as e:
        LOGGER.warning("Could not generate chart for %s: %s", name, e)

    return results


def _plot_sweep(name, param_name, param_values, results, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    matched = [r for r in results if r["status"] == "MATCHED"]
    if not matched:
        LOGGER.warning("No matched results to plot for %s", name)
        return

    vals = [r["value"] for r in matched]
    auc_pr = [r["auc_pr"] for r in matched]
    auc_roc = [r["auc_roc"] for r in matched]
    f1s = [r["f1"] for r in matched]

    is_line = len(vals) > 4
    x = range(len(vals))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"HP Sweep: {name} — {param_name}", fontsize=13, fontweight="bold")

    for ax, metric, label, color in zip(
        axes,
        [auc_pr, auc_roc, f1s],
        ["AUC-PR", "AUC-ROC", "F1"],
        ["#2196F3", "#4CAF50", "#FF9800"],
    ):
        ax.plot(x, metric, marker="o", color=color, linewidth=2, markersize=6)
        if is_line:
            ax.fill_between(x, metric, alpha=0.15, color=color)
        else:
            ax.bar(x, metric, color=color, alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in vals], rotation=45, ha="right")
        ax.set_xlabel(param_name)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        best_idx = int(np.argmax(metric))
        ax.axhline(metric[best_idx], color="red", linestyle="--", alpha=0.5, linewidth=0.8)

    plt.tight_layout()
    out_path = out_dir / f"{name}_sweep.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    LOGGER.info("  Chart saved: %s", out_path)


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="HP Correlation Sweeps — Part 4")
    parser.add_argument("--max-train", type=int, default=500000)
    parser.add_argument("--max-val", type=int, default=500000)
    parser.add_argument("--sweep", choices=["all", "4a", "4b", "4c", "4d"], default="all")
    args = parser.parse_args()

    out_dir = RESULTS_DIR / "hp_correlation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    LOGGER.info("Loading data...")
    X_train, hours_train, dows_train, rcs_train, nb_train = extract_features_from_parquet(
        str(TRAIN_PATH), max_rows=args.max_train
    )
    X_val, hours_val, dows_val, rcs_val, nb_val = extract_features_from_parquet(
        str(VALID_PATH), max_rows=args.max_val
    )
    gt_mask = np.load(str(GT_MASK_PATH))
    gt_mask = gt_mask[-len(X_val):]
    LOGGER.info("Train: %d, Val: %d (GT: %d anomalies, %.2f%%)",
               len(X_train), len(X_val), gt_mask.sum(), gt_mask.mean() * 100)

    # Load base config
    base = load_best_config()

    # Sweep configs
    sweeps = {}

    if args.sweep in ("all", "4a"):
        sweeps["4a_memory"] = {
            "param": "memory_len",
            "values": [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
        }

    if args.sweep in ("all", "4b"):
        sweeps["4b_outdim"] = {
            "param": "out_dim",
            "values": [19, 38, 76, 152],
        }

    if args.sweep in ("all", "4c"):
        sweeps["4c_beta"] = {
            "param": "beta",
            "values": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0],
        }

    if args.sweep in ("all", "4d"):
        sweeps["4d_gamma"] = {
            "param": "gamma",
            "values": [0, 0.25, 0.5, 1.0],
        }

    for sweep_name, cfg in sweeps.items():
        run_sweep(
            name=sweep_name,
            param_name=cfg["param"],
            param_values=cfg["values"],
            base_config=base,
            X_train=X_train, hours_train=hours_train,
            dows_train=dows_train, rcs_train=rcs_train, nb_train=nb_train,
            X_val=X_val, hours_val=hours_val,
            dows_val=dows_val, rcs_val=rcs_val, nb_val=nb_val,
            gt_mask=gt_mask,
            out_dir=out_dir,
        )

    LOGGER.info("")
    LOGGER.info("=" * 60)
    LOGGER.info("ALL SWEEPS COMPLETE")
    LOGGER.info("Results: %s", out_dir)
    LOGGER.info("=" * 60)


if __name__ == "__main__":
    main()
