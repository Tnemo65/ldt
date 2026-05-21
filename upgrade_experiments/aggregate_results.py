#!/usr/bin/env python3
"""
Aggregate all experiment results and generate charts + report.

Usage:
    python upgrade_experiments/aggregate_results.py

Run after all experiments have completed.
"""
from __future__ import annotations

import os, sys, json, glob
import numpy as np

OUT_DIR = r"c:\proj\ldt\upgrade_experiments"
CHART_DIR = r"c:\proj\ldt\upgrade_experiments\charts"
os.makedirs(CHART_DIR, exist_ok=True)

PYTHON = r"C:\Users\Administrator\Desktop\AI ComfyUI\system\python\python.exe"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def best_of_dir(glob_pattern, key="auc_pr"):
    files = glob.glob(glob_pattern)
    if not files:
        return None, None
    results = []
    for f in files:
        try:
            r = load_json(f)
            if "auc_pr" in r:
                results.append(r)
            elif "results" in r and isinstance(r["results"], list):
                for item in r["results"]:
                    if "auc_pr" in item:
                        results.append(item)
        except Exception:
            pass
    if not results:
        return None, None
    results.sort(key=lambda x: x.get(key, 0), reverse=True)
    return results[0], results


def main():
    print("=" * 80)
    print("UPGRADE EXPERIMENTS — Results Aggregation")
    print("=" * 80)

    all_data = {}

    # ---- Grid Search Stage 2 ----
    best_s2, all_s2 = best_of_dir(os.path.join(OUT_DIR, "grid_search_s2", "*.json"))
    if best_s2:
        print(f"\nStage 2 (noise/lr/epochs): Best = {best_s2.get('config_id')}, AUC-PR = {best_s2.get('auc_pr', 0):.4f}")
        all_data["stage2"] = {"best": best_s2, "all": all_s2[:20]}
    else:
        print("\nStage 2: No results yet")
        all_data["stage2"] = {"best": None, "all": []}

    # ---- Grid Search Stage 3 ----
    best_s3, all_s3 = best_of_dir(os.path.join(OUT_DIR, "grid_search_s3", "*.json"))
    if best_s3:
        print(f"\nStage 3 (out_dim/memory/k): Best = {best_s3.get('config_id')}, AUC-PR = {best_s3.get('auc_pr', 0):.4f}")
        all_data["stage3"] = {"best": best_s3, "all": all_s3[:20]}
    else:
        print("\nStage 3: No results yet")
        all_data["stage3"] = {"best": None, "all": []}

    # ---- Context Stratified ----
    ctx_path = os.path.join(OUT_DIR, "context_stratified", "context_ablation_results.json")
    if os.path.exists(ctx_path):
        ctx = load_json(ctx_path)
        summary = ctx.get("summary", {})
        cb_pr = summary.get("cb_global_auc_pr", 0)
        gb_pr = summary.get("gb_global_auc_pr", 0)
        print(f"\nContextBeta Ablation: CB={cb_pr:.4f}, GB={gb_pr:.4f}, diff={cb_pr-gb_pr:+.4f}")
        all_data["context"] = ctx
    else:
        print("\nContextBeta Ablation: No results yet")
        all_data["context"] = None

    # ---- Concept Drift ----
    cd_path = os.path.join(OUT_DIR, "concept_drift", "concept_drift_results.json")
    if os.path.exists(cd_path):
        cd = load_json(cd_path)
        print(f"\nConcept Drift:")
        for dt, dr in cd.get("drift_results", {}).items():
            ms_s = dr["memstream_streaming"]["auc_pr"]
            ms_st = dr["memstream_static"]["auc_pr"]
            iff = dr["isolation_forest"]["auc_pr"]
            adv = ms_s - ms_st
            print(f"  {dr['drift_name']}: Stream={ms_s:.4f}, Static={ms_st:.4f}, IF={iff:.4f}, Adv={adv:+.4f}")
        all_data["concept_drift"] = cd
    else:
        print("\nConcept Drift: No results yet")
        all_data["concept_drift"] = None

    # ---- Full Data ----
    fd_path = os.path.join(OUT_DIR, "full_data", "full_data_results.json")
    if os.path.exists(fd_path):
        fd = load_json(fd_path)
        print(f"\nFull Data Training:")
        for key, r in sorted(fd.get("results", {}).items()):
            print(f"  {key}: AUC-PR={r.get('auc_pr',0):.4f}, F1={r.get('f1',0):.4f}")
        all_data["full_data"] = fd
    else:
        print("\nFull Data: No results yet")
        all_data["full_data"] = None

    # ---- Comparison v2 (existing) ----
    cmp_path = r"c:\proj\ldt\HP_benchmark_v5\results\comparison\comparison_results.json"
    if os.path.exists(cmp_path):
        cmp = load_json(cmp_path)
        results_dict = cmp.get("results", {})
        print(f"\nComparison v2 ({len(results_dict)} methods):")
        sortable = [(k, v) for k, v in results_dict.items() if "auc_pr_mean" in v]
        sortable.sort(key=lambda x: x[1].get("auc_pr_mean", 0), reverse=True)
        for k, v in sortable:
            pr = v.get("auc_pr_mean", 0)
            f1 = v.get("f1_mean", 0)
            cat = v.get("category", "?")
            print(f"  {v.get('method','?')}: AUC-PR={pr:.4f}, F1={f1:.4f} [{cat}]")
        all_data["comparison"] = cmp
    else:
        print("\nComparison v2: Not found")
        all_data["comparison"] = None

    # ---- Overall Best ----
    overall_best = None
    best_pr = 0
    for stage_name, stage_data in [("S2", all_data.get("stage2")),
                                     ("S3", all_data.get("stage3"))]:
        if stage_data and stage_data.get("best"):
            pr = stage_data["best"].get("auc_pr", 0)
            if pr > best_pr:
                best_pr = pr
                overall_best = (stage_name, stage_data["best"])

    print("")
    print("=" * 80)
    print("OVERALL BEST CONFIG:")
    if overall_best:
        stage, cfg = overall_best
        print(f"  From: {stage}")
        print(f"  Config: {cfg.get('config_id')}")
        print(f"  memory_len={cfg.get('memory_len')}, k={cfg.get('k')}, "
              f"noise={cfg.get('noise_std')}, lr={cfg.get('lr')}, "
              f"out_dim={cfg.get('out_dim')}, epochs={cfg.get('epochs')}")
        print(f"  AUC-PR={cfg.get('auc_pr',0):.4f}, AUC-ROC={cfg.get('auc_roc',0):.4f}, "
              f"F1={cfg.get('f1',0):.4f}, Recall={cfg.get('recall',0):.4f}")
    else:
        print("  No results yet")

    # ---- Save aggregated JSON ----
    agg_path = os.path.join(OUT_DIR, "aggregated_results.json")
    with open(agg_path, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"\nAggregated results saved to: {agg_path}")

    # ---- Generate charts ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except Exception as e:
        print(f"\nCharts skipped (matplotlib unavailable): {e}")
        return

    plt.rcParams.update({"font.size": 10, "font.family": "sans-serif"})

    # Chart 1: Grid search results heatmap (S2 noise vs epochs vs lr)
    if all_s2 and len(all_s2) >= 4:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Stage 2: Fine-Tuning — Effect of noise_std, epochs, lr on AUC-PR",
                     fontsize=14, fontweight="bold")

        noise_vals = sorted(set(r.get("noise_std", 0) for r in all_s2 if "error" not in r))
        epoch_vals = sorted(set(r.get("epochs", 0) for r in all_s2 if "error" not in r))
        lr_vals = sorted(set(r.get("lr", 0) for r in all_s2 if "error" not in r))

        for ax, param, param_vals, param_name in zip(
            axes,
            ["noise_std", "epochs", "lr"],
            [noise_vals, epoch_vals, lr_vals],
            ["noise_std", "epochs", "lr"]
        ):
            # Group by that param and show all others as different lines
            for val in param_vals:
                subset = [r for r in all_s2
                         if r.get(param, 0) == val and "error" not in r]
                if not subset:
                    continue
                # Average across other params
                avg_pr = np.mean([r.get("auc_pr", 0) for r in subset])
                label = f"{param_name}={val}"
                ax.bar([label], [avg_pr], alpha=0.7)

            ax.set_ylabel("AUC-PR")
            ax.set_title(f"Grouped by {param_name}")
            ax.axhline(y=0.9, color="red", linestyle="--", alpha=0.4)
            ax.set_ylim(0, 1.05)

        plt.tight_layout()
        path = os.path.join(CHART_DIR, "stage2_heatmaps.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Chart: {path}")

    # Chart 2: Stage 3 architecture variation
    if all_s3 and len(all_s3) >= 4:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Stage 3: Architecture Variation — out_dim, memory_len, k",
                     fontsize=14, fontweight="bold")

        od_vals = sorted(set(r.get("out_dim", 0) for r in all_s3 if "error" not in r))
        ml_vals = sorted(set(r.get("memory_len", 0) for r in all_s3 if "error" not in r))
        k_vals = sorted(set(r.get("k", 0) for r in all_s3 if "error" not in r))

        for ax, param, param_vals, param_name in zip(
            axes,
            ["out_dim", "memory_len", "k"],
            [od_vals, ml_vals, k_vals],
            ["out_dim (bottleneck dim)", "memory_len", "k (neighbors)"]
        ):
            means, stds, labels = [], [], []
            for val in param_vals:
                subset = [r for r in all_s3 if r.get(param, 0) == val and "error" not in r]
                if subset:
                    vals = [r.get("auc_pr", 0) for r in subset]
                    means.append(np.mean(vals))
                    stds.append(np.std(vals) if len(vals) > 1 else 0)
                    labels.append(str(val))
            bars = ax.bar(labels, means, yerr=stds, alpha=0.75, capsize=4,
                         color=plt.cm.viridis(np.linspace(0.2, 0.8, len(labels))))
            for bar, v in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                        f"{v:.4f}", ha="center", va="bottom", fontsize=9)
            ax.set_ylabel("AUC-PR")
            ax.set_title(f"AUC-PR vs {param_name}")
            ax.axhline(y=0.9, color="red", linestyle="--", alpha=0.4)
            ax.set_ylim(0, 1.05)

        plt.tight_layout()
        path = os.path.join(CHART_DIR, "stage3_architecture.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Chart: {path}")

    # Chart 3: ContextBeta ablation
    if all_data.get("context"):
        ctx = all_data["context"]
        comp = ctx.get("comparison", {})

        # Filter to real segments
        segments = [(k, v) for k, v in comp.items()
                   if k not in ("GLOBAL",) and isinstance(v, dict)
                   and v.get("n", 0) > 50]
        segments.sort(key=lambda x: x[1].get("n_anomaly", 0), reverse=True)
        segments = segments[:15]  # top 15

        if segments:
            labels = [v.get("name", k)[:15] for k, v in segments]
            cb_pr = [v.get("cb_auc_pr", 0) for k, v in segments]
            gb_pr = [v.get("gb_auc_pr", 0) for k, v in segments]

            x = np.arange(len(labels))
            w = 0.35
            fig, ax = plt.subplots(figsize=(14, 6))
            b1 = ax.bar(x - w/2, cb_pr, w, label="ContextBeta ON", color="#4CAF50", alpha=0.8)
            b2 = ax.bar(x + w/2, gb_pr, w, label="Global Beta", color="#888888", alpha=0.8)
            for bar, v in zip(b1, cb_pr):
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("AUC-PR", fontsize=11)
            ax.set_title("ContextBeta ON vs Global Beta — Per-Segment AUC-PR",
                        fontsize=13, fontweight="bold")
            ax.legend(fontsize=10)
            ax.axhline(y=0.9, color="red", linestyle="--", alpha=0.4)
            ax.grid(True, alpha=0.3, axis="y")

            # Mark winners
            for i, (k, v) in enumerate(segments):
                if v.get("cb_auc_pr", 0) > v.get("gb_auc_pr", 0):
                    ax.text(i, max(cb_pr[i], gb_pr[i]) + 0.04, "*", ha="center",
                            color="green", fontsize=14, fontweight="bold")

            plt.tight_layout()
            path = os.path.join(CHART_DIR, "contextbeta_ablation.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Chart: {path}")

    # Chart 4: Concept Drift comparison
    if all_data.get("concept_drift"):
        cd = all_data["concept_drift"]
        dr_results = cd.get("drift_results", {})

        drift_names = []
        ms_stream_pr, ms_static_pr, if_pr, pca_pr = [], [], [], []

        for dt, dr in dr_results.items():
            drift_names.append(dr.get("drift_name", dt)[:20])
            ms_stream_pr.append(dr["memstream_streaming"]["auc_pr"])
            ms_static_pr.append(dr["memstream_static"]["auc_pr"])
            if_pr.append(dr["isolation_forest"]["auc_pr"])
            pca_pr.append(dr["inc_pca"]["auc_pr"])

        x = np.arange(len(drift_names))
        w = 0.2
        fig, ax = plt.subplots(figsize=(14, 6))
        bars1 = ax.bar(x - 1.5*w, ms_stream_pr, w, label="MemStream + Streaming", color="#4CAF50", alpha=0.85)
        bars2 = ax.bar(x - 0.5*w, ms_static_pr, w, label="MemStream + Static", color="#8BC34A", alpha=0.85)
        bars3 = ax.bar(x + 0.5*w, if_pr, w, label="IsolationForest", color="#2196F3", alpha=0.85)
        bars4 = ax.bar(x + 1.5*w, pca_pr, w, label="Inc. PCA", color="#9C27B0", alpha=0.85)

        for bars, vals in [(bars1, ms_stream_pr), (bars2, ms_static_pr),
                          (bars3, if_pr), (bars4, pca_pr)]:
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(drift_names, rotation=15, ha="right", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("AUC-PR", fontsize=11)
        ax.set_title("Concept Drift Injection — AUC-PR by Method",
                    fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.axhline(y=0.9, color="red", linestyle="--", alpha=0.4)
        ax.grid(True, alpha=0.3, axis="y")

        # Shade drift advantage
        for i, (s, st) in enumerate(zip(ms_stream_pr, ms_static_pr)):
            if s > st:
                ax.text(i, 0.02, f"+{s-st:.3f}", ha="center", fontsize=8, color="#4CAF50")

        plt.tight_layout()
        path = os.path.join(CHART_DIR, "concept_drift_comparison.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Chart: {path}")

    # Chart 5: Full data training
    if all_data.get("full_data"):
        fd = all_data["full_data"]
        results = fd.get("results", {})

        labels = sorted(results.keys())
        pr_vals = [results[k].get("auc_pr", 0) for k in labels]
        f1_vals = [results[k].get("f1", 0) for k in labels]
        recall_vals = [results[k].get("recall", 0) for k in labels]

        x = np.arange(len(labels))
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Full Data Training — Effect of Train/Val Size on Performance",
                     fontsize=13, fontweight="bold")

        metrics = [("AUC-PR", pr_vals), ("F1 Score", f1_vals), ("Recall", recall_vals)]
        for ax, (title, vals) in zip(axes, metrics):
            bars = ax.bar(x, vals, alpha=0.8,
                         color=["#4CAF50" if "FULL" in l else "#2196F3" for l in labels])
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                        f"{v:.4f}", ha="center", va="bottom", fontsize=9)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
            ax.set_title(title, fontweight="bold")
            ax.set_ylim(0, 1.05)
            ax.axhline(y=0.9, color="red", linestyle="--", alpha=0.4)
            ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        path = os.path.join(CHART_DIR, "full_data_training.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Chart: {path}")

    # Chart 6: Overall summary bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    entries = []
    labels_all = []
    colors_all = []

    # Comparison v2 methods
    cmp_results = all_data.get("comparison", {})
    if cmp_results:
        for k, v in cmp_results.get("results", {}).items():
            if "auc_pr_mean" in v:
                entries.append((v.get("method", k), v.get("auc_pr_mean", 0),
                               v.get("category", "?"), "#888888"))

    # Stage 2 best
    if all_data.get("stage2", {}).get("best"):
        b = all_data["stage2"]["best"]
        entries.append((f"S2: {b.get('config_id', '')[:25]}", b.get("auc_pr", 0),
                      "streaming-opt", "#4CAF50"))

    # Stage 3 best
    if all_data.get("stage3", {}).get("best"):
        b = all_data["stage3"]["best"]
        entries.append((f"S3: {b.get('config_id', '')[:25]}", b.get("auc_pr", 0),
                      "streaming-arch", "#E91E63"))

    if entries:
        names = [e[0] for e in entries]
        vals = [e[1] for e in entries]
        colors = [e[2] for e in entries]
        colors = ["#4CAF50" if "streaming" in c else "#2196F3" if "offline" in c else c
                 for c in colors]

        bars = ax.bar(names, vals, color=colors, alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("AUC-PR", fontsize=11)
        ax.set_title("All Experiments — AUC-PR Comparison",
                    fontsize=13, fontweight="bold")
        stream_patch = mpatches.Patch(color="#4CAF50", label="Streaming", alpha=0.85)
        offline_patch = mpatches.Patch(color="#2196F3", label="Offline", alpha=0.85)
        ax.legend(handles=[stream_patch, offline_patch], fontsize=10)
        ax.axhline(y=0.9, color="red", linestyle="--", alpha=0.4, linewidth=1)
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        path = os.path.join(CHART_DIR, "overall_summary.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Chart: {path}")

    print(f"\nAll charts saved to: {CHART_DIR}")
    print("\nDone.")


if __name__ == "__main__":
    main()
