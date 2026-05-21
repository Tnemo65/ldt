import sys, os, json, time, io, pickle, hashlib, hmac
sys.path.insert(0, r"c:\proj\ldt\HP_benchmark_v4")
import numpy as np
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTHONLEGACYWINDOWSSTDIO"] = "1"
os.environ["ARROW_DISABLE_MMAP"] = "1"

from benchmark_core import (
    MemStreamPipeline, extract_features_from_parquet,
    evaluate_scores, find_best_threshold, DEVICE
)
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stderr)
LOGGER = logging.getLogger("stage3_manual")

base = r"c:\proj\ldt\HP_benchmark_v4"
v3 = os.path.join(base, "..", "HP_benchmark_v3")
train_path = os.path.join(v3, "train_clean.parquet")
valid_path = os.path.join(v3, "valid_polluted.parquet")
output_dir = os.path.join(base, "results", "grid_search")
stage3_dir = os.path.join(output_dir, "stage3")

# Delete corrupted Stage 3 results first
LOGGER.info("Cleaning corrupted Stage 3 results...")
if os.path.exists(stage3_dir):
    for fn in os.listdir(stage3_dir):
        if fn.endswith('.json'):
            fp = os.path.join(stage3_dir, fn)
            with open(fp) as f:
                d = json.load(f)
            if d.get('auc_pr', 0) < 0.25:
                os.remove(fp)
                LOGGER.info("  Deleted: %s (AUC-PR=%.4f)", fn, d.get('auc_pr', 0))
os.makedirs(stage3_dir, exist_ok=True)

# Load FULL training data
LOGGER.info("Loading FULL training data (500K rows)...")
X_train, hours_train, dows_train, rcs_train, nb_train = extract_features_from_parquet(
    train_path, max_rows=0  # 0 = full
)
LOGGER.info("Training: %d rows", len(X_train))

LOGGER.info("Loading validation data (500K rows)...")
X_val, hours_val, dows_val, rcs_val, nb_val = extract_features_from_parquet(
    valid_path, max_rows=500000
)
gt_mask = np.load(os.path.join(v3, "valid", "ground_truth_mask.npy"))
gt_mask = gt_mask[-len(X_val):]
LOGGER.info("Validation: %d rows (GT: %d anomalies, %.2f%%)",
            len(X_val), gt_mask.sum(), gt_mask.mean() * 100)

# Best Stage 2 config
LOGGER.info("Reading Stage 2 summary for top configs...")
s2_path = os.path.join(output_dir, "stage2", "summary.json")
with open(s2_path) as f:
    s2_data = json.load(f)
s2_results = s2_data.get('results', [])
LOGGER.info("Stage 2: %d configs available", len(s2_results))

# Combine S1 + S2 for Stage 3 top selection
s1_path = os.path.join(output_dir, "stage1", "summary.json")
with open(s1_path) as f:
    s1_data = json.load(f)
combined = s1_data.get('results', []) + s2_results
combined.sort(key=lambda x: x.get('auc_pr', 0), reverse=True)
top_configs = [r for r in combined if r.get('auc_pr', 0) > 0.001][:10]
LOGGER.info("Combined: %d configs for Stage 3", len(top_configs))

# Stage 3 grid: out_dim x memory_len x k (4 configs)
arch_configs = [
    {"out_dim": 34, "memory_len": 1024, "k": 5},
    {"out_dim": 34, "memory_len": 1024, "k": 10},
    {"out_dim": 68, "memory_len": 1024, "k": 5},
    {"out_dim": 68, "memory_len": 1024, "k": 10},
]

best_overall = combined[0] if combined else {}
LOGGER.info("Best overall base: %s AUC-PR=%.4f",
            best_overall.get('config_id', '?'), best_overall.get('auc_pr', 0))

all_results = []
for arch in arch_configs:
    base_cfg = best_overall
    ml = arch["memory_len"]
    k = arch["k"]
    od = arch["out_dim"]
    epochs = 5000

    cfg_id = (
        f"M{ml}_k{k}_g{int(base_cfg.get('gamma', 0)*100)}_"
        f"b{str(base_cfg.get('beta', 0)).replace('.','p')}_"
        f"e{base_cfg.get('epochs', 2000)}_lr{int(base_cfg.get('lr', 0.01)*1000)}_"
        f"n{base_cfg.get('noise_std', 0.001)}_od{od}"
    )
    result_path = os.path.join(stage3_dir, cfg_id + ".json")

    if os.path.exists(result_path):
        with open(result_path) as f:
            result = json.load(f)
        if result.get('auc_pr', 0) > 0.25:
            LOGGER.info("  SKIP: %s AUC-PR=%.4f", cfg_id, result.get('auc_pr', 0))
            all_results.append(result)
            continue

    LOGGER.info("  %s", cfg_id)
    t0 = time.time()

    try:
        pipeline = MemStreamPipeline(
            d=34, out_dim=od,
            memory_len=ml,
            k=k,
            gamma=base_cfg.get('gamma', 0.0),
            beta=base_cfg.get('beta', 0.0001),
            noise_std=base_cfg.get('noise_std', 0.001),
            lr=base_cfg.get('lr', 0.01),
            epochs=epochs,
            batch_size=1024,
            seed=42,
            cb_warmup=min(4096, ml * 4),
            verbose=False,
            adam_betas=(0.9, 0.999),
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
            gt_mask=gt_mask,
        )
        best_t, best_f1 = find_best_threshold(adj_scores, gt_mask)
        metrics_opt = evaluate_scores(adj_scores, gt_mask, threshold=best_t)

        result = {
            "config_id": cfg_id,
            "stage": 3,
            "memory_len": ml,
            "k": k,
            "gamma": base_cfg.get('gamma', 0.0),
            "beta": base_cfg.get('beta', 0.0001),
            "noise_std": base_cfg.get('noise_std', 0.001),
            "lr": base_cfg.get('lr', 0.01),
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

        all_results.append(result)
        LOGGER.info(
            "    -> AUC-PR=%.4f, AUC-ROC=%.4f, F1=%.4f [%.1fs]",
            result["auc_pr"], result["auc_roc"], result["f1"], time.time() - t0
        )

        # Clear GPU
        del pipeline
        torch.cuda.empty_cache()

    except Exception as e:
        LOGGER.error("    [ERROR] %s: %s", cfg_id, e)
        import traceback
        traceback.print_exc()
        result = {
            "config_id": cfg_id, "stage": 3,
            "error": str(e), "out_dim": od, "memory_len": ml, "k": k,
            "auc_pr": 0.0, "train_time_s": float(time.time() - t0),
        }
        all_results.append(result)

# Save Stage 3 summary
all_results.sort(key=lambda x: x.get('auc_pr', 0), reverse=True)
summary_path = os.path.join(stage3_dir, "summary.json")
with open(summary_path, "w") as f:
    json.dump({"stage": 3, "results": all_results, "best": all_results[0] if all_results else {}}, f, indent=2)
LOGGER.info("Stage 3 summary saved: %d configs", len(all_results))

# Overall best
all_stage_results = []
for stage_dir in ["stage1", "stage2", "stage3"]:
    sp = os.path.join(output_dir, stage_dir, "summary.json")
    if os.path.exists(sp):
        with open(sp) as f:
            data = json.load(f)
        all_stage_results.extend(data.get("results", []))

all_stage_results.sort(key=lambda x: x.get('auc_pr', 0), reverse=True)
best = all_stage_results[0] if all_stage_results else {}

with open(os.path.join(output_dir, "best_config.json"), "w") as f:
    json.dump(best, f, indent=2)
with open(os.path.join(output_dir, "final_results.json"), "w") as f:
    json.dump({"all_results": all_stage_results[:100], "best_config": best}, f, indent=2)

LOGGER.info("")
LOGGER.info("BEST OVERALL: %s AUC-PR=%.4f AUC-ROC=%.4f F1=%.4f",
            best.get('config_id', '?'), best.get('auc_pr', 0),
            best.get('auc_roc', 0), best.get('f1', 0))
