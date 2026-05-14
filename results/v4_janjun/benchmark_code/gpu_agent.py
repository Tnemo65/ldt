"""
GPUAgent — LSTM-Autoencoder evaluation on NVIDIA RTX 3090 Ti (CUDA).
Runs in a single process to maximize GPU utilization.
Process-per-fold strategy to avoid CUDA OOM: one fold loaded at a time.

Evaluates: 11 folds × 3 difficulties × 10 seeds = 330 runs.
"""
from __future__ import annotations

import time
import gc
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve, f1_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


OUT_DIR   = Path(__file__).parent.parent / 'results' / 'v3'
CACHE_DIR = OUT_DIR / 'features' / 'fold_data'
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS        = [42, 123, 456, 789, 1024, 2048, 4096, 8192, 16384, 32768]
DIFFICULTIES = ['easy', 'medium', 'hard']
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
METRICS      = ['AUC_PR', 'AUC_ROC', 'F1']


# ─── LSTM-AE Model ─────────────────────────────────────────────────────────

class LSTMAE(nn.Module):
    """LSTM Autoencoder for anomaly detection."""
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.enc = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dec = nn.LSTM(hidden_dim, input_dim, batch_first=True)

    def forward(self, x):
        _, (h, _) = self.enc(x)
        h_perm = h.permute(1, 0, 2)
        dec, _ = self.dec(h_perm.repeat(1, 1, 1))
        return dec

    def encode_decode(self, x):
        _, (h, _) = self.enc(x)
        h_perm = h.permute(1, 0, 2)
        dec, _ = self.dec(h_perm.repeat(1, 1, 1))
        return dec


def train_lstm_ae(X_train: np.ndarray, seed: int,
                  hidden_dim: int = 64, epochs: int = 10,
                  batch_size: int = 256) -> tuple[nn.Module, StandardScaler, float]:
    """Train LSTM-AE on GPU, return (model, scaler, threshold)."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train).astype(np.float32)

    # Wrap as sequence: (N, 1, D)
    seq = torch.FloatTensor(Xs.reshape(-1, 1, Xs.shape[1])).to(DEVICE)
    n_samples = len(seq)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    model = LSTMAE(Xs.shape[1], hidden_dim).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    ds    = TensorDataset(seq, seq)
    dl    = DataLoader(ds, batch_size=batch_size, shuffle=True,
                       num_workers=0, pin_memory=True)

    for epoch in range(epochs):
        for bx, by in dl:
            bx, by = bx.to(DEVICE, non_blocking=True), by.to(DEVICE, non_blocking=True)
            pred = model(bx)
            loss = loss_fn(pred, by)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    # Compute threshold on training reconstruction errors
    with torch.no_grad():
        preds = model(seq).cpu().numpy()
    errors  = np.mean(np.abs(seq.cpu().numpy() - preds), axis=(1, 2))
    threshold = float(np.percentile(errors, 95))

    return model, scaler, threshold


def score_lstm_ae(model: nn.Module, scaler: StandardScaler,
                  X_test: np.ndarray, threshold: float) -> tuple[np.ndarray, float]:
    """Score test data, return (anomaly_scores, inference_time_ms)."""
    Xs = scaler.transform(X_test).astype(np.float32)
    seq = torch.FloatTensor(Xs.reshape(-1, 1, Xs.shape[1])).to(DEVICE)

    t0 = time.perf_counter()
    with torch.no_grad():
        preds = model(seq).cpu().numpy()
    scores = np.mean(np.abs(seq.cpu().numpy() - preds), axis=(1, 2)).astype(np.float64)
    inference_ms = (time.perf_counter() - t0) * 1000

    return scores, inference_ms


def evaluate_one_run(X_train: np.ndarray, X_test: np.ndarray,
                     y_test: np.ndarray, seed: int) -> dict:
    """Train LSTM-AE and evaluate on test set."""
    try:
        t0 = time.perf_counter()
        model, scaler, threshold = train_lstm_ae(X_train, seed)
        t_train = time.perf_counter() - t0

        scores, t_score = score_lstm_ae(model, scaler, X_test, threshold)

        # Clear GPU memory for next fold
        del model
        torch.cuda.empty_cache()
        gc.collect()

        if len(scores) < 10 or y_test.sum() == 0:
            return {'error': 'too few samples'}

        pr_curve, rc_curve, _ = precision_recall_curve(y_test, scores)
        auc_pr   = auc(rc_curve, pr_curve) if len(rc_curve) > 1 else 0.0
        fpr_arr, tpr_arr, _   = roc_curve(y_test, scores)
        auc_roc  = auc(fpr_arr, tpr_arr) if len(fpr_arr) > 1 else 0.5

        # Optimal threshold from scores (use top 5% as anomaly)
        best_t = float(np.percentile(scores, 95))
        preds  = (scores >= best_t).astype(int)
        f1     = f1_score(y_test, preds, zero_division=0)

        return {
            'AUC_PR': auc_pr, 'AUC_ROC': auc_roc, 'F1': f1,
            'train_ms': t_train * 1000,
            'score_ms': t_score,
        }
    except Exception as e:
        torch.cuda.empty_cache()
        gc.collect()
        return {'error': str(e)}


class GPUAgent:
    """
    Runs LSTM-Autoencoder on CUDA (RTX 3090 Ti).
    Processes folds one at a time to avoid OOM.
    Each fold processes all 3 difficulties and 10 seeds.
    Results saved to results/v3/gpu_lstm_results.csv
    """

    def __init__(self, base_dir: Path | None = None, out_dir: Path | None = None):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent.parent
        self.out_dir  = Path(out_dir)  if out_dir  else OUT_DIR
        self.out_file = self.out_dir  / 'gpu_lstm_results.csv'

    def run(self) -> pd.DataFrame:
        if not torch.cuda.is_available():
            print('[GPUAgent] No CUDA device found. Skipping.')
            return pd.DataFrame()

        print(f'\n[GPUAgent] Starting on {torch.cuda.get_device_name(0)}')
        print(f'  Device: {DEVICE}  |  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

        # Collect fold paths
        fold_paths = sorted(CACHE_DIR.glob('fold*.npz'))
        total_runs = len(fold_paths) * len(SEEDS)

        t0 = time.perf_counter()
        all_results = []

        for fi, path in enumerate(fold_paths):
            data = np.load(path, allow_pickle=True)
            fold      = int(data['fold'])
            diff      = str(data['difficulty'])
            X_train   = data['X_train']         # already scaled
            X_test_inj= data['X_test_inj']       # already scaled
            y_test    = data['y_test']

            fold_results = []
            for seed_val in SEEDS:
                row = {
                    'fold': fold, 'difficulty': diff,
                    'algorithm': 'LSTM-AE', 'seed': seed_val,
                }
                result = evaluate_one_run(X_train, X_test_inj, y_test, seed_val)
                row.update(result)
                fold_results.append(row)
                all_results.append(row)

            # Progress
            elapsed = time.perf_counter() - t0
            done    = fi + 1
            rate    = done / elapsed
            remain  = (len(fold_paths) - done) / rate / 60
            print(f'  Fold {fold+1:02d}/{len(fold_paths)} ({diff:7s}): '
                  f'{len(fold_results)} runs done, '
                  f'~{remain:.1f}m remaining')

        df = pd.DataFrame(all_results)
        df.to_csv(self.out_file, index=False)

        elapsed = time.perf_counter() - t0
        print(f'\n[GPUAgent] Done. {len(all_results)} rows saved in {elapsed:.1f}s ({elapsed/60:.1f} min)')

        # Summary
        if not df.empty and 'AUC_PR' in df.columns:
            print(f'\n  Mean AUC-PR: {df["AUC_PR"].mean():.4f}')
            print(f'  Mean train time: {df["train_ms"].mean():.1f} ms')
            print(f'  Mean inference time: {df["score_ms"].mean():.1f} ms')

        return df


if __name__ == '__main__':
    GPUAgent().run()
