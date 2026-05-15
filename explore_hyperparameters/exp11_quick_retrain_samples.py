#!/usr/bin/env python3
"""
Exp 11: Quick Retrain Samples
Priority: MEDIUM
Primary Metric: Fine-tune convergence
"""

import sys, json, time, warnings
from pathlib import Path
warnings.filterwarnings('ignore')

import numpy as np

ROOT = Path(__file__).parent
OUT  = ROOT / 'results'
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))
from shared import load_data, GPUExperimentModel, compute_all_metrics


def quick_retrain_quality(X_samples, n_samples, n_epochs=50):
    """Quick retrain AE on X_samples and measure convergence."""
    if len(X_samples) < n_samples:
        return {'converged': False, 'final_loss': np.nan, 'quality': np.nan}

    from sklearn.preprocessing import StandardScaler
    import torch, torch.nn.functional as F

    idx = np.random.RandomState(42).choice(len(X_samples), size=n_samples, replace=False)
    X_retrain = X_samples[idx]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_retrain.astype(np.float64)).astype(np.float32)
    d, latent = Xs.shape[1], 60

    torch.manual_seed(42)
    W1 = torch.nn.Parameter(torch.randn(d, latent, dtype=torch.float32, device='cuda') * np.sqrt(2.0/d))
    b1 = torch.nn.Parameter(torch.zeros(latent, dtype=torch.float32, device='cuda'))
    W2 = torch.nn.Parameter(torch.randn(latent, d, dtype=torch.float32, device='cuda') * np.sqrt(2.0/latent))
    b2 = torch.nn.Parameter(torch.zeros(d, dtype=torch.float32, device='cuda'))
    opt = torch.optim.Adam([W1, b1, W2, b2], lr=1e-3)

    Xt = torch.from_numpy(Xs.astype(np.float32)).to('cuda')
    losses = []
    for _ in range(n_epochs):
        x_noisy = Xt + torch.randn_like(Xt) * 0.1
        z = F.relu(x_noisy @ W1 + b1)
        x_rec = z @ W2 + b2
        loss = F.mse_loss(x_rec, Xt)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())

    final_loss = losses[-1]
    loss_reduction = (losses[0] - final_loss) / (losses[0] + 1e-8)
    quality = 1.0 / (final_loss + 1e-8)
    return {
        'converged': loss_reduction > 0.95,
        'initial_loss': losses[0],
        'final_loss': final_loss,
        'loss_reduction': loss_reduction,
        'quality': quality,
    }


def run():
    ts = time.strftime('%Y%m%d_%H%M%S')
    print("EXP 11: Quick Retrain Samples")
    print("Loading data...")
    data = load_data(n_warmup=10000, n_test=15000)

    grid = [256, 512, 1024, 2048]
    results = []

    for n_samples in grid:
        t0 = time.time()
        print(f"  n_samples={n_samples}")

        model = GPUExperimentModel(
            memory_len=256, k=10, gamma=0.0, latent_dim=60,
            default_beta=0.5, seed=42, device='cuda'
        )
        model.fit(
            data['X_warmup'],
            neighborhood_ids=data['nb_warmup'],
            hour_vals=data['hr_warmup'],
            dow_vals=data['dw_warmup'],
            ratecode_vals=data['rc_warmup'],
            epochs=20, batch_size=256
        )

        scores, _ = model.score_streaming(
            data['X_test'],
            neighborhood_ids=data['nb_test'],
            hour_vals=data['hr_test'],
            dow_vals=data['dw_test'],
            ratecode_vals=data['rc_test'],
        )
        m = compute_all_metrics(data['y_test'], scores)
        m['elapsed_s'] = time.time() - t0

        # Retrain quality
        recent_normal = data['X_warmup'][-5000:]
        rq = quick_retrain_quality(recent_normal, n_samples=n_samples, n_epochs=50)
        m['retrain_converged'] = rq.get('converged', False)
        m['retrain_initial_loss'] = rq.get('initial_loss', np.nan)
        m['retrain_final_loss'] = rq.get('final_loss', np.nan)
        m['retrain_quality'] = rq.get('quality', np.nan)

        print(f"    AUC-ROC={m['AUC-ROC']:.4f}  Recon error={rq.get('final_loss', 0):.4f}  Converged={rq.get('converged', False)}")
        results.append({'n_samples': n_samples, 'metrics': m})

    best = max(results, key=lambda r: r['metrics'].get('retrain_quality', 0))

    output = {
        'experiment': 'exp11_quick_retrain_samples',
        'hyperparameter': 'quick_retrain_samples',
        'timestamp': ts,
        'priority': 'MEDIUM',
        'grid': grid,
        'results': results,
        'best_config': best,
        'recommendation': f"n_samples={best["n_samples"]} is optimal for quick retrain.",
    }
    out_path = OUT / f'exp11_quick_retrain_samples_{ts}.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    run()
