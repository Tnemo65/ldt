================================================================================
SESSION LOG: HP_benchmark_v5 - Ablation + Comparison
Date: 2026-05-20
================================================================================

================================================================================
TERMINAL 637328: Ablation run #1 (6 setups, before user changed to 4 setups)
================================================================================
Start: 16:37:56, End: 16:47:41, Duration: ~10 min
Status: Completed (but A=shape error, D=SkPCA error)

Results:
  [A] AUC-PR=0.1424, AUC-ROC=0.8167, F1=0.2318, sep=2.51x  -- ERROR: shape mismatch (7,)(14,)
  [B] AUC-PR=0.9077, AUC-ROC=0.9953, F1=0.8689, sep=8.41x  -- OK
  [C] AUC-PR=0.9073, AUC-ROC=0.9953, F1=0.8736, sep=8.30x  -- OK (after ContextBeta fix)
  [D] AUC-PR=??, AUC-ROC=??, F1=??, sep=??                 -- ERROR: name 'SkPCA' is not defined
  [E] AUC-PR=0.8543, AUC-ROC=0.9878, F1=0.5099, sep=5069.89x  -- OK
  [F] AUC-PR=0.9073, AUC-ROC=0.9953, F1=0.8733, sep=8.30x    -- OK

================================================================================
TERMINAL 65734: Comparison run (previous session, max-train=full)
================================================================================
Start: 13:51:06, End: ~15:33 (killed after ~102 min)
Status: Incomplete - IF grid done, RCF failed

IF Grid Results (6 configs):
  [1/6] max_samples128_n_estimators100  -> AUC-PR=0.1993, F1=0.3043
  [2/6] max_samples256_n_estimators100  -> AUC-PR=0.3467, F1=0.4544
  [3/6] max_samples256_n_estimators200  -> AUC-PR=0.3233, F1=0.4554
  [4/6] max_samples512_n_estimators200  -> AUC-PR=0.4123, F1=0.5213  <-- BEST
  [5/6] max_samples256_n_estimators300  -> AUC-PR=0.3109, F1=0.4481
  [6/6] max_samples512_n_estimators300  -> AUC-PR=0.3811, F1=0.5011

RCF: FAILED - robustcutforest not available, falling back to RandomTreesEmbedding

================================================================================
TERMINAL (various ablation runs before new 4-setup design):
================================================================================

Run 1 (16:07): A=MemStreamAE not defined, B=OK, C=OK, D=SkPCA, E=OK, F=OK
  - Fix applied: added MemStreamAE import in train() method

Run 2 (16:29): A=(7,)(14,) shape error, B=OK, C=OK, D=SkPCA, E=OK, F=OK
  - Fix applied: changed Memory(..., d) to Memory(..., self.out_dim) in train()

================================================================================
FINAL ABLATION RUN (new 4-setups design): SUCCESS
================================================================================
Start: 16:44:03, End: 16:47:52, Duration: ~4 min

Results:
  [A] Normal Autoencoder (no noise, no memory)
      AUC-PR=0.0814, AUC-ROC=0.8231, F1=0.0576, Precision=0.0297, Recall=1.0000, sep=0.96x [18.1s]
      
  [B] Denoise Autoencoder (raw features)
      AUC-PR=0.0934, AUC-ROC=0.5004, F1=0.0576, Precision=0.0297, Recall=1.0000, sep=1.00x [42.1s]
      NOTE: AUC-ROC=0.50 means essentially random - noise hurts 7D raw features
      
  [C] Denoise AE + Feature Engineering (streaming memory)
      AUC-PR=0.9077, AUC-ROC=0.9953, F1=0.8689, Precision=0.8238, Recall=0.9191, sep=8.41x [75.8s]
      
  [D] Denoise AE + FE + ContextBeta
      AUC-PR=0.9073, AUC-ROC=0.9953, F1=0.8736, Precision=0.8280, Recall=0.9245, sep=8.30x [85.3s]

Delta Analysis:
  B vs A (Effect of denoise noise):     auc_roc=-0.3226, auc_pr=+0.0120, f1=+0.0000
  C vs B (Effect of feature engineering): auc_roc=+0.4948, auc_pr=+0.8143, f1=+0.8112
  D vs C (Effect of ContextBeta):        auc_roc=-0.0000, auc_pr=-0.0004, f1=+0.0047
  C vs A (Cumulative: FE+streaming):      auc_roc=+0.1722, auc_pr=+0.8263, f1=+0.8112
  D vs A (Cumulative: all components):    auc_roc=+0.1722, auc_pr=+0.8259, f1=+0.8160

KEY INSIGHT: Feature Engineering is the primary driver of performance.
The jump from B (raw, denoised) to C (FE) is massive: AUC-PR +0.81, F1 +0.81
ContextBeta adds only marginal improvement (F1 +0.005)

================================================================================
NEXT STEPS (pending):
================================================================================
1. Run comparison.py (skip IF grid, methods: RCF,MemStream,HSTrees,LODA,NormalAE,OCSVM,DAGMM,DeepSVDD)
2. Verify comparison_results.json is complete

================================================================================
