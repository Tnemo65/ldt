"""
ADWIN-U: ADWIN with Higher-Order Statistics for CA-DQStream.

Extends SimpleADWIN (memstream_core.py) with multiple statistical tests
for detecting different types of concept drift:

  1. Mean Shift  — Student's t-test on window means
  2. Variance Shift — Levene's test on window variances
  3. Skewness Drift — Checks if distribution asymmetry changes
  4. Kurtosis Drift — Checks if tail-mass distribution changes

Unlike the original ADWIN_U paper (ADWIN_U_detector.py) which computes
statistics on a single sample, this implementation computes statistics over
the sliding window, making it suitable for scalar meta-metrics (null_rate,
violation_rate, etc.) produced by the MetaAggregator.

Design goals:
- Backward compatible with SimpleADWIN (statistic='mean' is identical)
- Pure Python (no skmultiflow dependency — compatible with PyFlink UDF)
- Zero external dependencies beyond the standard library + numpy + scipy
- Integrates with existing MultiInstanceADWIN in adwin_multi_instance.py

Usage:
  # Replace SimpleADWIN with ADWIN_U in MultiInstanceADWIN
  detector = ADWIN_U(delta=0.002, max_window=500, statistic='skewness')

  # For volume metrics (spike detection)
  vol_detector = ADWIN_U(delta=0.005, statistic='variance')

  # For rate metrics (burst detection)
  null_detector = ADWIN_U(delta=0.001, statistic='skewness')

  # For score metrics (distribution shift)
  score_detector = ADWIN_U(delta=0.001, statistic='kurtosis')

  if detector.update(value):
      print("Drift detected via", detector.last_drift_type)

References:
  - Bifet & Gavalda (2007). Learning from Time-Changing Data with Adaptive Windowing.
  - ADWIN_U paper (ADWIN_U_detector.py): ADWIN-U with higher-order statistics
  - METER (Zhu et al., VLDB 2024): META: A Dynamic Concept Adaptation Framework
"""

from __future__ import annotations

import math
from collections import deque
from typing import Optional, Tuple, Literal

import numpy as np

# Optional scipy dependency — falls back to numpy-only implementations
try:
    from scipy import stats as scipy_stats

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    scipy_stats = None

import logging as _logging

logger = _logging.getLogger("adwin-u")


# ---------------------------------------------------------------------------
# Statistical helpers (pure Python + numpy fallback)
# ---------------------------------------------------------------------------

def _nan_mean(arr) -> float:
    """Compute mean, ignoring NaN values."""
    valid = [x for x in arr if x == x]  # filter out NaN
    return float(sum(valid) / len(valid)) if valid else 0.0


def _nan_std(arr, ddof=0) -> float:
    """Compute standard deviation, ignoring NaN values."""
    valid = [x for x in arr if x == x]
    if len(valid) < 2:
        return 0.0
    m = sum(valid) / len(valid)
    variance = sum((x - m) ** 2 for x in valid) / (len(valid) - ddof if ddof > 0 else len(valid))
    return float(math.sqrt(variance))


def _nan_var(arr, ddof=0) -> float:
    """Compute variance, ignoring NaN values."""
    valid = [x for x in arr if x == x]
    if len(valid) < 2:
        return 0.0
    m = sum(valid) / len(valid)
    return float(sum((x - m) ** 2 for x in valid) / (len(valid) - ddof if ddof > 0 else len(valid)))


def _nan_skewness(arr) -> float:
    """Compute sample skewness, ignoring NaN values.

    Skewness = E[(X - mu)^3] / sigma^3
    - skewness > 0: right-tailed (positive outliers)
    - skewness < 0: left-tailed (negative outliers)
    - skewness ~= 0: symmetric
    """
    valid = [x for x in arr if x == x]
    n = len(valid)
    if n < 3:
        return 0.0

    if HAS_SCIPY:
        return float(scipy_stats.skew(valid, bias=False))

    m = sum(valid) / n
    sigma = math.sqrt(sum((x - m) ** 2 for x in valid) / n)
    if sigma < 1e-10:
        return 0.0
    skew = sum((x - m) ** 3 for x in valid) / (n * sigma ** 3)
    return float(skew)


def _nan_kurtosis(arr) -> float:
    """Compute sample excess kurtosis, ignoring NaN values.

    Excess Kurtosis = E[(X - mu)^4] / sigma^4 - 3
    - kurtosis > 0: heavier tails than normal (leptokurtic)
    - kurtosis < 0: lighter tails than normal (platykurtic)
    - kurtosis ~= 0: similar to normal distribution (mesokurtic)
    """
    valid = [x for x in arr if x == x]
    n = len(valid)
    if n < 4:
        return 0.0

    if HAS_SCIPY:
        return float(scipy_stats.kurtosis(valid, bias=False))

    m = sum(valid) / n
    sigma_sq = sum((x - m) ** 2 for x in valid) / n
    if sigma_sq < 1e-10:
        return 0.0
    kurt = (sum((x - m) ** 4 for x in valid) / (n * sigma_sq ** 2)) - 3
    return float(kurt)


def _nan_median(arr) -> float:
    """Compute median, ignoring NaN values."""
    valid = sorted([x for x in arr if x == x])
    if not valid:
        return 0.0
    n = len(valid)
    if n % 2 == 1:
        return float(valid[n // 2])
    return float((valid[n // 2 - 1] + valid[n // 2]) / 2.0)


def _t_test_statistic(w1: list, w2: list) -> Tuple[float, float]:
    """Two-sample Welch's t-test (unequal variances, unequal sizes).

    Returns:
        (t_statistic, p_approximation)
    """
    n1, n2 = len(w1), len(w2)
    if n1 < 2 or n2 < 2:
        return 0.0, 1.0

    m1 = _nan_mean(w1)
    m2 = _nan_mean(w2)
    v1 = _nan_var(w1, ddof=1)
    v2 = _nan_var(w2, ddof=1)

    denom = math.sqrt(v1 / n1 + v2 / n2)
    if denom < 1e-10:
        return 0.0, 1.0

    t = (m1 - m2) / denom

    # Welch-Satterthwaite degrees of freedom approximation
    num = (v1 / n1 + v2 / n2) ** 2
    denom_df = ((v1 / n1) ** 2 / (n1 - 1)) + ((v2 / n2) ** 2 / (n2 - 1))
    df = num / denom_df if denom_df > 1e-10 else 1.0

    # Approximate two-tailed p-value using standard normal for large df
    # For very small df, use a conservative bound
    if df > 30:
        # Standard normal approximation
        abs_t = abs(t)
        # CDF of standard normal via error function approximation
        p = 2.0 * (1.0 - _norm_cdf(abs_t))
    else:
        # Conservative: use a fixed threshold equivalent to alpha=0.05
        p = 1.0 if abs(t) < 2.0 else 0.01

    return float(t), float(p)


def _norm_cdf(x: float) -> float:
    """Approximate standard normal CDF using error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _levene_statistic(w1: list, w2: list) -> Tuple[float, float]:
    """Two-sample Levene's test for equality of variances.

    More robust to non-normality than Bartlett's test. Uses deviations
    from the median (Brown-Forsythe variant) which is the default in scipy.

    Returns:
        (levene_statistic, p_approximation)

    Per KAIS 2025 paper: Levene's test is a key component of variance shift
    detection alongside skewness/kurtosis for unsupervised drift detection.
    """
    n1, n2 = len(w1), len(w2)
    if n1 < 2 or n2 < 2:
        return 0.0, 1.0

    median_w1 = _nan_median(w1)
    median_w2 = _nan_median(w2)

    z1 = [abs(v - median_w1) for v in w1 if v == v]
    z2 = [abs(v - median_w2) for v in w2 if v == v]

    z1_valid = [x for x in z1 if x == x]
    z2_valid = [x for x in z2 if x == x]

    if len(z1_valid) < 2 or len(z2_valid) < 2:
        return 0.0, 1.0

    z1_mean = _nan_mean(z1_valid)
    z2_mean = _nan_mean(z2_valid)

    nn1, nn2 = len(z1_valid), len(z2_valid)

    # Levene's W statistic (Brown-Forsythe variant)
    numerator = (nn1 + nn2 - 2) * nn1 * (z1_mean - (nn1 * z1_mean + nn2 * z2_mean) / (nn1 + nn2))**2
    denominator = (nn1 - 1) * sum((z - z1_mean) ** 2 for z in z1_valid) + \
                  (nn2 - 1) * sum((z - z2_mean) ** 2 for z in z2_valid)

    if denominator < 1e-10:
        return 0.0, 1.0

    W = numerator / denominator

    # Approximate p-value using chi-squared distribution with df=1
    # For large W, p small. W ~ chi-squared(1) under null.
    # Conservative: W > 4.0 -> p < 0.05, W > 6.6 -> p < 0.01
    if W > 10.0:
        p_approx = 0.001
    elif W > 7.0:
        p_approx = 0.01
    elif W > 4.0:
        p_approx = 0.05
    else:
        p_approx = min(W / 4.0, 1.0)

    return float(W), float(p_approx)


# ---------------------------------------------------------------------------
# Statistic registry
# ---------------------------------------------------------------------------

_STAT_FUNCS = {
    'mean':       lambda w: _nan_mean(w),
    'median':     lambda w: _nan_median(w),
    'variance':   lambda w: _nan_var(w, ddof=1),
    'std':        lambda w: _nan_std(w, ddof=1),
    'skewness':   _nan_skewness,
    'kurtosis':   _nan_kurtosis,
}

_STAT_NAMES = list(_STAT_FUNCS.keys())

# Public alias for exports
STAT_NAMES = _STAT_NAMES

# Default statistic per meta-metric (CA-DQStream metaaggregator)
#
# Per KAIS 2025 paper findings:
#   - Skewness: Rank 1 for BAR (accuracy-based), Rank 1 for accuracy
#   - Kurtosis: Rank 1 (tied) for BAR, Rank 2 for accuracy
#   - Variance: Rank 4 for BAR, Rank 2 for accuracy
#   - Mean: Rank 5 for BAR, Rank 3 for accuracy
#
# Higher-order statistics (skewness/kurtosis) capture distributional shifts
# that mean/variance miss. They are most effective for unsupervised drift detection.
DEFAULT_STATISTIC_PER_METRIC = {
    'volume':             'variance',  # Volume spike = variance shift (OK)
    'null_rate':          'skewness', # NULL burst = asymmetric distribution (rank 1)
    'violation_rate':     'skewness', # Violation bursts create asymmetric patterns (rank 1)
    'anomaly_rate':       'skewness', # Anomaly rate spikes = distribution shift (rank 1)
    'avg_anomaly_score':  'kurtosis', # Score tail changes = kurtosis shift (rank 1)
    'delta_score':        'kurtosis', # Canary-ML disagreement = tail divergence (rank 1)
}


# ---------------------------------------------------------------------------
# ADWIN-U main class
# ---------------------------------------------------------------------------

class ADWIN_U:
    """ADWIN with Higher-Order Statistics for multi-type drift detection.

    This class extends SimpleADWIN (memstream_core.py lines 1388-1450) with
    support for detecting three types of concept drift:

      Type 1 — Mean Shift:    Student's t-test on window means
      Type 2 — Variance Shift: Levene's test on window variances
      Type 3 — Distribution:  Skewness / Kurtosis divergence

    Each ADWIN_U instance monitors ONE (neighborhood, metric) pair, matching
    the architecture of MultiInstanceADWIN in adwin_multi_instance.py.

    Key differences from SimpleADWIN:
      - Uses Welch's t-test (instead of Hoeffding bound) for mean comparison
      - Adds variance comparison via Levene's test
      - Adds skewness and kurtosis comparison for distribution shift
      - Reports which type of drift was detected (last_drift_type)

    Args:
        delta: Confidence parameter (lower = more sensitive).
               Default 0.002. Critical metrics use 0.001.
        max_window: Maximum window size. Default 500 (1-min windows * ~500 neighborhoods).
        statistic: Primary statistic for drift detection.
                   Options: 'mean', 'median', 'variance', 'std',
                            'skewness', 'kurtosis'.
                   Default: 'mean' (backward compatible with SimpleADWIN).
        min_window_size: Minimum window size before drift checking. Default 50.
        secondary_stat: Secondary statistic for cross-validation.
                        When primary detects drift, secondary must confirm.
                        Default: 'variance' for mean-based primary.
        sensitivity: Predefined sensitivity presets:
                     'low'    -> delta=0.01,   high tolerance
                     'medium' -> delta=0.002,  balanced (default)
                     'high'   -> delta=0.0005,  sensitive
                     'critical' -> delta=0.0001, very sensitive
        use_secondary_check: If True, secondary statistic must agree before
                             reporting drift. Reduces false positives.

    Example:
        # Backward compatible with SimpleADWIN
        det = ADWIN_U(delta=0.002, statistic='mean')

        # Skewness-based detection for rate metrics
        det = ADWIN_U(delta=0.001, statistic='skewness',
                       secondary_stat='kurtosis')

        # Production config for null_rate monitoring
        det = ADWIN_U(delta=0.001, statistic='skewness',
                       use_secondary_check=True)

        # Stream update
        if det.update(null_rate_value):
            print(f"Drift detected: {det.last_drift_type}")
            print(f"  Recent stat: {det.last_recent_stat:.4f}")
            print(f"  Old stat:    {det.last_old_stat:.4f}")
            print(f"  Drift magnitude: {det.last_drift_magnitude:.4f}")
    """

    # Drift type constants
    DRIFT_MEAN_SHIFT: int = 1
    DRIFT_VARIANCE_SHIFT: int = 2
    DRIFT_SKEWNESS_SHIFT: int = 3
    DRIFT_KURTOSIS_SHIFT: int = 4
    DRIFT_NONE: int = 0

    def __init__(
        self,
        delta: float = 0.002,
        max_window: int = 500,
        statistic: Literal["mean", "median", "variance", "std",
                            "skewness", "kurtosis"] = "mean",
        min_window_size: int = 50,
        secondary_stat: Optional[str] = None,
        use_secondary_check: bool = False,
        sensitivity: Optional[Literal["low", "medium", "high", "critical"]] = None,
    ):
        if statistic not in _STAT_FUNCS:
            raise ValueError(
                f"statistic must be one of {_STAT_NAMES}, got '{statistic}'"
            )

        # Apply sensitivity presets
        if sensitivity is not None:
            _sensitivity_map = {
                'low': 0.01,
                'medium': 0.002,
                'high': 0.0005,
                'critical': 0.0001,
            }
            delta = _sensitivity_map.get(sensitivity, delta)

        self.delta: float = delta
        self.max_window: int = max_window
        self.statistic: str = statistic
        self.min_window_size: int = min_window_size
        self.use_secondary_check: bool = use_secondary_check

        # Primary statistic function
        self._stat_fn = _STAT_FUNCS[statistic]

        # Secondary statistic (for cross-validation)
        # Per KAIS 2025 paper: skewness and kurtosis are the most effective
        # statistics for unsupervised drift detection (rank 1 BAR).
        # We recommend cross-checking with complementary statistics.
        if secondary_stat is None:
            # Primary skewness -> cross-check with kurtosis (both rank 1)
            # Primary kurtosis -> cross-check with skewness
            # Primary mean -> cross-check with variance (traditional)
            # Primary variance -> cross-check with kurtosis (higher-order)
            if statistic == 'skewness':
                secondary_stat = 'kurtosis'
            elif statistic == 'kurtosis':
                secondary_stat = 'skewness'
            elif statistic in ('mean', 'median'):
                secondary_stat = 'variance'
            elif statistic == 'variance':
                secondary_stat = 'kurtosis'
            else:
                secondary_stat = 'mean'
        self.secondary_stat: str = secondary_stat
        self._secondary_stat_fn = _STAT_FUNCS.get(secondary_stat, _nan_mean)

        # Sliding window (stores individual scalar values)
        self._window: deque = deque(maxlen=max_window)

        # Tracking sums for efficient mean (not used for other stats)
        self._total: float = 0.0

        # Drift detection state
        self.drift_detected: bool = False
        self.last_drift_type: int = self.DRIFT_NONE
        self.last_drift_magnitude: float = 0.0

        # Per-call diagnostic state
        self.last_recent_stat: float = 0.0
        self.last_old_stat: float = 0.0
        self.last_p_value: float = 1.0

        # Hoeffding threshold for the primary comparison
        self.last_hoeffding_threshold: float = 0.0

        # Drift history for severity assessment
        self._drift_history: deque = deque(maxlen=100)
        self._drift_count: int = 0

        # Statistics tracking
        self._update_count: int = 0

    # ------------------------------------------------------------------
    # Core: update and drift check
    # ------------------------------------------------------------------

    def update(self, value: float) -> bool:
        """Add a value to the stream and check for drift.

        Args:
            value: Scalar meta-metric value (e.g., null_rate=0.02)

        Returns:
            True if drift is detected, False otherwise.

        Drift detection logic:
            1. Append value to sliding window
            2. If window < min_window_size: return False
            3. Split window into recent (W1) and old (W0) sub-windows
            4. Compute primary statistic on both sub-windows
            5. Compute secondary statistic on both sub-windows
            6. If use_secondary_check=True: require BOTH to indicate drift
            7. If drift: shrink window to W1, record event, return True
        """
        self._window.append(value)
        self._total += value
        self._update_count += 1
        self.drift_detected = False
        self.last_drift_type = self.DRIFT_NONE

        # Enforce max_window (deque auto-trims)
        n = len(self._window)

        # Need enough data before checking
        if n < self.min_window_size:
            return False

        # Split into recent and old sub-windows
        window_size = min(n // 4, 100)
        recent = list(self._window)[-window_size:]
        old = list(self._window)[:-window_size]

        if len(old) < 10 or len(recent) < 10:
            return False

        # ---- Primary statistic check ----
        primary_recent = self._stat_fn(recent)
        primary_old = self._stat_fn(old)

        self.last_recent_stat = primary_recent
        self.last_old_stat = primary_old

        # Compute thresholds
        hoeffding_thresh = self._hoeffding_threshold(n)

        # For 'mean' statistic: use Welch's t-test for statistical rigor
        # For 'variance' statistic: use Levene's test (Brown-Forsythe variant)
        # For other statistics: use relative difference + Hoeffding bound
        if self.statistic == 'mean':
            t_stat, p_val = _t_test_statistic(recent, old)
            self.last_p_value = p_val
            primary_drift = p_val < self.delta
            self.last_hoeffding_threshold = hoeffding_thresh
            drift_magnitude = abs(primary_recent - primary_old)
        elif self.statistic == 'variance':
            # Levene's test for variance equality (per KAIS 2025)
            levene_stat, levene_p = _levene_statistic(recent, old)
            self.last_p_value = levene_p
            primary_drift = levene_p < self.delta
            self.last_hoeffding_threshold = hoeffding_thresh
            # Magnitude: relative difference in variance
            var_old = primary_old
            var_recent = primary_recent
            denom = abs(var_old) + 1e-10
            drift_magnitude = abs(var_recent - var_old) / denom
        else:
            # Relative difference with Hoeffding guard
            denom = (abs(primary_old) + 1e-10)
            rel_diff = abs(primary_recent - primary_old) / denom
            primary_drift = rel_diff > hoeffding_thresh
            self.last_hoeffding_threshold = hoeffding_thresh
            drift_magnitude = rel_diff

        # ---- Secondary statistic check (optional) ----
        secondary_ok = True
        if self.use_secondary_check and primary_drift:
            secondary_recent = self._secondary_stat_fn(recent)
            secondary_old = self._secondary_stat_fn(old)
            sec_denom = (abs(secondary_old) + 1e-10)
            sec_rel_diff = abs(secondary_recent - secondary_old) / sec_denom
            secondary_ok = sec_rel_diff > hoeffding_thresh

        # ---- Final drift decision ----
        if primary_drift and secondary_ok:
            self.drift_detected = True
            self.last_drift_magnitude = float(drift_magnitude)

            # Classify drift type
            if self.statistic == 'mean':
                self.last_drift_type = self.DRIFT_MEAN_SHIFT
            elif self.statistic in ('variance', 'std'):
                self.last_drift_type = self.DRIFT_VARIANCE_SHIFT
            elif self.statistic == 'skewness':
                self.last_drift_type = self.DRIFT_SKEWNESS_SHIFT
            elif self.statistic == 'kurtosis':
                self.last_drift_type = self.DRIFT_KURTOSIS_SHIFT
            else:
                self.last_drift_type = self.DRIFT_MEAN_SHIFT

            # Shrink window: keep only the recent half (W1 becomes new window)
            cut = len(self._window) // 2
            new_window_vals = list(self._window)[cut:]
            self._window.clear()
            self._window.extend(new_window_vals)
            self._total = sum(new_window_vals)

            # Record drift event
            self._drift_count += 1
            self._drift_history.append({
                'drift_type': self.last_drift_type,
                'magnitude': self.last_drift_magnitude,
                'update_count': self._update_count,
                'window_size': len(self._window),
                'statistic': self.statistic,
                'recent_stat': self.last_recent_stat,
                'old_stat': self.last_old_stat,
            })

            return True

        return False

    def _hoeffding_threshold(self, n: int) -> float:
        """Compute Hoeffding bound threshold.

        Hoeffding inequality: P(|X_n - E[X]| > epsilon) <= 2 * exp(-2 * n * epsilon^2)
        Solving for epsilon: epsilon = sqrt((1/(2*n)) * ln(2/delta))
        """
        if n < 1:
            return float('inf')
        return math.sqrt((1.0 / (2.0 * n)) * math.log(2.0 / max(self.delta, 1e-10)))

    # ------------------------------------------------------------------
    # Drift type helpers
    # ------------------------------------------------------------------

    @staticmethod
    def drift_type_name(drift_type: int) -> str:
        """Human-readable name for drift type constant."""
        names = {
            ADWIN_U.DRIFT_NONE: "none",
            ADWIN_U.DRIFT_MEAN_SHIFT: "mean_shift",
            ADWIN_U.DRIFT_VARIANCE_SHIFT: "variance_shift",
            ADWIN_U.DRIFT_SKEWNESS_SHIFT: "skewness_shift",
            ADWIN_U.DRIFT_KURTOSIS_SHIFT: "kurtosis_shift",
        }
        return names.get(drift_type, "unknown")

    def get_last_drift_description(self) -> str:
        """Human-readable description of the last detected drift."""
        if self.last_drift_type == self.DRIFT_NONE:
            return "No drift"
        name = self.drift_type_name(self.last_drift_type)
        return (
            f"Drift type={name} | "
            f"stat={self.statistic} | "
            f"recent={self.last_recent_stat:.4f} | "
            f"old={self.last_old_stat:.4f} | "
            f"mag={self.last_drift_magnitude:.4f}"
        )

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the detector to initial state."""
        self._window.clear()
        self._total = 0.0
        self.drift_detected = False
        self.last_drift_type = self.DRIFT_NONE
        self.last_drift_magnitude = 0.0
        self.last_recent_stat = 0.0
        self.last_old_stat = 0.0
        self.last_p_value = 1.0
        self.last_hoeffding_threshold = 0.0
        self._drift_count = 0
        self._drift_history.clear()
        self._update_count = 0

    def get_window_size(self) -> int:
        """Return current window size."""
        return len(self._window)

    def get_window(self) -> list:
        """Return a copy of the current window."""
        return list(self._window)

    def get_drift_count(self) -> int:
        """Return total number of drift events detected."""
        return self._drift_count

    def get_drift_history(self) -> list:
        """Return list of recent drift events."""
        return list(self._drift_history)

    def get_recent_drift_rate(self, window_updates: int = 100) -> float:
        """Compute drift rate over recent updates.

        Args:
            window_updates: Number of recent updates to consider.

        Returns:
            Drift events per update (0.0 to 1.0).
        """
        if self._update_count == 0:
            return 0.0
        return self._drift_count / min(self._update_count, window_updates)

    # ------------------------------------------------------------------
    # Serialization (for Flink state checkpointing)
    # ------------------------------------------------------------------

    def get_state_dict(self) -> dict:
        """Get serializable state for Flink checkpointing."""
        return {
            'delta': self.delta,
            'max_window': self.max_window,
            'statistic': self.statistic,
            'min_window_size': self.min_window_size,
            'secondary_stat': self.secondary_stat,
            'use_secondary_check': self.use_secondary_check,
            '_window': list(self._window),
            '_total': self._total,
            'drift_detected': self.drift_detected,
            'last_drift_type': self.last_drift_type,
            'last_drift_magnitude': self.last_drift_magnitude,
            'last_recent_stat': self.last_recent_stat,
            'last_old_stat': self.last_old_stat,
            'last_p_value': self.last_p_value,
            'last_hoeffding_threshold': self.last_hoeffding_threshold,
            '_drift_count': self._drift_count,
            '_drift_history': list(self._drift_history),
            '_update_count': self._update_count,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore state from Flink checkpoint."""
        self.delta = float(state.get('delta', self.delta))
        self.max_window = int(state.get('max_window', self.max_window))
        self.statistic = str(state.get('statistic', self.statistic))
        self.min_window_size = int(state.get('min_window_size', self.min_window_size))
        self.secondary_stat = str(state.get('secondary_stat', self.secondary_stat))
        self.use_secondary_check = bool(state.get('use_secondary_check', self.use_secondary_check))

        self._window = deque(state.get('_window', []), maxlen=self.max_window)
        self._total = float(state.get('_total', 0.0))
        self.drift_detected = bool(state.get('drift_detected', False))
        self.last_drift_type = int(state.get('last_drift_type', self.DRIFT_NONE))
        self.last_drift_magnitude = float(state.get('last_drift_magnitude', 0.0))
        self.last_recent_stat = float(state.get('last_recent_stat', 0.0))
        self.last_old_stat = float(state.get('last_old_stat', 0.0))
        self.last_p_value = float(state.get('last_p_value', 1.0))
        self.last_hoeffding_threshold = float(state.get('last_hoeffding_threshold', 0.0))
        self._drift_count = int(state.get('_drift_count', 0))
        self._drift_history = deque(state.get('_drift_history', []), maxlen=100)
        self._update_count = int(state.get('_update_count', 0))

        # Re-bind function references
        self._stat_fn = _STAT_FUNCS.get(self.statistic, _nan_mean)
        self._secondary_stat_fn = _STAT_FUNCS.get(self.secondary_stat, _nan_mean)

    # ------------------------------------------------------------------
    # Convenience factories
    # ------------------------------------------------------------------

    @classmethod
    def for_metric(
        cls,
        metric_name: str,
        delta: Optional[float] = None,
        use_secondary_check: bool = False,
    ) -> "ADWIN_U":
        """Factory: create ADWIN_U configured for a specific CA-DQStream metric.

        Args:
            metric_name: One of the 6 meta-aggregator metrics.
            delta: Override sensitivity (default per metric).
            use_secondary_check: Require secondary statistic confirmation.

        Returns:
            Pre-configured ADWIN_U instance.

        Example:
            det = ADWIN_U.for_metric('null_rate', use_secondary_check=True)
            if det.update(0.02):
                print("NULL rate drift detected")
        """
        stat = DEFAULT_STATISTIC_PER_METRIC.get(metric_name, 'mean')

        # Default delta per metric (from adwin_multi_instance.py delta_config)
        _delta_defaults = {
            'volume': 0.005,
            'null_rate': 0.001,
            'violation_rate': 0.005,
            'anomaly_rate': 0.005,
            'avg_anomaly_score': 0.005,
            'delta_score': 0.005,
        }

        final_delta = delta if delta is not None else _delta_defaults.get(metric_name, 0.002)

        # Sensitivity presets based on delta
        if final_delta <= 0.0005:
            sensitivity: Optional[str] = 'high'
        elif final_delta <= 0.001:
            sensitivity = 'critical'
        elif final_delta <= 0.002:
            sensitivity = 'medium'
        else:
            sensitivity = 'low'

        return cls(
            delta=final_delta,
            statistic=stat,
            sensitivity=sensitivity,
            use_secondary_check=use_secondary_check,
        )

    def __repr__(self) -> str:
        return (
            f"ADWIN_U(statistic={self.statistic}, "
            f"delta={self.delta}, "
            f"window={len(self._window)}/{self.max_window}, "
            f"drifts={self._drift_count}, "
            f"updates={self._update_count})"
        )


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    'ADWIN_U',
    'ADWIN_U_detector',   # Alias for backward compatibility
    'DEFAULT_STATISTIC_PER_METRIC',
    'STAT_NAMES',
    'DRIFT_MEAN_SHIFT',
    'DRIFT_VARIANCE_SHIFT',
    'DRIFT_SKEWNESS_SHIFT',
    'DRIFT_KURTOSIS_SHIFT',
    'DRIFT_NONE',
]

# Backward-compatible alias
ADWIN_U_detector = ADWIN_U

# Top-level drift type constants for direct import
DRIFT_NONE = ADWIN_U.DRIFT_NONE
DRIFT_MEAN_SHIFT = ADWIN_U.DRIFT_MEAN_SHIFT
DRIFT_VARIANCE_SHIFT = ADWIN_U.DRIFT_VARIANCE_SHIFT
DRIFT_SKEWNESS_SHIFT = ADWIN_U.DRIFT_SKEWNESS_SHIFT
DRIFT_KURTOSIS_SHIFT = ADWIN_U.DRIFT_KURTOSIS_SHIFT
