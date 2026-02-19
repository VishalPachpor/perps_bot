"""
Quant-Grade Adaptive Weight Engine

Computes Factor Quality Score (FQS) for each signal factor:
    FQS = (Mean R at extremes / Std Dev of R) - Downside Penalty

This mirrors Sharpe-ratio logic applied to features, not portfolios.
Optimizes for CONSISTENT edge, not coincidental correlation.

Learning pipeline:
    1. Bin factors into quintiles (edge lives at extremes)
    2. Compute edge stability per bin (Sharpe-style)
    3. Penalize downside risk (large losses when factor is "strong")
    4. Scale by sample confidence
    5. Smooth weight transitions (EMA)
"""
import json
import os
import pandas as pd
import numpy as np
from loguru import logger

# ── Configuration ─────────────────────────────────────────────────────
WEIGHTS_FILE = "data/adaptive_weights.json"
TRADE_LOG = "data/trade_features.csv"
WINDOW = 300          # Rolling window for learning
MIN_TRADES = 50       # Minimum trades before adapting
LEARNING_RATE = 0.2   # EMA smoothing alpha
CONFIDENCE_N = 300.0  # Trades needed for full confidence
NUM_BINS = 5          # Quintile bins for factor analysis

# ── Factor Keys ───────────────────────────────────────────────────────
FACTORS = [
    "mtf_alignment",
    "ofi",
    "cvd",
    "sweep",
    "liquidity",
    "vol_expansion",
    "correlation",
    "signal_strength",
    "execution_quality",
]

# Initial defaults (uniform-ish, research-informed priors)
DEFAULT_WEIGHTS = {
    "mtf_alignment":    0.15,
    "ofi":              0.15,
    "cvd":              0.10,
    "sweep":            0.10,
    "liquidity":        0.15,
    "vol_expansion":    0.10,
    "correlation":      0.10,
    "signal_strength":  0.10,
    "execution_quality": 0.05,
}


# ── Load / Save ───────────────────────────────────────────────────────
def load_weights() -> dict:
    """Load current weights from JSON or return defaults."""
    if os.path.exists(WEIGHTS_FILE):
        try:
            with open(WEIGHTS_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"[WEIGHTS] Failed to load: {e}")
    return DEFAULT_WEIGHTS.copy()


def save_weights(weights: dict):
    """Persist updated weights to JSON."""
    os.makedirs(os.path.dirname(WEIGHTS_FILE), exist_ok=True)
    try:
        with open(WEIGHTS_FILE, "w") as f:
            json.dump(weights, f, indent=2)
        logger.info("[WEIGHTS] Saved adaptive weights")
    except Exception as e:
        logger.error(f"[WEIGHTS] Failed to save: {e}")


# ── Core Statistical Functions ────────────────────────────────────────
def _bin_factor(series: pd.Series) -> pd.Series:
    """
    Bin a continuous factor into quintiles (0–4).
    Uses qcut for equal-frequency bins.
    Falls back to simple rank bins if qcut fails (too few unique values).
    """
    try:
        return pd.qcut(series, NUM_BINS, labels=False, duplicates='drop')
    except (ValueError, IndexError):
        # Fallback for low-variance factors (e.g. mostly 0/1)
        try:
            return pd.cut(series, NUM_BINS, labels=False, duplicates='drop')
        except Exception:
            return pd.Series(0, index=series.index)


def _factor_quality_score(df: pd.DataFrame, factor: str) -> float:
    """
    Compute Factor Quality Score (FQS) — Sharpe-style edge metric.

    FQS = max across bins of: (mean_R / std_R) - downside_penalty

    This captures:
      - Edge magnitude (mean R when factor is strong)
      - Edge consistency (low std = reliable)
      - Downside risk (penalize large losses even when factor looks good)

    Prop Desk: Uses recency-weighted means if available.
    """
    col = df[factor]
    if col.std() < 1e-9:
        return 0.0  # No variance = no information

    bins = _bin_factor(col)
    if bins.nunique() < 2:
        return 0.0

    # Prop Desk: Recency-weighted grouped statistics
    has_recency = '_recency_weight' in df.columns
    if has_recency:
        # Weighted mean per bin
        weighted_df = df.assign(_bin=bins)
        mean_r = weighted_df.groupby('_bin').apply(
            lambda g: np.average(g['result_r'], weights=g['_recency_weight'])
            if len(g) > 0 else 0.0
        )
        std_r = weighted_df.groupby('_bin')['result_r'].std().replace(0, np.nan).fillna(1e-6)
    else:
        grouped = df.groupby(bins)["result_r"]
        mean_r = grouped.mean()
        std_r = grouped.std().replace(0, np.nan).fillna(1e-6)

    # Sharpe-style: edge / volatility
    sharpe = mean_r / std_r

    # Downside penalty: worst loss in the top quintile
    penalty = _downside_penalty(df, factor)

    # FQS = best bin Sharpe - penalty
    fqs = sharpe.max() - penalty
    return float(fqs) if np.isfinite(fqs) else 0.0


def _downside_penalty(df: pd.DataFrame, factor: str) -> float:
    """
    Penalize factors that cause large losses when they appear strong.
    Quant Upgrade: Use 5th percentile of losses (tail risk) instead of min().
    """
    threshold = df[factor].quantile(0.8)
    if pd.isna(threshold):
        return 0.0

    strong_trades = df[df[factor] > threshold]["result_r"]

    if len(strong_trades) == 0:
        return 0.0
    
    # Filter for losses only
    losses = strong_trades[strong_trades < 0]
    if len(losses) == 0:
        return 0.0

    # Use 5th percentile of losses (robust tail risk)
    # If fewer than 20 trades, fallback to min()
    if len(losses) >= 20:
        tail_loss = losses.quantile(0.05)
    else:
        tail_loss = losses.min()
        
    return abs(tail_loss)


def _confidence_scale(n: int) -> float:
    """
    Quant Upgrade: Use effective sample size scaling.
    sqrt(N) / 17 gives 1.0 at ~289 trades, but learns faster early on.
    """
    if n < 1: return 0.0
    return min(1.0, np.sqrt(n) / 17.0)


def _normalize_weights(scores: dict) -> dict:
    """Normalize raw scores into weights summing to 1.0."""
    positive = {k: max(0, v) for k, v in scores.items()}
    total = sum(positive.values())
    if total <= 0:
        return {k: 1.0 / len(scores) for k in scores}
    return {k: v / total for k, v in positive.items()}


def _smooth_weights(old: dict, new: dict, alpha: float) -> dict:
    """
    EMA smoothing with dynamic alpha.
    """
    return {
        k: round(alpha * new.get(k, 0) + (1 - alpha) * old.get(k, 0), 5)
        for k in DEFAULT_WEIGHTS.keys()
    }


# ── Main Learning Loop ───────────────────────────────────────────────
def update_weights() -> dict:
    """
    Full adaptive weight update pipeline (Quant-Grade).
    """
    if not os.path.exists(TRADE_LOG):
        logger.info("[WEIGHTS] No trade log — using defaults")
        return load_weights()

    try:
        df = pd.read_csv(TRADE_LOG)
    except Exception as e:
        logger.error(f"[WEIGHTS] Failed to read log: {e}")
        return load_weights()

    # Only completed trades
    df = df.dropna(subset=["result_r"])
    
    # Audit Fix: Minimum sample for stability
    # qcut requires sufficient data to bin correctly
    if len(df) < 80:
        logger.info(f"[WEIGHTS] Insufficient data ({len(df)}/80) — using defaults")
        return load_weights()

    # Rolling window
    df = df.tail(WINDOW)

    # ── Prop Desk: Recency Weighting ─────────────────────────────────
    # Recent trades matter more. Decay old trade influence.
    if 'timestamp' in df.columns:
        try:
            ts = pd.to_datetime(df['timestamp'], errors='coerce')
            now = pd.Timestamp.now()
            hours_ago = (now - ts).dt.total_seconds() / 3600
            # <24h → 2.0x, <7d → 1.0x, >7d → 0.5x
            recency_w = np.where(hours_ago < 24, 2.0,
                        np.where(hours_ago < 168, 1.0, 0.5))
            df = df.copy()
            df['_recency_weight'] = recency_w
        except Exception:
            df = df.copy()
            df['_recency_weight'] = 1.0
    else:
        df = df.copy()
        df['_recency_weight'] = 1.0
    
    # Calc dynamic smoothing alpha based on R-volatility
    # If results are stable, learn faster (0.35). If chaotic, slow down (0.15).
    r_std = df["result_r"].std()
    dynamic_alpha = 0.35 if r_std < 0.5 else 0.15

    # ── Step 1: Compute raw FQS per factor ───────────────────────────
    raw_scores = {}
    for f in FACTORS:
        if f not in df.columns:
            raw_scores[f] = 0.0
            continue
        try:
            raw_scores[f] = _factor_quality_score(df, f)
        except Exception as e:
            logger.warning(f"[WEIGHTS] FQS failed for {f}: {e}")
            raw_scores[f] = 0.0

    # ── Step 2: Confidence scaling ───────────────────────────────────
    confidence = _confidence_scale(len(df))
    scaled = {k: v * confidence for k, v in raw_scores.items()}

    # ── Step 3: Normalize to weights ─────────────────────────────────
    learned = _normalize_weights(scaled)

    # ── Step 4: Smooth with old weights ──────────────────────────────
    old = load_weights()
    # Use dynamic alpha for smoothing
    final = _smooth_weights(old, learned, alpha=dynamic_alpha)

    # Re-normalize after smoothing
    total = sum(final.values())
    if total > 0:
        final = {k: round(v / total, 5) for k, v in final.items()}

    save_weights(final)

    logger.info(
        f"[WEIGHTS] FQS scores: "
        + ", ".join(f"{k}={v:.3f}" for k, v in raw_scores.items())
    )
    logger.info(
        f"[WEIGHTS] New weights (N={len(df)}, conf={confidence:.2f}, α={dynamic_alpha}): "
        + ", ".join(f"{k}={v:.4f}" for k, v in final.items())
    )

    return final
