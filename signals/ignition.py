"""
Momentum Ignition Detector — detects when microstructure pressure cascades.

Layers on top of existing signal factors (OFI, Sweeps, CVD, Vol, Liquidity)
to identify when an imbalance becomes a cascade. When ignition is confirmed,
the execution engine switches from micro-scalp mode to momentum capture mode
with expanded TP, runner logic, and trail stops.

Architecture:
  Layer 1 (existing) → detects pressure (OFI, sweep, CVD)
  Layer 2 (THIS)     → detects cascade (sweep cluster, delta accel, depth collapse)
  Layer 3 (execution)→ adjusts TP/SL/sizing based on ignition regime
"""
from dataclasses import dataclass
from loguru import logger


# ── Ignition Thresholds ──────────────────────────────────────────────
SWEEP_CLUSTER_THRESHOLD   = 0.35   # sweep_val above this = cluster detected
DELTA_ACCEL_THRESHOLD     = 0.40   # CVD factor above this = delta acceleration
DEPTH_COLLAPSE_THRESHOLD  = 0.30   # liquidity_val BELOW this = depth collapse
VELOCITY_SPIKE_THRESHOLD  = 0.55   # vol_expansion factor above this = velocity spike
BTC_IMPULSE_THRESHOLD     = 0.35   # correlation divergence above this = BTC impulse
VOL_EXPANSION_THRESHOLD   = 0.50   # vol factor above this = volatility expanding

IGNITION_MIN_SCORE        = 3      # minimum factors to confirm ignition


@dataclass
class IgnitionResult:
    """Result of ignition detection."""
    is_ignition: bool
    score: int           # 0-6 ignition factor count
    factors: dict        # which factors triggered
    tp_multiplier: float # how much to expand TP (1.0 = normal, 2.5 = full ignition)
    size_multiplier: float # how much to scale position size
    runner_enabled: bool # whether to enable runner logic (partial profit + trail)


def detect_ignition(
    sweep_val: float,
    cvd_val: float,
    liquidity_val: float,
    vol_val: float,
    corr_val: float,
    vol_expanding: bool,
    regime: str,
    symbol: str = '',
) -> IgnitionResult:
    """
    Detect momentum ignition from existing signal factors.
    
    This does NOT replace the existing signal stack — it layers on top.
    All inputs are the same normalized 0-1 factors already computed in scanner.py.
    
    Returns:
        IgnitionResult with score, multipliers, and whether runner is enabled.
    """
    score = 0
    factors = {}

    # 1. Sweep Cluster — large aggressive orders hitting multiple levels
    if sweep_val >= SWEEP_CLUSTER_THRESHOLD:
        score += 1
        factors['sweep_cluster'] = round(sweep_val, 4)

    # 2. Delta Acceleration — CVD surging in one direction
    if cvd_val >= DELTA_ACCEL_THRESHOLD:
        score += 1
        factors['delta_accel'] = round(cvd_val, 4)

    # 3. Depth Collapse — liquidity retreating (book thinning out)
    #    LOW liquidity_val means the book is thin/collapsing
    if liquidity_val < DEPTH_COLLAPSE_THRESHOLD:
        score += 1
        factors['depth_collapse'] = round(liquidity_val, 4)

    # 4. Velocity Spike — short-term vol >> long-term vol
    if vol_val >= VELOCITY_SPIKE_THRESHOLD:
        score += 1
        factors['velocity_spike'] = round(vol_val, 4)

    # 5. BTC Impulse — cross-asset divergence suggesting macro move
    if corr_val >= BTC_IMPULSE_THRESHOLD:
        score += 1
        factors['btc_impulse'] = round(corr_val, 4)

    # 6. Volatility Expansion — ATR expanding, range breaking out
    if vol_expanding:
        score += 1
        factors['vol_expansion'] = True

    is_ignition = score >= IGNITION_MIN_SCORE

    # ── Compute execution adjustments ────────────────────────────────
    if is_ignition:
        # Scale TP expansion based on ignition strength
        # Score 3 → 1.8x TP, Score 4 → 2.2x, Score 5 → 2.6x, Score 6 → 3.0x
        tp_multiplier = 1.0 + (score - 2) * 0.4
        tp_multiplier = min(tp_multiplier, 3.0)

        # Scale position size up slightly in confirmed cascades
        # Score 3 → 1.2x, Score 6 → 1.5x
        size_multiplier = 1.0 + (score - 2) * 0.1
        size_multiplier = min(size_multiplier, 1.5)

        runner_enabled = True

        logger.info(
            f'[IGNITION] 🔥 {symbol} CONFIRMED | score={score}/6 | '
            f'tp_mult={tp_multiplier:.1f}x | size_mult={size_multiplier:.1f}x | '
            f'factors={list(factors.keys())}'
        )
    else:
        tp_multiplier = 1.0
        size_multiplier = 1.0
        runner_enabled = False

        if score >= 2:
            logger.debug(
                f'[IGNITION] {symbol} near-ignition score={score}/6 | '
                f'factors={list(factors.keys())}'
            )

    return IgnitionResult(
        is_ignition=is_ignition,
        score=score,
        factors=factors,
        tp_multiplier=tp_multiplier,
        size_multiplier=size_multiplier,
        runner_enabled=runner_enabled,
    )
