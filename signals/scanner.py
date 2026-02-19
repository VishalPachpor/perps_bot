"""
Signals Adapter — wraps the 6-layer signal stack for perps context.

The signal layers themselves (MTF, OFI, CVD, VWAP, microstructure, funding)
are identical to the Polymarket bot — they read from order-flow data which
is native to perps markets.

This adapter:
  1. Bridges the perps buffer (data/buffer.py) to signal layer expectations
  2. Filters by vol regime BEFORE running expensive signals
  3. Checks volatility expansion (only enter expanding vol)
  4. Returns a unified scan result dict
"""
import time
import numpy as np
from loguru import logger

from config import (
    SYMBOLS, MIN_SCORE, MIN_EDGE_BY_REGIME,
    ATR_WINDOW_SECONDS,
)
from data.buffer import buffer
from execution.fee_model import compute_edge, min_edge_for_regime


def compute_atr(symbol: str, window_seconds: int = ATR_WINDOW_SECONDS) -> float:
    """
    Compute Average True Range from recent trades.
    Used for dynamic SL distance calculation.
    """
    trades = list(buffer.get_trades(symbol))
    if not trades:
        return 0.0

    now_ms   = time.time() * 1000
    cutoff   = now_ms - (window_seconds * 1000)
    recent   = [t for t in trades if t.get('time', 0) > cutoff]

    if len(recent) < 10:
        return 0.0

    prices = [t['price'] for t in recent]
    highs  = [max(prices[i:i+10]) for i in range(0, len(prices)-10, 5)]
    lows   = [min(prices[i:i+10]) for i in range(0, len(prices)-10, 5)]
    ranges = [h - l for h, l in zip(highs, lows)]
    return float(np.mean(ranges)) if ranges else 0.0


def is_volatility_expanding(symbol: str, short_sec: int = 60, long_sec: int = 300) -> bool:
    """
    Volatility expansion filter — only take trades when vol is increasing.
    Prevents entries in flat/chop regimes.

    Returns True if short-window vol > long-window vol (expanding).
    """
    trades = list(buffer.get_trades(symbol))
    if not trades:
        return False

    now_ms = time.time() * 1000

    def _vol(secs: int) -> float:
        cutoff  = now_ms - (secs * 1000)
        prices  = [t['price'] for t in trades if t.get('time', 0) > cutoff]
        if len(prices) < 5:
            return 0.0
        returns = np.diff(np.log(np.array(prices, dtype=float) + 1e-9))
        return float(np.std(returns))

    short_vol = _vol(short_sec)
    long_vol  = _vol(long_sec)

    if long_vol < 1e-9:
        return False
    return (short_vol / long_vol) > 1.05  # 5% expansion threshold


def _vol_ratio(symbol: str, short_sec: int = 60, long_sec: int = 300) -> float:
    """
    Continuous volatility expansion ratio.
    Returns short_vol / long_vol as a float (>1 = expanding, <1 = contracting).
    Used by adaptive weight engine for continuous factor logging.
    """
    trades = list(buffer.get_trades(symbol))
    if not trades:
        return 0.0

    now_ms = time.time() * 1000

    def _vol(secs: int) -> float:
        cutoff = now_ms - (secs * 1000)
        prices = [t['price'] for t in trades if t.get('time', 0) > cutoff]
        if len(prices) < 5:
            return 0.0
        returns = np.diff(np.log(np.array(prices, dtype=float) + 1e-9))
        return float(np.std(returns))

    short_vol = _vol(short_sec)
    long_vol = _vol(long_sec)

    if long_vol < 1e-9:
        return 0.0
    ratio = short_vol / long_vol
    # Squash into 0-1 range: ratio of 2.0 → ~0.96, ratio of 0.5 → ~0.46
    return float(np.tanh(ratio))


def is_liquidity_stable(symbol: str, instability_threshold: float = 0.6) -> bool:
    """
    Liquidity stability check — skip trading when depth is flickering.
    Detects unstable order books where depth appears/disappears rapidly.

    Returns False if depth changed >60% in the last second.
    """
    snapshots = list(buffer.get_book_snapshots(symbol))
    if len(snapshots) < 3:
        return True  # not enough data — assume stable

    now = time.time()
    recent = [s for s in snapshots if now - s['time'] < 2.0]
    if len(recent) < 2:
        return True

    depths = [(s['bid_depth'] + s['ask_depth']) for s in recent]
    if depths[0] < 1e-6:
        return True

    max_change = max(abs(depths[i] - depths[i-1]) / max(depths[i-1], 1e-6)
                     for i in range(1, len(depths)))
    if max_change > instability_threshold:
        logger.debug(f'[SIGNALS] {symbol} unstable liquidity: {max_change:.0%} depth swing')
        return False
    return True


from signals.adaptive_weights import load_weights

def run_scan(symbol: str, regime: str = 'NORMAL') -> dict:
    """
    Run the full signal scan for a single perps symbol.
    """
    t0 = time.time()

    # ── GATE 1: Extreme regime — no trading ─────────────────────────
    min_edge = min_edge_for_regime(regime)
    if min_edge is None:
        return _skip(symbol, 'EXTREME regime — no trading', t0)

    # ── GATE 2: Buffer readiness ─────────────────────────────────────
    ready, reason = buffer.is_ready(symbol)
    if not ready:
        return _skip(symbol, f'Buffer not ready: {reason}', t0)

    # ── GATE 3: Volatility expansion filter ──────────────────────────
    vol_expanding = is_volatility_expanding(symbol)

    # ── GATE 4: Liquidity stability ──────────────────────────────────
    if not is_liquidity_stable(symbol):
        return _skip(symbol, 'Unstable liquidity — skipping', t0)

    # ── ATR for position sizing ──────────────────────────────────────
    atr = compute_atr(symbol)

    # ── Signal Layers ────────────────────────────────────────────────
    try:
        from perp_signals.layer1_mtf import get_mtf_bias
        from perp_signals.layer2_orderflow import get_orderflow_signals
        from perp_signals.layer3_correlation import compute_correlation_signal
        from perp_signals.layer5_microstructure import get_microstructure_signals
        from perp_signals.layer6_cvd import compute_cvd
        from perp_signals.session_gate import get_session

        mtf  = get_mtf_bias(f'{symbol}USDT')
        of   = get_orderflow_signals(symbol)
        corr = compute_correlation_signal(f'{symbol}USDT')
        ms   = get_microstructure_signals(symbol)
        fund = buffer.get_funding(symbol)
        session = get_session()

        # Cap to last 2000 trades — enough for 15-min CVD window,
        # avoids iterating full 10k-entry deque for BTC every scan cycle.
        trades_raw = list(buffer.get_trades(symbol))[-2000:]
        cvd = compute_cvd(trades_raw, window_seconds=900)

    except ImportError as e:
        logger.warning(f'[SIGNALS] Signal layer import failed: {e}')
        return _skip(symbol, f'Signal layer unavailable: {e}', t0)
    except Exception as e:
        logger.warning(f'[SIGNALS] Signal layer error: {e}')
        return _skip(symbol, f'Signal error: {e}', t0)

    # ── Extract CONTINUOUS Factor Values ────────────────────────────────
    # Quant-grade: continuous floats using soft-scaling to avoid saturation.
    # Formula: x / (abs(x) + k) -> preserves rank at extremes better than tanh.

    # 1. MTF Alignment — use the raw MTF score (typically 0-3 range)
    #    Normalize: WEAK=0.2, STRONG=0.6, ELITE=1.0, plus raw score blend
    mtf_raw_score = mtf.get('score', 0) if mtf else 0
    mtf_strength = mtf.get('strength', 'WEAK') if mtf else 'WEAK'
    mtf_base = {'ELITE': 1.0, 'STRONG': 0.7, 'WEAK': 0.3}.get(mtf_strength, 0.1)
    mtf_val = round(mtf_base * 0.6 + min(1.0, abs(mtf_raw_score) / 3.0) * 0.4, 4)

    # 2. OFI — use raw OFI magnitude
    #    Typical strong OFI is ~500-1000. Saturation at k=1000.
    ofi_raw = of.get('ofi', 0) if of else 0
    k_ofi = 1000.0
    ofi_val = round(abs(ofi_raw) / (abs(ofi_raw) + k_ofi), 4)

    # 3. CVD — use raw CVD value magnitude
    #    Typical strong CVD delta 15m is ~50k-100k. k=100000.
    cvd_raw = cvd.get('cvd', 0) if cvd else 0
    k_cvd = 100000.0
    cvd_val = round(abs(cvd_raw) / (abs(cvd_raw) + k_cvd), 4)

    # 4. Sweep — use sweep magnitude
    #    Typical sweep is 50-200k? Let's say k=50000 represents a solid sweep.
    sweep_mag = ms.get('sweep_magnitude', 0.0) if ms else 0.0
    k_sweep = 50000.0
    sweep_val = round(sweep_mag / (sweep_mag + k_sweep), 4) if sweep_mag > 0 else 0.0

    # 5. Liquidity — keep existing logic but ensure no negative
    state = buffer.snapshot(symbol)
    bid_depth = state.get('bid_depth', 0)
    ask_depth = state.get('ask_depth', 0)
    total_depth = bid_depth + ask_depth
    spread = state.get('spread', 0.0)
    
    # Depth score: k=100k USD depth
    k_depth = 100000.0
    depth_score = total_depth / (total_depth + k_depth) if total_depth > 0 else 0.0
    
    # Spread score: 1.0 at 0bps, 0.0 at 10bps
    spread_score = max(0.0, 1.0 - (spread * 10000 / 10)) 
    liquidity_val = round((depth_score * 0.6 + spread_score * 0.4), 4)

    # 6. Vol Expansion — use actual ratio
    short_vol_val = _vol_ratio(symbol)
    vol_val = round(short_vol_val, 4)

    # 7. Correlation — use raw divergence magnitude
    #    Typical divergence 0.001 - 0.005? k=0.005
    corr_div = corr.get('divergence', 0) if corr else 0
    k_corr = 0.005
    corr_val = round(abs(corr_div) / (abs(corr_div) + k_corr), 4)

    # 8. Signal Strength — composite
    sig_str_val = 0.0
    if ms and ms.get('in_lvn'):
        sig_str_val += 0.35
    if ms and ms.get('vwap_active'):
        sig_str_val += 0.35
    fund_rate = fund.get('rate', 0) if isinstance(fund, dict) else 0
    if abs(fund_rate) > 0.0001:
        sig_str_val += 0.30
    sig_str_val = round(min(1.0, sig_str_val), 4)

    # 9. Execution Quality — spread tightness
    exec_val = round(spread_score, 4)

    # ── Adaptive Scoring ─────────────────────────────────────────────
    weights = load_weights()
    
    score_sum = 0.0
    score_sum += mtf_val * weights.get('mtf_alignment', 0)
    score_sum += ofi_val * weights.get('ofi', 0)
    score_sum += cvd_val * weights.get('cvd', 0)
    score_sum += sweep_val * weights.get('sweep', 0)
    score_sum += liquidity_val * weights.get('liquidity', 0)
    score_sum += vol_val * weights.get('vol_expansion', 0)
    score_sum += corr_val * weights.get('correlation', 0)
    score_sum += sig_str_val * weights.get('signal_strength', 0)
    score_sum += exec_val * weights.get('execution_quality', 0)

    # Normalize score to 0-100 scale (or 0-11 to match old system for compatibility)
    # Old system max score was ~11. 
    # New weighted sum max is ~1.0 (if all weights sum to 1.0 and all vals are 1.0)
    # So we multiply by 100 to get TQS (0-100) or by 11 to match old.
    # The user manual says "Signal Strength = aggregator_score / 11", so let's aim for TQS 0-100.
    # User's request: "Trade Quality Score (TQS) = 0 → 100".
    # BUT the old code expects a score (integer-ish) to compare with MIN_SCORE (7).
    # If MIN_SCORE is 7, and max old score was 11, that's ~63/100.
    # Let's return TQS (0-100) but also a 'score' compatible with 0-11 range for legacy checks if needed.
    
    tqs = score_sum * 100
    legacy_score = score_sum * 11  # Rough conversion for existing thresholds

    # ── Raw Factors Bundle for Logging ───────────────────────────────
    raw_factors = {
        "mtf_alignment": mtf_val,
        "ofi": ofi_val,
        "cvd": cvd_val,
        "sweep": sweep_val,
        "liquidity": liquidity_val,
        "vol_expansion": vol_val,
        "correlation": corr_val,
        "signal_strength": sig_str_val,
        "execution_quality": exec_val,
    }

    # ── Signal Layers Data for Dashboard ─────────────────────────────
    signal_data = {
        'mtf_score':  round(mtf.get('score', 0), 2) if mtf else 0,
        'mtf_bias':   mtf.get('direction', 'MIXED') if mtf else 'MIXED',
        'ofi_score':  round(of.get('ofi', 0), 3) if of else 0,
        'corr_score': round(corr.get('divergence', 0), 4) if corr else 0,
        'fund_score': round(fund.get('rate', 0) * 10000, 2) if isinstance(fund, dict) else 0,
        'vp_score':   1 if (ms and ms.get('in_lvn')) else 0,
        'cvd_score':  round(cvd.get('cvd', 0), 2) if cvd else 0,
        'tqs': round(tqs, 1)  # Add TQS for visibility
    }

    # ── Direction Logic ──────────────────────────────────────────────
    mtf_bull = 'BULL' in mtf.get('direction', '')
    ofi_bull = of.get('direction') == 'bull'
    direction = 'long' if (mtf_bull and ofi_bull) else 'short' if (not mtf_bull and not ofi_bull) else (
        'long' if mtf_bull else 'short'
    )

    # ── Quality Gate (Adaptive Thresholding — Prop Desk Upgrade) ───────
    # Rolling 70th percentile with a floor calibrated for quiet regimes.
    # Floor = 20 (was 25) — at avg TQS 18.6, floor 25 blocks everything.
    if not hasattr(run_scan, '_tqs_history'):
        run_scan._tqs_history = []

    run_scan._tqs_history.append(tqs)
    if len(run_scan._tqs_history) > 1000:
        run_scan._tqs_history = run_scan._tqs_history[-1000:]

    if len(run_scan._tqs_history) >= 50:
        MIN_TQS = max(20.0, float(np.percentile(run_scan._tqs_history, 70)))
    else:
        MIN_TQS = 30.0  # Fallback until enough history (was 35)

    # Fix 3: Factor visibility — log per-factor contributions each scan
    logger.debug(
        f"[FACTORS] {symbol}: mtf={mtf_val:.2f} ofi={ofi_val:.2f} cvd={cvd_val:.2f} "
        f"sweep={sweep_val:.2f} liq={liquidity_val:.2f} vol={vol_val:.2f} "
        f"corr={corr_val:.2f} sig={sig_str_val:.2f} exec={exec_val:.2f} → TQS={tqs:.1f}/{MIN_TQS:.1f}"
    )

    # Fix 2: Use <= so exact threshold ties trigger (not silently skipped)
    if tqs <= MIN_TQS:
        return {
            'trade': False, 'symbol': symbol, 'direction': direction,
            'score': f'{legacy_score:.1f}',
            'reason': f'TQS {tqs:.1f} ≤ {MIN_TQS:.1f} (Quality Low)',
            'vol_expanding': vol_expanding, 'atr': atr,
            'cycle_ms': round((time.time() - t0) * 1000, 1),
            'raw_factors': raw_factors,
            **signal_data,
        }

    # ── Edge Calculation ─────────────────────────────────────────────
    # Use TQS to boost probability
    mark = buffer.get_mark_price(symbol)
    
    # Probability Mapping: TQS 35 -> 51%, TQS 100 -> 75%
    # Base 0.5. Add up to 0.25 based on TQS.
    # prob = 0.5 + (tqs / 400) was for TQS 50+. 
    # Let's adjust: prob = 0.5 + (tqs / 200) -> TQS 35 = 0.675 (too high?)
    # Let's stick to conservative: prob = 0.5 + (tqs / 400)
    # TQS 35 -> 0.5 + 0.0875 = 0.58. Good.
    prob = 0.5 + (tqs / 400.0)
    my_prob = 0.5 + (tqs / 500) # Conservative: TQS 100 = 0.70 prob.

    edge = compute_edge(my_prob, mark, order_type='taker', spread=spread, regime=regime)

    if edge['net'] < min_edge:
        return _skip(symbol, f'Edge {edge["net"]:.4f} < threshold {min_edge:.4f}', t0)

    # ── Determine OFI/CVD trend for exit monitoring ──────────────────
    ofi_direction = of.get('direction', 'neutral') 
    cvd_trend     = cvd.get('trend', 'neutral')        

    return {
        'trade': True,
        'symbol': symbol,
        'direction': direction,
        'score': f'{tqs:.0f}',  # Return TQS as score for display
        'legacy_score': legacy_score,
        'reason': f'TQS {tqs:.0f}/100, edge {edge["net"]:.4f}',
        'net_edge': edge['net'],
        'mark_price': mark,
        'spread': spread,
        'atr': round(atr, 4),
        'vol_expanding': vol_expanding,
        'regime': regime,
        'session': session.get('name', ''),
        'session_mult': session.get('size_mult', 1.0),
        'funding_rate': fund.get('rate', 0.0),
        'ofi_direction': ofi_direction,
        'cvd_trend': cvd_trend,
        'sweep_depth': ms.get('sweep_magnitude', 0.0) if ms.get('sweep_detected') else 0.0,
        'cycle_ms': round((time.time() - t0) * 1000, 1),
        'raw_factors': raw_factors,  # <--- CRITICAL for logging
        **signal_data,
    }

def _skip(symbol: str, reason: str, t0: float) -> dict:
    return {
        'trade': False, 'symbol': symbol, 'direction': 'none',
        'score': '0', 'reason': reason, 'atr': 0.0,
        'vol_expanding': False,
        'cycle_ms': round((time.time() - t0) * 1000, 1),
        'raw_factors': {},
    }
