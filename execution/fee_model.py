"""
Perps Fee Model — compute net edge after fees and expected slippage.

Key difference from Polymarket:
  Polymarket fees: 1–2% taker
  Perps fees:      0.035–0.055% taker (100× lower)
  → MIN_EDGE threshold must be recalibrated accordingly
"""
from config import TAKER_FEE, MAKER_FEE


def compute_edge(
    my_prob: float,
    market_price: float,
    order_type: str = 'taker',
    spread: float = 0.0,
    regime: str = 'NORMAL',
) -> dict:
    """
    Compute gross and net edge for a perps trade.

    Args:
        my_prob:      our estimated win probability for the direction
        market_price: current mark price (used to estimate mid-market fair value)
        order_type:   'taker' | 'maker'
        spread:       current bid-ask spread (for slippage estimate)
        regime:       volatility regime (slippage scales in HIGH vol)

    Returns:
        {
          'gross': float,      gross edge before costs
          'fee':   float,      one-way fee
          'slippage': float,   expected slippage on exit
          'net':   float,      net edge (what matters)
          'ev':    float,      expected value per $1 risked
        }
    """
    fee = TAKER_FEE if order_type == 'taker' else max(MAKER_FEE, 0.0)

    # Slippage model: 0.5× spread in NORMAL, 2× spread in HIGH regime
    slip_mult = {'LOW': 0.3, 'NORMAL': 0.5, 'HIGH': 2.0, 'EXTREME': 5.0}.get(regime, 0.5)
    expected_slippage = (spread / max(market_price, 1)) * slip_mult

    # Gross edge = our probability estimate vs market-implied cost
    # For perps: market_price ≈ entry cost (we win if price moves in our direction)
    gross = my_prob - 0.5  # edge over random

    # Net edge = gross - round-trip fees - slippage
    round_trip_fee = fee * 2  # entry + exit
    net = gross - round_trip_fee - expected_slippage

    return {
        'gross':     round(gross, 5),
        'fee':       round(fee, 6),
        'slippage':  round(expected_slippage, 6),
        'net':       round(net, 5),
        'ev':        round(net, 5),   # alias for external callers
    }


def min_edge_for_regime(regime: str) -> float | None:
    """
    Return the minimum net edge required to trade in a given regime.
    None = no trading allowed (EXTREME).
    """
    from config import MIN_EDGE_BY_REGIME
    return MIN_EDGE_BY_REGIME.get(regime, 0.0015)
