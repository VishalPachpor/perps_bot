"""
Perps Risk Manager — tightened version of Polymarket risk.py for leveraged trading.

Key changes from Polymarket:
  - MAX_DAILY_LOSS is -$50 (not -$500) for early live phase
  - MAX_OPEN_POSITIONS capped at 2 (correlated assets)
  - Volatility regime blocks trading at EXTREME
  - Portfolio exposure guard replaces simple position count
  - Cascade detection mode added
"""
import time
from loguru import logger

from config import (
    MAX_DAILY_LOSS, MAX_OPEN_POSITIONS,
    COOLDOWN_AFTER_LOSS_SEC, MAX_CORRELATED_EXPOSURE,
    BANKROLL,
)


# ── Volatility Regime Monitor ────────────────────────────────────────
class PerpsVolatilityMonitor:
    """
    Tracks current vol regime based on recent ATR readings.
    Regime affects position sizing and edge thresholds.
    """

    def __init__(self):
        self._regime = 'NORMAL'
        self._atr_history: list[float] = []
        self._max_history = 200

    def update(self, atr: float):
        self._atr_history.append(atr)
        if len(self._atr_history) > self._max_history:
            self._atr_history.pop(0)
        self._regime = self._classify()

    def _classify(self) -> str:
        if len(self._atr_history) < 20:
            return 'NORMAL'
        recent  = self._atr_history[-10:]
        baseline = self._atr_history[:-10]
        if not baseline:
            return 'NORMAL'

        ratio = (sum(recent) / len(recent)) / max(sum(baseline) / len(baseline), 1e-9)

        if ratio > 3.0: return 'EXTREME'
        if ratio > 1.8: return 'HIGH'
        if ratio < 0.6: return 'LOW'
        return 'NORMAL'

    @property
    def regime(self) -> str:
        return self._regime


vol_monitor = PerpsVolatilityMonitor()


# ── Liquidation Cascade Detector ──────────────────────────────────────
class CascadeDetector:
    """
    Detect liquidation cascades: OI drops sharply while price spikes + volume surges.
    When active: widen stops, skip new entries.
    """
    def __init__(self):
        self._oi_history: list[float] = []
        self._price_history: list[float] = []
        self._active = False
        self._active_until = 0.0

    def update(self, oi_usd: float, mark_price: float):
        self._oi_history.append(oi_usd)
        self._price_history.append(mark_price)
        if len(self._oi_history) > 30:
            self._oi_history.pop(0)
            self._price_history.pop(0)

        self._active = self._detect()
        if self._active:
            self._active_until = time.time() + 120  # 2 min cooldown

    def _detect(self) -> bool:
        if len(self._oi_history) < 5:
            return False

        oi_drop = (self._oi_history[-1] - self._oi_history[-4]) / max(self._oi_history[-4], 1e-9)
        p_move  = abs((self._price_history[-1] - self._price_history[-4]) / max(self._price_history[-4], 1e-9))

        return oi_drop < -0.03 and p_move > 0.015  # OI -3% + price +1.5%

    @property
    def is_active(self) -> bool:
        if time.time() < self._active_until:
            return True
        return self._active


cascade_detector = CascadeDetector()


# ── Risk Manager ─────────────────────────────────────────────────────
class PerpsRiskManager:
    """
    Tightened risk manager for perpetual futures trading.
    Gates every trade entry with layered checks.
    """

    def __init__(self):
        self.daily_pnl      = 0.0
        self.daily_trades   = 0
        self.last_loss_time = 0.0
        self.open_positions: dict[str, float] = {}  # symbol → size_usd
        self._trade_results: list[float] = []

    def can_trade(self, symbol: str = '') -> tuple[bool, str]:
        """Run all risk gates. Returns (allowed, reason)."""

        # Gate 1: Daily loss limit
        if self.daily_pnl <= MAX_DAILY_LOSS:
            return False, f'Daily loss limit hit: ${self.daily_pnl:.2f}'

        # Gate 2: Max open positions (correlated assets)
        if len(self.open_positions) >= MAX_OPEN_POSITIONS:
            return False, f'Max {MAX_OPEN_POSITIONS} positions open'

        # Gate 3: Cooldown after a loss
        elapsed = time.time() - self.last_loss_time
        if 0 < self.last_loss_time and elapsed < COOLDOWN_AFTER_LOSS_SEC:
            remaining = int(COOLDOWN_AFTER_LOSS_SEC - elapsed)
            return False, f'Cooldown: {remaining}s remaining'

        # Gate 4: Correlated exposure cap
        total_exposure = sum(self.open_positions.values())
        max_exp = BANKROLL * MAX_CORRELATED_EXPOSURE
        if total_exposure >= max_exp:
            return False, f'Correlated exposure cap: ${total_exposure:.0f} / ${max_exp:.0f}'

        # Gate 5: Extreme vol regime
        if vol_monitor.regime == 'EXTREME':
            return False, 'EXTREME volatility — no new positions'

        # Gate 6: Liquidation cascade active
        if cascade_detector.is_active:
            return False, 'Liquidation cascade detected — no new entries'

        # Gate 7: Duplicate symbol
        if symbol and symbol in self.open_positions:
            return False, f'Already positioned in {symbol}'

        return True, 'OK'

    def record_open(self, symbol: str, size_usd: float):
        self.open_positions[symbol] = size_usd
        self.daily_trades += 1
        logger.info(f'[RISK] Opened {symbol} ${size_usd:.2f} | '
                    f'Exposure: ${sum(self.open_positions.values()):.0f}')

    def record_outcome(self, symbol: str, pnl: float):
        """Call when a position is closed."""
        self.daily_pnl += pnl
        self._trade_results.append(pnl)
        self.open_positions.pop(symbol, None)

        if pnl < 0:
            self.last_loss_time = time.time()
            logger.warning(f'[RISK] Loss ${pnl:.2f} | Daily: ${self.daily_pnl:.2f}')
        else:
            logger.info(f'[RISK] Win  ${pnl:.2f} | Daily: ${self.daily_pnl:.2f}')

    def status(self) -> dict:
        return {
            'daily_pnl': round(self.daily_pnl, 2),
            'daily_trades': self.daily_trades,
            'open_positions': dict(self.open_positions),
            'regime': vol_monitor.regime,
            'cascade': cascade_detector.is_active,
            'blocked': self.daily_pnl <= MAX_DAILY_LOSS,
        }

    def reset_daily(self):
        self.daily_pnl    = 0.0
        self.daily_trades = 0
        self._trade_results.clear()
        logger.info('[RISK] Daily counters reset')


# Global singleton
risk_manager = PerpsRiskManager()
