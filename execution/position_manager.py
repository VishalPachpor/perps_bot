"""
Position Manager — Perps Position Lifecycle Engine.

═══════════════════════════════════════════════════════════════
PURPOSE:
  Manages the full lifecycle of a perpetual futures position:
    OPEN → monitor (mark price) → EXIT (signal or price trigger)

ARCHITECTURE:
  - Standalone module — no dependency on Polymarket executor
  - Integrates with: risk.py (daily PnL), signals (exit triggers)
  - Designed for drop-in use with any perps venue (Hyperliquid, Binance, etc.)

EXIT TRIGGERS (in priority order):
  1. Hard stop loss     — dynamic ATR-based price level
  2. Liquidation guard  — exit at 120% of liquidation distance
  3. OFI flip           — L2 order flow reverses against position
  4. CVD divergence     — CVD diverges from price direction
  5. Time stop          — max hold window exceeded
  6. Take profit        — static TP or trailing stop activation

FAILURE PATTERN TRACKING:
  Every closed position writes a structured failure/success record.
  After 300+ trades this log becomes the primary edge refinement tool.

USAGE:
  from execution.position_manager import position_manager

  # Open a position
  pos_id = position_manager.open_position(
      symbol='ETHUSDT',
      direction='long',
      entry_price=2850.0,
      size_usd=150.0,
      leverage=3,
      atr=8.5,
      regime='NORMAL',
      signal_type='sweep_reversal',
      session='us_core',
      funding_rate=0.00003,
  )

  # On every price tick (call from your main loop):
  exits = position_manager.check_all_exits(
      symbol='ETHUSDT',
      mark_price=2848.0,
      ofi_direction='bear',     # from layer2_orderflow
      cvd_trend='negative',     # from layer6_cvd
      regime='NORMAL',
  )
  for event in exits:
      venue_client.close_position(event['symbol'], event['reason'])

═══════════════════════════════════════════════════════════════
"""
import time
import json
import os
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger

from execution.risk import risk_manager, vol_monitor
from config import MAX_NET_DIRECTIONAL, BANKROLL


# ── Constants ────────────────────────────────────────────────────────
MAX_HOLD_SECONDS        = 600    # 10 min default scalp window
TRAILING_ACTIVATE_PCT   = 0.004  # activate trailing at +0.4% profit
TRAILING_LOCK_PCT       = 0.50   # lock in 50% of peak profit
MIN_RR_RATIO            = 1.5    # minimum reward:risk at entry
MAX_LEVERAGE            = 5      # absolute cap — never exceed
MAX_NET_DIRECTIONAL_EXP = MAX_NET_DIRECTIONAL  # From config.py (25%), was hardcoded 12%
FAILURE_LOG_PATH        = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'failure_patterns.jsonl'
)


# ── Position Dataclass ───────────────────────────────────────────────
@dataclass
class PerpPosition:
    """
    Complete state of a single perpetual futures position.
    Tracks entry, exit triggers, mark price, and realized outcome.
    """
    # Identity
    pos_id:        str
    symbol:        str
    direction:     str            # 'long' | 'short'

    # Entry
    entry_price:   float
    size_usd:      float
    leverage:      int
    entry_time:    float = field(default_factory=time.time)

    # Trigger levels (set at open, updated dynamically)
    sl_price:      float = 0.0   # hard stop loss price
    tp_price:      float = 0.0   # take profit price
    liq_price:     float = 0.0   # estimated liquidation price

    # Trailing stop state
    peak_price:    float = 0.0   # best mark price seen since open
    trailing_active: bool = False
    trailing_stop: float = 0.0   # current trailing stop level

    # Context at entry (for failure pattern log)
    signal_type:   str = ''
    regime:        str = ''
    session:       str = ''
    funding_at_entry: float = 0.0
    atr_at_entry:  float = 0.0
    spread_at_entry: float = 0.0

    # Live state
    mark_price:    float = 0.0
    unrealized_pnl: float = 0.0
    status:        str = 'OPEN'  # 'OPEN' | 'CLOSED'

    # Exit outcome
    exit_price:    float = 0.0
    exit_time:     float = 0.0
    exit_reason:   str = ''
    realized_pnl:  float = 0.0
    slippage:      float = 0.0


# ── Position Manager ─────────────────────────────────────────────────
class PositionManager:
    """
    Manages all open perps positions.
    Handles entry validation, active monitoring, exit logic,
    and failure pattern logging.

    Designed to be venue-agnostic — the caller places the actual
    exchange order, then calls open_position() / close_position().
    """

    def __init__(self, bankroll: float = BANKROLL):
        self.bankroll = bankroll
        self.positions: dict[str, PerpPosition] = {}  # pos_id → PerpPosition
        self._counter = 0

        # Ensure failure log directory exists
        os.makedirs(os.path.dirname(os.path.abspath(FAILURE_LOG_PATH)), exist_ok=True)

    # ── Entry ─────────────────────────────────────────────────────────
    def open_position(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        size_usd: float,
        leverage: int,
        atr: float,
        regime: str = 'NORMAL',
        signal_type: str = '',
        session: str = '',
        funding_rate: float = 0.0,
        spread: float = 0.0,
        sweep_depth: float = 0.0,
    ) -> Optional[str]:
        """
        Register a new open position and compute all trigger levels.

        Args:
            symbol:       e.g. 'ETHUSDT'
            direction:    'long' or 'short'
            entry_price:  fill price (not mark price)
            size_usd:     position size in USD (notional)
            leverage:     effective leverage (1–MAX_LEVERAGE)
            atr:          current ATR from price data (sets SL distance)
            regime:       current volatility regime
            signal_type:  which setup fired (for failure log)
            session:      trading session (for failure log)
            funding_rate: current 8h funding rate (for failure log)
            spread:       bid-ask spread at entry (for failure log)
            sweep_depth:  detected sweep magnitude (for dynamic SL)

        Returns:
            pos_id string if opened, None if blocked by risk checks.
        """
        # ── Guard: leverage cap ──────────────────────────────────────
        leverage = min(leverage, MAX_LEVERAGE)

        # ── Guard: portfolio directional exposure ────────────────────
        allowed, reason = self._check_directional_exposure(direction, size_usd)
        if not allowed:
            logger.warning(f'[PM] Entry blocked — {reason}')
            return None

        # ── Guard: no duplicate symbol ───────────────────────────────
        existing = [p for p in self.positions.values()
                    if p.symbol == symbol and p.status == 'OPEN']
        if existing:
            logger.warning(f'[PM] Already have open {symbol} position — skipping')
            return None

        # ── Compute dynamic SL distance ──────────────────────────────
        sl_dist = self._compute_sl_distance(atr, sweep_depth, regime)

        # ── Compute trigger prices ───────────────────────────────────
        if direction == 'long':
            sl_price  = entry_price - sl_dist
            tp_price  = entry_price + (sl_dist * MIN_RR_RATIO)
            liq_price = entry_price * (1 - (1 / leverage) * 0.85)
        else:
            sl_price  = entry_price + sl_dist
            tp_price  = entry_price - (sl_dist * MIN_RR_RATIO)
            liq_price = entry_price * (1 + (1 / leverage) * 0.85)

        # ── Liquidation guard: skip if SL is deeper than liq ────────
        if direction == 'long' and sl_price < liq_price * 1.20:
            logger.warning(
                f'[PM] SL ({sl_price:.2f}) too close to liquidation '
                f'({liq_price:.2f}) — widening not possible, skip entry'
            )
            return None

        # ── Create position record ───────────────────────────────────
        self._counter += 1
        pos_id = f'{symbol}_{direction}_{self._counter}_{int(time.time())}'

        pos = PerpPosition(
            pos_id=pos_id,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            size_usd=size_usd,
            leverage=leverage,
            sl_price=sl_price,
            tp_price=tp_price,
            liq_price=liq_price,
            peak_price=entry_price,
            signal_type=signal_type,
            regime=regime,
            session=session,
            funding_at_entry=funding_rate,
            atr_at_entry=atr,
            spread_at_entry=spread,
            mark_price=entry_price,
        )

        self.positions[pos_id] = pos

        logger.info(
            f'[PM] OPEN {direction.upper()} {symbol} | '
            f'Entry: {entry_price:.4f} | SL: {sl_price:.4f} | '
            f'TP: {tp_price:.4f} | Liq: {liq_price:.4f} | '
            f'${size_usd:.2f} @ {leverage}x | '
            f'Signal: {signal_type} | Regime: {regime}'
        )

        return pos_id

    # ── Tick Update ───────────────────────────────────────────────────
    def update_mark_price(self, pos_id: str, mark_price: float):
        """
        Feed the latest mark price into an open position.
        Updates unrealized PnL and trailing stop state.
        Call on every price tick from your data feed.
        """
        pos = self.positions.get(pos_id)
        if not pos or pos.status != 'OPEN':
            return

        pos.mark_price = mark_price

        # Unrealized PnL (without fees — approximation)
        if pos.direction == 'long':
            pos.unrealized_pnl = (mark_price - pos.entry_price) / pos.entry_price * pos.size_usd
        else:
            pos.unrealized_pnl = (pos.entry_price - mark_price) / pos.entry_price * pos.size_usd

        # Update peak price for trailing stop
        if pos.direction == 'long':
            pos.peak_price = max(pos.peak_price, mark_price)
        else:
            pos.peak_price = min(pos.peak_price, mark_price)

        # Activate trailing stop if profit threshold reached
        if not pos.trailing_active:
            peak_return = abs(pos.peak_price - pos.entry_price) / pos.entry_price
            if peak_return >= TRAILING_ACTIVATE_PCT:
                pos.trailing_active = True
                self._update_trailing_stop(pos)
                logger.debug(
                    f'[PM] Trailing stop activated for {pos.symbol} — '
                    f'peak: {pos.peak_price:.4f}, trail: {pos.trailing_stop:.4f}'
                )
        elif pos.trailing_active:
            self._update_trailing_stop(pos)

    def _update_trailing_stop(self, pos: PerpPosition):
        """Recalculate trailing stop to lock in TRAILING_LOCK_PCT of peak profit."""
        if pos.direction == 'long':
            locked = pos.entry_price + (pos.peak_price - pos.entry_price) * TRAILING_LOCK_PCT
            pos.trailing_stop = max(pos.trailing_stop, locked)
        else:
            locked = pos.entry_price - (pos.entry_price - pos.peak_price) * TRAILING_LOCK_PCT
            pos.trailing_stop = min(
                pos.trailing_stop if pos.trailing_stop > 0 else float('inf'),
                locked
            )

    # ── Exit Checks ───────────────────────────────────────────────────
    def check_all_exits(
        self,
        symbol: str,
        mark_price: float,
        ofi_direction: str = '',
        cvd_trend: str = '',
        regime: str = '',
        cascade_detected: bool = False,
    ) -> list[dict]:
        """
        Check all open positions for this symbol against exit triggers.
        Returns a list of exit events — caller must execute the close.

        Args:
            symbol:            'ETHUSDT' etc.
            mark_price:        current mark price
            ofi_direction:     current OFI signal ('bull'|'bear'|'neutral')
            cvd_trend:         current CVD trend ('positive'|'negative'|'neutral')
            regime:            current vol regime (for cascade SL widening)
            cascade_detected:  True if liquidation cascade detected

        Returns:
            List of exit dicts, each containing:
              { 'pos_id', 'symbol', 'direction', 'reason', 'mark_price',
                'size_usd', 'unrealized_pnl', 'action': 'CLOSE' }
        """
        exits = []

        for pos_id, pos in list(self.positions.items()):
            if pos.symbol != symbol or pos.status != 'OPEN':
                continue

            # Update mark price first
            self.update_mark_price(pos_id, mark_price)

            # ── EXIT 1: Liquidation guard ────────────────────────────
            if self._check_liquidation_guard(pos):
                exits.append(self._build_exit_event(pos, 'LIQUIDATION_GUARD'))
                continue

            # ── EXIT 2: Hard stop loss ───────────────────────────────
            if self._check_hard_sl(pos, mark_price, cascade_detected):
                reason = 'CASCADE_SL' if cascade_detected else 'HARD_SL'
                exits.append(self._build_exit_event(pos, reason))
                continue

            # ── EXIT 3: Trailing stop ────────────────────────────────
            if pos.trailing_active and self._check_trailing_stop(pos, mark_price):
                exits.append(self._build_exit_event(pos, 'TRAILING_STOP'))
                continue

            # ── EXIT 4: OFI signal flip ──────────────────────────────
            if self._check_ofi_exit(pos, ofi_direction):
                exits.append(self._build_exit_event(pos, 'OFI_FLIP'))
                continue

            # ── EXIT 5: CVD divergence ───────────────────────────────
            if self._check_cvd_exit(pos, cvd_trend):
                exits.append(self._build_exit_event(pos, 'CVD_DIVERGENCE'))
                continue

            # ── EXIT 6: Take profit ──────────────────────────────────
            if self._check_take_profit(pos, mark_price):
                exits.append(self._build_exit_event(pos, 'TAKE_PROFIT'))
                continue

            # ── EXIT 7: Time stop ────────────────────────────────────
            if self._check_time_stop(pos):
                exits.append(self._build_exit_event(pos, 'TIME_STOP'))
                continue

        return exits

    # ── Individual Exit Checks ────────────────────────────────────────
    def _check_liquidation_guard(self, pos: PerpPosition) -> bool:
        """Exit if mark price approaches 120% of liquidation distance."""
        if pos.liq_price == 0:
            return False
        if pos.direction == 'long':
            # Danger zone: within 20% of distance between entry and liq
            buffer = (pos.entry_price - pos.liq_price) * 0.20
            return pos.mark_price <= pos.liq_price + buffer
        else:
            buffer = (pos.liq_price - pos.entry_price) * 0.20
            return pos.mark_price >= pos.liq_price - buffer

    def _check_hard_sl(
        self, pos: PerpPosition, mark_price: float, cascade: bool
    ) -> bool:
        """Check hard stop loss. Widen by 50% if cascade detected."""
        sl = pos.sl_price

        if cascade:
            # During cascade: widen SL to avoid being stopped by the cascade itself
            # but immediately exit if liquidation guard triggers
            dist = abs(pos.entry_price - sl)
            if pos.direction == 'long':
                sl = pos.entry_price - (dist * 1.5)
            else:
                sl = pos.entry_price + (dist * 1.5)

        if pos.direction == 'long':
            return mark_price <= sl
        else:
            return mark_price >= sl

    def _check_trailing_stop(self, pos: PerpPosition, mark_price: float) -> bool:
        """Check if trailing stop has been hit."""
        if not pos.trailing_active or pos.trailing_stop == 0:
            return False
        if pos.direction == 'long':
            return mark_price <= pos.trailing_stop
        else:
            return mark_price >= pos.trailing_stop

    def _check_ofi_exit(self, pos: PerpPosition, ofi_direction: str) -> bool:
        """
        Exit if OFI has flipped against the position AND position is in profit.
        Require profit guard to avoid letting OFI flip stop out small losers
        (those should hit SL instead).
        """
        if not ofi_direction:
            return False
        in_profit = pos.unrealized_pnl > 0

        if pos.direction == 'long' and ofi_direction == 'bear' and in_profit:
            return True
        if pos.direction == 'short' and ofi_direction == 'bull' and in_profit:
            return True
        return False

    def _check_cvd_exit(self, pos: PerpPosition, cvd_trend: str) -> bool:
        """
        Exit on CVD divergence — price moving against CVD direction.
        Only exit if in meaningful profit (avoid noise exits on flat positions).
        """
        if not cvd_trend:
            return False
        min_profit_pct = 0.002  # 0.2% minimum profit before CVD exit
        profit_pct = pos.unrealized_pnl / pos.size_usd

        if profit_pct < min_profit_pct:
            return False

        if pos.direction == 'long' and cvd_trend == 'negative':
            return True
        if pos.direction == 'short' and cvd_trend == 'positive':
            return True
        return False

    def _check_take_profit(self, pos: PerpPosition, mark_price: float) -> bool:
        """Check static TP level."""
        if pos.tp_price == 0:
            return False
        if pos.direction == 'long':
            return mark_price >= pos.tp_price
        else:
            return mark_price <= pos.tp_price

    def _check_time_stop(self, pos: PerpPosition) -> bool:
        """Close position if max hold window exceeded."""
        age = time.time() - pos.entry_time
        return age > MAX_HOLD_SECONDS

    # ── Close Position ────────────────────────────────────────────────
    def close_position(
        self,
        pos_id: str,
        exit_price: float,
        slippage: float = 0.0,
        fee_pct: float = 0.00035,  # Hyperliquid default taker fee
    ) -> Optional[dict]:
        """
        Record a closed position. Called AFTER the exchange close is confirmed.

        Args:
            pos_id:     position ID from open_position()
            exit_price: actual fill price (not mark price)
            slippage:   realized slippage (exit_price vs expected)
            fee_pct:    taker fee rate for this venue

        Returns:
            Failure pattern record dict (also written to JSONL log).
        """
        pos = self.positions.get(pos_id)
        if not pos or pos.status != 'OPEN':
            logger.warning(f'[PM] close_position: {pos_id} not found or already closed')
            return None

        pos.exit_price = exit_price
        pos.exit_time = time.time()
        pos.slippage = slippage
        pos.status = 'CLOSED'

        hold_seconds = pos.exit_time - pos.entry_time

        # Realized PnL (gross)
        if pos.direction == 'long':
            gross_pnl = (exit_price - pos.entry_price) / pos.entry_price * pos.size_usd
        else:
            gross_pnl = (pos.entry_price - exit_price) / pos.entry_price * pos.size_usd

        # Deduct fees (entry + exit round trip)
        fee_cost = pos.size_usd * fee_pct * 2
        pos.realized_pnl = gross_pnl - fee_cost - (slippage * pos.size_usd)

        # Update risk manager daily PnL
        risk_manager.record_outcome(pos.symbol, pos.realized_pnl)

        result = 'WIN' if pos.realized_pnl > 0 else 'LOSS'
        logger.info(
            f'[PM] CLOSE {result} {pos.symbol} {pos.direction.upper()} | '
            f'Entry: {pos.entry_price:.4f} → Exit: {exit_price:.4f} | '
            f'PnL: ${pos.realized_pnl:.2f} | '
            f'Reason: {pos.exit_reason} | '
            f'Hold: {hold_seconds:.0f}s'
        )

        # Write failure pattern record
        record = self._build_failure_record(pos, hold_seconds)
        self._write_failure_record(record)

        return record

    # ── Failure Pattern Tracker ────────────────────────────────────────
    def _build_failure_record(self, pos: PerpPosition, hold_seconds: float) -> dict:
        """Build a structured record for failure pattern analysis."""
        profit_pct = (
            (pos.exit_price - pos.entry_price) / pos.entry_price
            if pos.direction == 'long'
            else (pos.entry_price - pos.exit_price) / pos.entry_price
        )
        return {
            'timestamp': pos.exit_time,
            'pos_id': pos.pos_id,
            'symbol': pos.symbol,
            'direction': pos.direction,
            # Entry context
            'signal_type': pos.signal_type,
            'regime': pos.regime,
            'session': pos.session,
            'funding_at_entry': pos.funding_at_entry,
            'atr_at_entry': pos.atr_at_entry,
            'spread_at_entry': pos.spread_at_entry,
            # Price levels
            'entry_price': pos.entry_price,
            'exit_price': pos.exit_price,
            'sl_price': pos.sl_price,
            'tp_price': pos.tp_price,
            # Outcome
            'exit_reason': pos.exit_reason,
            'hold_seconds': round(hold_seconds, 1),
            'gross_pnl_pct': round(profit_pct, 5),
            'realized_pnl': round(pos.realized_pnl, 4),
            'slippage': round(pos.slippage, 5),
            'result': 'WIN' if pos.realized_pnl > 0 else 'LOSS',
        }

    def _write_failure_record(self, record: dict):
        """Append failure/success record to JSONL log."""
        try:
            with open(FAILURE_LOG_PATH, 'a') as f:
                f.write(json.dumps(record) + '\n')
        except Exception as e:
            logger.warning(f'[PM] Could not write failure record: {e}')

    # ── Portfolio Exposure Guard ──────────────────────────────────────
    def _check_directional_exposure(
        self, direction: str, size_usd: float
    ) -> tuple[bool, str]:
        """
        Enforce max net directional exposure across all open positions.
        e.g. all longs combined < 12% of bankroll.
        """
        open_positions = [p for p in self.positions.values() if p.status == 'OPEN']

        net_long  = sum(p.size_usd for p in open_positions if p.direction == 'long')
        net_short = sum(p.size_usd for p in open_positions if p.direction == 'short')

        max_exposure = self.bankroll * MAX_NET_DIRECTIONAL_EXP

        if direction == 'long' and (net_long + size_usd) > max_exposure:
            return False, (
                f'Net long exposure would be ${net_long + size_usd:.0f} '
                f'> limit ${max_exposure:.0f} (bankroll × {MAX_NET_DIRECTIONAL_EXP:.0%})'
            )
        if direction == 'short' and (net_short + size_usd) > max_exposure:
            return False, (
                f'Net short exposure would be ${net_short + size_usd:.0f} '
                f'> limit ${max_exposure:.0f}'
            )
        return True, 'OK'

    # ── Dynamic SL Distance ───────────────────────────────────────────
    def _compute_sl_distance(
        self, atr: float, sweep_depth: float, regime: str
    ) -> float:
        """
        Compute SL distance from entry price.
        Stops must sit beyond sweep depth and LVN — not at obvious round levels.
        """
        base = atr * 1.5
        if regime == 'HIGH':
            base *= 1.5    # wider stops in high vol
        elif regime == 'LOW':
            base *= 0.8    # tighter in low vol chop

        if sweep_depth > 0:
            base = max(base, sweep_depth * 1.1)  # stop beyond the sweep

        return max(base, atr * 0.5)  # minimum floor

    # ── Build Exit Event ──────────────────────────────────────────────
    def _build_exit_event(self, pos: PerpPosition, reason: str) -> dict:
        """Build a standardized exit event dict for the caller to act on."""
        pos.exit_reason = reason
        logger.info(
            f'[PM] EXIT SIGNAL: {pos.symbol} {pos.direction.upper()} | '
            f'Reason: {reason} | Mark: {pos.mark_price:.4f} | '
            f'uPnL: ${pos.unrealized_pnl:.2f}'
        )
        return {
            'action': 'CLOSE',
            'pos_id': pos.pos_id,
            'symbol': pos.symbol,
            'direction': pos.direction,
            'reason': reason,
            'mark_price': pos.mark_price,
            'size_usd': pos.size_usd,
            'unrealized_pnl': round(pos.unrealized_pnl, 4),
        }

    # ── Status & Analytics ────────────────────────────────────────────
    def get_open_positions(self) -> list[dict]:
        """Return summary of all open positions."""
        return [
            {
                'pos_id': p.pos_id,
                'symbol': p.symbol,
                'direction': p.direction,
                'entry_price': p.entry_price,
                'mark_price': p.mark_price,
                'sl_price': p.sl_price,
                'tp_price': p.tp_price,
                'size_usd': p.size_usd,
                'leverage': p.leverage,
                'unrealized_pnl': round(p.unrealized_pnl, 4),
                'trailing_active': p.trailing_active,
                'trailing_stop': p.trailing_stop,
                'hold_seconds': round(time.time() - p.entry_time, 0),
                'signal_type': p.signal_type,
                'regime': p.regime,
            }
            for p in self.positions.values()
            if p.status == 'OPEN'
        ]

    def get_net_exposure(self) -> dict:
        """Return net directional exposure across all open positions."""
        open_pos = [p for p in self.positions.values() if p.status == 'OPEN']
        net_long  = sum(p.size_usd for p in open_pos if p.direction == 'long')
        net_short = sum(p.size_usd for p in open_pos if p.direction == 'short')
        return {
            'net_long': round(net_long, 2),
            'net_short': round(net_short, 2),
            'net_exposure': round(net_long - net_short, 2),
            'max_allowed': round(self.bankroll * MAX_NET_DIRECTIONAL_EXP, 2),
            'open_count': len(open_pos),
        }

    def load_failure_patterns(self, n: int = 100) -> list[dict]:
        """
        Load last N closed trade records from the failure pattern log.
        Use this to analyze which setups win/lose by regime, session, etc.
        """
        records = []
        try:
            with open(FAILURE_LOG_PATH, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        except FileNotFoundError:
            pass
        return records[-n:]


# ── Sizing Helper ─────────────────────────────────────────────────────
def compute_perp_size(
    bankroll: float,
    risk_pct: float,
    sl_pct: float,
    leverage: int = 3,
    depth_scalar: float = 1.0,
    max_notional_pct: float = 0.15,
) -> float:
    """
    Compute position size using risk-first sizing.

    Based on: dollar_risk = bankroll × risk_pct
              position    = (dollar_risk / sl_pct) × leverage

    Args:
        bankroll:         account size in USD
        risk_pct:         fraction of bankroll to risk (e.g. 0.005 = 0.5%)
        sl_pct:           stop loss distance as fraction of price (e.g. 0.003 = 0.3%)
        leverage:         effective leverage (capped at MAX_LEVERAGE)
        depth_scalar:     scale down in thin books (0.0–1.0)
        max_notional_pct: hard cap on notional as fraction of bankroll

    Returns:
        USD notional size to place.
    """
    if sl_pct <= 0:
        return 0.0

    leverage = min(leverage, MAX_LEVERAGE)
    dollar_risk  = bankroll * risk_pct
    position_usd = (dollar_risk / sl_pct) * leverage
    position_usd *= max(0.0, min(1.0, depth_scalar))
    return round(min(position_usd, bankroll * max_notional_pct), 2)


# ── Global Singleton ──────────────────────────────────────────────────
position_manager = PositionManager()
