"""
Perps Data Buffer — in-memory ring buffer for mark price, OI, trades, and order book.

Key differences from Polymarket buffer:
  - Stores mark_price (not last price) — used for liquidation checks
  - Stores open_interest (OI) — used for cascade detection
  - Separate trade deques per symbol for signal layers
  - Thread-safe via simple dict + deque (no lock needed for GIL-protected ops)
"""
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from config import BUFFER_SIZE, SYMBOLS


@dataclass
class SymbolState:
    """Current market state for a single perps symbol."""
    symbol: str

    # Mark price (liquidation + position monitoring use this)
    mark_price:   float = 0.0
    last_price:   float = 0.0
    index_price:  float = 0.0
    mark_updated: float = 0.0

    # Order book (top 5 levels)
    best_bid:  float = 0.0
    best_ask:  float = 0.0
    bid_depth: float = 0.0  # USD depth at best bid
    ask_depth: float = 0.0  # USD depth at best ask

    # Funding
    funding_rate:      float = 0.0
    next_funding_time: float = 0.0

    # Open Interest (for cascade detection)
    open_interest:       float = 0.0  # in coin
    open_interest_usd:   float = 0.0
    oi_updated:          float = 0.0

    # Trade history ring buffer
    trades: deque = field(default_factory=lambda: deque(maxlen=BUFFER_SIZE))
    
    # Binance Reference Data
    binance_mark: float = 0.0
    binance_trades: deque = field(default_factory=lambda: deque(maxlen=BUFFER_SIZE))

    # Order book snapshots (for OFI)
    book_snapshots: deque = field(default_factory=lambda: deque(maxlen=500))

    def is_stale(self, max_age: float = 3.0) -> bool:
        """True if mark price hasn't updated in >max_age seconds."""
        return (time.time() - self.mark_updated) > max_age

    def spread(self) -> float:
        """Current bid-ask spread."""
        if self.best_bid > 0 and self.best_ask > 0:
            return self.best_ask - self.best_bid
        return 0.0

    def mid_price(self) -> float:
        """Mid price from order book."""
        if self.best_bid > 0 and self.best_ask > 0:
            return (self.best_bid + self.best_ask) / 2
        return self.mark_price


class PerpsBuffer:
    """
    Central data store for all symbols.
    Updated by the WebSocket feed; read by signal layers.
    """

    def __init__(self):
        self.states: dict[str, SymbolState] = {
            sym: SymbolState(symbol=sym)
            for sym in SYMBOLS
        }

    def _get(self, symbol: str) -> Optional[SymbolState]:
        return self.states.get(symbol)

    # ── Mark Price ───────────────────────────────────────────────────
    def update_mark_price(self, symbol: str, mark: float, last: float = 0.0, index: float = 0.0):
        s = self._get(symbol)
        if not s:
            return
        s.mark_price   = mark
        s.last_price   = last or mark
        s.index_price  = index or mark
        s.mark_updated = time.time()

    def get_mark_price(self, symbol: str) -> float:
        s = self._get(symbol)
        return s.mark_price if s else 0.0

    def update_binance_mark(self, symbol: str, mark: float):
        """Update reference price from Binance."""
        s = self._get(symbol)
        if s:
            s.binance_mark = mark

    # ── Order Book ───────────────────────────────────────────────────
    def update_book(
        self, symbol: str,
        best_bid: float, best_ask: float,
        bid_depth: float = 0.0, ask_depth: float = 0.0,
    ):
        s = self._get(symbol)
        if not s:
            return
        s.best_bid  = best_bid
        s.best_ask  = best_ask
        s.bid_depth = bid_depth
        s.ask_depth = ask_depth
        s.book_snapshots.append({
            'time': time.time(),
            'bid': best_bid, 'ask': best_ask,
            'bid_depth': bid_depth, 'ask_depth': ask_depth,
        })

    def get_depth(self, symbol: str) -> tuple[list, list]:
        """
        Return current order book depth for signal layers.
        Format: (bids, asks) where each is list of (price, size).
        Perps buffer currently only tracks BBO, so we return 1 level.
        """
        s = self._get(symbol)
        if not s or s.best_bid == 0:
            return [], []
        # Return 1-level depth: [(price, size)]
        return (
            [(s.best_bid, s.bid_depth)],
            [(s.best_ask, s.ask_depth)]
        )

    # ── Trades ───────────────────────────────────────────────────────
    def add_trade(self, symbol: str, trade: dict):
        """
        Add a raw trade event (Primary/Hyperliquid).
        Expected keys: { price, qty, side ('buy'|'sell'), time (ms) }
        """
        s = self._get(symbol)
        if s:
            s.trades.append(trade)

    def add_binance_trade(self, symbol: str, trade: dict):
        """Add a trade event from Binance (Reference)."""
        s = self._get(symbol)
        if s:
            s.binance_trades.append(trade)

    def get_trades(self, symbol: str) -> deque:
        s = self._get(symbol)
        return s.trades if s else deque()

    def get_book_snapshots(self, symbol: str) -> deque:
        s = self._get(symbol)
        return s.book_snapshots if s else deque()

    # ── Funding ──────────────────────────────────────────────────────
    def update_funding(self, symbol: str, rate: float, next_time: float = 0.0):
        s = self._get(symbol)
        if s:
            s.funding_rate      = rate
            s.next_funding_time = next_time

    def get_funding(self, symbol: str) -> dict:
        s = self._get(symbol)
        if not s:
            return {'rate': 0.0, 'next_time': 0.0}
        return {'rate': s.funding_rate, 'next_time': s.next_funding_time}

    # ── Open Interest ────────────────────────────────────────────────
    def update_oi(self, symbol: str, oi: float, oi_usd: float = 0.0):
        s = self._get(symbol)
        if s:
            s.open_interest     = oi
            s.open_interest_usd = oi_usd
            s.oi_updated        = time.time()

    def get_oi(self, symbol: str) -> dict:
        s = self._get(symbol)
        if not s:
            return {'oi': 0.0, 'oi_usd': 0.0, 'updated': 0.0}
        return {
            'oi': s.open_interest,
            'oi_usd': s.open_interest_usd,
            'updated': s.oi_updated,
        }

    # ── State Snapshot ───────────────────────────────────────────────
    def snapshot(self, symbol: str) -> dict:
        """Return a dict snapshot of current state (for logging/dashboard)."""
        s = self._get(symbol)
        if not s:
            return {}
        return {
            'symbol':       s.symbol,
            'mark_price':   s.mark_price,
            'last_price':   s.last_price,
            'best_bid':     s.best_bid,
            'best_ask':     s.best_ask,
            'spread':       round(s.spread(), 5),
            'funding_rate': s.funding_rate,
            'open_interest_usd': s.open_interest_usd,
            'open_interest_usd': s.open_interest_usd,
            'trade_count':  len(s.trades),
            'binance_trade_count': len(s.binance_trades),
            'stale':        s.is_stale(),
        }

    def is_ready(self, symbol: str, min_trades: int = 50) -> tuple[bool, str]:
        """Check if buffer has enough data to run signal layers."""
        s = self._get(symbol)
        if not s:
            return False, f'Unknown symbol: {symbol}'
        if s.is_stale():
            return False, f'{symbol} mark price is stale'
        if len(s.trades) < min_trades:
            return False, f'{symbol} only {len(s.trades)}/{min_trades} trades buffered'
        return True, 'OK'


# Global singleton
buffer = PerpsBuffer()
