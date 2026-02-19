"""
Perps Bot — Central Configuration
All thresholds, constants, and env vars for perpetual futures trading.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── Venue ────────────────────────────────────────────────────────────
VENUE              = os.getenv('VENUE', 'hyperliquid')   # hyperliquid | binance | bybit
VENUE_TESTNET      = os.getenv('VENUE_TESTNET', 'true').lower() == 'true'

# ── Assets ──────────────────────────────────────────────────────────
SYMBOLS        = os.getenv('SYMBOLS', 'ETH,BTC,SOL').split(',')
PRIMARY_SYMBOL = os.getenv('PRIMARY_SYMBOL', 'ETH')

# ── Bankroll & Sizing ────────────────────────────────────────────────
BANKROLL              = float(os.getenv('BANKROLL', '1000'))
MAX_LEVERAGE          = int(os.getenv('MAX_LEVERAGE', '3'))         # hard cap, never exceed
MAX_POSITION_FRACTION = float(os.getenv('MAX_POSITION_FRACTION', '0.10'))  # 10% max per trade
RISK_PER_TRADE_PCT    = float(os.getenv('RISK_PER_TRADE_PCT', '0.01'))    # 1% risk per trade
MAX_NOTIONAL_PCT      = float(os.getenv('MAX_NOTIONAL_PCT', '0.60'))       # 60% notional (allows leverage)
MAX_NET_DIRECTIONAL   = float(os.getenv('MAX_NET_DIRECTIONAL', '0.40'))    # 40% net long/short

# ── Perps Fee Model ──────────────────────────────────────────────────
# Taker fees by venue (round-trip = fee × 2)
TAKER_FEE = {
    'hyperliquid': 0.00035,   # 0.035%
    'binance':     0.00040,   # 0.04%
    'bybit':       0.00055,   # 0.055%
}.get(VENUE, 0.00040)

MAKER_FEE = {
    'hyperliquid': -0.00010,  # -0.01% (rebate)
    'binance':     -0.00020,  # -0.02% (rebate)
    'bybit':        0.00010,  # 0.01%
}.get(VENUE, 0.00000)

# ── Edge Thresholds (regime-aware, replaces static 4% from Polymarket) ──
# Rationale: perps fees ~0.035–0.05% vs Polymarket 1–2%
MIN_EDGE_BY_REGIME = {
    'LOW':     0.0020,   # 0.20% — chop protection, skip most
    'NORMAL':  0.0015,   # 0.15% — standard threshold
    'HIGH':    0.0010,   # 0.10% — fast moves, tighter size compensates
    'EXTREME': None,     # no trading in extreme vol
}

# ── Position Lifecycle ───────────────────────────────────────────────
MAX_HOLD_SECONDS      = int(os.getenv('MAX_HOLD_SECONDS', '600'))     # 10 min
TRAILING_ACTIVATE_PCT = float(os.getenv('TRAILING_ACTIVATE_PCT', '0.004'))  # +0.4%
TRAILING_LOCK_PCT     = float(os.getenv('TRAILING_LOCK_PCT', '0.50'))       # lock 50%
MIN_RR_RATIO          = float(os.getenv('MIN_RR_RATIO', '1.5'))             # reward:risk

# ── Risk Management ──────────────────────────────────────────────────
MAX_DAILY_LOSS          = float(os.getenv('MAX_DAILY_LOSS', '-50'))        # -$50 early live
MAX_OPEN_POSITIONS      = int(os.getenv('MAX_OPEN_POSITIONS', '2'))
COOLDOWN_AFTER_LOSS_SEC = int(os.getenv('COOLDOWN_AFTER_LOSS_SEC', '300')) # 5 min
MAX_CORRELATED_EXPOSURE = float(os.getenv('MAX_CORRELATED_EXPOSURE', '0.08'))

# ── Volatility Regimes ───────────────────────────────────────────────
REGIME_SIZE_MULT = {
    'LOW':     1.2,   # low vol → slightly more aggressive
    'NORMAL':  1.0,
    'HIGH':    0.5,   # high vol → smaller size
    'EXTREME': 0.0,   # no trading
}

# ── Signal Thresholds ────────────────────────────────────────────────
MIN_SCORE          = int(os.getenv('MIN_SCORE', '5'))     # min score /11 to trade
SCAN_INTERVAL      = float(os.getenv('SCAN_INTERVAL', '2.0'))  # seconds
ATR_WINDOW_SECONDS = int(os.getenv('ATR_WINDOW_SECONDS', '300'))  # 5-min ATR

# ── Data Feed ────────────────────────────────────────────────────────
BINANCE_WS_URL   = 'wss://fstream.binance.com/ws'      # futures WebSocket
HYPERLIQUID_WS   = 'wss://api.hyperliquid.xyz/ws'

BUFFER_SIZE      = int(os.getenv('BUFFER_SIZE', '10000'))

# ── Venue API Credentials ────────────────────────────────────────────
HL_PRIVATE_KEY   = os.getenv('HL_PRIVATE_KEY', '')     # Hyperliquid wallet private key
HL_WALLET_ADDR   = os.getenv('HL_WALLET_ADDR', '')
BINANCE_API_KEY  = os.getenv('BINANCE_API_KEY', '')
BINANCE_SECRET   = os.getenv('BINANCE_SECRET', '')

# ── System ──────────────────────────────────────────────────────────
PAPER_TRADE = os.getenv('PAPER_TRADE', 'true').lower() == 'true'
LOG_LEVEL   = os.getenv('LOG_LEVEL', 'INFO')

# ── Failure Pattern Log ──────────────────────────────────────────────
FAILURE_LOG_PATH = os.getenv('FAILURE_LOG_PATH', 'data/failure_patterns.jsonl')

# ── Telegram ─────────────────────────────────────────────────────────
TELEGRAM_TOKEN   = os.getenv('TELEGRAM_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
