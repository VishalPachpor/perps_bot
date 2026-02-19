# Perps Bot

Perpetual futures scalping bot — Hyperliquid (primary) / Binance Futures.

Spun out from the Polymarket bot. Shares signal layers, adds perps-specific execution.

---

## Repo Structure

```
perps-bot/
├── config.py                  ← all thresholds, fees, risk params
├── main.py                    ← async event loop
├── start.sh                   ← run script
│
├── data/
│   ├── buffer.py              ← mark price, OI, trades, order book per symbol
│   └── feed.py                ← Hyperliquid / Binance WS feed
│
├── execution/
│   ├── position_manager.py    ← full position lifecycle, 7 exit triggers
│   ├── venue_client.py        ← Hyperliquid order placement
│   ├── fee_model.py           ← perps edge calculation (0.035% fees)
│   └── risk.py                ← daily loss limit, cascade detector, exposure cap
│
├── signals/
│   └── scanner.py             ← adapts 6-layer signal stack for perps
│
└── monitoring/
    ├── trade_logger.py        ← JSONL trade log for failure pattern analysis
    └── telegram.py            ← async Telegram alerts
```

> The 6 signal layers (MTF, OFI, CVD, VWAP, microstructure, correlation) are
> imported from the Polymarket bot's `signals/` directory via a shared path or
> pip-installed package. They are venue-agnostic — no changes needed.

---

## Setup

```bash
cp .env.example .env
# Edit .env — set HL_PRIVATE_KEY, HL_WALLET_ADDR, TELEGRAM_TOKEN

pip install -r requirements.txt
chmod +x start.sh
./start.sh
```

---

## Start in Paper Mode (default)

`PAPER_TRADE=true` in `.env` — no real orders placed. Run paper for at least
**2 weeks / 300 trades** before flipping to live.

---

## Go Live Checklist

- [ ] 300+ paper trades logged
- [ ] Win rate ≥ 52% in NORMAL regime
- [ ] Average slippage < 0.05%
- [ ] No regime where avg PnL is negative
- [ ] `MAX_DAILY_LOSS` set to `-$50` (raise only after 1000 trades)
- [ ] `PAPER_TRADE=false`
- [ ] `MAX_LEVERAGE=3` (never higher for first live month)

---

## Failure Pattern Analysis

After 300 trades:
```python
from monitoring.trade_logger import load_recent, win_rate_by_regime
records = load_recent(300)
print(win_rate_by_regime(records))
```
