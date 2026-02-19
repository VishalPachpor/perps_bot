"""
Hyperliquid WebSocket Feed — streams mark price, trades, order book, funding, and OI.

Connects to Hyperliquid's WebSocket API and populates the perps buffer.
Falls back to Binance Futures WS if VENUE=binance.

Hyperliquid WS docs: https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/websocket
Binance Futures WS:  https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams
"""
import asyncio
import json
import time
import websockets
from loguru import logger

from config import (
    VENUE, HYPERLIQUID_WS, BINANCE_WS_URL,
    SYMBOLS, VENUE_TESTNET,
)
from data.buffer import buffer


# ── Symbol mapping ────────────────────────────────────────────────────
# Internal symbol ('ETH') → venue symbol ('ETH' for HL, 'ETHUSDT' for Binance)
def _to_venue_symbol(sym: str) -> str:
    if VENUE == 'binance':
        return f'{sym}USDT'
    return sym  # Hyperliquid uses bare symbol


# ══════════════════════════════════════════════════════════════════════
# Hyperliquid Feed
# ══════════════════════════════════════════════════════════════════════
async def _hl_feed():
    """
    Hyperliquid WebSocket subscription.
    Subscribes to: allMids (mark price), trades, l2Book per symbol.
    """
    url = 'wss://api.hyperliquid.xyz/ws'
    retry_delay = 2

    while True:
        try:
            logger.info(f'[FEED-HL] Connecting to {url}')
            async with websockets.connect(url, ping_interval=20) as ws:
                # Subscribe to all mids (mark prices for all assets)
                await ws.send(json.dumps({
                    'method': 'subscribe',
                    'subscription': {'type': 'allMids'},
                }))

                # Subscribe per-asset streams
                for sym in SYMBOLS:
                    # Trades
                    await ws.send(json.dumps({
                        'method': 'subscribe',
                        'subscription': {'type': 'trades', 'coin': sym},
                    }))
                    # L2 order book
                    await ws.send(json.dumps({
                        'method': 'subscribe',
                        'subscription': {'type': 'l2Book', 'coin': sym},
                    }))

                logger.info(f'[FEED-HL] Subscribed to {SYMBOLS}')
                retry_delay = 2

                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                        _hl_dispatch(msg)
                    except Exception as e:
                        logger.warning(f'[FEED-HL] Dispatch error: {e}')

        except Exception as e:
            logger.error(f'[FEED-HL] Connection lost: {e} — retry in {retry_delay}s')
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 30)


def _hl_dispatch(msg: dict):
    """Route Hyperliquid WS message to buffer."""
    channel = msg.get('channel', '')
    data    = msg.get('data', {})

    if channel == 'allMids':
        # { 'mids': { 'ETH': '2850.5', 'BTC': '54000.0', ... } }
        mids = data.get('mids', {})
        for sym, price_str in mids.items():
            if sym in SYMBOLS:
                try:
                    buffer.update_mark_price(sym, float(price_str))
                except ValueError:
                    pass

    elif channel == 'trades':
        # List of trade events per coin
        coin = data[0].get('coin') if data else None
        if coin and coin in SYMBOLS:
            for t in data:
                buffer.add_trade(coin, {
                    'price': float(t['px']),
                    'qty':   float(t['sz']),
                    'side':  'buy' if t['side'] == 'B' else 'sell',
                    'is_maker': t['side'] != 'B',  # False=Buy, True=Sell (perp-signals convention)
                    'time':  t.get('time', int(time.time() * 1000)),
                })

    elif channel == 'l2Book':
        coin   = data.get('coin', '')
        levels = data.get('levels', [[], []])
        if coin in SYMBOLS and levels:
            bids = levels[0]
            asks = levels[1]
            best_bid = float(bids[0]['px']) if bids else 0.0
            best_ask = float(asks[0]['px']) if asks else 0.0
            bid_depth = sum(float(b['sz']) * float(b['px']) for b in bids[:5])
            ask_depth = sum(float(a['sz']) * float(a['px']) for a in asks[:5])
            buffer.update_book(coin, best_bid, best_ask, bid_depth, ask_depth)
            buffer.update_mark_price(coin, (best_bid + best_ask) / 2)


# ══════════════════════════════════════════════════════════════════════
# Binance Futures Feed (fallback)
# ══════════════════════════════════════════════════════════════════════
async def _binance_feed():
    """
    Binance Futures combined stream.
    Subscribes to: markPrice + aggTrade + bookTicker per symbol.
    """
    streams = []
    for sym in SYMBOLS:
        # Force USDT suffix for Binance streams regardless of VENUE config
        vsym = f'{sym.lower()}usdt'
        streams += [
            f'{vsym}@markPrice',
            f'{vsym}@aggTrade',
            f'{vsym}@bookTicker',
        ]

    url = f'wss://fstream.binance.com/stream?streams={"/".join(streams)}'
    retry_delay = 2

    while True:
        try:
            logger.info(f'[FEED-BN] Connecting: {len(streams)} streams')
            async with websockets.connect(url, ping_interval=20) as ws:
                retry_delay = 2
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                        _binance_dispatch(msg.get('data', msg))
                    except Exception as e:
                        logger.warning(f'[FEED-BN] Dispatch error: {e}')

        except Exception as e:
            logger.error(f'[FEED-BN] Connection lost: {e} — retry in {retry_delay}s')
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 30)


def _binance_dispatch(data: dict):
    """Route Binance Futures WS event to buffer."""
    event = data.get('e', '')

    if event == 'markPriceUpdate':
        sym = data.get('s', '').replace('USDT', '')
        if sym in SYMBOLS:
            # Update REFERENCE price (do not overwrite execution mark price)
            buffer.update_binance_mark(sym, float(data.get('p', 0)))
            
            # Funding is global enough to update? Maybe keep it for reference too?
            # For now, let HL drive funding. Binance funding just for logging if needed.
            # r = data.get('r')
            # if r: buffer.update_funding(sym, rate=float(r))

    elif event == 'aggTrade':
        sym = data.get('s', '').replace('USDT', '')
        if sym in SYMBOLS:
            buffer.add_binance_trade(sym, {
                'price': float(data['p']),
                'qty':   float(data['q']),
                'side':  'sell' if data.get('m') else 'buy',
                'is_maker': data.get('m', False),
                'time':  data.get('T', int(time.time() * 1000)),
            })

    elif event == 'bookTicker':
        sym = data.get('s', '').replace('USDT', '')
        if sym in SYMBOLS:
            buffer.update_book(
                sym,
                best_bid=float(data.get('b', 0)),
                best_ask=float(data.get('a', 0)),
            )


# ── Entry point ────────────────────────────────────────────────────────
async def start_feed():
    """Start BOTH feeds concurrently (Hyperliquid Primary + Binance Reference)."""
    logger.info('[FEED] Starting Dual Feed: Hyperliquid (Primary) + Binance (Ref)')
    
    # Run both forever
    await asyncio.gather(
        _hl_feed(),
        _binance_feed(),
    )
