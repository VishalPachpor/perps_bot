"""
Perps Bot — Main Event Loop

Architecture:
  - asyncio dual-task: (1) WebSocket feed, (2) scan + execute loop
  - Feed runs continuously, updating buffer
  - Scanner runs every SCAN_INTERVAL seconds per symbol
  - Position monitor runs every 0.5s to check exit triggers

Flow per scan cycle:
  1. Check risk gates (daily PnL, cooldown, exposure)
  2. Run signal scanner → scan result
  3. If trade: compute size → open position → notify
  4. Check all open positions for exit triggers → close → notify
"""
import asyncio
import signal
import time
from datetime import datetime, timezone
from loguru import logger

from config import (
    SYMBOLS, SCAN_INTERVAL, BANKROLL, MAX_LEVERAGE,
    PAPER_TRADE, LOG_LEVEL, RISK_PER_TRADE_PCT,
)
from data.buffer import buffer
from data.feed import start_feed

# ── Wire signal package to this bot's buffer ─────────────────────────
# Must be called before any signal layer is imported or used.
import perp_signals
perp_signals.set_buffer(buffer)

from execution.risk import risk_manager, vol_monitor, cascade_detector
from execution.position_manager import position_manager, compute_perp_size
from execution.venue_client import venue
from monitoring.trade_logger import log_trade
from monitoring import telegram
from dashboard.server import record_scan, record_price_tick

logger.remove()
logger.add(
    'logs/perps_bot.log',
    level=LOG_LEVEL,
    rotation='50 MB',
    retention='7 days',
    format='{time:HH:mm:ss.SSS} | {level:<7} | {message}',
)
logger.add(
    lambda msg: print(msg, end=''),
    level='INFO',
    format='{time:HH:mm:ss} | {level:<7} | {message}',
)


from signals.adaptive_weights import update_weights
from execution.trade_logger import log_trade_entry, log_trade_exit

# ── Scan + Execute ────────────────────────────────────────────────────
async def execute_and_register_trade(symbol, direction, size_usd, mark_price, atr, regime, scan_data):
    """Background task to execute trade and register position without blocking the scan loop."""
    try:
        is_buy = (direction == 'long')
        
        # Get inside spread to act as Maker
        state = buffer.snapshot(symbol)
        best_bid = state.get('best_bid', mark_price)
        best_ask = state.get('best_ask', mark_price)
        limit_px = best_bid if is_buy else best_ask
        if limit_px <= 0:
            limit_px = mark_price
            
        coin_size = size_usd / limit_px if limit_px > 0 else 0.0

        loop = asyncio.get_event_loop()
        
        # 1. Place Post-Only Limit Order
        logger.info(f'[EXEC] {symbol} routing Maker limit at {limit_px:.4f}')
        result = await loop.run_in_executor(None, venue.limit_order, symbol, is_buy, coin_size, limit_px, True)
        
        filled_size = 0.0
        avg_px = limit_px
        
        if result['success']:
            oid = result['order_id']
            # 2. Wait up to 5 seconds
            for _ in range(10):
                await asyncio.sleep(0.5)
                status = await loop.run_in_executor(None, venue.get_order_status, symbol, oid)
                if status['status'] == 'filled':
                    break
            
            status = await loop.run_in_executor(None, venue.get_order_status, symbol, oid)
            filled_size = status.get('filled_size', 0.0)
            if status.get('avg_price', 0.0) > 0:
                avg_px = status['avg_price']
                
            # If partially unfilled or open
            if status['status'] != 'filled' and filled_size < coin_size - 0.0001:
                # Cancel remaining
                await loop.run_in_executor(None, venue.cancel_order, symbol, oid)
                
                # Fetch final status after cancellation
                status = await loop.run_in_executor(None, venue.get_order_status, symbol, oid)
                filled_size = status.get('filled_size', 0.0)
                
                remaining_size = coin_size - filled_size
                if remaining_size > (size_usd * 0.01 / limit_px): # at least 1% of required trade
                    logger.info(f'[EXEC] {symbol} limit unfilled. Routing remaining {remaining_size:.4f} to market taker.')
                    mkt_res = await loop.run_in_executor(None, venue.market_order, symbol, is_buy, remaining_size)
                    if mkt_res['success']:
                        mkt_price = mkt_res.get('filled_price', mark_price)
                        total_size = filled_size + remaining_size
                        if total_size > 0:
                            avg_px = ((filled_size * avg_px) + (remaining_size * mkt_price)) / total_size
                            filled_size = total_size

            if filled_size <= 0:
                logger.error(f'[EXEC] {symbol} order entirely failed/unfilled.')
                return
        else:
            # Fallback direct market
            logger.warning(f'[EXEC] Maker limit failed on submission ({result.get("reason")}). Fallback to Market.')
            mkt_res = await loop.run_in_executor(None, venue.market_order, symbol, is_buy, coin_size)
            if not mkt_res['success']:
                return
            avg_px = mkt_res.get('filled_price', mark_price)
            filled_size = coin_size

        fill_price = avg_px
        actual_size_usd = filled_size * fill_price

        # Register position
        pos_id = position_manager.open_position(
            symbol=symbol,
            direction=direction,
            entry_price=fill_price,
            size_usd=actual_size_usd,
            leverage=MAX_LEVERAGE,
            atr=atr,
            regime=regime,
            signal_type=scan_data.get('score', ''),
            session=scan_data.get('session', ''),
            funding_rate=scan_data.get('funding_rate', 0.0),
            sweep_depth=scan_data.get('sweep_depth', 0.0),
        )

        if not pos_id:
            await loop.run_in_executor(None, venue.close_position, symbol, not is_buy, filled_size)
            return

        risk_manager.record_open(symbol, actual_size_usd)

        open_positions = position_manager.get_open_positions()
        pos_info = next((p for p in open_positions if p['pos_id'] == pos_id), {})
        
        # Fire off notification silently
        asyncio.create_task(telegram.notify_open(
            symbol, direction,
            entry=fill_price,
            size=actual_size_usd,
            sl=pos_info.get('sl_price', 0),
            tp=pos_info.get('tp_price', 0),
            reason=scan_data.get('score', ''),
        ))

        log_trade({
            'event': 'open', 'symbol': symbol, 'direction': direction,
            'entry_price': fill_price, 'size_usd': actual_size_usd,
            'regime': regime, 'signal_score': scan_data.get('score'),
            'pos_id': pos_id,
        })
        
        # Quant Log Entry Factors
        raw_factors = scan_data.get('raw_factors', {})
        raw_factors.update({
            'symbol': symbol,
            'direction': direction,
            'regime': regime,
            'session': scan_data.get('session', ''),
            'spread_at_entry': scan_data.get('spread', 0),
            'entry_price': fill_price,
            'signal_strength': scan_data.get('raw_factors', {}).get('signal_strength', 0),
        })
        log_trade_entry(pos_id, raw_factors)

    except Exception as e:
        logger.error(f'[EXEC] Async execution failed for {symbol}: {e}')


async def execute_and_register_close(symbol, direction, coin_size, mark_price, exit_event):
    """Background task to execute trade exit and register outcome without blocking the monitor loop."""
    try:
        is_buy = (direction == 'short')  # closing a short = buy
        
        state = buffer.snapshot(symbol)
        best_bid = state.get('best_bid', mark_price)
        best_ask = state.get('best_ask', mark_price)
        limit_px = best_bid if is_buy else best_ask
        if limit_px <= 0:
            limit_px = mark_price

        loop = asyncio.get_event_loop()
        
        # 1. Place Post-Only Limit Order
        logger.info(f'[EXEC] {symbol} routing Maker limit close at {limit_px:.4f}')
        result = await loop.run_in_executor(None, venue.limit_order, symbol, is_buy, coin_size, limit_px, True)
        
        filled_size = 0.0
        avg_px = limit_px
        
        if result['success']:
            oid = result['order_id']
            # 2. Wait 5s
            for _ in range(10):
                await asyncio.sleep(0.5)
                status = await loop.run_in_executor(None, venue.get_order_status, symbol, oid)
                if status['status'] == 'filled':
                    break
            
            status = await loop.run_in_executor(None, venue.get_order_status, symbol, oid)
            filled_size = status.get('filled_size', 0.0)
            if status.get('avg_price', 0.0) > 0:
                avg_px = status['avg_price']
                
            if status['status'] != 'filled' and filled_size < coin_size - 0.0001:
                await loop.run_in_executor(None, venue.cancel_order, symbol, oid)
                status = await loop.run_in_executor(None, venue.get_order_status, symbol, oid)
                filled_size = status.get('filled_size', 0.0)
                
                remaining = coin_size - filled_size
                if remaining > (coin_size * 0.01):
                    logger.info(f'[EXEC] {symbol} limit close unfilled. Force market for {remaining:.4f}')
                    mkt_res = await loop.run_in_executor(None, venue.close_position, symbol, is_buy, remaining)
                    if mkt_res.get('success'):
                        mkt_px = mkt_res.get('filled_price', mark_price)
                        total = filled_size + remaining
                        if total > 0:
                            avg_px = ((filled_size * avg_px) + (remaining * mkt_px)) / total
                        filled_size = total

            if filled_size <= 0:
                logger.error(f'[EXEC] {symbol} close order entirely failed.')
                return
        else:
            logger.warning(f'[EXEC] Maker limit close failed ({result.get("reason")}). Fallback market.')
            mkt_res = await loop.run_in_executor(None, venue.close_position, symbol, is_buy, coin_size)
            if not mkt_res.get('success'):
                return
            avg_px = mkt_res.get('filled_price') or mark_price
            filled_size = coin_size

        fill = avg_px

        # Register closure locally
        pos_id = exit_event['pos_id']
        record = position_manager.close_position(
            pos_id=pos_id,
            exit_price=fill,
            slippage=abs(fill - mark_price) / mark_price if fill != mark_price and mark_price > 0 else 0,
        )

        if record:
            risk_manager.record_outcome(symbol, record.get('realized_pnl', 0))
            
            asyncio.create_task(telegram.notify_close(
                symbol, direction,
                pnl=record['realized_pnl'],
                reason=exit_event['reason'],
            ))
            
            log_trade({**record, 'event': 'close'})
            
            # Quant Log Exit Result
            expected_risk = BANKROLL * RISK_PER_TRADE_PCT
            if expected_risk > 0:
                r_multiple = record.get('realized_pnl', 0) / expected_risk
            else:
                r_multiple = 0.0
                
            log_trade_exit(
                pos_id=pos_id,
                result_r=r_multiple,
                exit_reason=exit_event['reason'],
                exit_price=fill,
                hold_time=record.get('hold_seconds', 0)
            )

    except Exception as e:
        logger.error(f'[EXEC] Async close failed for {symbol}: {e}')


async def scan_and_execute():
    """Main scan loop — runs every SCAN_INTERVAL seconds per symbol."""
    from signals.scanner import run_scan

    while True:
        try:
            for symbol in SYMBOLS:
                regime = vol_monitor.regime
                scan   = run_scan(symbol, regime=regime)

                # ── Push to dashboard ─────────────────────────────────
                record_scan({
                    **scan,
                    'market_price': buffer.get_mark_price(symbol),
                    'daily_pnl': risk_manager.status()['daily_pnl'],
                })
                mark = buffer.get_mark_price(symbol)
                if mark > 0:
                    state = buffer.snapshot(symbol)
                    record_price_tick(
                        price=mark,
                        bid=state.get('best_bid', 0),
                        ask=state.get('best_ask', 0),
                        asset=symbol,
                    )

                # ── Log scan result ──────────────────────────────────
                reason_short = scan.get('reason', '')[:60]
                logger.info(
                    f'[SCAN] {symbol} | trade={scan["trade"]} | '
                    f'score={scan["score"]} | {reason_short}'
                )

                if not scan.get('trade'):
                    continue

                # ── Risk gate ────────────────────────────────────────
                allowed, risk_reason = risk_manager.can_trade(symbol)
                if not allowed:
                    logger.info(f'[RISK] {symbol} blocked: {risk_reason}')
                    continue

                # ── Position sizing ──────────────────────────────────
                mark_price = scan.get('mark_price', buffer.get_mark_price(symbol))
                atr        = scan.get('atr', 0.0)
                sl_pct     = (atr / mark_price) * 1.5 if mark_price > 0 and atr > 0 else 0.003
                regime_mult = {
                    'LOW': 1.2, 'NORMAL': 1.0, 'HIGH': 0.5, 'EXTREME': 0.0
                }.get(regime, 1.0)
                session_mult = scan.get('session_mult', 1.0)

                size_usd = compute_perp_size(
                    bankroll=BANKROLL,
                    risk_pct=RISK_PER_TRADE_PCT,
                    sl_pct=sl_pct,
                    leverage=MAX_LEVERAGE,
                    depth_scalar=regime_mult * session_mult,
                )

                if size_usd < 5.0:
                    logger.debug(f'[EXEC] {symbol} size ${size_usd:.2f} too small — skip')
                    continue

                # ── Dispatch async execution ────────────────────────────
                asyncio.create_task(execute_and_register_trade(
                    symbol=symbol,
                    direction=direction,
                    size_usd=size_usd,
                    mark_price=mark_price,
                    atr=atr,
                    regime=regime,
                    scan_data=scan
                ))

        except Exception as e:
            logger.exception(f'[SCAN] Unhandled error: {e}')

        await asyncio.sleep(SCAN_INTERVAL)


# ── Position Monitor ──────────────────────────────────────────────────
async def monitor_positions():
    """Runs every 0.5s to check exit triggers across all open positions."""
    while True:
        try:
            for symbol in SYMBOLS:
                mark_price = buffer.get_mark_price(symbol)
                if mark_price <= 0:
                    continue

                # Update cascade detector
                oi_data = buffer.get_oi(symbol)
                cascade_detector.update(oi_data.get('oi_usd', 0), mark_price)

                # Update vol monitor
                # (ATR update happens in scanner, not here)

                # Check exits
                exits = position_manager.check_all_exits(
                    symbol=symbol,
                    mark_price=mark_price,
                    ofi_direction='',   # TODO: pull live from buffer
                    cvd_trend='',       # TODO: pull live from buffer
                    regime=vol_monitor.regime,
                    cascade_detected=cascade_detector.is_active,
                )

                for exit_event in exits:
                    pos_id    = exit_event['pos_id']
                    direction = exit_event['direction']
                    is_buy    = (direction == 'short')  # closing a short = buy
                    coin_size = exit_event['size_usd'] / mark_price

                    # Dispatch async execution
                    asyncio.create_task(execute_and_register_close(
                        symbol=symbol,
                        direction=direction,
                        coin_size=coin_size,
                        mark_price=mark_price,
                        exit_event=exit_event
                    ))

        except Exception as e:
            logger.exception(f'[MONITOR] Error: {e}')

        await asyncio.sleep(0.5)


# ── Daily Reset ───────────────────────────────────────────────────────
async def daily_reset():
    """Runs at UTC midnight to reset daily counters."""
    while True:
        now    = datetime.now(timezone.utc)
        secs_until_midnight = (
            (86400 - (now.hour * 3600 + now.minute * 60 + now.second))
        )
        await asyncio.sleep(secs_until_midnight)
        status = risk_manager.status()
        asyncio.create_task(telegram.notify_daily_summary(
            status['daily_pnl'], status['daily_trades'], status['regime']
        ))
        risk_manager.reset_daily()
        logger.info('[MAIN] Daily reset complete')


# ── Adaptive Weight Learning Loop ─────────────────────────────────────
async def _adaptive_learning_loop():
    """Periodically update signal weights based on trade history."""
    while True:
        await asyncio.sleep(60 * 30) # Run every 30 minutes
        logger.info("[LEARNING] Running adaptive weight update...")
        try:
            new_weights = update_weights()
            logger.info(f"[LEARNING] Weights updated: {new_weights}")
        except Exception as e:
            logger.error(f"[LEARNING] Update failed: {e}")


# ── Status Logger ─────────────────────────────────────────────────────
async def log_status():
    """Log a brief status line every 60s."""
    while True:
        await asyncio.sleep(60)
        status = risk_manager.status()
        open_p = position_manager.get_open_positions()
        exposure = position_manager.get_net_exposure()
        logger.info(
            f'[STATUS] Regime={status["regime"]} | '
            f'Daily PnL=${status["daily_pnl"]:+.2f} | '
            f'Open={len(open_p)} | NetLong=${exposure["net_long"]:.0f} | '
            f'Cascade={"YES" if status["cascade"] else "no"}'
        )


async def _poll_funding():
    """Fetch funding rates from Hyperliquid API every 60s."""
    import aiohttp
    url = "https://api.hyperliquid.xyz/info"
    payload = {"type": "metaAndAssetCtxs"}
    
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        # data[0] is universe, data[1] is assetCtxs
                        universe = data[0]['universe']
                        asset_ctxs = data[1]
                        
                        for i, asset_info in enumerate(universe):
                            sym = asset_info['name']
                            if sym in SYMBOLS:
                                # Funding is in assetCtx as string, e.g. "0.0000125"
                                funding = float(asset_ctxs[i].get('funding', 0))
                                buffer.update_funding(sym, funding)
                        
                        logger.info(f"[FUNDING] Updated for {SYMBOLS}")
        except Exception as e:
            logger.warning(f"[FUNDING] Poll error: {e}")
        
        await asyncio.sleep(60)


# ── Graceful Shutdown ─────────────────────────────────────────────────
def _handle_shutdown(loop: asyncio.AbstractEventLoop):
    logger.warning('[MAIN] Shutdown signal received — stopping...')
    for task in asyncio.all_tasks(loop):
        task.cancel()


# ── Entry Point ───────────────────────────────────────────────────────
# ── Dashboard Integration ──────────────────────────────────────────
import threading
def _start_dashboard_server(port: int):
    """Start the FastAPI dashboard server in a background thread."""
    try:
        import uvicorn
        from dashboard.server import app
        uvicorn.run(app, host='0.0.0.0', port=port, log_level='warning')
    except ImportError:
        logger.warning("Dashboard dependencies not installed. Skipping.")

async def main():
    logger.info(f"Starting Perps Bot — {PAPER_TRADE=}")
    logger.info(f"Market: Hyperliquid | Symbols: {SYMBOLS}")

    # ── Prop Desk: Orphan Trade Reconciliation ────────────────────────
    _reconcile_orphan_trades()
    
    # Start Dashboard
    dash_port = 8081
    logger.info(f"🖥  Dashboard starting at http://localhost:{dash_port}")
    threading.Thread(
        target=_start_dashboard_server,
        args=(dash_port,),
        daemon=True,
    ).start()

    # Initialize Telegram
    await telegram.send_startup_alert(SYMBOLS)
    loop = asyncio.get_event_loop()
    loop.add_signal_handler(signal.SIGINT,  lambda: _handle_shutdown(loop))
    loop.add_signal_handler(signal.SIGTERM, lambda: _handle_shutdown(loop))

    await asyncio.gather(
        start_feed(),
        scan_and_execute(),
        monitor_positions(),
        daily_reset(),
        log_status(),
        _poll_funding(),
        _adaptive_learning_loop(), # Add learning loop
    )


def _reconcile_orphan_trades():
    """
    Prop Desk: On startup, find trades with entry but no exit (crash survivors).
    Mark them as ORPHAN_CLOSE to prevent dataset corruption.
    """
    import os
    import pandas as pd
    log_path = "data/trade_features.csv"
    if not os.path.exists(log_path):
        return

    try:
        df = pd.read_csv(log_path)
        orphans = df['result_r'].isna()
        n_orphans = orphans.sum()

        if n_orphans > 0:
            df.loc[orphans, 'exit_reason'] = 'ORPHAN_CLOSE'
            df.loc[orphans, 'result_r'] = 0.0  # Neutral — don't corrupt learning
            df.loc[orphans, 'hold_time'] = 0.0
            df.to_csv(log_path, index=False)
            logger.warning(f"[RECONCILE] Closed {n_orphans} orphan trade(s) from prior crash")
        else:
            logger.info("[RECONCILE] No orphan trades found — clean startup")
    except Exception as e:
        logger.error(f"[RECONCILE] Failed: {e}")


if __name__ == '__main__':
    asyncio.run(main())

