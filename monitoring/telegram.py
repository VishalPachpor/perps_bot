"""
Telegram Notifier — async Telegram bot alerts for trade events.
"""
import aiohttp
from loguru import logger

from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID


async def _send(text: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage'
    payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': text, 'parse_mode': 'HTML'}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    logger.warning(f'[TG] Failed: {resp.status}')
    except Exception as e:
        logger.warning(f'[TG] Error: {e}')


async def notify_open(symbol: str, direction: str, entry: float, size: float, sl: float, tp: float, reason: str):
    emoji = '🟢' if direction == 'long' else '🔴'
    text = (
        f'{emoji} <b>OPEN {direction.upper()} {symbol}</b>\n'
        f'Entry: <code>{entry:.4f}</code>  Size: <code>${size:.2f}</code>\n'
        f'SL: <code>{sl:.4f}</code>  TP: <code>{tp:.4f}</code>\n'
        f'Signal: {reason}'
    )
    await _send(text)


async def notify_close(symbol: str, direction: str, pnl: float, reason: str):
    emoji = '✅' if pnl > 0 else '❌'
    text = (
        f'{emoji} <b>CLOSE {direction.upper()} {symbol}</b>\n'
        f'PnL: <code>${pnl:+.2f}</code>  Reason: {reason}'
    )
    await _send(text)


async def notify_risk_block(reason: str):
    await _send(f'⛔ <b>Risk block:</b> {reason}')


async def notify_daily_summary(daily_pnl: float, trades: int, regime: str):
    emoji = '📈' if daily_pnl > 0 else '📉'
    text = (
        f'{emoji} <b>Daily Summary</b>\n'
        f'PnL: <code>${daily_pnl:+.2f}</code>  Trades: {trades}\n'
        f'Regime: {regime}'
    )
    await _send(text)


async def send_startup_alert(symbols: list):
    """Notify that the bot has started and is monitoring these symbols."""
    text = (
        f'🚀 <b>Perps Bot Started</b>\n'
        f'Monitoring: {", ".join(symbols)}\n'
        f'Ready to trade.'
    )
    await _send(text)
