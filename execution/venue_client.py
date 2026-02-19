"""
Venue Client — Hyperliquid order execution.

Wraps the Hyperliquid Python SDK to place market and limit orders.
Falls back to a paper-trade stub when PAPER_TRADE=true or SDK not installed.

Hyperliquid SDK: pip install hyperliquid-python-sdk
Docs: https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api
"""
import time
from loguru import logger

from config import (
    VENUE, PAPER_TRADE,
    HL_PRIVATE_KEY, HL_WALLET_ADDR,
    TAKER_FEE,
)


class HyperliquidClient:
    """
    Thin wrapper around the Hyperliquid Python SDK.
    All methods return a standardised result dict:
      { 'success': bool, 'order_id': str, 'reason': str }
    """

    def __init__(self):
        self.exchange = None
        self.info     = None
        self._paper   = PAPER_TRADE

        if not self._paper:
            self._init_sdk()

    def _init_sdk(self):
        try:
            from hyperliquid.exchange import Exchange
            from hyperliquid.info import Info
            import eth_account

            account = eth_account.Account.from_key(HL_PRIVATE_KEY)
            self.info     = Info(skip_ws=True)
            self.exchange = Exchange(account, skip_ws=True)
            logger.info(f'[VENUE] Hyperliquid client ready — wallet: {HL_WALLET_ADDR[:10]}...')
        except ImportError:
            logger.warning('[VENUE] hyperliquid-python-sdk not installed — forcing paper mode')
            self._paper = True
        except Exception as e:
            logger.error(f'[VENUE] Failed to init Hyperliquid client: {e}')
            self._paper = True

    # ── Market Order (Taker) ─────────────────────────────────────────
    def market_order(
        self,
        symbol: str,
        is_buy: bool,
        size: float,         # in coin (not USD)
        slippage: float = 0.001,  # 0.1% slippage tolerance
    ) -> dict:
        """
        Place a market order (crosses spread → taker fee applies).

        Args:
            symbol:   e.g. 'ETH'
            is_buy:   True = long/cover, False = short/sell
            size:     coin amount (USD / mark_price)
            slippage: max acceptable slippage fraction

        Returns:
            { 'success', 'order_id', 'filled_price', 'reason' }
        """
        side_str = 'BUY' if is_buy else 'SELL'

        if self._paper:
            oid = f'paper_{symbol}_{side_str}_{int(time.time()*1000)}'
            logger.info(f'[VENUE-PAPER] Market {side_str} {size:.4f} {symbol}')
            return {'success': True, 'order_id': oid, 'filled_price': 0.0, 'reason': 'paper'}

        try:
            result = self.exchange.market_open(
                coin=symbol,
                is_buy=is_buy,
                sz=round(size, 4),
                slippage=slippage,
            )
            status = result.get('status', 'unknown')
            if status == 'ok':
                filled = result['response']['data']['statuses'][0]
                oid    = filled.get('resting', {}).get('oid', 'filled')
                price  = float(filled.get('filled', {}).get('avgPx', 0))
                logger.info(f'[VENUE] Market {side_str} {size:.4f} {symbol} @ {price:.4f} — {oid}')
                return {'success': True, 'order_id': str(oid), 'filled_price': price, 'reason': 'ok'}
            else:
                reason = str(result)
                logger.error(f'[VENUE] Market order failed: {reason}')
                return {'success': False, 'order_id': '', 'filled_price': 0.0, 'reason': reason}

        except Exception as e:
            logger.error(f'[VENUE] Market order exception: {e}')
            return {'success': False, 'order_id': '', 'filled_price': 0.0, 'reason': str(e)}

    # ── Limit Order (Maker) ──────────────────────────────────────────
    def limit_order(
        self,
        symbol: str,
        is_buy: bool,
        size: float,
        limit_px: float,
        post_only: bool = True
    ) -> dict:
        """Place a Maker Limit Order to earn the spread/avoid taker fees."""
        side_str = 'BUY' if is_buy else 'SELL'
        if self._paper:
            oid = f'paper_limit_{symbol}_{side_str}_{int(time.time()*1000)}'
            logger.info(f'[VENUE-PAPER] Limit {side_str} {size:.4f} {symbol} @ {limit_px:.4f} (Alo={post_only})')
            # Simulated paper order fills 100% instantly for ease unless a robust sim is built
            return {'success': True, 'order_id': oid, 'reason': 'paper'}

        try:
            # tif: 'Alo' = Add liquidity only (Post-only), 'Gtc' = Good til cancelled
            order_type = {"limit": {"tif": "Alo" if post_only else "Gtc"}}
            
            result = self.exchange.order(
                name=symbol,
                is_buy=is_buy,
                sz=round(size, 4),
                limit_px=limit_px,
                order_type=order_type,
                reduce_only=False
            )
            status = result.get('status', 'unknown')
            if status == 'ok':
                filled = result['response']['data']['statuses'][0]
                oid = filled.get('resting', {}).get('oid')
                if not oid:
                    oid = filled.get('filled', {}).get('oid', 'filled')
                logger.info(f'[VENUE] Limit {side_str} {size:.4f} {symbol} @ {limit_px:.4f} — {oid}')
                return {'success': True, 'order_id': str(oid), 'reason': 'ok'}
            else:
                reason = str(result)
                logger.error(f'[VENUE] Limit order failed: {reason}')
                return {'success': False, 'order_id': '', 'reason': reason}
        except Exception as e:
            logger.error(f'[VENUE] Limit order exception: {e}')
            return {'success': False, 'order_id': '', 'reason': str(e)}

    # ── Order Management ─────────────────────────────────────────────
    def cancel_order(self, symbol: str, order_id: str) -> dict:
        """Cancel an existing order by its ID."""
        if self._paper or order_id.startswith('paper_'):
            logger.info(f'[VENUE-PAPER] Cancel {symbol} order {order_id}')
            return {'success': True, 'reason': 'paper'}
            
        if not order_id or not order_id.isdigit():
            return {'success': True, 'reason': 'already_filled_or_invalid'}
            
        try:
            result = self.exchange.cancel(name=symbol, oid=int(order_id))
            status = result.get('status', 'unknown')
            if status == 'ok':
                logger.info(f'[VENUE] Cancelled {symbol} limit order {order_id}')
                return {'success': True, 'reason': 'ok'}
            else:
                return {'success': False, 'reason': str(result)}
        except Exception as e:
            logger.error(f'[VENUE] Cancel order exception: {e}')
            return {'success': False, 'reason': str(e)}

    def get_order_status(self, symbol: str, order_id: str) -> dict:
        """Check order status: 'open', 'filled', 'canceled', 'unknown'"""
        if self._paper or order_id.startswith('paper_'):
             return {'status': 'filled', 'filled_size': 0.0, 'avg_price': 0.0} # simulate fill immediately for limit
             
        if not order_id or not order_id.isdigit():
             return {'status': 'filled', 'filled_size': 0.0, 'avg_price': 0.0}
             
        try:
            res = self.info.query_order_by_oid(HL_WALLET_ADDR, int(order_id))
            if isinstance(res, dict) and 'order' in res:
                order_data = res['order'].get('order', {})
                status = res['order'].get('status', 'unknown').lower()
                sz = float(order_data.get('origSz', 0.0))
                remaining = float(order_data.get('sz', 0.0))
                filled_size = sz - remaining
                return {'status': status, 'filled_size': filled_size, 'avg_price': float(order_data.get('limitPx', 0))}
                
            return {'status': 'unknown', 'filled_size': 0.0, 'avg_price': 0.0}
        except Exception as e:
            logger.error(f'[VENUE] Get order status exception: {e}')
            return {'status': 'unknown', 'filled_size': 0.0, 'avg_price': 0.0}

    # ── Close Position (Market) ──────────────────────────────────────
    def close_position(
        self,
        symbol: str,
        is_buy: bool,   # True if closing a short (buy to close), False if closing a long
        size: float,
        slippage: float = 0.002,  # slightly wider slippage for exits
    ) -> dict:
        """Close an open position at market price."""
        if self._paper:
            oid = f'paper_close_{symbol}_{int(time.time()*1000)}'
            logger.info(f'[VENUE-PAPER] Close {"SHORT" if is_buy else "LONG"} {size:.4f} {symbol}')
            return {'success': True, 'order_id': oid, 'filled_price': 0.0, 'reason': 'paper'}

        try:
            # Hyperliquid uses market_close for position reduction
            result = self.exchange.market_close(
                coin=symbol,
                sz=round(size, 4),
                slippage=slippage,
            )
            status = result.get('status', 'unknown')
            if status == 'ok':
                filled = result['response']['data']['statuses'][0]
                price  = float(filled.get('filled', {}).get('avgPx', 0))
                logger.info(f'[VENUE] Closed {size:.4f} {symbol} @ {price:.4f}')
                return {'success': True, 'order_id': 'closed', 'filled_price': price, 'reason': 'ok'}
            else:
                reason = str(result)
                logger.error(f'[VENUE] Close failed: {reason}')
                return {'success': False, 'filled_price': 0.0, 'reason': reason}

        except Exception as e:
            logger.error(f'[VENUE] Close exception: {e}')
            return {'success': False, 'order_id': '', 'filled_price': 0.0, 'reason': str(e)}

    # ── Account Info ─────────────────────────────────────────────────
    def get_account_value(self) -> float:
        """Return current account equity in USD."""
        if self._paper or not self.info:
            return 0.0
        try:
            state = self.info.user_state(HL_WALLET_ADDR)
            return float(state.get('marginSummary', {}).get('accountValue', 0))
        except Exception as e:
            logger.warning(f'[VENUE] get_account_value failed: {e}')
            return 0.0

    def get_open_positions(self) -> list[dict]:
        """Return current open perps positions from the venue."""
        if self._paper or not self.info:
            return []
        try:
            state = self.info.user_state(HL_WALLET_ADDR)
            return state.get('assetPositions', [])
        except Exception as e:
            logger.warning(f'[VENUE] get_open_positions failed: {e}')
            return []


# Global singleton — shared across main loop + position manager
venue = HyperliquidClient()
