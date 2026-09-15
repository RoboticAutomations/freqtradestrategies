"""
Crypto data collector for MultibotdashboardV7
Fetches data from CoinGecko API
"""

import asyncio
import aiohttp
import asyncpg
from datetime import datetime
from typing import List, Dict, Any
import logging

from src.services.runtime_settings import create_finance_db_pool

logger = logging.getLogger(__name__)

# Top coins to track
DEFAULT_COINS = [
    'bitcoin', 'ethereum', 'tether', 'binancecoin', 'solana', 'ripple', 'usd-coin',
    'cardano', 'dogecoin', 'avalanche-2', 'tron', 'chainlink', 'polkadot', 'polygon',
    'wrapped-bitcoin', 'litecoin', 'internet-computer', 'shiba-inu', 'dai', 'uniswap',
    'bitcoin-cash', 'stellar', 'cosmos', 'leo-token', 'filecoin', 'okb', 'ethereum-classic',
    'aptos', 'hedera-hashgraph', 'near', 'lido-dao', 'vechain', 'crypto-com-chain',
    'quant-network', 'mantle', 'arbitrum', 'optimism', 'maker', 'aave', 'rocket-pool',
    'the-graph', 'elrond-erd-2', 'algorand', 'fantom', 'eos', 'blockstack', 'render-token',
    'theta-token', 'immutable-x', 'tezos', 'decentraland', 'axie-infinity', 'flow',
    'neo', 'kava', 'curve-dao-token', 'chiliz', 'paxos-standard', 'frax', 'gala',
    'kucoin-shares', 'whitebit', 'iota', 'bitget-token', 'dydx', 'bittorrent',
    'kronos', 'neo-name-service', 'gatechain-token', 'pepe', 'floki', 'sui',
    'bone-shibaswap', 'radix', 'ecash', 'casper-network', 'injective-protocol',
    'gemini-dollar', 'conflux-token', 'mina-protocol', 'pancakeswap-token',
    '1inch', 'loopring', 'zilliqa', 'enjincoin', 'basic-attention-token',
    'stepn', 'nxm', 'oasis-network', 'gnosis', 'dash', 'trust-wallet-token',
    'theta-fuel', 'illuvium', ' Convex-finance', 'celsius-degree-token'
]


def sanitize(value: Any) -> Any:
    """
    Remove boolean values - they break PostgreSQL driver.
    CoinGecko returns 'false' (JSON) which Python converts to False (bool).
    """
    if isinstance(value, bool):
        return None
    if value is None:
        return None
    return value

def to_str(value: Any) -> str | None:
    """Safe string conversion, rejects booleans."""
    val = sanitize(value)
    if val is None:
        return None
    try:
        return str(val)
    except:
        return None

def to_float(value: Any) -> float | None:
    """Safe float conversion, rejects booleans."""
    val = sanitize(value)
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None

class CryptoCollector:
    """Collects cryptocurrency data from CoinGecko."""
    
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
    
    async def fetch_prices(self) -> List[Dict]:
        """Fetch current prices from CoinGecko."""
        url = f"{self.base_url}/coins/markets"
        params = {
            'vs_currency': 'usd',
            'ids': ','.join(DEFAULT_COINS),
            'order': 'market_cap_desc',
            'per_page': 100,
            'page': 1,
            'sparkline': 'false',  # Changed: string 'false' not boolean False
            'price_change_percentage': '24h'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"CoinGecko API error: {response.status}")
                    return []
    
    async def save_to_db(self, data: List[Dict]):
        """Save crypto data to database with aggressive type checking."""
        if not data:
            logger.warning("No data to save")
            return
        
        pool = await create_finance_db_pool()
        
        async with pool.acquire() as conn:
            for coin in data:
                try:
                    # EXPLICIT extraction with sanitization
                    coin_id = to_str(coin.get('id'))
                    symbol = to_str(coin.get('symbol'))
                    name = to_str(coin.get('name'))
                    price = to_float(coin.get('current_price'))
                    market_cap = to_float(coin.get('market_cap'))
                    volume = to_float(coin.get('total_volume'))
                    change_24h = to_float(coin.get('price_change_percentage_24h'))
                    
                    # Debug line - check your logs for this
                    logger.info(f"Processing {coin_id}: price={price}, cap={market_cap}")
                    
                    # Ensure symbol is uppercase only if it exists
                    if symbol:
                        symbol = symbol.upper()
                    
                    # Final safety check - if any critical field is None, skip or handle
                    if coin_id is None:
                        logger.error(f"Skipping coin with missing ID: {coin}")
                        continue
                    
                    await conn.execute(
                        """
                        INSERT INTO crypto_prices 
                        (coin_id, symbol, name, price_usd, market_cap, volume_24h, change_24h_pct, source)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, 'coingecko')
                        """,
                        coin_id,      # $1 - text
                        symbol,       # $2 - text  
                        name,         # $3 - text
                        price,        # $4 - numeric/float
                        market_cap,   # $5 - numeric/float
                        volume,       # $6 - numeric/float
                        change_24h    # $7 - numeric/float
                    )
                    
                except Exception as e:
                    logger.error(f"Failed to insert coin {coin.get('id', 'UNKNOWN')}: {e}")
                    logger.error(f"Problematic data: {coin}")
                    raise  # Re-raise to trigger error logging
        
        await pool.close()
        logger.info(f"Saved {len(data)} crypto prices to DB")
    
    async def run(self):
        """Run the collector."""
        try:
            logger.info("Starting crypto collection...")
            data = await self.fetch_prices()
            if data:
                await self.save_to_db(data)
                await self._log_sync('crypto_prices', 'success', len(data), None)
            else:
                await self._log_sync('crypto_prices', 'error', 0, 'No data received')
        except Exception as e:
            logger.error(f"Crypto collector error: {e}")
            # Ensure we convert error to string explicitly
            error_msg = str(e) if e else "Unknown error"
            await self._log_sync('crypto_prices', 'error', 0, error_msg)
    
    async def _log_sync(self, source: str, status: str, records: int, error: str | None):
        """Log sync status."""
        # Sanitize inputs to _log_sync too!
        source = to_str(source) or 'unknown'
        status = to_str(status) or 'unknown'
        error = to_str(error)  # Converts None to None, False to None, etc.
        
        pool = await create_finance_db_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO sync_log (source, status, records_processed, error_message, started_at, completed_at)
                VALUES ($1, $2, $3, $4, NOW(), NOW())
                """,
                source, 
                status, 
                int(records) if records is not None else 0, 
                error
            )
        await pool.close()

# Singleton instance
crypto_collector = CryptoCollector()
