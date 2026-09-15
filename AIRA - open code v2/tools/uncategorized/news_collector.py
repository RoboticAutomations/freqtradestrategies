"""
News collector for MultibotdashboardV7
Fetches financial news from various sources
"""

import asyncio
import aiohttp
import asyncpg
from datetime import datetime, timedelta
from typing import List, Dict
import logging

from src.services.runtime_settings import create_finance_db_pool, get_runtime_setting

logger = logging.getLogger(__name__)

class NewsCollector:
    """Collects financial news."""
    
    def __init__(self):
        self.newsapi_key = None  # Add to config if available

    async def fetch_market_news(self) -> List[Dict]:
        """Fetch general market news from News API when a key is configured."""
        if not self.newsapi_key:
            return []

        url = "https://newsapi.org/v2/top-headlines"
        params = {
            'category': 'business',
            'language': 'en',
            'pageSize': 20,
        }
        headers = {
            'X-Api-Key': self.newsapi_key,
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, params=params, headers=headers, timeout=10) as response:
                    if response.status != 200:
                        logger.warning(f"NewsAPI fetch failed with status {response.status}")
                        return []

                    data = await response.json()
                    articles = data.get('articles', [])
                    return [
                        {
                            'title': item.get('title'),
                            'source': item.get('source', {}).get('name', 'NewsAPI'),
                            'url': item.get('url'),
                            'category': 'market',
                            'published_at': item.get('publishedAt'),
                            'sentiment_score': None,
                        }
                        for item in articles
                        if item.get('title') and item.get('url')
                    ]
            except Exception as e:
                logger.warning(f"NewsAPI fetch failed: {e}")

        return []
    
    async def fetch_crypto_news(self) -> List[Dict]:
        """Fetch crypto news from CryptoPanic or similar."""
        # Free endpoint without API key
        url = "https://cryptopanic.com/api/v1/posts/"
        params = {'public': 'true'}
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, params=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = data.get('results', [])
                        return [
                            {
                                'title': item.get('title'),
                                'source': item.get('source', {}).get('title', 'Unknown'),
                                'url': item.get('url'),
                                'category': 'crypto',
                                'published_at': item.get('created_at'),
                                'sentiment_score': None
                            }
                            for item in results[:20]
                        ]
            except Exception as e:
                logger.warning(f"CryptoPanic fetch failed: {e}")
        
        return []
    
    async def fetch_fallback_news(self) -> List[Dict]:
        """Fallback: Generate sample news for demo."""
        return [
            {
                'title': 'Fed Signals Potential Rate Cuts in Coming Months',
                'source': 'Bloomberg',
                'url': 'https://bloomberg.com',
                'category': 'market',
                'published_at': datetime.now().isoformat(),
                'sentiment_score': 0.5
            },
            {
                'title': 'Bitcoin ETF Inflows Reach Record High',
                'source': 'CoinDesk',
                'url': 'https://coindesk.com',
                'category': 'crypto',
                'published_at': datetime.now().isoformat(),
                'sentiment_score': 0.8
            },
            {
                'title': 'Tech Stocks Rally on AI Optimism',
                'source': 'Reuters',
                'url': 'https://reuters.com',
                'category': 'market',
                'published_at': datetime.now().isoformat(),
                'sentiment_score': 0.6
            }
        ]
    
    async def save_to_db(self, news: List[Dict]):
        """Save news to database."""
        from datetime import datetime
        pool = await create_finance_db_pool()
        
        async with pool.acquire() as conn:
            for item in news:
                # Check if news already exists (by URL)
                existing = await conn.fetchval(
                    "SELECT id FROM news WHERE url = $1",
                    item.get('url')
                )
                if existing:
                    continue

                category = item.get('category') or 'market'
                if category not in {'market', 'crypto', 'stock', 'forex'}:
                    category = 'market'
                
                # Convert published_at string to datetime
                published_at = item.get('published_at')
                if isinstance(published_at, str):
                    try:
                        # Handle ISO format with or without microseconds
                        if '.' in published_at:
                            published_at = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                        else:
                            published_at = datetime.fromisoformat(published_at.replace('Z', ''))
                    except:
                        published_at = None
                
                await conn.execute(
                    """
                    INSERT INTO news 
                    (title, source, url, category, published_at, sentiment_score)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    item['title'],
                    item['source'],
                    item.get('url'),
                    category,
                    published_at,
                    item.get('sentiment_score')
                )
        
        await pool.close()
        logger.info(f"Saved {len(news)} news items to DB")
    
    async def run(self):
        """Run the collector."""
        try:
            logger.info("Starting news collection...")
            self.newsapi_key = await get_runtime_setting("newsapi_key")
            news: List[Dict] = []

            if self.newsapi_key:
                news.extend(await self.fetch_market_news())

            news.extend(await self.fetch_crypto_news())
            
            if not news:
                news = await self.fetch_fallback_news()
            
            if news:
                await self.save_to_db(news)
                await self._log_sync('news', 'success', len(news))
            else:
                await self._log_sync('news', 'error', 0, 'No news received')
        except Exception as e:
            logger.error(f"News collector error: {e}")
            await self._log_sync('news', 'error', 0, str(e))
    
    async def _log_sync(self, source: str, status: str, records: int, error: str = None):
        """Log sync status."""
        pool = await create_finance_db_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO sync_log (source, status, records_processed, error_message, started_at, completed_at)
                VALUES ($1, $2, $3, $4, NOW(), NOW())
                """,
                source, status, records, error
            )
        await pool.close()

# Singleton instance
news_collector = NewsCollector()
