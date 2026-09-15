"""
Finviz Stock Screener API endpoints
"""

import logging
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import CurrentUser
from src.models import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/finviz", tags=["finviz"])


@router.get("/screener/presets")
async def get_screener_presets(
    current_user: CurrentUser,
) -> dict:
    """Get available Finviz screener presets/filters."""
    try:
        # Import finvizfinance here to avoid startup issues
        from finvizfinance.screener.overview import Overview
        
        # Get available filters
        overview = Overview()
        filters = overview.get_filters()
        
        return {
            "status": "success",
            "data": {
                "filters": filters,
                "presets": [
                    {"id": "top_gainers", "name": "Top Gainers", "description": "Stocks with highest daily gain"},
                    {"id": "top_losers", "name": "Top Losers", "description": "Stocks with highest daily loss"},
                    {"id": "unusual_volume", "name": "Unusual Volume", "description": "High volume stocks"},
                    {"id": "oversold", "name": "Oversold (RSI < 30)", "description": "Potentially oversold stocks"},
                    {"id": "overbought", "name": "Overbought (RSI > 70)", "description": "Potentially overbought stocks"},
                    {"id": "breakout", "name": "Breakout", "description": "Stocks breaking out of range"},
                ]
            }
        }
    except Exception as e:
        logger.error(f"Failed to get screener presets: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get presets: {str(e)}")


@router.get("/screener")
async def run_screener(
    current_user: CurrentUser,
    preset: Optional[str] = Query(None, description="Preset filter name"),
    price_min: Optional[float] = Query(None, description="Minimum price"),
    price_max: Optional[float] = Query(None, description="Maximum price"),
    market_cap_min: Optional[str] = Query(None, description="Minimum market cap (e.g., '1B')"),
    volume_min: Optional[int] = Query(None, description="Minimum average volume"),
    rsi_min: Optional[float] = Query(None, description="Minimum RSI"),
    rsi_max: Optional[float] = Query(None, description="Maximum RSI"),
    limit: int = Query(50, ge=1, le=100, description="Max results to return"),
) -> dict:
    """
    Run Finviz stock screener with specified filters.
    
    Returns list of stocks matching the criteria with key metrics.
    """
    try:
        from src.finvizfinance.screener.overview import Overview
        from src.finvizfinance.screener.ticker import Ticker
        
        # Build filters dictionary
        filters_dict = {}
        
        if preset:
            # Map presets to valid Finviz filters
            # Valid values from finvizfinance constants
            preset_filters = {
                "oversold": {"RSI (14)": "Oversold (30)"},
                "overbought": {"RSI (14)": "Overbought (70)"},
                "breakout": {"Performance": "Today Up"},
            }
            if preset in preset_filters:
                filters_dict.update(preset_filters[preset])
            # Note: top_gainers, top_losers, unusual_volume don't have direct filters
            # They will be sorted after fetching results
        
        # Add custom filters
        # Map price to valid Finviz options
        if price_min is not None:
            # Find the closest valid price option
            price_options = [1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            valid_price = None
            for p in price_options:
                if price_min <= p:
                    valid_price = p
                    break
            if valid_price:
                filters_dict["Price"] = f"Over ${valid_price}"
        
        if price_max is not None and price_max < 100:
            # Under options are different
            under_options = [1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            valid_price = None
            for p in reversed(under_options):
                if price_max >= p:
                    valid_price = p
                    break
            if valid_price:
                filters_dict["Price"] = f"Under ${valid_price}"
        
        if market_cap_min:
            filters_dict["Market Cap."] = f"+{market_cap_min}"
        
        if volume_min:
            if volume_min >= 1000000:
                filters_dict["Average Volume"] = "Over 1M"
            elif volume_min >= 500000:
                filters_dict["Average Volume"] = "Over 500K"
            elif volume_min >= 200000:
                filters_dict["Average Volume"] = "Over 200K"
        
        if rsi_min is not None:
            if rsi_min <= 30:
                filters_dict["RSI (14)"] = "Oversold (30)"
        
        if rsi_max is not None:
            if rsi_max >= 70:
                filters_dict["RSI (14)"] = "Overbought (70)"
        
        # Run screener
        overview = Overview()
        if filters_dict:
            overview.set_filter(filters_dict=filters_dict)
        
        # Fetch first page only to avoid loading all pages (551 pages = 11000+ stocks!)
        # We fetch more than needed for sorting, then limit results
        fetch_limit = min(limit * 3, 100)  # Fetch up to 100 rows or 3x the limit
        df = overview.screener_view(limit=fetch_limit, verbose=0)
        
        logger.info(f"Screener returned df: {df is not None}, empty: {df.empty if df is not None else 'N/A'}, shape: {df.shape if df is not None else 'N/A'}")
        if df is not None and not df.empty:
            logger.info(f"Columns: {list(df.columns)}")
            logger.info(f"First row: {df.iloc[0].to_dict() if len(df) > 0 else 'N/A'}")
        
        if df is None or df.empty:
            return {
                "status": "success",
                "data": {
                    "stocks": [],
                    "count": 0,
                    "filters_applied": filters_dict
                }
            }
        
        # Convert to list of dicts
        stocks = df.to_dict('records')
        logger.info(f"Converted to {len(stocks)} stocks")
        
        # Helper function to parse numeric values safely
        def parse_value(val):
            if val is None or pd.isna(val):
                return 0.0
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, str):
                # Remove %, K, M, B, $, commas
                cleaned = val.replace('%', '').replace('K', '').replace('M', '').replace('B', '').replace('$', '').replace(',', '').strip()
                try:
                    return float(cleaned) if cleaned else 0.0
                except ValueError:
                    return 0.0
            return 0.0
        
        # Apply preset sorting for filters that don't have direct filter options
        if preset == "top_gainers":
            stocks = sorted(stocks, key=lambda x: parse_value(x.get('Change')), reverse=True)
        elif preset == "top_losers":
            stocks = sorted(stocks, key=lambda x: parse_value(x.get('Change')))
        elif preset == "unusual_volume":
            stocks = sorted(stocks, key=lambda x: parse_value(x.get('Volume')), reverse=True)
        
        # Clean up data
        for stock in stocks:
            # Remove NaN values
            for k, v in list(stock.items()):
                if pd.isna(v):
                    stock[k] = None
        
        # Limit results
        stocks = stocks[:limit]
        
        logger.info(f"Returning {len(stocks)} stocks")
        return {
            "status": "success",
            "data": {
                "stocks": stocks,
                "count": len(stocks),
                "filters_applied": filters_dict,
                "preset": preset
            }
        }
        
    except Exception as e:
        logger.error(f"Screener failed: {e}")
        raise HTTPException(status_code=500, detail=f"Screener error: {str(e)}")


@router.get("/quote/{ticker}")
async def get_stock_quote(
    ticker: str,
    current_user: CurrentUser,
) -> dict:
    """
    Get detailed quote information for a specific stock.
    
    Includes fundamentals, technicals, and news.
    """
    try:
        from src.finvizfinance.quote import finvizfinance
        
        logger.info(f"Fetching quote for {ticker}")
        stock = finvizfinance(ticker.upper())
        
        # Get all available data
        fundament = stock.ticker_fundament()
        logger.info(f"Got fundamentals for {ticker}: {len(fundament)} fields")
        description = stock.ticker_description()
        logger.info(f"Got description for {ticker}: {description[:50] if description else 'None'}...")
        
        try:
            news = stock.ticker_news()
        except:
            news = []
        
        try:
            insider = stock.ticker_insider()
        except:
            insider = []
        
        return {
            "ticker": ticker.upper(),
            "fundamentals": fundament,
            "description": description,
            "news": news[:10] if isinstance(news, list) else [],  # Limit to 10 items
            "insider_trades": insider[:10] if isinstance(insider, list) else [],
        }
        
    except Exception as e:
        logger.error(f"Quote failed for {ticker}: {e}")
        raise HTTPException(status_code=404, detail=f"Could not fetch data for {ticker}: {str(e)}")


@router.get("/groups/{group_by}")
async def get_group_performance(
    group_by: str,  # sector, industry, country
    current_user: CurrentUser,
) -> dict:
    """
    Get performance data grouped by sector, industry, or country.
    """
    try:
        from finvizfinance.group import performance
        
        group_map = {
            "sector": "Sector",
            "industry": "Industry",
            "country": "Country",
        }
        
        if group_by not in group_map:
            raise HTTPException(status_code=400, detail=f"Invalid group_by. Use: {list(group_map.keys())}")
        
        group = performance.Performance()
        df = group.screener_view(group=group_map[group_by])
        
        if df is None or df.empty:
            return {
                "status": "success",
                "data": {
                    "groups": [],
                    "group_by": group_by
                }
            }
        
        return {
            "status": "success",
            "data": {
                "groups": df.to_dict('records'),
                "group_by": group_by
            }
        }
        
    except Exception as e:
        logger.error(f"Group performance failed: {e}")
        raise HTTPException(status_code=500, detail=f"Group data error: {str(e)}")
