"""API connector for Freqtrade REST API access."""

import time
from datetime import datetime
from typing import Any, Optional

import httpx
import structlog

from src.services.connectors.base import (
    BaseConnector,
    BotBalance,
    BotProfit,
    BotStatus,
    ConnectorResult,
    Trade,
)

logger = structlog.get_logger()


class APIConnector(BaseConnector):
    """Connector for Freqtrade REST API access.

    Provides real-time data access via the bot's HTTP API.
    Requires API to be enabled and accessible.
    """

    def __init__(
        self,
        bot_id: str,
        api_url: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        timeout: float = 10.0,
    ):
        """Initialize API connector.

        Args:
            bot_id: UUID of the bot.
            api_url: Base URL of the Freqtrade API (e.g., http://localhost:8080).
            username: API username if authentication required.
            password: API password if authentication required.
            timeout: Request timeout in seconds.
        """
        super().__init__(bot_id)
        self.api_url = api_url.rstrip("/")
        self.username = username
        self.password = password
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        self._token: Optional[str] = None
        self._token_expires: Optional[datetime] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                follow_redirects=True,
            )
        return self._client

    async def _authenticate(self) -> bool:
        """Authenticate with the API if credentials provided.

        Returns:
            True if authentication successful or not required.
        """
        if not self.username or not self.password:
            return True

        # Check if token is still valid
        if self._token and self._token_expires:
            if datetime.utcnow() < self._token_expires:
                return True

        try:
            client = await self._get_client()
            # Freqtrade requires HTTP Basic Auth for the token endpoint
            response = await client.post(
                f"{self.api_url}/api/v1/token/login",
                auth=(self.username, self.password),
            )

            if response.status_code == 200:
                data = response.json()
                self._token = data.get("access_token")
                # Freqtrade tokens typically last 15 minutes
                self._token_expires = datetime.utcnow()
                return True

            logger.warning(
                "API authentication failed",
                bot_id=self.bot_id,
                status=response.status_code,
            )
            return False

        except Exception as e:
            logger.error(
                "API authentication error",
                bot_id=self.bot_id,
                error=str(e),
            )
            return False

    def _get_headers(self) -> dict[str, str]:
        """Get request headers with authentication if available."""
        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    async def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs: Any,
    ) -> ConnectorResult:
        """Make API request with timing and error handling.

        Args:
            method: HTTP method (GET, POST, etc.).
            endpoint: API endpoint path.
            **kwargs: Additional arguments for httpx request.

        Returns:
            ConnectorResult with response data or error.
        """
        start = time.perf_counter()

        try:
            client = await self._get_client()
            url = f"{self.api_url}{endpoint}"
            
            # Use basic auth directly
            auth = None
            if self.username and self.password:
                auth = (self.username, self.password)

            response = await client.request(
                method,
                url,
                headers=self._get_headers(),
                auth=auth,
                **kwargs,
            )

            latency = (time.perf_counter() - start) * 1000

            if response.status_code == 200:
                self._available = True
                self._last_error = None
                return ConnectorResult(
                    success=True,
                    data=response.json(),
                    latency_ms=latency,
                )

            self._available = False
            self._last_error = f"HTTP {response.status_code}"
            return ConnectorResult(
                success=False,
                error=f"HTTP {response.status_code}: {response.text[:200]}",
                latency_ms=latency,
            )

        except httpx.TimeoutException:
            latency = (time.perf_counter() - start) * 1000
            self._available = False
            self._last_error = "Request timeout"
            return ConnectorResult(
                success=False,
                error="Request timeout",
                latency_ms=latency,
            )

        except httpx.ConnectError as e:
            latency = (time.perf_counter() - start) * 1000
            self._available = False
            self._last_error = "Connection failed"
            return ConnectorResult(
                success=False,
                error=f"Connection failed: {str(e)}",
                latency_ms=latency,
            )

        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            self._available = False
            self._last_error = str(e)
            logger.error(
                "API request error",
                bot_id=self.bot_id,
                endpoint=endpoint,
                error=str(e),
            )
            return ConnectorResult(
                success=False,
                error=str(e),
                latency_ms=latency,
            )

    async def check_health(self) -> ConnectorResult:
        """Check API health via ping endpoint."""
        result = await self._request("GET", "/api/v1/ping")

        if result.success:
            self._last_check = datetime.utcnow()

        return result

    async def get_status(self) -> ConnectorResult:
        """Get bot status from API."""
        result = await self._request("GET", "/api/v1/show_config")

        if not result.success:
            return result

        data = result.data
        return ConnectorResult(
            success=True,
            data=BotStatus(
                state=data.get("state", "unknown"),
                strategy=data.get("strategy"),
                exchange=data.get("exchange"),
                trading_mode=data.get("trading_mode"),
                is_dryrun=data.get("dry_run", True),
                version=data.get("version"),
            ),
            latency_ms=result.latency_ms,
        )

    async def get_profit(self) -> ConnectorResult:
        """Get profit metrics from API."""
        result = await self._request("GET", "/api/v1/profit")

        if not result.success:
            return result

        data = result.data

        # Parse dates if present (Freqtrade returns timestamps in milliseconds)
        first_trade = None
        latest_trade = None
        if data.get("first_trade_timestamp"):
            first_trade = datetime.fromtimestamp(data["first_trade_timestamp"] / 1000)
        if data.get("latest_trade_timestamp"):
            latest_trade = datetime.fromtimestamp(data["latest_trade_timestamp"] / 1000)

        return ConnectorResult(
            success=True,
            data=BotProfit(
                profit_closed_coin=data.get("profit_closed_coin", 0.0),
                profit_closed_percent=data.get("profit_closed_percent", 0.0),
                profit_closed_fiat=data.get("profit_closed_fiat", 0.0),
                profit_all_coin=data.get("profit_all_coin", 0.0),
                profit_all_percent=data.get("profit_all_percent", 0.0),
                profit_all_fiat=data.get("profit_all_fiat", 0.0),
                trade_count=data.get("trade_count", 0),
                closed_trade_count=data.get("closed_trade_count", 0),
                first_trade_date=first_trade,
                latest_trade_date=latest_trade,
                winning_trades=data.get("winning_trades", 0),
                losing_trades=data.get("losing_trades", 0),
            ),
            latency_ms=result.latency_ms,
        )

    async def get_balance(self) -> ConnectorResult:
        """Get balance from API."""
        result = await self._request("GET", "/api/v1/balance")

        if not result.success:
            return result

        data = result.data

        # Find stake currency balance
        stake_currency = data.get("stake", "USDT")
        stake_balance = 0.0
        currencies = data.get("currencies", [])

        for currency in currencies:
            if currency.get("currency") == stake_currency:
                stake_balance = currency.get("balance", 0.0)
                break

        return ConnectorResult(
            success=True,
            data=BotBalance(
                currency=stake_currency,
                total=data.get("total", 0.0),
                free=data.get("free", 0.0),
                used=data.get("used", 0.0),
                stake_currency=stake_currency,
                stake_currency_balance=stake_balance,
            ),
            latency_ms=result.latency_ms,
        )

    async def get_trades(
        self,
        limit: int = 50,
        offset: int = 0,
        is_open: Optional[bool] = None,
    ) -> ConnectorResult:
        """Get trades from API."""
        params: dict[str, Any] = {
            "limit": limit,
            "offset": offset,
        }

        result = await self._request("GET", "/api/v1/trades", params=params)

        if not result.success:
            return result

        trades_data = result.data.get("trades", [])
        trades = []

        for t in trades_data:
            # Filter by is_open if specified
            if is_open is not None and t.get("is_open") != is_open:
                continue

            trade = self._parse_trade(t)
            trades.append(trade)

        return ConnectorResult(
            success=True,
            data=trades,
            latency_ms=result.latency_ms,
        )

    async def get_open_trades(self) -> ConnectorResult:
        """Get open trades from API."""
        result = await self._request("GET", "/api/v1/status")

        if not result.success:
            return result

        trades = [self._parse_trade(t) for t in result.data]

        return ConnectorResult(
            success=True,
            data=trades,
            latency_ms=result.latency_ms,
        )

    async def get_closed_trades(self, limit: int = 50) -> ConnectorResult:
        """Get closed trades from API."""
        result = await self._request("GET", "/api/v1/trades", params={"limit": limit})

        if not result.success:
            return result

        trades_data = result.data.get("trades", [])
        # Filter to only closed trades
        trades = [self._parse_trade(t) for t in trades_data if not t.get("is_open", True)]

        return ConnectorResult(
            success=True,
            data=trades,
            latency_ms=result.latency_ms,
        )

    def _parse_trade(self, data: dict) -> Trade:
        """Parse trade data from API response."""
        open_date = datetime.fromisoformat(
            data["open_date"].replace("Z", "+00:00")
        ) if data.get("open_date") else datetime.utcnow()

        close_date = None
        if data.get("close_date"):
            close_date = datetime.fromisoformat(
                data["close_date"].replace("Z", "+00:00")
            )

        return Trade(
            trade_id=data.get("trade_id", 0),
            pair=data.get("pair", ""),
            is_open=data.get("is_open", True),
            open_rate=data.get("open_rate", 0.0),
            open_date=open_date,
            close_rate=data.get("close_rate"),
            close_date=close_date,
            stake_amount=data.get("stake_amount", 0.0),
            amount=data.get("amount", 0.0),
            profit_abs=data.get("profit_abs"),
            profit_ratio=data.get("profit_ratio"),
            stop_loss=data.get("stop_loss"),
            stop_loss_abs=data.get("stop_loss_abs"),
            take_profit=data.get("take_profit"),
            sell_reason=data.get("exit_reason") or data.get("sell_reason"),
            min_rate=data.get("min_rate"),
            max_rate=data.get("max_rate"),
            enter_tag=data.get("enter_tag"),
            exit_reason=data.get("exit_reason"),
            leverage=data.get("leverage", 1.0),
            is_short=data.get("is_short", False),
        )

    # Bot Control Methods

    async def start_bot(self) -> ConnectorResult:
        """Start the trading bot."""
        return await self._request("POST", "/api/v1/start")

    async def stop_bot(self) -> ConnectorResult:
        """Stop the trading bot."""
        return await self._request("POST", "/api/v1/stop")

    async def reload_config(self) -> ConnectorResult:
        """Reload bot configuration."""
        return await self._request("POST", "/api/v1/reload_config")

    async def force_exit(self, pair: str | None = None) -> ConnectorResult:
        """Force exit open trades.

        Args:
            pair: Optional pair to exit. If None, exits all trades.
        """
        if pair:
            return await self._request(
                "POST",
                "/api/v1/forceexit",
                json={"tradeid": "all", "pair": pair},
            )
        return await self._request(
            "POST",
            "/api/v1/forceexit",
            json={"tradeid": "all"},
        )

    async def force_exit_trade(self, trade_id: int) -> ConnectorResult:
        """Force exit a specific trade.

        Args:
            trade_id: ID of the trade to exit.
        """
        return await self._request(
            "POST",
            "/api/v1/forceexit",
            json={"tradeid": trade_id},
        )

    async def get_locks(self) -> ConnectorResult:
        """Get current pair locks."""
        return await self._request("GET", "/api/v1/locks")

    async def delete_lock(self, lock_id: int) -> ConnectorResult:
        """Delete a pair lock.

        Args:
            lock_id: ID of the lock to delete.
        """
        return await self._request("DELETE", f"/api/v1/locks/{lock_id}")

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
