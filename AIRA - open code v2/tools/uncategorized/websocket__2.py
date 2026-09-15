"""WebSocket manager for real-time updates."""

import asyncio
import json
from datetime import datetime
from typing import Any, Optional

import structlog
from fastapi import WebSocket, WebSocketDisconnect

logger = structlog.get_logger()


class ConnectionManager:
    """Manages WebSocket connections and broadcasts."""

    def __init__(self):
        """Initialize connection manager."""
        self._active_connections: dict[str, list[WebSocket]] = {}
        self._user_connections: dict[str, list[WebSocket]] = {}
        self._broadcast_task: Optional[asyncio.Task] = None
        self._running = False

    async def connect(
        self,
        websocket: WebSocket,
        user_id: str,
        channel: str = "global",
    ) -> None:
        """Accept a WebSocket connection.

        Args:
            websocket: WebSocket connection to accept.
            user_id: ID of the authenticated user.
            channel: Channel to subscribe to (global, bot:{id}, etc.).
        """
        await websocket.accept()

        # Add to channel connections
        if channel not in self._active_connections:
            self._active_connections[channel] = []
        self._active_connections[channel].append(websocket)

        # Add to user connections
        if user_id not in self._user_connections:
            self._user_connections[user_id] = []
        self._user_connections[user_id].append(websocket)

        logger.info(
            "WebSocket connected",
            user_id=user_id,
            channel=channel,
            total_connections=sum(len(c) for c in self._active_connections.values()),
        )

    def disconnect(self, websocket: WebSocket, user_id: str) -> None:
        """Remove a WebSocket connection.

        Args:
            websocket: WebSocket connection to remove.
            user_id: ID of the user.
        """
        # Remove from all channels
        for channel, connections in list(self._active_connections.items()):
            if websocket in connections:
                connections.remove(websocket)
                if not connections:
                    del self._active_connections[channel]

        # Remove from user connections
        if user_id in self._user_connections:
            if websocket in self._user_connections[user_id]:
                self._user_connections[user_id].remove(websocket)
            if not self._user_connections[user_id]:
                del self._user_connections[user_id]

        logger.info(
            "WebSocket disconnected",
            user_id=user_id,
            total_connections=sum(len(c) for c in self._active_connections.values()),
        )

    async def send_personal_message(
        self,
        message: dict[str, Any],
        user_id: str,
    ) -> None:
        """Send a message to all connections for a specific user.

        Args:
            message: Message to send.
            user_id: ID of the user to send to.
        """
        if user_id not in self._user_connections:
            return

        message_data = json.dumps(message)
        disconnected = []

        for websocket in self._user_connections[user_id]:
            try:
                await websocket.send_text(message_data)
            except Exception:
                disconnected.append(websocket)

        # Clean up disconnected
        for ws in disconnected:
            self.disconnect(ws, user_id)

    async def broadcast_to_channel(
        self,
        message: dict[str, Any],
        channel: str,
    ) -> None:
        """Broadcast a message to all connections in a channel.

        Args:
            message: Message to broadcast.
            channel: Channel to broadcast to.
        """
        if channel not in self._active_connections:
            return

        message_data = json.dumps(message)
        disconnected = []

        for websocket in self._active_connections[channel]:
            try:
                await websocket.send_text(message_data)
            except Exception:
                disconnected.append(websocket)

        # Clean up disconnected (find user_id for each)
        for ws in disconnected:
            for user_id, connections in list(self._user_connections.items()):
                if ws in connections:
                    self.disconnect(ws, user_id)
                    break

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Broadcast a message to all connected clients.

        Args:
            message: Message to broadcast.
        """
        await self.broadcast_to_channel(message, "global")

    async def broadcast_bot_update(
        self,
        bot_id: str,
        event_type: str,
        data: dict[str, Any],
    ) -> None:
        """Broadcast a bot-specific update.

        Args:
            bot_id: ID of the bot.
            event_type: Type of event (metrics, health, trade, etc.).
            data: Event data.
        """
        message = {
            "type": event_type,
            "bot_id": bot_id,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Broadcast to bot-specific channel
        await self.broadcast_to_channel(message, f"bot:{bot_id}")

        # Also broadcast to global channel for dashboard updates
        await self.broadcast_to_channel(message, "global")

    async def broadcast_portfolio_update(self, data: dict[str, Any]) -> None:
        """Broadcast portfolio summary update.

        Args:
            data: Portfolio data.
        """
        message = {
            "type": "portfolio_update",
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
        }
        await self.broadcast_to_channel(message, "global")

    def get_connection_count(self) -> int:
        """Get total number of active connections."""
        return sum(len(c) for c in self._active_connections.values())

    def get_channel_count(self, channel: str) -> int:
        """Get number of connections in a channel."""
        return len(self._active_connections.get(channel, []))


# Singleton instance
ws_manager = ConnectionManager()
