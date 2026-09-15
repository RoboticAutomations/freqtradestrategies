"""Proxy routes for embedded Freqtrade UI access."""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
from datetime import timedelta
from typing import Annotated
import httpx
import websockets
from fastapi import APIRouter, Depends, HTTPException, Query, Request, WebSocket, WebSocketDisconnect, status
from fastapi.responses import HTMLResponse, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import CurrentUser
from src.models import get_db
from src.models.bot import Bot
from src.services.runtime_settings import get_runtime_setting
from src.utils.security import create_access_token, decode_token

router = APIRouter()
root_router = APIRouter()

BOT_UI_TOKEN_TTL_MINUTES = 5
FREQTRADE_AUTH_STORAGE_KEY = "ftAuthLoginInfo"
FREQTRADE_SELECTED_BOT_KEY = "ftSelectedBot"
FREQTRADE_EMBED_BOT_ID = "ftbot.0"

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "content-encoding",
    "content-length",
    "host",
}


def _bot_ui_cookie_name(bot_id: str) -> str:
    return f"bot_ui_token_{bot_id.replace('-', '_')}"


def _normalize_proxy_path(path: str | None) -> str:
    return re.sub(r"^/+", "", path or "")


def _extract_bot_id_from_request(request: Request) -> str | None:
    referer = request.headers.get("referer", "")
    match = re.search(r"/api/v1/bot-ui/([0-9a-f-]+)/", referer)
    if match:
        return match.group(1)

    cookie_prefix = "bot_ui_token_"
    matching_keys = [key for key in request.cookies if key.startswith(cookie_prefix)]
    if len(matching_keys) == 1:
        return matching_keys[0][len(cookie_prefix) :].replace("_", "-")

    return None


def _build_bot_ui_token(bot_id: str, username: str) -> str:
    return create_access_token(
        {
            "sub": username,
            "bot_id": bot_id,
            "type": "bot_ui",
        },
        expires_delta=timedelta(minutes=BOT_UI_TOKEN_TTL_MINUTES),
    )


def _validate_bot_ui_token(token: str | None, bot_id: str) -> None:
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bot UI token")
    payload = decode_token(token)
    if payload is None or payload.get("type") != "bot_ui" or payload.get("bot_id") != bot_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid bot UI token")


async def _get_bot_or_404(bot_id: str, db: AsyncSession) -> Bot:
    result = await db.execute(select(Bot).where(Bot.id == bot_id))
    bot = result.scalar_one_or_none()
    if bot is None or not bot.api_url:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Bot UI not available")
    return bot


async def _get_bot_api_credentials() -> tuple[str, str]:
    username = await get_runtime_setting("api_username", "user")
    password = await get_runtime_setting("api_password", "pass")
    if not username or not password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Bot API credentials are not configured",
        )
    return username, password


async def _login_to_bot(bot: Bot, proxy_base_url: str) -> dict[str, str | int | bool]:
    username, password = await _get_bot_api_credentials()
    login_url = f"{bot.api_url.rstrip('/')}/api/v1/token/login"

    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.post(login_url, json={}, auth=(username, password))
        response.raise_for_status()
        payload = response.json()

    access_token = payload.get("access_token")
    refresh_token = payload.get("refresh_token")
    if not access_token or not refresh_token:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Bot login did not return tokens")

    return {
        "botName": bot.name,
        "apiUrl": proxy_base_url.rstrip("/"),
        "username": username,
        "accessToken": access_token,
        "refreshToken": refresh_token,
        "autoRefresh": True,
        "sortId": 0,
    }


def _rewrite_html(html: str, proxy_prefix: str, login_info: dict[str, str | int | bool]) -> str:
    prefix = proxy_prefix.rstrip("/")
    prefix_with_slash = f"{prefix}/"
    storage_map = {
        FREQTRADE_EMBED_BOT_ID: login_info,
    }
    bootstrap = json.dumps(storage_map)

    html = re.sub(r'href="/', f'href="{prefix}/', html)
    html = re.sub(r'src="/', f'src="{prefix}/', html)

    injection = f"""
<base href="{prefix_with_slash}">
<script>
(() => {{
  const authKey = {json.dumps(FREQTRADE_AUTH_STORAGE_KEY)};
  const selectedKey = {json.dumps(FREQTRADE_SELECTED_BOT_KEY)};
  const botId = {json.dumps(FREQTRADE_EMBED_BOT_ID)};
  const authMap = {bootstrap};
  localStorage.setItem(authKey, JSON.stringify(authMap));
  localStorage.setItem(selectedKey, botId);
}})();
</script>
"""
    if "</head>" in html:
        html = html.replace("</head>", f"{injection}</head>", 1)
    else:
        html = injection + html
    return html


def _rewrite_location_header(location: str, proxy_prefix: str, target_base: str) -> str:
    if location.startswith(target_base):
        return proxy_prefix.rstrip("/") + location[len(target_base.rstrip("/")) :]
    if location.startswith("/"):
        return proxy_prefix.rstrip("/") + location
    return location


async def _proxy_bot_ui_http_request(
    request: Request,
    bot: Bot,
    proxy_prefix: str,
    target_path: str,
) -> Response:
    target_base = bot.api_url.rstrip("/")
    normalized_target_path = _normalize_proxy_path(target_path)
    if normalized_target_path == "api/v1/ui_version":
        normalized_target_path = "ui_version"
    target_url = f"{target_base}/{normalized_target_path}" if normalized_target_path else f"{target_base}/"

    query_params = [(key, value) for key, value in request.query_params.multi_items() if key != "token"]
    if query_params:
        target_url = f"{target_url}?{httpx.QueryParams(query_params)}"

    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS
    }
    headers["accept-encoding"] = "identity"

    body = await request.body()

    async with httpx.AsyncClient(follow_redirects=False, timeout=30.0) as client:
        upstream = await client.request(
            request.method,
            target_url,
            content=body if body else None,
            headers=headers,
        )

    response_headers = {
        key: value
        for key, value in upstream.headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS
    }

    if "location" in response_headers:
        response_headers["location"] = _rewrite_location_header(
            response_headers["location"],
            proxy_prefix,
            target_base,
        )

    content_type = upstream.headers.get("content-type", "")
    is_root_request = not normalized_target_path
    if is_root_request and "text/html" in content_type:
        proxy_base_url = str(request.base_url).rstrip("/") + proxy_prefix
        login_info = await _login_to_bot(bot, proxy_base_url)
        html = _rewrite_html(upstream.text, proxy_prefix, login_info)
        return HTMLResponse(content=html, status_code=upstream.status_code, headers=response_headers)

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=response_headers,
        media_type=upstream.headers.get("content-type"),
    )


@router.post("/bots/{bot_id}/ui-token")
async def create_bot_ui_token(
    bot_id: str,
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict[str, object]:
    await _get_bot_or_404(bot_id, db)
    token = _build_bot_ui_token(bot_id, current_user.username)
    return {"status": "success", "data": {"token": token}}


@root_router.api_route("/ui_version", methods=["GET", "HEAD"])
@root_router.api_route("/assets/{path:path}", methods=["GET", "HEAD"])
@root_router.api_route("/favicon.ico", methods=["GET", "HEAD"])
async def proxy_root_bot_ui_assets(
    request: Request,
    path: str = "",
    db: AsyncSession = Depends(get_db),
) -> Response:
    bot_id = _extract_bot_id_from_request(request)
    if not bot_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Bot UI context not found")

    token = request.cookies.get(_bot_ui_cookie_name(bot_id))
    _validate_bot_ui_token(token, bot_id)
    bot = await _get_bot_or_404(bot_id, db)

    target_path = request.url.path.lstrip("/")
    return await _proxy_bot_ui_http_request(
        request=request,
        bot=bot,
        proxy_prefix=f"/api/v1/bot-ui/{bot_id}",
        target_path=target_path,
    )


@router.websocket("/bot-ui/{bot_id}//api/v1/message/ws")
@router.websocket("/bot-ui/{bot_id}/api/v1/message/ws")
async def proxy_bot_ui_websocket(
    websocket: WebSocket,
    bot_id: str,
    token: str | None = Query(None),
    db: AsyncSession = Depends(get_db),
) -> None:
    ui_token = websocket.cookies.get(_bot_ui_cookie_name(bot_id)) or token
    _validate_bot_ui_token(ui_token, bot_id)
    bot = await _get_bot_or_404(bot_id, db)

    target_base = bot.api_url.rstrip("/")
    target_ws_base = re.sub(r"^http", "ws", target_base, count=1)
    query_string = websocket.scope.get("query_string", b"").decode("utf-8")
    target_ws_url = f"{target_ws_base}/api/v1/message/ws"
    if query_string:
        target_ws_url = f"{target_ws_url}?{query_string}"

    await websocket.accept()

    async with websockets.connect(target_ws_url, ping_interval=None) as upstream:
        async def client_to_upstream() -> None:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                if message.get("text") is not None:
                    await upstream.send(message["text"])
                elif message.get("bytes") is not None:
                    await upstream.send(message["bytes"])

        async def upstream_to_client() -> None:
            async for message in upstream:
                if isinstance(message, bytes):
                    await websocket.send_bytes(message)
                else:
                    await websocket.send_text(message)

        tasks = [
            asyncio.create_task(client_to_upstream()),
            asyncio.create_task(upstream_to_client()),
        ]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        for task in done:
            task.result()


@router.api_route("/bot-ui/{bot_id}/", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
@router.api_route("/bot-ui/{bot_id}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
@router.api_route("/bot-ui/{bot_id}//{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
@router.api_route("/bot-ui/{bot_id}/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
async def proxy_bot_ui(
    request: Request,
    bot_id: str,
    path: str = "",
    token: str | None = Query(None, description="Short-lived bot UI token"),
    db: AsyncSession = Depends(get_db),
) -> Response:
    token = token or request.cookies.get(_bot_ui_cookie_name(bot_id))
    _validate_bot_ui_token(token, bot_id)
    bot = await _get_bot_or_404(bot_id, db)
    proxy_prefix = f"/api/v1/bot-ui/{bot_id}"
    response = await _proxy_bot_ui_http_request(
        request=request,
        bot=bot,
        proxy_prefix=proxy_prefix,
        target_path=path,
    )
    response.set_cookie(
        key=_bot_ui_cookie_name(bot_id),
        value=token,
        max_age=BOT_UI_TOKEN_TTL_MINUTES * 60,
        httponly=True,
        samesite="lax",
        path="/",
    )
    return response
