#!/usr/bin/env python3
"""Smoke-test a running dashboard instance."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def unwrap_payload(payload: Any) -> Any:
    if isinstance(payload, dict) and "data" in payload and payload.get("status") in {"success", "ok"}:
        return payload["data"]
    return payload


def record(results: list[CheckResult], name: str, ok: bool, detail: str) -> None:
    results.append(CheckResult(name=name, ok=ok, detail=detail))


def require(
    results: list[CheckResult], name: str, response: requests.Response, predicate, detail: str
) -> Any:
    try:
        response.raise_for_status()
        payload = unwrap_payload(response.json())
        ok = bool(predicate(payload))
        record(results, name, ok, detail if ok else f"{detail} [unexpected payload]")
        return payload
    except Exception as exc:
        record(results, name, False, f"{detail} [{exc}]")
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test a running dashboard")
    parser.add_argument("--frontend-url", required=True)
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--min-bots", type=int, default=1)
    parser.add_argument("--min-strategies", type=int, default=1)
    parser.add_argument("--require-historic", action="store_true")
    parser.add_argument("--timeout", type=int, default=20)
    args = parser.parse_args()

    results: list[CheckResult] = []
    session = requests.Session()

    try:
        frontend = session.get(args.frontend_url, timeout=args.timeout)
        record(
            results,
            "frontend",
            frontend.ok and "text/html" in frontend.headers.get("content-type", ""),
            f"GET {args.frontend_url} returned HTML",
        )
    except Exception as exc:
        record(results, "frontend", False, f"Frontend unreachable [{exc}]")

    try:
        login = session.post(
            f"{args.api_url}/auth/token",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={"username": args.username, "password": args.password},
            timeout=args.timeout,
        )
        login.raise_for_status()
        login_payload = unwrap_payload(login.json())
        token = login_payload["access_token"]
        session.headers.update({"Authorization": f"Bearer {token}"})
        record(results, "login", True, "Auth token issued")
    except Exception as exc:
        record(results, "login", False, f"Login failed [{exc}]")
        token = None

    if token is None:
        for result in results:
            print(f"[{'PASS' if result.ok else 'FAIL'}] {result.name}: {result.detail}")
        return 1

    setup_status = require(
        results,
        "setup-status",
        session.get(f"{args.api_url}/settings/setup-status", timeout=args.timeout),
        lambda payload: payload.get("needs_setup") is False,
        "Setup wizard completed",
    )

    bots = require(
        results,
        "bots",
        session.get(f"{args.api_url}/bots", timeout=args.timeout),
        lambda payload: isinstance(payload, list) and len(payload) >= args.min_bots,
        f"At least {args.min_bots} bots discovered",
    ) or []

    require(
        results,
        "portfolio-summary",
        session.get(f"{args.api_url}/portfolio/summary", timeout=args.timeout),
        lambda payload: payload.get("total_bots", 0) >= args.min_bots,
        "Portfolio summary available",
    )

    require(
        results,
        "discovery-status",
        session.get(f"{args.api_url}/discovery/status", timeout=args.timeout),
        lambda payload: payload.get("docker_available") is True,
        "Discovery status available",
    )

    require(
        results,
        "alerts",
        session.get(f"{args.api_url}/alerts", timeout=args.timeout),
        lambda payload: isinstance(payload, list) or (isinstance(payload, dict) and isinstance(payload.get("data"), list)),
        "Alerts endpoint responds",
    )

    require(
        results,
        "strategy-lab",
        session.get(f"{args.api_url}/strategy-lab/strategies", timeout=args.timeout),
        lambda payload: isinstance(payload, list) and len(payload) >= args.min_strategies,
        f"At least {args.min_strategies} strategies visible",
    )

    require(
        results,
        "pairlist-jobs",
        session.get(f"{args.api_url}/pairlist-selector/jobs", timeout=args.timeout),
        lambda payload: isinstance(payload, list),
        "Pairlist jobs endpoint responds",
    )

    require(
        results,
        "backtest-summary",
        session.get(f"{args.api_url}/backtest/summary", timeout=args.timeout),
        lambda payload: "total_strategies" in payload,
        "Backtest summary responds",
    )

    require(
        results,
        "finance-stocks",
        session.get(f"{args.api_url}/finance/stocks?limit=5", timeout=args.timeout),
        lambda payload: isinstance(payload, list) and len(payload) >= 1,
        "Finance stocks available",
    )

    require(
        results,
        "finance-news",
        session.get(f"{args.api_url}/finance/news?limit=5", timeout=args.timeout),
        lambda payload: isinstance(payload, list) and len(payload) >= 1,
        "Finance news available",
    )

    require(
        results,
        "finance-economic",
        session.get(f"{args.api_url}/finance/economic", timeout=args.timeout),
        lambda payload: isinstance(payload, list) and len(payload) >= 1,
        "Finance economic indicators available",
    )

    historic = require(
        results,
        "historic-bots",
        session.get(f"{args.api_url}/historic/bots", timeout=args.timeout),
        lambda payload: isinstance(payload.get("bots"), list),
        "Historic endpoint responds",
    ) or {"bots": []}

    if args.require_historic:
        record(
            results,
            "historic-populated",
            len(historic.get("bots", [])) >= 1,
            "Historic dataset contains at least one bot",
        )

    if bots:
        first_bot_id = bots[0]["id"]
        require(
            results,
            "bot-metrics",
            session.get(f"{args.api_url}/bots/{first_bot_id}/metrics", timeout=args.timeout),
            lambda payload: isinstance(payload, dict) and payload.get("bot_id") == first_bot_id,
            "Bot metrics endpoint responds",
        )

    failed = [result for result in results if not result.ok]
    for result in results:
        print(f"[{'PASS' if result.ok else 'FAIL'}] {result.name}: {result.detail}")

    print()
    print(f"Summary: {len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
