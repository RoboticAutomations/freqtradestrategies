"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api import api_router
from src.api.bot_ui import root_router as bot_ui_root_router
from src.config import settings

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
        if settings.logging.format == "json"
        else structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

# Set log level
logging.basicConfig(
    level=getattr(logging, settings.logging.level.upper()),
    format="%(message)s",
)

logger = structlog.get_logger()


async def _seed_settings_from_env() -> None:
    """Seed system settings from .env file so setup wizard shows pre-populated values.
    
    Skips gracefully if system_settings table doesn't exist yet (init scripts may still be running).
    """
    import os
    from sqlalchemy import select, text
    from sqlalchemy.exc import ProgrammingError
    from src.models import SystemSetting, async_session_maker

    # Map .env variable names to DB setting keys
    env_to_db = {
        "PUBLIC_HOST": "discovery_host_ip",
        "FREQTRADE_USERNAME": "api_username",
        "FREQTRADE_PASSWORD": "api_password",
        "FINANCE_DB_HOST": "finance_db_host",
        "FINANCE_DB_PORT": "finance_db_port",
        "FINANCE_DB_USER": "finance_db_user",
        "FINANCE_DB_PASSWORD": "finance_db_password",
        "FINANCE_DB_NAME": "finance_db_name",
        "DB_NAME": "analytics_db_url",
    }

    try:
        async with async_session_maker() as session:
            # Check if system_settings table exists before querying
            result = await session.execute(text("SELECT to_regclass('system_settings') IS NOT NULL"))
            table_exists = result.scalar()
            if not table_exists:
                logger.warning("system_settings table does not exist yet, skipping env seeding")
                return

            for env_var, db_key in env_to_db.items():
                env_value = os.environ.get(env_var, "").strip()
                if not env_value:
                    continue

                result = await session.execute(select(SystemSetting).where(SystemSetting.key == db_key))
                setting = result.scalar_one_or_none()

                if setting is None:
                    session.add(SystemSetting(
                        key=db_key,
                        value=env_value,
                        description=f"Auto-seeded from {env_var}"
                    ))
                    logger.info("Seeded setting from env", key=db_key, source=env_var)
                elif not setting.value.strip():
                    setting.value = env_value
                    logger.info("Updated empty setting from env", key=db_key, source=env_var)

            await session.commit()
    except ProgrammingError as e:
        logger.warning("Cannot seed settings from env — table may not exist yet", error=str(e))
    except Exception as e:
        logger.error("Unexpected error seeding settings from env", error=str(e))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager for startup/shutdown events."""
    # Startup
    logger.info("Starting Freqtrade Dashboard API", port=settings.server.port)

    # Initialize database connection
    # NOTE: Schema is managed by Alembic migrations (run in Docker CMD).
    logger.info("Database connection initialized (alembic-managed)")
    # NOTE: Admin user seeding is handled via Alembic migration 002_seed_admin.
    # No runtime user creation here (keeps startup deterministic).

    # Seed system settings from .env (so setup wizard sees pre-populated values)
    await _seed_settings_from_env()
    logger.info("System settings seeded from environment")

    # Start discovery service
    from src.services.discovery.scheduler import discovery_scheduler

    await discovery_scheduler.start()
    logger.info("Discovery scheduler started")

    # Start health monitoring
    from src.services.health import health_monitor

    await health_monitor.start()
    logger.info("Health monitor started")

    # Start trade monitoring for live updates
    from src.services.trade_monitor import trade_monitor

    await trade_monitor.start()
    logger.info("Trade monitor started")

    # Start log monitoring for rate limits
    from src.services.log_monitor import log_monitor

    await log_monitor.start()
    logger.info("Log monitor started")

    # Start cache service
    from src.services.cache import cache

    await cache.start()
    logger.info("Cache service started")

    # Start Strategy Lab (V6) - initialize ftmanager components
    from src.api.strategy_lab import startup_strategy_lab

    await startup_strategy_lab()
    logger.info("Strategy Lab initialized")

    # Start Finance Data Collectors
    from src.services.finance_collectors import finance_scheduler

    await finance_scheduler.start()
    logger.info("Finance Data Collectors started")

    yield

    # Shutdown
    logger.info("Shutting down Freqtrade Dashboard API")

    # Stop Finance Data Collectors
    await finance_scheduler.stop()
    logger.info("Finance Data Collectors stopped")

    # Stop Strategy Lab (V6)
    from src.api.strategy_lab import app_state

    if app_state:
        # Clean up any running workflows
        logger.info("Strategy Lab shutting down")
    logger.info("Strategy Lab stopped")

    # Stop cache service
    await cache.stop()
    logger.info("Cache service stopped")

    # Stop log monitor
    await log_monitor.stop()
    logger.info("Log monitor stopped")

    # Stop trade monitor
    await trade_monitor.stop()
    logger.info("Trade monitor stopped")

    # Stop health monitor
    await health_monitor.stop()
    logger.info("Health monitor stopped")

    # Stop discovery scheduler
    await discovery_scheduler.stop()
    logger.info("Discovery scheduler stopped")

    # Clean up connector manager
    from src.services.connectors.manager import connector_manager

    await connector_manager.close_all()

    # Clean up analytics DB
    from src.db.analytics import close_analytics_db

    await close_analytics_db()
    logger.info("Analytics DB connection closed")

    # Clean up main DB engine
    from src.models import engine

    await engine.dispose()
    logger.info("Main DB connection closed")


# Create FastAPI application
app = FastAPI(
    title="Freqtrade Multi-Bot Dashboard",
    description="API for monitoring and controlling multiple Freqtrade trading bots",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.server.cors_origins,
    allow_origin_regex=settings.server.cors_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle uncaught exceptions globally."""
    logger.error(
        "Unhandled exception",
        path=request.url.path,
        method=request.method,
        error=str(exc),
        exc_info=True,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "error": "Internal server error",
            "data": None,
        },
    )


# Health check endpoint
@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint for container orchestration."""
    return {"status": "healthy", "service": "freqtrade-dashboard"}


# Root-level FreqUI asset fallback routes for embedded bot UIs
app.include_router(bot_ui_root_router)


# Include API routes
app.include_router(api_router, prefix="/api/v1")

# Include Strategy Lab routes (V6)
from src.api.strategy_lab import router as strategy_lab_router
app.include_router(strategy_lab_router, prefix="/api/v1", tags=["strategy-lab"])

# Include FinanceData routes (AlexFinanceData integration)
from src.api.finance import router as finance_router
app.include_router(finance_router, prefix="/api/v1", tags=["finance"])

# Include Agent routes (V8 Agent Strategy)
from src.api.agent import router as agent_router
app.include_router(agent_router, prefix="/api/v1", tags=["agent"])

# Include WebSocket routes (directly on app, not under /api/v1)
from src.api.websocket import router as ws_router
app.include_router(ws_router, prefix="/api/v1", tags=["websocket"])

# Include Pairlist Selector routes
from src.api.pairlist_selector import router as pairlist_router
app.include_router(pairlist_router, prefix="/api/v1", tags=["pairlist-selector"])

# Include Settings routes (OLD - disabled, using unified settings instead)
# from src.api.settings import router as settings_router
# app.include_router(settings_router, prefix="/api/v1", tags=["settings"])

# Include Unified Settings routes (at /api/v1/settings)
from src.api.unified_settings import router as unified_settings_router
app.include_router(unified_settings_router, prefix="/api/v1", tags=["settings"])

# Include Pairlist Results routes
from src.api.pairlist_results import router as pairlist_results_router
app.include_router(pairlist_results_router, prefix="/api/v1", tags=["pairlist-results"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=settings.server.host,
        port=settings.server.port,
        workers=settings.server.workers,
        reload=True,
    )
