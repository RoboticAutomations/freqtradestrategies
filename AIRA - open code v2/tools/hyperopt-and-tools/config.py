"""Configuration loader for the dashboard backend."""

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class ServerConfig(BaseModel):
    """Server configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    cors_origins: list[str] = Field(default_factory=lambda: [
        "http://localhost:5000",
        "http://localhost:5173",
        "http://127.0.0.1:5000",
    ])
    cors_origin_regex: str | None = (
        r"^https?://("
        r"localhost|127\.0\.0\.1|"
        r"10\.\d+\.\d+\.\d+|"
        r"192\.168\.\d+\.\d+|"
        r"172\.(1[6-9]|2\d|3[0-1])\.\d+\.\d+|"
        r"100\.\d+\.\d+\.\d+|"
        r"([a-zA-Z0-9-]+\.)*ts\.net"
        r")(?::\d+)?$"
    )


class DatabaseConfig(BaseModel):
    """Database configuration."""

    url: str = ""
    pool_size: int = 5
    echo: bool = False


class AnalyticsDatabaseConfig(BaseModel):
    """Analytics database configuration (read-only, separate from main DB)."""

    url: str = ""
    pool_size: int = 3
    echo: bool = False


class RedisConfig(BaseModel):
    """Redis cache configuration."""

    enabled: bool = False
    url: str = ""


class DockerDiscoveryConfig(BaseModel):
    """Docker discovery configuration."""

    enabled: bool = True
    socket: str = "unix://var/run/docker.sock"
    labels: list[str] = Field(default_factory=lambda: ["com.freqtrade.bot_name"])
    image_patterns: list[str] = Field(
        default_factory=lambda: ["freqtradeorg/freqtrade", "freqtrade/*"]
    )


class FilesystemDiscoveryConfig(BaseModel):
    """Filesystem discovery configuration."""

    enabled: bool = True
    scan_paths: list[str] = Field(default_factory=lambda: ["/opt/freqtrade/*/user_data"])
    patterns: list[str] = Field(
        default_factory=lambda: ["tradesv3.sqlite", "tradesv3.dryrun.sqlite"]
    )


class DiscoveryConfig(BaseModel):
    """Discovery configuration."""

    docker: DockerDiscoveryConfig = Field(default_factory=DockerDiscoveryConfig)
    filesystem: FilesystemDiscoveryConfig = Field(default_factory=FilesystemDiscoveryConfig)
    interval_seconds: int = 60


class HealthConfig(BaseModel):
    """Health check configuration."""

    check_interval_seconds: int = 10
    latency_threshold_ms: float = 5000
    error_rate_threshold: float = 0.3
    recovery_window_seconds: int = 60
    request_timeout_seconds: float = 10.0


class ApiDefaultsConfig(BaseModel):
    """Default API settings for Freqtrade bots."""

    timeout_seconds: int = 5
    username: str = "user"
    password: str = "pass"


class AuthConfig(BaseModel):
    """Authentication configuration."""

    jwt_secret: str = "development-secret-change-in-production"
    jwt_algorithm: str = "HS256"
    token_expire_minutes: int = 10080  # 7 days (7 * 24 * 60)
    refresh_expire_days: int = 30  # Extended to 30 days


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    format: str = "json"
    file: str | None = None


class Settings(BaseSettings):
    """Application settings loaded from YAML config and environment."""

    server: ServerConfig = Field(default_factory=ServerConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    analytics: AnalyticsDatabaseConfig = Field(default_factory=AnalyticsDatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)
    health: HealthConfig = Field(default_factory=HealthConfig)
    api_defaults: ApiDefaultsConfig = Field(default_factory=ApiDefaultsConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    class Config:
        env_prefix = "DASHBOARD_"
        env_nested_delimiter = "__"


def load_config(config_path: str | Path | None = None) -> Settings:
    """Load configuration from YAML file with environment variable substitution.

    Args:
        config_path: Path to the YAML configuration file.
                    If None, looks for config in standard locations.

    Returns:
        Settings object with loaded configuration.
    """
    if config_path is None:
        # Search for config in standard locations
        search_paths = [
            Path("config/dashboard.yaml"),
            Path("dashboard.yaml"),
            Path("/app/config/dashboard.yaml"),
            Path.home() / ".config" / "freqtrade-dashboard" / "dashboard.yaml",
        ]
        for path in search_paths:
            if path.exists():
                config_path = path
                break

    config_data: dict[str, Any] = {}

    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            raw_config = f.read()

        # Substitute environment variables (${VAR} and ${VAR:-default} syntax)
        import re
        def env_substitute(match):
            var_name = match.group(1)
            default = match.group(2)
            return os.environ.get(var_name, default or "")
        raw_config = re.sub(r'\$\{(\w+)(?::-([^}]*))?\}', env_substitute, raw_config)

        config_data = yaml.safe_load(raw_config) or {}

    # Override JWT secret from environment if set
    if jwt_secret := os.environ.get("JWT_SECRET"):
        if "auth" not in config_data:
            config_data["auth"] = {}
        config_data["auth"]["jwt_secret"] = jwt_secret

    # Override analytics database URL from environment if set
    if analytics_url := os.environ.get("ANALYTICS_DATABASE_URL"):
        if "analytics" not in config_data:
            config_data["analytics"] = {}
        config_data["analytics"]["url"] = analytics_url

    # Override Redis URL from environment if set
    if redis_url := os.environ.get("REDIS_URL"):
        if "redis" not in config_data:
            config_data["redis"] = {}
        config_data["redis"]["url"] = redis_url
        config_data["redis"]["enabled"] = True

    # Override database URL from environment if set
    if db_url := os.environ.get("DATABASE_URL"):
        if "database" not in config_data:
            config_data["database"] = {}
        config_data["database"]["url"] = db_url

    return Settings(**config_data)


# Global settings instance
settings = load_config()
