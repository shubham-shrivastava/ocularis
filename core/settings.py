from __future__ import annotations

import os
from pathlib import Path
from typing import ClassVar

from pydantic import Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

try:
    from pydantic_settings import YamlConfigSettingsSource
except ImportError:  # pragma: no cover
    YamlConfigSettingsSource = None  # type: ignore[assignment,misc]


_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


class BrowserSettings(BaseSettings):
    headless: bool = False
    viewport_width: int = 1280
    viewport_height: int = 800
    screenshot_quality: int = 90
    timeout_ms: int = 30000


class ModelSettings(BaseSettings):
    base_url: str = "http://localhost:8000/v1"
    api_key: str = ""
    model_name: str = "ui-tars"
    timeout_s: int = 300
    max_tokens: int = 1024


class TextModelSettings(BaseSettings):
    base_url: str = ""
    api_key: str = ""
    model_name: str = "openai/gpt-4o-mini"
    timeout_s: int = 60
    max_tokens: int = 1200


class SupervisorSettings(BaseSettings):
    stuck_threshold: int = 3
    history_size: int = 5
    ssim_similarity_floor: float = 0.98
    max_retries: int = 3
    recovery_order: list[str] = Field(
        default=["SCROLL_DOWN", "REFRESH_PAGE", "GO_BACK", "CLICK_OFFSET", "escalate"]
    )


class SecuritySettings(BaseSettings):
    allowed_domains: list[str] = Field(default_factory=list)
    confirm_patterns: list[str] = Field(default_factory=list)
    block_password_fields: bool = True
    cdp_host: str = "127.0.0.1"
    cdp_port: int = 9222


class DbSettings(BaseSettings):
    url: str = "postgresql+asyncpg://ocularis:ocularis@localhost:5432/ocularis"
    pool_size: int = 10
    max_overflow: int = 20


class MemorySettings(BaseSettings):
    enabled: bool = False
    embedding_model: str = "text-embedding-3-small"
    top_k: int = 5
    similarity_threshold: float = 0.75


class GoalEvaluatorSettings(BaseSettings):
    confidence_threshold: float = 0.8


class LoggingSettings(BaseSettings):
    level: str = "INFO"
    json_format: bool = True
    file: str = "logs/ocularis.jsonl"


class TracesSettings(BaseSettings):
    dir: str = "traces"
    ttl_seconds: int = 604800


class PlannerSettings(BaseSettings):
    enabled: bool = False
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model_name: str = "gpt-4o-mini"
    max_sub_steps: int = 8


class ContextAwareSettings(BaseSettings):
    enabled: bool = False
    max_candidates: int = 12
    max_main_text_chars: int = 15000


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OCULARIS_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        yaml_file=str(_CONFIG_PATH),
    )

    _yaml_path: ClassVar[Path] = _CONFIG_PATH

    browser: BrowserSettings = Field(default_factory=BrowserSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    text_model: TextModelSettings = Field(default_factory=TextModelSettings)
    supervisor: SupervisorSettings = Field(default_factory=SupervisorSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    db: DbSettings = Field(default_factory=DbSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    goal_evaluator: GoalEvaluatorSettings = Field(default_factory=GoalEvaluatorSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    traces: TracesSettings = Field(default_factory=TracesSettings)
    planner: PlannerSettings = Field(default_factory=PlannerSettings)
    context_aware: ContextAwareSettings = Field(default_factory=ContextAwareSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Priority: init kwargs > env vars > .env file > YAML file > defaults."""
        sources: list[PydanticBaseSettingsSource] = [
            init_settings,
            env_settings,
            dotenv_settings,
        ]
        if YamlConfigSettingsSource is not None:
            yaml_path = str(cls._yaml_path)
            sources.append(YamlConfigSettingsSource(settings_cls, yaml_file=yaml_path))
        sources.append(file_secret_settings)
        return tuple(sources)


def _resolve_api_key_for_url(base_url: str) -> str:
    """Pick the right API key from env based on the service base URL."""
    url = base_url.lower()
    if "openrouter" in url:
        return os.environ.get("OPENROUTER_API_KEY", "")
    if "openai" in url:
        return os.environ.get("OPENAI_API_KEY", "")
    return os.environ.get("OPENAI_API_KEY", "") or os.environ.get("OPENROUTER_API_KEY", "")


def _resolve_api_keys(s: Settings) -> None:
    """Fill empty api_key fields from OPENAI_API_KEY / OPENROUTER_API_KEY env vars."""
    if not s.model.api_key:
        s.model.api_key = _resolve_api_key_for_url(s.model.base_url)

    if not s.text_model.base_url:
        s.text_model.base_url = s.model.base_url
    if not s.text_model.api_key:
        s.text_model.api_key = _resolve_api_key_for_url(s.text_model.base_url)

    if not s.planner.api_key:
        s.planner.api_key = _resolve_api_key_for_url(s.planner.base_url)


def load_settings(config_path: str | Path | None = None) -> Settings:
    """Load settings from config.yaml, then overlay environment variables.

    Priority: env vars (OCULARIS_*) > YAML file > defaults.
    API keys: resolved from OPENAI_API_KEY / OPENROUTER_API_KEY based on each
    service's base_url when the per-service api_key is empty.
    """
    from dotenv import load_dotenv  # noqa: PLC0415

    load_dotenv(override=False)

    if config_path is not None:
        Settings._yaml_path = Path(config_path)
    else:
        Settings._yaml_path = _CONFIG_PATH
    s = Settings()
    _resolve_api_keys(s)
    return s


# Module-level singleton; override in tests via dependency injection.
settings = load_settings()
