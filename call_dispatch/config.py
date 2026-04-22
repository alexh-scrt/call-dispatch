"""Configuration module for call_dispatch.

Loads and validates all required environment variables for Twilio, OpenAI,
and Deepgram credentials using pydantic-settings. A single `settings` singleton
is exported for use throughout the application.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or a .env file.

    All fields with no default value are required. Fields with defaults are
    optional and will use the provided value if not overridden in the environment.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # Twilio credentials
    # ------------------------------------------------------------------ #
    twilio_account_sid: str = Field(
        ...,
        description="Twilio Account SID, found in the Twilio Console dashboard.",
    )
    twilio_auth_token: str = Field(
        ...,
        description="Twilio Auth Token, found in the Twilio Console dashboard.",
    )
    twilio_phone_number: str = Field(
        ...,
        description="Twilio phone number in E.164 format, e.g. +15551234567.",
    )

    # ------------------------------------------------------------------ #
    # OpenAI credentials
    # ------------------------------------------------------------------ #
    openai_api_key: str = Field(
        ...,
        description="OpenAI API key for GPT-4 conversation and summarisation.",
    )
    openai_model: str = Field(
        default="gpt-4o",
        description="OpenAI model identifier to use for agent and summariser calls.",
    )

    # ------------------------------------------------------------------ #
    # Deepgram credentials
    # ------------------------------------------------------------------ #
    deepgram_api_key: str = Field(
        ...,
        description="Deepgram API key for real-time audio transcription.",
    )

    # ------------------------------------------------------------------ #
    # Server / application settings
    # ------------------------------------------------------------------ #
    host: str = Field(
        default="0.0.0.0",
        description="Host address the Uvicorn server will bind to.",
    )
    port: int = Field(
        default=8000,
        description="Port the Uvicorn server will listen on.",
    )
    public_base_url: str = Field(
        ...,
        description=(
            "Publicly accessible base URL of this server, used to construct "
            "Twilio webhook callback URLs. Example: https://abc123.ngrok.io"
        ),
    )

    # ------------------------------------------------------------------ #
    # Database settings
    # ------------------------------------------------------------------ #
    database_url: str = Field(
        default="sqlite:///./call_dispatch.db",
        description="SQLite database URL. Supports relative and absolute paths.",
    )

    # ------------------------------------------------------------------ #
    # Optional tuning parameters
    # ------------------------------------------------------------------ #
    agent_max_tokens: int = Field(
        default=256,
        description="Maximum tokens for agent response generation.",
    )
    agent_temperature: float = Field(
        default=0.4,
        description="Sampling temperature for agent response generation (0.0-2.0).",
    )
    summarizer_max_tokens: int = Field(
        default=512,
        description="Maximum tokens for post-call summary generation.",
    )
    summarizer_temperature: float = Field(
        default=0.2,
        description="Sampling temperature for summariser (lower = more deterministic).",
    )
    call_timeout_seconds: int = Field(
        default=300,
        description="Maximum duration in seconds before a call is forcibly ended.",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR, or CRITICAL.",
    )

    # ------------------------------------------------------------------ #
    # Validators
    # ------------------------------------------------------------------ #
    @field_validator("twilio_phone_number")
    @classmethod
    def validate_e164(cls, v: str) -> str:
        """Ensure the Twilio phone number is in E.164 format."""
        stripped = v.strip()
        if not stripped.startswith("+"):
            raise ValueError(
                f"twilio_phone_number must be in E.164 format (start with '+'). Got: {stripped!r}"
            )
        digits = stripped[1:]
        if not digits.isdigit() or len(digits) < 7 or len(digits) > 15:
            raise ValueError(
                f"twilio_phone_number contains invalid digits: {stripped!r}"
            )
        return stripped

    @field_validator("agent_temperature", "summarizer_temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Ensure temperature values are within the valid OpenAI range."""
        if not 0.0 <= v <= 2.0:
            raise ValueError(f"Temperature must be between 0.0 and 2.0, got {v}")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure the log level is a recognised Python logging level name."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid_levels:
            raise ValueError(
                f"log_level must be one of {sorted(valid_levels)}, got {v!r}"
            )
        return upper

    @field_validator("public_base_url")
    @classmethod
    def validate_public_base_url(cls, v: str) -> str:
        """Strip trailing slash from the public base URL for consistency."""
        return v.rstrip("/")

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        """Ensure the port is in the valid TCP range."""
        if not 1 <= v <= 65535:
            raise ValueError(f"port must be between 1 and 65535, got {v}")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings singleton.

    Uses ``lru_cache`` so that the .env file and environment are only parsed
    once per process. Call ``get_settings.cache_clear()`` in tests to force
    re-loading.

    Returns:
        Settings: Validated application settings instance.

    Raises:
        pydantic.ValidationError: If any required variable is missing or invalid.
    """
    return Settings()  # type: ignore[call-arg]


# Module-level convenience alias
settings: Settings


def __getattr__(name: str) -> object:
    """Lazy-load `settings` at module attribute access time to allow tests
    to patch the environment before the singleton is created."""
    if name == "settings":
        return get_settings()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
