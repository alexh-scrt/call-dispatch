"""Unit tests for call_dispatch.config - settings loading and validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from call_dispatch.config import Settings, get_settings


MINIMAL_VALID_ENV: dict[str, str] = {
    "TWILIO_ACCOUNT_SID": "ACtest1234567890abcdef1234567890ab",
    "TWILIO_AUTH_TOKEN": "test_auth_token",
    "TWILIO_PHONE_NUMBER": "+15550001234",
    "OPENAI_API_KEY": "sk-test-key",
    "DEEPGRAM_API_KEY": "dg-test-key",
    "PUBLIC_BASE_URL": "https://example.ngrok.io",
}


def make_settings(**overrides: str) -> Settings:
    """Construct a Settings instance from the minimal valid env plus any overrides."""
    env = {**MINIMAL_VALID_ENV, **overrides}
    return Settings(**env)  # type: ignore[arg-type]


class TestSettingsDefaults:
    """Verify that optional fields use expected default values."""

    def test_default_openai_model(self) -> None:
        s = make_settings()
        assert s.openai_model == "gpt-4o"

    def test_default_host(self) -> None:
        s = make_settings()
        assert s.host == "0.0.0.0"

    def test_default_port(self) -> None:
        s = make_settings()
        assert s.port == 8000

    def test_default_database_url(self) -> None:
        s = make_settings()
        assert s.database_url == "sqlite:///./call_dispatch.db"

    def test_default_agent_max_tokens(self) -> None:
        s = make_settings()
        assert s.agent_max_tokens == 256

    def test_default_agent_temperature(self) -> None:
        s = make_settings()
        assert s.agent_temperature == 0.4

    def test_default_summarizer_max_tokens(self) -> None:
        s = make_settings()
        assert s.summarizer_max_tokens == 512

    def test_default_summarizer_temperature(self) -> None:
        s = make_settings()
        assert s.summarizer_temperature == 0.2

    def test_default_call_timeout_seconds(self) -> None:
        s = make_settings()
        assert s.call_timeout_seconds == 300

    def test_default_log_level(self) -> None:
        s = make_settings()
        assert s.log_level == "INFO"


class TestE164Validator:
    """Verify the E.164 phone number validator."""

    def test_valid_e164(self) -> None:
        s = make_settings(TWILIO_PHONE_NUMBER="+442071234567")
        assert s.twilio_phone_number == "+442071234567"

    def test_missing_plus_raises(self) -> None:
        with pytest.raises(ValidationError, match="E.164"):
            make_settings(TWILIO_PHONE_NUMBER="15550001234")

    def test_non_digit_raises(self) -> None:
        with pytest.raises(ValidationError):
            make_settings(TWILIO_PHONE_NUMBER="+1555-000-1234")

    def test_too_short_raises(self) -> None:
        with pytest.raises(ValidationError):
            make_settings(TWILIO_PHONE_NUMBER="+123")


class TestTemperatureValidator:
    """Verify temperature field validation."""

    def test_valid_temperature_zero(self) -> None:
        s = make_settings(AGENT_TEMPERATURE="0.0")
        assert s.agent_temperature == 0.0

    def test_valid_temperature_max(self) -> None:
        s = make_settings(AGENT_TEMPERATURE="2.0")
        assert s.agent_temperature == 2.0

    def test_negative_temperature_raises(self) -> None:
        with pytest.raises(ValidationError, match="Temperature"):
            make_settings(AGENT_TEMPERATURE="-0.1")

    def test_too_high_temperature_raises(self) -> None:
        with pytest.raises(ValidationError, match="Temperature"):
            make_settings(SUMMARIZER_TEMPERATURE="2.1")


class TestLogLevelValidator:
    """Verify log level normalisation and validation."""

    def test_lowercase_accepted(self) -> None:
        s = make_settings(LOG_LEVEL="debug")
        assert s.log_level == "DEBUG"

    def test_invalid_level_raises(self) -> None:
        with pytest.raises(ValidationError, match="log_level"):
            make_settings(LOG_LEVEL="VERBOSE")


class TestPublicBaseUrlValidator:
    """Verify trailing slash stripping."""

    def test_trailing_slash_stripped(self) -> None:
        s = make_settings(PUBLIC_BASE_URL="https://example.ngrok.io/")
        assert s.public_base_url == "https://example.ngrok.io"

    def test_no_trailing_slash_unchanged(self) -> None:
        s = make_settings(PUBLIC_BASE_URL="https://example.ngrok.io")
        assert s.public_base_url == "https://example.ngrok.io"


class TestPortValidator:
    """Verify port range validation."""

    def test_valid_port(self) -> None:
        s = make_settings(PORT="8080")
        assert s.port == 8080

    def test_port_zero_raises(self) -> None:
        with pytest.raises(ValidationError, match="port"):
            make_settings(PORT="0")

    def test_port_too_high_raises(self) -> None:
        with pytest.raises(ValidationError, match="port"):
            make_settings(PORT="65536")


class TestGetSettings:
    """Verify the get_settings caching behaviour."""

    def test_cache_clear_and_reload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """After clearing the cache, get_settings returns a fresh instance."""
        for key, val in MINIMAL_VALID_ENV.items():
            monkeypatch.setenv(key, val)

        get_settings.cache_clear()
        s1 = get_settings()
        s2 = get_settings()
        # Same cached instance
        assert s1 is s2
        get_settings.cache_clear()
