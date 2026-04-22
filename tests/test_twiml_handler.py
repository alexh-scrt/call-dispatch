"""Unit tests for call_dispatch.twiml_handler - TwiML builders and webhook handlers.

Tests cover:
- TwiML builder functions (build_answer_twiml, build_say_twiml, etc.)
- Twilio status mapping (map_twilio_status)
- Twilio stream event helper functions
- Answer webhook endpoint
- Status callback endpoint
- Media stream WebSocket handler (basic path)
"""

from __future__ import annotations

import base64
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from xml.etree.ElementTree import fromstring

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from call_dispatch.models import CallRecord, CallStatus
from call_dispatch.twiml_handler import (
    build_answer_twiml,
    build_clear_message,
    build_hangup_twiml,
    build_mark_message,
    build_reject_twiml,
    build_say_twiml,
    extract_stream_sid,
    map_twilio_status,
    parse_media_event,
    router,
    ws_router,
)


# ---------------------------------------------------------------------------
# Test environment
# ---------------------------------------------------------------------------

_FAKE_ENV = {
    "TWILIO_ACCOUNT_SID": "ACtest1234567890abcdef1234567890ab",
    "TWILIO_AUTH_TOKEN": "test_auth_token",
    "TWILIO_PHONE_NUMBER": "+15550001234",
    "OPENAI_API_KEY": "sk-test-key",
    "DEEPGRAM_API_KEY": "dg-test-key",
    "PUBLIC_BASE_URL": "https://example.ngrok.io",
}


@pytest.fixture(autouse=True)
def patch_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, val in _FAKE_ENV.items():
        monkeypatch.setenv(key, val)
    from call_dispatch.config import get_settings
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# FastAPI test app
# ---------------------------------------------------------------------------


def make_test_app(store: Any = None) -> FastAPI:
    """Create a minimal FastAPI app with the TwiML routers mounted."""
    app = FastAPI()
    app.include_router(router)
    app.include_router(ws_router)
    if store is not None:
        app.state.store = store
    app.state.active_calls = {}
    return app


def make_mock_store(
    call_record: Any = None,
    update_status_result: bool = True,
    update_call_result: Any = None,
) -> MagicMock:
    """Create a mock store with configurable return values."""
    store = MagicMock()
    store.get_call = AsyncMock(return_value=call_record)
    store.update_status = AsyncMock(return_value=update_status_result)
    store.update_call = AsyncMock(return_value=update_call_result)
    return store


def make_call_record(call_id: str = "test-call-id-123") -> CallRecord:
    """Create a minimal CallRecord for testing."""
    return CallRecord(
        call_id=call_id,
        to_number="+15550001234",
        from_number="+15559876543",
        goal="Book a dentist appointment",
        status=CallStatus.INITIATING,
        twilio_call_sid="CA_test_sid",
    )


# ---------------------------------------------------------------------------
# build_answer_twiml tests
# ---------------------------------------------------------------------------


class TestBuildAnswerTwiml:
    """Tests for build_answer_twiml."""

    def test_returns_string(self) -> None:
        result = build_answer_twiml("wss://example.ngrok.io/ws/stream/abc")
        assert isinstance(result, str)

    def test_valid_xml(self) -> None:
        """The output is well-formed XML."""
        result = build_answer_twiml("wss://example.ngrok.io/ws/stream/abc")
        # Remove XML declaration for parsing
        xml_body = result.split("?>", 1)[-1].strip()
        root = fromstring(xml_body)
        assert root.tag == "Response"

    def test_contains_stream_element(self) -> None:
        """The TwiML contains a <Stream> element."""
        result = build_answer_twiml("wss://example.ngrok.io/ws/stream/abc")
        assert "Stream" in result
        assert "wss://example.ngrok.io/ws/stream/abc" in result

    def test_contains_start_element(self) -> None:
        """The TwiML contains a <Start> element."""
        result = build_answer_twiml("wss://example.ngrok.io/ws/stream/abc")
        assert "<Start>" in result

    def test_contains_pause_element(self) -> None:
        """The TwiML contains a <Pause> element."""
        result = build_answer_twiml("wss://example.ngrok.io/ws/stream/abc")
        assert "Pause" in result

    def test_pause_length_is_timeout(self) -> None:
        """The Pause length matches the provided timeout."""
        result = build_answer_twiml(
            "wss://example.ngrok.io/ws/stream/abc",
            timeout_seconds=120,
        )
        assert 'length="120"' in result

    def test_welcome_message_included(self) -> None:
        """A welcome message appears in the TwiML as a <Say> element."""
        result = build_answer_twiml(
            "wss://example.ngrok.io/ws/stream/abc",
            welcome_message="Please hold.",
        )
        assert "Please hold." in result
        assert "Say" in result

    def test_no_welcome_message_no_say(self) -> None:
        """Without a welcome message, no <Say> element is included."""
        result = build_answer_twiml(
            "wss://example.ngrok.io/ws/stream/abc",
            welcome_message=None,
        )
        assert "<Say" not in result

    def test_xml_declaration_present(self) -> None:
        """The output starts with an XML declaration."""
        result = build_answer_twiml("wss://example.ngrok.io/ws/stream/abc")
        assert result.startswith("<?xml version")

    def test_stream_url_in_attribute(self) -> None:
        """The stream URL is set as the 'url' attribute on <Stream>."""
        url = "wss://example.ngrok.io/ws/stream/my-call"
        result = build_answer_twiml(url)
        assert f'url="{url}"' in result

    def test_inbound_track_attribute(self) -> None:
        """The <Stream> element has track='inbound_track'."""
        result = build_answer_twiml("wss://example.ngrok.io/ws/stream/abc")
        assert 'track="inbound_track"' in result


# ---------------------------------------------------------------------------
# build_say_twiml tests
# ---------------------------------------------------------------------------


class TestBuildSayTwiml:
    """Tests for build_say_twiml."""

    def test_contains_message(self) -> None:
        result = build_say_twiml("Hello, how can I help you?")
        assert "Hello, how can I help you?" in result

    def test_valid_xml(self) -> None:
        result = build_say_twiml("Test message")
        xml_body = result.split("?>", 1)[-1].strip()
        root = fromstring(xml_body)
        assert root.tag == "Response"

    def test_say_element_present(self) -> None:
        result = build_say_twiml("Some text")
        assert "<Say" in result

    def test_no_hangup_by_default(self) -> None:
        result = build_say_twiml("Some text")
        assert "Hangup" not in result

    def test_hangup_included_when_requested(self) -> None:
        result = build_say_twiml("Goodbye.", hang_up=True)
        assert "Hangup" in result

    def test_default_voice_is_joanna(self) -> None:
        result = build_say_twiml("Hello")
        assert "Polly.Joanna" in result

    def test_custom_voice(self) -> None:
        result = build_say_twiml("Hello", voice="Polly.Matthew")
        assert "Polly.Matthew" in result


# ---------------------------------------------------------------------------
# build_hangup_twiml tests
# ---------------------------------------------------------------------------


class TestBuildHangupTwiml:
    """Tests for build_hangup_twiml."""

    def test_contains_hangup_verb(self) -> None:
        result = build_hangup_twiml()
        assert "Hangup" in result

    def test_valid_xml(self) -> None:
        result = build_hangup_twiml()
        xml_body = result.split("?>", 1)[-1].strip()
        root = fromstring(xml_body)
        assert root.tag == "Response"


# ---------------------------------------------------------------------------
# build_reject_twiml tests
# ---------------------------------------------------------------------------


class TestBuildRejectTwiml:
    """Tests for build_reject_twiml."""

    def test_contains_reject_verb(self) -> None:
        result = build_reject_twiml()
        assert "Reject" in result

    def test_default_reason_is_rejected(self) -> None:
        result = build_reject_twiml()
        assert 'reason="rejected"' in result

    def test_busy_reason(self) -> None:
        result = build_reject_twiml(reason="busy")
        assert 'reason="busy"' in result

    def test_invalid_reason_defaults_to_rejected(self) -> None:
        result = build_reject_twiml(reason="invalid_reason")
        assert 'reason="rejected"' in result


# ---------------------------------------------------------------------------
# map_twilio_status tests
# ---------------------------------------------------------------------------


class TestMapTwilioStatus:
    """Tests for map_twilio_status."""

    def test_queued_maps_to_initiating(self) -> None:
        assert map_twilio_status("queued") == CallStatus.INITIATING

    def test_ringing_maps_to_initiating(self) -> None:
        assert map_twilio_status("ringing") == CallStatus.INITIATING

    def test_in_progress_maps_to_in_progress(self) -> None:
        assert map_twilio_status("in-progress") == CallStatus.IN_PROGRESS

    def test_completed_maps_to_completed(self) -> None:
        assert map_twilio_status("completed") == CallStatus.COMPLETED

    def test_busy_maps_to_busy(self) -> None:
        assert map_twilio_status("busy") == CallStatus.BUSY

    def test_no_answer_maps_to_no_answer(self) -> None:
        assert map_twilio_status("no-answer") == CallStatus.NO_ANSWER

    def test_canceled_maps_to_cancelled(self) -> None:
        assert map_twilio_status("canceled") == CallStatus.CANCELLED

    def test_failed_maps_to_failed(self) -> None:
        assert map_twilio_status("failed") == CallStatus.FAILED

    def test_case_insensitive(self) -> None:
        assert map_twilio_status("COMPLETED") == CallStatus.COMPLETED
        assert map_twilio_status("In-Progress") == CallStatus.IN_PROGRESS

    def test_unknown_status_returns_none(self) -> None:
        assert map_twilio_status("unknown_status") is None
        assert map_twilio_status("") is None


# ---------------------------------------------------------------------------
# Webhook route tests
# ---------------------------------------------------------------------------


class TestAnswerWebhook:
    """Tests for the POST /twiml/answer/{call_id} endpoint."""

    def test_answer_returns_200(self) -> None:
        """Answer webhook returns HTTP 200."""
        store = make_mock_store(call_record=make_call_record())
        app = make_test_app(store=store)
        client = TestClient(app)
        resp = client.post("/twiml/answer/test-call-id")
        assert resp.status_code == 200

    def test_answer_returns_twiml_content_type(self) -> None:
        """Answer webhook returns application/xml content type."""
        store = make_mock_store(call_record=make_call_record())
        app = make_test_app(store=store)
        client = TestClient(app)
        resp = client.post("/twiml/answer/test-call-id")
        assert "xml" in resp.headers["content-type"]

    def test_answer_returns_valid_twiml(self) -> None:
        """Answer webhook returns well-formed TwiML with a Stream element."""
        store = make_mock_store(call_record=make_call_record())
        app = make_test_app(store=store)
        client = TestClient(app)
        resp = client.post("/twiml/answer/test-call-id")
        body = resp.text
        assert "Stream" in body
        assert "Start" in body

    def test_answer_stream_url_uses_public_base_url(self) -> None:
        """The stream URL in the TwiML uses the configured public base URL."""
        store = make_mock_store(call_record=make_call_record())
        app = make_test_app(store=store)
        client = TestClient(app)
        resp = client.post("/twiml/answer/my-call-id")
        # Should contain wss://example.ngrok.io (https -> wss conversion)
        assert "wss://example.ngrok.io" in resp.text
        assert "my-call-id" in resp.text

    def test_answer_without_store_still_returns_twiml(self) -> None:
        """Answer webhook works even when store is not configured."""
        app = make_test_app(store=None)
        client = TestClient(app)
        resp = client.post("/twiml/answer/test-call-id")
        assert resp.status_code == 200
        assert "Stream" in resp.text


class TestStatusCallback:
    """Tests for the POST /twiml/status/{call_id} endpoint."""

    def test_status_callback_returns_204(self) -> None:
        """Status callback returns HTTP 204 No Content."""
        store = make_mock_store(call_record=make_call_record())
        app = make_test_app(store=store)
        client = TestClient(app)
        resp = client.post(
            "/twiml/status/test-call-id",
            data={"CallStatus": "completed", "CallSid": "CA_test"},
        )
        assert resp.status_code == 204

    def test_status_callback_updates_store(self) -> None:
        """Status callback calls store.update_status with the correct status."""
        store = make_mock_store(call_record=make_call_record())
        app = make_test_app(store=store)
        client = TestClient(app)
        client.post(
            "/twiml/status/test-call-id",
            data={"CallStatus": "completed", "CallSid": "CA_test"},
        )
        store.update_status.assert_called_once()
        args, kwargs = store.update_status.call_args
        assert args[0] == "test-call-id"
        assert args[1] == CallStatus.COMPLETED

    def test_status_callback_failed_includes_error(self) -> None:
        """A failed call status callback includes the error message."""
        store = make_mock_store(call_record=make_call_record())
        app = make_test_app(store=store)
        client = TestClient(app)
        client.post(
            "/twiml/status/test-call-id",
            data={
                "CallStatus": "failed",
                "CallSid": "CA_test",
                "ErrorCode": "21215",
                "ErrorMessage": "Invalid phone number",
            },
        )
        _, kwargs = store.update_status.call_args
        assert kwargs.get("error_message") is not None
        assert "21215" in kwargs["error_message"]

    def test_status_callback_without_store_returns_204(self) -> None:
        """Status callback without a store configured still returns 204."""
        app = make_test_app(store=None)
        client = TestClient(app)
        resp = client.post(
            "/twiml/status/test-call-id",
            data={"CallStatus": "completed"},
        )
        assert resp.status_code == 204

    def test_status_callback_no_status_field_returns_204(self) -> None:
        """Status callback with no CallStatus field returns 204 without error."""
        store = make_mock_store(call_record=make_call_record())
        app = make_test_app(store=store)
        client = TestClient(app)
        resp = client.post("/twiml/status/test-call-id", data={})
        assert resp.status_code == 204
        store.update_status.assert_not_called()

    def test_status_callback_unknown_status_skips_update(self) -> None:
        """An unrecognised Twilio status does not trigger a store update."""
        store = make_mock_store(call_record=make_call_record())
        app = make_test_app(store=store)
        client = TestClient(app)
        client.post(
            "/twiml/status/test-call-id",
            data={"CallStatus": "some-unknown-status"},
        )
        store.update_status.assert_not_called()

    def test_status_callback_sets_ended_at_for_terminal_status(self) -> None:
        """Terminal statuses (completed, failed, etc.) include ended_at."""
        store = make_mock_store(call_record=make_call_record())
        app = make_test_app(store=store)
        client = TestClient(app)
        client.post(
            "/twiml/status/test-call-id",
            data={"CallStatus": "completed"},
        )
        _, kwargs = store.update_status.call_args
        assert kwargs.get("ended_at") is not None

    def test_status_callback_in_progress_no_ended_at(self) -> None:
        """Non-terminal status (in-progress) does not set ended_at."""
        store = make_mock_store(call_record=make_call_record())
        app = make_test_app(store=store)
        client = TestClient(app)
        client.post(
            "/twiml/status/test-call-id",
            data={"CallStatus": "in-progress"},
        )
        _, kwargs = store.update_status.call_args
        assert kwargs.get("ended_at") is None


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestBuildMarkMessage:
    """Tests for build_mark_message."""

    def test_returns_json_string(self) -> None:
        result = build_mark_message("stream-sid-123", "agent-done")
        assert isinstance(result, str)
        data = json.loads(result)
        assert data["event"] == "mark"

    def test_stream_sid_included(self) -> None:
        result = build_mark_message("stream-sid-123", "my-mark")
        data = json.loads(result)
        assert data["streamSid"] == "stream-sid-123"

    def test_mark_name_included(self) -> None:
        result = build_mark_message("ssid", "end-of-speech")
        data = json.loads(result)
        assert data["mark"]["name"] == "end-of-speech"


class TestBuildClearMessage:
    """Tests for build_clear_message."""

    def test_returns_json_string(self) -> None:
        result = build_clear_message("stream-sid-456")
        data = json.loads(result)
        assert data["event"] == "clear"

    def test_stream_sid_included(self) -> None:
        result = build_clear_message("stream-sid-456")
        data = json.loads(result)
        assert data["streamSid"] == "stream-sid-456"


class TestExtractStreamSid:
    """Tests for extract_stream_sid."""

    def test_extracts_from_top_level(self) -> None:
        event = {"event": "media", "streamSid": "SS123"}
        assert extract_stream_sid(event) == "SS123"

    def test_extracts_from_start_nested(self) -> None:
        event = {
            "event": "start",
            "start": {"streamSid": "SS456"},
        }
        assert extract_stream_sid(event) == "SS456"

    def test_returns_none_when_absent(self) -> None:
        event = {"event": "media", "media": {}}
        assert extract_stream_sid(event) is None

    def test_top_level_takes_precedence(self) -> None:
        event = {
            "streamSid": "top-level",
            "start": {"streamSid": "nested"},
        }
        assert extract_stream_sid(event) == "top-level"


class TestParseMediaEvent:
    """Tests for parse_media_event."""

    def test_valid_event_returns_tuple(self) -> None:
        raw = b"raw audio"
        b64 = base64.b64encode(raw).decode()
        event = {"event": "media", "media": {"track": "inbound", "payload": b64}}
        result = parse_media_event(event)
        assert result is not None
        track, audio = result
        assert track == "inbound"
        assert audio == raw

    def test_missing_payload_returns_none(self) -> None:
        event = {"event": "media", "media": {"track": "inbound"}}
        assert parse_media_event(event) is None

    def test_outbound_track(self) -> None:
        raw = b"outbound audio"
        b64 = base64.b64encode(raw).decode()
        event = {"event": "media", "media": {"track": "outbound", "payload": b64}}
        result = parse_media_event(event)
        assert result is not None
        track, _ = result
        assert track == "outbound"
