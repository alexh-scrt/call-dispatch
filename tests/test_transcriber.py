"""Unit tests for call_dispatch.transcriber - Deepgram WebSocket transcription client.

All network calls are mocked so tests run fully offline and deterministically.

Test coverage includes:
- DeepgramTranscriber initialisation and URL construction
- Connection establishment (success and failure paths)
- Disconnection and cleanup
- send_audio / send_audio_base64 routing
- _handle_message for all Deepgram message types
- _process_transcript_result with valid, empty, and malformed data
- Callback invocation with TranscriptionResult objects
- Context manager protocol
- TranscriberError custom exception
- Reconnection retry behaviour
"""

from __future__ import annotations

import asyncio
import base64
import json
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from call_dispatch.models import TranscriptionResult
from call_dispatch.transcriber import (
    DeepgramTranscriber,
    TranscriberError,
    _DEFAULT_PARAMS,
    _DEEPGRAM_WS_URL,
    parse_media_event,
)


# ---------------------------------------------------------------------------
# Test environment
# ---------------------------------------------------------------------------

_FAKE_ENV = {
    "TWILIO_ACCOUNT_SID": "ACtest1234567890abcdef1234567890ab",
    "TWILIO_AUTH_TOKEN": "test_auth_token",
    "TWILIO_PHONE_NUMBER": "+15550001234",
    "OPENAI_API_KEY": "sk-test-key",
    "DEEPGRAM_API_KEY": "dg-test-key-12345",
    "PUBLIC_BASE_URL": "https://example.ngrok.io",
}


@pytest.fixture(autouse=True)
def patch_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject environment variables and clear settings cache."""
    for key, val in _FAKE_ENV.items():
        monkeypatch.setenv(key, val)
    from call_dispatch.config import get_settings
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_transcript_message(
    text: str = "Hello there",
    confidence: float = 0.95,
    is_final: bool = True,
    channel_index: int = 0,
) -> str:
    """Build a Deepgram Results message JSON string."""
    return json.dumps({
        "type": "Results",
        "channel": {
            "alternatives": [
                {"transcript": text, "confidence": confidence}
            ]
        },
        "is_final": is_final,
        "channel_index": [channel_index],
    })


def make_mock_ws() -> MagicMock:
    """Create a mock WebSocket connection."""
    ws = AsyncMock()
    ws.send = AsyncMock()
    ws.close = AsyncMock()
    return ws


@pytest.fixture
async def transcriber_with_callback() -> tuple[DeepgramTranscriber, list[TranscriptionResult]]:
    """Provide a transcriber and a list that collects callback results."""
    results: list[TranscriptionResult] = []

    async def callback(result: TranscriptionResult) -> None:
        results.append(result)

    t = DeepgramTranscriber(
        callback=callback,
        api_key="dg-test-key-12345",
    )
    return t, results


# ---------------------------------------------------------------------------
# Initialisation tests
# ---------------------------------------------------------------------------


class TestDeepgramTranscriberInit:
    """Tests for DeepgramTranscriber construction."""

    def test_is_not_connected_initially(self) -> None:
        """A freshly created transcriber is not connected."""
        results = []
        async def cb(r): results.append(r)
        t = DeepgramTranscriber(callback=cb, api_key="dg-key")
        assert t.is_connected is False

    def test_uses_provided_api_key(self) -> None:
        """The provided API key is stored on the instance."""
        async def cb(r): pass
        t = DeepgramTranscriber(callback=cb, api_key="custom-key")
        assert t._api_key == "custom-key"

    def test_uses_settings_api_key_when_not_provided(self) -> None:
        """Falls back to settings API key when none is explicitly given."""
        async def cb(r): pass
        t = DeepgramTranscriber(callback=cb)
        assert t._api_key == "dg-test-key-12345"

    def test_default_params_set(self) -> None:
        """Default connection parameters are set on the instance."""
        async def cb(r): pass
        t = DeepgramTranscriber(callback=cb, api_key="k")
        assert t._params["encoding"] == "mulaw"
        assert t._params["sample_rate"] == "8000"
        assert t._params["model"] == "nova-2-phonecall"

    def test_custom_params_override_defaults(self) -> None:
        """Constructor arguments override default parameters."""
        async def cb(r): pass
        t = DeepgramTranscriber(
            callback=cb,
            api_key="k",
            encoding="linear16",
            sample_rate=16000,
            model="nova-2",
        )
        assert t._params["encoding"] == "linear16"
        assert t._params["sample_rate"] == "16000"
        assert t._params["model"] == "nova-2"

    def test_extra_params_merged(self) -> None:
        """Extra params dict is merged into connection parameters."""
        async def cb(r): pass
        t = DeepgramTranscriber(
            callback=cb,
            api_key="k",
            extra_params={"diarize": "true", "multichannel": "false"},
        )
        assert t._params["diarize"] == "true"
        assert t._params["multichannel"] == "false"

    def test_interim_results_false(self) -> None:
        """Setting interim_results=False is reflected in params."""
        async def cb(r): pass
        t = DeepgramTranscriber(callback=cb, api_key="k", interim_results=False)
        assert t._params["interim_results"] == "false"


# ---------------------------------------------------------------------------
# URL construction tests
# ---------------------------------------------------------------------------


class TestBuildUrl:
    """Tests for _build_url."""

    def test_url_starts_with_deepgram_base(self) -> None:
        """Built URL starts with the Deepgram WSS base URL."""
        async def cb(r): pass
        t = DeepgramTranscriber(callback=cb, api_key="k")
        url = t._build_url()
        assert url.startswith(_DEEPGRAM_WS_URL)

    def test_url_contains_encoding(self) -> None:
        """Built URL contains the encoding parameter."""
        async def cb(r): pass
        t = DeepgramTranscriber(callback=cb, api_key="k")
        url = t._build_url()
        assert "encoding=mulaw" in url

    def test_url_contains_sample_rate(self) -> None:
        """Built URL contains the sample_rate parameter."""
        async def cb(r): pass
        t = DeepgramTranscriber(callback=cb, api_key="k")
        url = t._build_url()
        assert "sample_rate=8000" in url

    def test_url_has_query_separator(self) -> None:
        """Built URL has a '?' separating base from parameters."""
        async def cb(r): pass
        t = DeepgramTranscriber(callback=cb, api_key="k")
        url = t._build_url()
        assert "?" in url


# ---------------------------------------------------------------------------
# Connection tests
# ---------------------------------------------------------------------------


class TestConnect:
    """Tests for DeepgramTranscriber.connect."""

    @pytest.mark.asyncio
    async def test_connect_opens_websocket(self) -> None:
        """connect() opens a WebSocket connection."""
        async def cb(r): pass
        t = DeepgramTranscriber(callback=cb, api_key="k")

        mock_ws = make_mock_ws()
        # Make the async for loop on mock_ws exit immediately
        mock_ws.__aiter__ = MagicMock(return_value=iter([]))

        with patch("call_dispatch.transcriber.websockets.connect",
                   new=AsyncMock(return_value=mock_ws)):
            await t.connect()

        assert t.is_connected is True
        await t.disconnect()

    @pytest.mark.asyncio
    async def test_connect_sets_receiver_task(self) -> None:
        """connect() creates a background receiver task."""
        async def cb(r): pass
        t = DeepgramTranscriber(callback=cb, api_key="k")

        mock_ws = make_mock_ws()
        mock_ws.__aiter__ = MagicMock(return_value=iter([]))

        with patch("call_dispatch.transcriber.websockets.connect",
                   new=AsyncMock(return_value=mock_ws)):
            await t.connect()

        assert t._receiver_task is not None
        await t.disconnect()

    @pytest.mark.asyncio
    async def test_connect_twice_is_noop(self) -> None:
        """Calling connect() on an already-connected transcriber does nothing."""
        async def cb(r): pass
        t = DeepgramTranscriber(callback=cb, api_key="k")

        mock_ws = make_mock_ws()
        mock_ws.__aiter__ = MagicMock(return_value=iter([]))

        connect_mock = AsyncMock(return_value=mock_ws)
        with patch("call_dispatch.transcriber.websockets.connect", new=connect_mock):
            await t.connect()
            await t.connect()  # Second call should be a no-op

        assert connect_mock.call_count == 1
        await t.disconnect()

    @pytest.mark.asyncio
    async def test_connect_failure_raises_transcriber_error(self) -> None:
        """connect() raises TranscriberError after all retry attempts fail."""
        async def cb(r): pass
        t = DeepgramTranscriber(callback=cb, api_key="k")

        from websockets.exceptions import WebSocketException
        with patch(
            "call_dispatch.transcriber.websockets.connect",
            new=AsyncMock(side_effect=WebSocketException("refused")),
        ):
            with patch("call_dispatch.transcriber.asyncio.sleep", new=AsyncMock()):
                with pytest.raises(TranscriberError, match="Failed to connect"):
                    await t.connect()

    @pytest.mark.asyncio
    async def test_connect_includes_auth_header(self) -> None:
        """connect() passes the Authorization header to websockets.connect."""
        async def cb(r): pass
        t = DeepgramTranscriber(callback=cb, api_key="my-dg-key")

        mock_ws = make_mock_ws()
        mock_ws.__aiter__ = MagicMock(return_value=iter([]))

        captured: dict[str, Any] = {}
        async def capture_connect(url, **kwargs):
            captured.update(kwargs)
            return mock_ws

        with patch("call_dispatch.transcriber.websockets.connect",
                   new=capture_connect):
            await t.connect()

        headers = captured.get("extra_headers", {})
        assert headers.get("Authorization") == "Token my-dg-key"
        await t.disconnect()


# ---------------------------------------------------------------------------
# Disconnection tests
# ---------------------------------------------------------------------------


class TestDisconnect:
    """Tests for DeepgramTranscriber.disconnect."""

    @pytest.mark.asyncio
    async def test_disconnect_sets_is_connected_false(self) -> None:
        """After disconnect(), is_connected is False."""
        async def cb(r): pass
        t = DeepgramTranscriber(callback=cb, api_key="k")

        mock_ws = make_mock_ws()
        mock_ws.__aiter__ = MagicMock(return_value=iter([]))

        with patch("call_dispatch.transcriber.websockets.connect",
                   new=AsyncMock(return_value=mock_ws)):
            await t.connect()

        await t.disconnect()
        assert t.is_connected is False

    @pytest.mark.asyncio
    async def test_disconnect_idempotent(self) -> None:
        """Calling disconnect() when not connected does not raise."""
        async def cb(r): pass
        t = DeepgramTranscriber(callback=cb, api_key="k")
        await t.disconnect()  # Should not raise

    @pytest.mark.asyncio
    async def test_disconnect_sends_close_stream(self) -> None:
        """disconnect() sends a CloseStream message to Deepgram."""
        async def cb(r): pass
        t = DeepgramTranscriber(callback=cb, api_key="k")

        mock_ws = make_mock_ws()
        mock_ws.__aiter__ = MagicMock(return_value=iter([]))

        with patch("call_dispatch.transcriber.websockets.connect",
                   new=AsyncMock(return_value=mock_ws)):
            await t.connect()

        await t.disconnect()

        # Check that send was called with a CloseStream message
        sent_calls = [str(c) for c in mock_ws.send.call_args_list]
        assert any("CloseStream" in c for c in sent_calls)

    @pytest.mark.asyncio
    async def test_disconnect_cancels_receiver_task(self) -> None:
        """disconnect() cancels the background receiver task."""
        async def cb(r): pass
        t = DeepgramTranscriber(callback=cb, api_key="k")

        mock_ws = make_mock_ws()
        # Make receiver loop block until cancelled
        async def blocking_iter():
            await asyncio.sleep(100)
            return
            yield  # Make it a generator

        mock_ws.__aiter__ = MagicMock(return_value=iter([]))

        with patch("call_dispatch.transcriber.websockets.connect",
                   new=AsyncMock(return_value=mock_ws)):
            await t.connect()

        task = t._receiver_task
        await t.disconnect()
        assert task is not None
        assert task.done()


# ---------------------------------------------------------------------------
# Audio sending tests
# ---------------------------------------------------------------------------


class TestSendAudio:
    """Tests for send_audio and send_audio_base64."""

    @pytest.mark.asyncio
    async def test_send_audio_forwards_bytes(self) -> None:
        """send_audio forwards raw bytes to the WebSocket."""
        async def cb(r): pass
        t = DeepgramTranscriber(callback=cb, api_key="k")
        mock_ws = make_mock_ws()
        mock_ws.__aiter__ = MagicMock(return_value=iter([]))

        with patch("call_dispatch.transcriber.websockets.connect",
                   new=AsyncMock(return_value=mock_ws)):
            await t.connect()

        audio = b"\xff\xfe" * 100
        await t.send_audio(audio)
        mock_ws.send.assert_any_call(audio)
        await t.disconnect()

    @pytest.mark.asyncio
    async def test_send_audio_when_not_connected_is_noop(self) -> None:
        """send_audio silently drops audio when not connected."""
        async def cb(r): pass
        t = DeepgramTranscriber(callback=cb, api_key="k")
        # Should not raise
        await t.send_audio(b"some audio")

    @pytest.mark.asyncio
    async def test_send_audio_base64_decodes_and_sends(self) -> None:
        """send_audio_base64 decodes base64 and sends the raw bytes."""
        async def cb(r): pass
        t = DeepgramTranscriber(callback=cb, api_key="k")
        mock_ws = make_mock_ws()
        mock_ws.__aiter__ = MagicMock(return_value=iter([]))

        with patch("call_dispatch.transcriber.websockets.connect",
                   new=AsyncMock(return_value=mock_ws)):
            await t.connect()

        raw = b"audio data"
        b64 = base64.b64encode(raw).decode()
        await t.send_audio_base64(b64)
        mock_ws.send.assert_any_call(raw)
        await t.disconnect()

    @pytest.mark.asyncio
    async def test_send_audio_base64_invalid_is_noop(self) -> None:
        """send_audio_base64 silently ignores invalid base64 strings."""
        async def cb(r): pass
        t = DeepgramTranscriber(callback=cb, api_key="k")
        mock_ws = make_mock_ws()
        mock_ws.__aiter__ = MagicMock(return_value=iter([]))

        with patch("call_dispatch.transcriber.websockets.connect",
                   new=AsyncMock(return_value=mock_ws)):
            await t.connect()

        # Should not raise
        await t.send_audio_base64("!!!not-valid-base64!!!")
        await t.disconnect()


# ---------------------------------------------------------------------------
# Message handling tests
# ---------------------------------------------------------------------------


class TestHandleMessage:
    """Tests for _handle_message dispatch."""

    @pytest.mark.asyncio
    async def test_results_message_triggers_callback(self) -> None:
        """A Results message with text triggers the transcript callback."""
        results: list[TranscriptionResult] = []
        async def cb(r): results.append(r)

        t = DeepgramTranscriber(callback=cb, api_key="k")
        msg = make_transcript_message("Hello world", confidence=0.9, is_final=True)
        await t._handle_message(msg)

        assert len(results) == 1
        assert results[0].text == "Hello world"
        assert results[0].confidence == pytest.approx(0.9)
        assert results[0].is_final is True

    @pytest.mark.asyncio
    async def test_results_interim_message_triggers_callback(self) -> None:
        """A non-final (interim) Results message also triggers the callback."""
        results: list[TranscriptionResult] = []
        async def cb(r): results.append(r)

        t = DeepgramTranscriber(callback=cb, api_key="k")
        msg = make_transcript_message("Hello", is_final=False)
        await t._handle_message(msg)

        assert len(results) == 1
        assert results[0].is_final is False

    @pytest.mark.asyncio
    async def test_empty_transcript_does_not_trigger_callback(self) -> None:
        """A Results message with empty transcript text does not invoke the callback."""
        results: list[TranscriptionResult] = []
        async def cb(r): results.append(r)

        t = DeepgramTranscriber(callback=cb, api_key="k")
        msg = make_transcript_message("", confidence=0.0, is_final=True)
        await t._handle_message(msg)

        assert results == []

    @pytest.mark.asyncio
    async def test_metadata_message_does_not_trigger_callback(self) -> None:
        """Metadata messages do not invoke the callback."""
        results: list[TranscriptionResult] = []
        async def cb(r): results.append(r)

        t = DeepgramTranscriber(callback=cb, api_key="k")
        msg = json.dumps({"type": "Metadata", "request_id": "abc"})
        await t._handle_message(msg)  # Should not raise

        assert results == []

    @pytest.mark.asyncio
    async def test_error_message_does_not_trigger_callback(self) -> None:
        """Error messages do not invoke the callback."""
        results: list[TranscriptionResult] = []
        async def cb(r): results.append(r)

        t = DeepgramTranscriber(callback=cb, api_key="k")
        msg = json.dumps({"type": "Error", "message": "Invalid API key"})
        await t._handle_message(msg)

        assert results == []

    @pytest.mark.asyncio
    async def test_invalid_json_does_not_raise(self) -> None:
        """Non-JSON messages are silently ignored."""
        async def cb(r): pass
        t = DeepgramTranscriber(callback=cb, api_key="k")
        await t._handle_message("not json at all")

    @pytest.mark.asyncio
    async def test_bytes_message_decoded_and_processed(self) -> None:
        """Binary WebSocket frames are decoded to UTF-8 and processed."""
        results: list[TranscriptionResult] = []
        async def cb(r): results.append(r)

        t = DeepgramTranscriber(callback=cb, api_key="k")
        msg = make_transcript_message("Binary test", is_final=True)
        await t._handle_message(msg.encode("utf-8"))

        assert len(results) == 1
        assert results[0].text == "Binary test"

    @pytest.mark.asyncio
    async def test_results_no_alternatives_does_not_trigger_callback(self) -> None:
        """Results message with no alternatives does not trigger the callback."""
        results: list[TranscriptionResult] = []
        async def cb(r): results.append(r)

        t = DeepgramTranscriber(callback=cb, api_key="k")
        msg = json.dumps({
            "type": "Results",
            "channel": {"alternatives": []},
            "is_final": True,
        })
        await t._handle_message(msg)
        assert results == []


# ---------------------------------------------------------------------------
# TranscriptionResult construction tests
# ---------------------------------------------------------------------------


class TestTranscriptionResult:
    """Tests that TranscriptionResult is correctly constructed from Deepgram data."""

    @pytest.mark.asyncio
    async def test_confidence_clamped_to_one(self) -> None:
        """Confidence > 1.0 from Deepgram is clamped to 1.0."""
        results: list[TranscriptionResult] = []
        async def cb(r): results.append(r)

        t = DeepgramTranscriber(callback=cb, api_key="k")
        msg = json.dumps({
            "type": "Results",
            "channel": {
                "alternatives": [{"transcript": "hello", "confidence": 1.5}]
            },
            "is_final": True,
            "channel_index": [0],
        })
        await t._handle_message(msg)
        assert results[0].confidence == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_confidence_clamped_to_zero(self) -> None:
        """Confidence < 0.0 from Deepgram is clamped to 0.0."""
        results: list[TranscriptionResult] = []
        async def cb(r): results.append(r)

        t = DeepgramTranscriber(callback=cb, api_key="k")
        msg = json.dumps({
            "type": "Results",
            "channel": {
                "alternatives": [{"transcript": "hello", "confidence": -0.5}]
            },
            "is_final": True,
            "channel_index": [0],
        })
        await t._handle_message(msg)
        assert results[0].confidence == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_channel_index_extracted(self) -> None:
        """Channel index is extracted correctly from the message."""
        results: list[TranscriptionResult] = []
        async def cb(r): results.append(r)

        t = DeepgramTranscriber(callback=cb, api_key="k")
        msg = json.dumps({
            "type": "Results",
            "channel": {
                "alternatives": [{"transcript": "test", "confidence": 0.8}]
            },
            "is_final": True,
            "channel_index": [1],
        })
        await t._handle_message(msg)
        assert results[0].channel == 1

    @pytest.mark.asyncio
    async def test_timestamp_is_recent(self) -> None:
        """The TranscriptionResult timestamp is a recent UTC datetime."""
        results: list[TranscriptionResult] = []
        async def cb(r): results.append(r)

        before = datetime.utcnow()
        t = DeepgramTranscriber(callback=cb, api_key="k")
        msg = make_transcript_message("test", is_final=True)
        await t._handle_message(msg)
        after = datetime.utcnow()

        assert before <= results[0].timestamp <= after


# ---------------------------------------------------------------------------
# Context manager tests
# ---------------------------------------------------------------------------


class TestContextManager:
    """Tests for async context manager protocol."""

    @pytest.mark.asyncio
    async def test_context_manager_connects_and_disconnects(self) -> None:
        """Using as async context manager connects on enter and disconnects on exit."""
        async def cb(r): pass
        t = DeepgramTranscriber(callback=cb, api_key="k")

        mock_ws = make_mock_ws()
        mock_ws.__aiter__ = MagicMock(return_value=iter([]))

        with patch("call_dispatch.transcriber.websockets.connect",
                   new=AsyncMock(return_value=mock_ws)):
            async with t as ctx:
                assert ctx is t
                assert t.is_connected is True

        assert t.is_connected is False

    @pytest.mark.asyncio
    async def test_context_manager_disconnects_on_exception(self) -> None:
        """Context manager disconnects even when the body raises an exception."""
        async def cb(r): pass
        t = DeepgramTranscriber(callback=cb, api_key="k")

        mock_ws = make_mock_ws()
        mock_ws.__aiter__ = MagicMock(return_value=iter([]))

        with patch("call_dispatch.transcriber.websockets.connect",
                   new=AsyncMock(return_value=mock_ws)):
            with pytest.raises(ValueError):
                async with t:
                    raise ValueError("test error")

        assert t.is_connected is False


# ---------------------------------------------------------------------------
# TranscriberError tests
# ---------------------------------------------------------------------------


class TestTranscriberError:
    """Tests for the TranscriberError custom exception."""

    def test_is_exception(self) -> None:
        assert issubclass(TranscriberError, Exception)

    def test_message_preserved(self) -> None:
        exc = TranscriberError("Something went wrong")
        assert "Something went wrong" in str(exc)


# ---------------------------------------------------------------------------
# parse_media_event helper tests
# ---------------------------------------------------------------------------


class TestParseMediaEvent:
    """Tests for the parse_media_event helper function."""

    def test_valid_event_returns_track_and_bytes(self) -> None:
        """A valid media event returns the track and decoded audio bytes."""
        raw = b"audio data here"
        b64 = base64.b64encode(raw).decode()
        event = {
            "event": "media",
            "media": {"track": "inbound", "payload": b64},
        }
        result = parse_media_event(event)
        assert result is not None
        track, audio = result
        assert track == "inbound"
        assert audio == raw

    def test_missing_payload_returns_none(self) -> None:
        """An event without a payload returns None."""
        event = {"event": "media", "media": {"track": "inbound"}}
        assert parse_media_event(event) is None

    def test_empty_payload_returns_none(self) -> None:
        """An event with an empty payload returns None."""
        event = {"event": "media", "media": {"track": "inbound", "payload": ""}}
        assert parse_media_event(event) is None

    def test_default_track_is_inbound(self) -> None:
        """When no track is specified, defaults to 'inbound'."""
        raw = b"test"
        b64 = base64.b64encode(raw).decode()
        event = {"event": "media", "media": {"payload": b64}}
        result = parse_media_event(event)
        assert result is not None
        track, _ = result
        assert track == "inbound"
