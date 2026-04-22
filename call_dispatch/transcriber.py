"""Deepgram WebSocket transcription client for call_dispatch.

Provides the :class:`DeepgramTranscriber` class that connects to Deepgram's
real-time transcription API via WebSocket, streams audio received from Twilio
Media Streams (mu-law encoded, 8kHz), and emits :class:`TranscriptionResult`
objects through an async callback.

The transcriber is designed to be used as an async context manager:

    async with DeepgramTranscriber(callback=my_handler) as transcriber:
        await transcriber.send_audio(mulaw_bytes)

Or managed manually:

    transcriber = DeepgramTranscriber(callback=my_handler)
    await transcriber.connect()
    await transcriber.send_audio(audio_bytes)
    await transcriber.disconnect()

The callback receives a :class:`~call_dispatch.models.TranscriptionResult`
for each transcript event (both interim and final results).

Usage example::

    async def handle_transcript(result: TranscriptionResult) -> None:
        if result.is_final:
            print(f"Final: {result.text} (conf={result.confidence:.2f})")

    async with DeepgramTranscriber(callback=handle_transcript) as t:
        # Feed raw mu-law audio bytes from Twilio
        await t.send_audio(mulaw_audio_bytes)
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from datetime import datetime
from typing import Awaitable, Callable, Optional

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from call_dispatch.models import TranscriptionResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Deepgram WebSocket endpoint and connection parameters
# ---------------------------------------------------------------------------

_DEEPGRAM_WS_URL = "wss://api.deepgram.com/v1/listen"

# Default query parameters for Deepgram's streaming API.
# These are tuned for Twilio Media Stream audio (8kHz mu-law, mono).
_DEFAULT_PARAMS: dict[str, str] = {
    "encoding": "mulaw",
    "sample_rate": "8000",
    "channels": "1",
    "model": "nova-2-phonecall",
    "language": "en-US",
    "punctuate": "true",
    "interim_results": "true",
    "endpointing": "500",
    "utterance_end_ms": "1000",
    "smart_format": "true",
}

# Reconnection settings
_MAX_RECONNECT_ATTEMPTS = 3
_RECONNECT_DELAY_SECONDS = 1.0

# Type alias for the transcript callback
TranscriptCallback = Callable[[TranscriptionResult], Awaitable[None]]


# ---------------------------------------------------------------------------
# DeepgramTranscriber
# ---------------------------------------------------------------------------


class DeepgramTranscriber:
    """Real-time speech-to-text client using Deepgram's WebSocket API.

    Streams mu-law encoded audio from Twilio Media Streams to Deepgram and
    delivers transcription results asynchronously via a user-supplied callback.

    Args:
        callback: An async callable that receives a
            :class:`~call_dispatch.models.TranscriptionResult` for each
            transcription event.
        api_key: Deepgram API key.  Falls back to the configured value from
            :func:`~call_dispatch.config.get_settings`.
        encoding: Audio encoding format.  Default: ``"mulaw"``.
        sample_rate: Audio sample rate in Hz.  Default: ``8000``.
        channels: Number of audio channels.  Default: ``1``.
        model: Deepgram model to use.  Default: ``"nova-2-phonecall"``.
        language: BCP-47 language code.  Default: ``"en-US"``.
        interim_results: Whether to emit interim (non-final) results.
        extra_params: Additional Deepgram query parameters to merge into
            the connection URL.
    """

    def __init__(
        self,
        callback: TranscriptCallback,
        *,
        api_key: Optional[str] = None,
        encoding: str = "mulaw",
        sample_rate: int = 8000,
        channels: int = 1,
        model: str = "nova-2-phonecall",
        language: str = "en-US",
        interim_results: bool = True,
        extra_params: Optional[dict[str, str]] = None,
    ) -> None:
        from call_dispatch.config import get_settings

        cfg = get_settings()
        self._api_key = api_key or cfg.deepgram_api_key
        self._callback = callback

        # Build connection parameters
        self._params: dict[str, str] = {
            **_DEFAULT_PARAMS,
            "encoding": encoding,
            "sample_rate": str(sample_rate),
            "channels": str(channels),
            "model": model,
            "language": language,
            "interim_results": "true" if interim_results else "false",
        }
        if extra_params:
            self._params.update(extra_params)

        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._receiver_task: Optional[asyncio.Task] = None
        self._connected = False
        self._closing = False

        logger.debug(
            "DeepgramTranscriber initialised model=%s encoding=%s sample_rate=%s",
            model,
            encoding,
            sample_rate,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        """Return ``True`` if the WebSocket connection is currently open."""
        return self._connected and self._ws is not None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _build_url(self) -> str:
        """Construct the Deepgram WebSocket URL with query parameters.

        Returns:
            str: The full WSS URL including all query parameters.
        """
        query_string = "&".join(f"{k}={v}" for k, v in self._params.items())
        return f"{_DEEPGRAM_WS_URL}?{query_string}"

    async def connect(self) -> None:
        """Open the WebSocket connection to Deepgram and start the receiver task.

        Raises:
            TranscriberError: If the connection cannot be established after
                all retry attempts.
        """
        if self._connected:
            logger.debug("DeepgramTranscriber already connected")
            return

        url = self._build_url()
        headers = {"Authorization": f"Token {self._api_key}"}

        last_exc: Optional[Exception] = None
        for attempt in range(1, _MAX_RECONNECT_ATTEMPTS + 1):
            try:
                self._ws = await websockets.connect(
                    url,
                    extra_headers=headers,
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=10,
                )
                self._connected = True
                self._closing = False
                logger.info(
                    "DeepgramTranscriber connected (attempt %d/%d)",
                    attempt,
                    _MAX_RECONNECT_ATTEMPTS,
                )
                # Start background receiver
                self._receiver_task = asyncio.create_task(
                    self._receive_loop(),
                    name="deepgram-receiver",
                )
                return
            except (WebSocketException, OSError, Exception) as exc:
                last_exc = exc
                logger.warning(
                    "DeepgramTranscriber connection attempt %d/%d failed: %s",
                    attempt,
                    _MAX_RECONNECT_ATTEMPTS,
                    exc,
                )
                if attempt < _MAX_RECONNECT_ATTEMPTS:
                    await asyncio.sleep(_RECONNECT_DELAY_SECONDS * attempt)

        raise TranscriberError(
            f"Failed to connect to Deepgram after {_MAX_RECONNECT_ATTEMPTS} attempts: {last_exc}"
        ) from last_exc

    async def disconnect(self) -> None:
        """Close the WebSocket connection and stop the receiver task gracefully.

        Safe to call even if already disconnected.  Sends a close-stream
        message to Deepgram before closing the WebSocket so that any final
        transcript results are flushed.
        """
        if not self._connected:
            return

        self._closing = True
        self._connected = False

        # Send close-stream sentinel to Deepgram
        if self._ws is not None:
            try:
                await self._ws.send(json.dumps({"type": "CloseStream"}))
            except Exception as exc:
                logger.debug("Error sending CloseStream to Deepgram: %s", exc)

            try:
                await self._ws.close()
            except Exception as exc:
                logger.debug("Error closing Deepgram WebSocket: %s", exc)
            finally:
                self._ws = None

        # Cancel and await the receiver task
        if self._receiver_task is not None and not self._receiver_task.done():
            self._receiver_task.cancel()
            try:
                await self._receiver_task
            except asyncio.CancelledError:
                pass
            self._receiver_task = None

        logger.info("DeepgramTranscriber disconnected")

    # ------------------------------------------------------------------
    # Audio ingestion
    # ------------------------------------------------------------------

    async def send_audio(self, audio_bytes: bytes) -> None:
        """Send a chunk of raw audio bytes to Deepgram for transcription.

        The bytes should be mu-law encoded at 8kHz mono (as produced by
        Twilio Media Streams).  The method silently drops the chunk if the
        connection is not currently open.

        Args:
            audio_bytes: Raw audio data to stream.
        """
        if not self._connected or self._ws is None:
            logger.debug("send_audio called while not connected; skipping chunk")
            return

        try:
            await self._ws.send(audio_bytes)
        except ConnectionClosed as exc:
            logger.warning("Deepgram connection closed while sending audio: %s", exc)
            self._connected = False
        except WebSocketException as exc:
            logger.warning("WebSocket error while sending audio: %s", exc)

    async def send_audio_base64(self, b64_audio: str) -> None:
        """Decode a base64-encoded audio string and send it to Deepgram.

        Twilio Media Stream payloads carry audio as base64-encoded strings.
        This convenience method decodes them before forwarding.

        Args:
            b64_audio: Base64-encoded mu-law audio string from Twilio.
        """
        try:
            audio_bytes = base64.b64decode(b64_audio)
        except Exception as exc:
            logger.warning("Failed to decode base64 audio payload: %s", exc)
            return
        await self.send_audio(audio_bytes)

    # ------------------------------------------------------------------
    # Receiver loop
    # ------------------------------------------------------------------

    async def _receive_loop(self) -> None:
        """Background task that reads messages from Deepgram and fires callbacks.

        Runs until the WebSocket is closed or :meth:`disconnect` is called.
        Each received message is parsed and, if it contains transcript text,
        the user callback is invoked.
        """
        if self._ws is None:
            return

        try:
            async for raw_message in self._ws:
                if self._closing:
                    break
                await self._handle_message(raw_message)
        except ConnectionClosed as exc:
            if not self._closing:
                logger.warning(
                    "Deepgram WebSocket closed unexpectedly: code=%s reason=%s",
                    exc.rcvd.code if exc.rcvd else "?",
                    exc.rcvd.reason if exc.rcvd else "?",
                )
        except asyncio.CancelledError:
            logger.debug("Deepgram receiver task cancelled")
            raise
        except Exception as exc:
            if not self._closing:
                logger.error("Unexpected error in Deepgram receiver loop: %s", exc)
        finally:
            self._connected = False
            logger.debug("Deepgram receiver loop exited")

    async def _handle_message(self, raw_message: str | bytes) -> None:
        """Parse a single Deepgram WebSocket message and invoke the callback.

        Deepgram sends JSON text frames.  We handle the following message types:
        - ``Results``: Transcript results (interim and final)
        - ``Metadata``: Connection metadata (logged at DEBUG level)
        - ``Error``: Deepgram error messages (logged at WARNING level)
        - ``UtteranceEnd``: Signals end of a spoken utterance (ignored here)

        Args:
            raw_message: The raw WebSocket message (string or bytes).
        """
        if isinstance(raw_message, bytes):
            try:
                raw_message = raw_message.decode("utf-8")
            except UnicodeDecodeError:
                logger.debug("Received non-UTF-8 binary message from Deepgram; skipping")
                return

        try:
            data: dict = json.loads(raw_message)
        except json.JSONDecodeError as exc:
            logger.debug("Failed to parse Deepgram message as JSON: %s", exc)
            return

        msg_type = data.get("type", "")

        if msg_type == "Results":
            await self._process_transcript_result(data)
        elif msg_type == "Metadata":
            logger.debug("Deepgram metadata: %s", data)
        elif msg_type == "UtteranceEnd":
            logger.debug("Deepgram utterance end signal received")
        elif msg_type == "Error":
            logger.warning(
                "Deepgram error message: %s",
                data.get("message", str(data)),
            )
        elif msg_type == "SpeechStarted":
            logger.debug("Deepgram speech started signal received")
        else:
            logger.debug("Unhandled Deepgram message type=%r: %s", msg_type, str(data)[:200])

    async def _process_transcript_result(self, data: dict) -> None:
        """Extract transcript text and confidence from a Deepgram Results message.

        Navigates the Deepgram response structure to extract the highest-
        confidence transcript alternative.  If a non-empty transcript is
        found, the user callback is invoked with a
        :class:`~call_dispatch.models.TranscriptionResult`.

        Args:
            data: Parsed Deepgram ``Results`` message dictionary.
        """
        try:
            channel = data.get("channel", {})
            alternatives = channel.get("alternatives", [])
            if not alternatives:
                return

            best = alternatives[0]
            transcript_text: str = best.get("transcript", "").strip()
            confidence: float = float(best.get("confidence", 0.0))

            if not transcript_text:
                return

            is_final: bool = data.get("is_final", False)
            channel_index: int = data.get("channel_index", [0])
            if isinstance(channel_index, list):
                channel_index = channel_index[0] if channel_index else 0

            result = TranscriptionResult(
                text=transcript_text,
                confidence=min(1.0, max(0.0, confidence)),
                is_final=bool(is_final),
                timestamp=datetime.utcnow(),
                channel=int(channel_index),
            )

            logger.debug(
                "Transcript result is_final=%s confidence=%.2f text=%r",
                result.is_final,
                result.confidence,
                result.text[:80],
            )

            await self._callback(result)

        except Exception as exc:
            logger.error(
                "Error processing Deepgram transcript result: %s. Data: %s",
                exc,
                str(data)[:200],
            )

    # ------------------------------------------------------------------
    # Async context manager support
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "DeepgramTranscriber":
        """Connect to Deepgram on context entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Disconnect from Deepgram on context exit."""
        await self.disconnect()


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class TranscriberError(Exception):
    """Raised when the :class:`DeepgramTranscriber` encounters a fatal error.

    This wraps underlying WebSocket and connection errors so callers can
    catch a single, application-specific exception type.
    """
