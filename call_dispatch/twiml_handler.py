"""Twilio webhook handler and TwiML response generator for call_dispatch.

This module handles all Twilio webhook callbacks for outbound calls and
generates TwiML (Twilio Markup Language) XML responses that control the
call flow, including:

- **Answer webhook** (``/twiml/answer``): Called by Twilio when the called
  party answers.  Returns TwiML that opens a Media Stream WebSocket to
  this server and plays the agent's opening utterance.

- **Status callback** (``/twiml/status``): Called by Twilio at call status
  transitions (ringing, in-progress, completed, failed, etc.).
  Updates the call record in the store.

- **Media stream handler** (``/ws/stream/{call_id}``): A FastAPI WebSocket
  endpoint that receives Twilio Media Stream events, feeds audio to
  the Deepgram transcriber, and orchestrates agent utterances.

The TwiML answer response instructs Twilio to:
1. Open a bidirectional Media Stream WebSocket back to this server.
2. Play a short greeting while the stream is being established.
3. Pause to allow the agent response to be spoken via ``<Say>`` or TTS.

Usage within a FastAPI app::

    from call_dispatch.twiml_handler import router as twiml_router
    app.include_router(twiml_router)
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from datetime import datetime
from typing import Any, Optional
from xml.etree.ElementTree import Element, SubElement, tostring

from fastapi import APIRouter, Form, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse

from call_dispatch.models import CallStatus, TranscriptEntry, TranscriptionResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/twiml", tags=["twiml"])
ws_router = APIRouter(tags=["websocket"])

# ---------------------------------------------------------------------------
# TwiML builders
# ---------------------------------------------------------------------------


def build_answer_twiml(
    stream_url: str,
    welcome_message: Optional[str] = None,
    timeout_seconds: int = 300,
) -> str:
    """Build the TwiML response for the answer webhook.

    Instructs Twilio to open a bidirectional Media Stream WebSocket to
    ``stream_url`` and optionally speak a brief welcome message while the
    WebSocket connection is being established.

    Args:
        stream_url: The WebSocket URL (``wss://...``) that Twilio should
            connect to for the Media Stream.  Must be the publicly accessible
            URL of this server's WebSocket endpoint.
        welcome_message: Optional text to speak immediately when the call is
            answered, before the agent's first utterance is ready.  Defaults
            to a neutral hold message.
        timeout_seconds: Maximum duration of the ``<Pause>`` used to keep the
            call open.  Should be at or below the call's configured timeout.

    Returns:
        str: Well-formed TwiML XML string.

    Example output::

        <?xml version="1.0" encoding="UTF-8"?>
        <Response>
            <Start>
                <Stream url="wss://example.ngrok.io/ws/stream/call-id-here"
                        track="inbound_track" />
            </Start>
            <Say voice="Polly.Joanna">Please hold for a moment.</Say>
            <Pause length="300" />
        </Response>
    """
    response_el = Element("Response")

    # Open bidirectional Media Stream
    start_el = SubElement(response_el, "Start")
    stream_el = SubElement(start_el, "Stream")
    stream_el.set("url", stream_url)
    stream_el.set("track", "inbound_track")

    # Optional welcome message to fill the gap before agent speaks
    if welcome_message:
        say_el = SubElement(response_el, "Say")
        say_el.set("voice", "Polly.Joanna")
        say_el.text = welcome_message

    # Pause to keep the call open until the stream takes over
    pause_el = SubElement(response_el, "Pause")
    pause_el.set("length", str(max(1, timeout_seconds)))

    xml_bytes = tostring(response_el, encoding="unicode", xml_declaration=False)
    twiml = f'<?xml version="1.0" encoding="UTF-8"?>{xml_bytes}'
    logger.debug("Built answer TwiML stream_url=%s", stream_url)
    return twiml


def build_say_twiml(
    message: str,
    voice: str = "Polly.Joanna",
    hang_up: bool = False,
) -> str:
    """Build a TwiML response that speaks a message and optionally hangs up.

    Useful for injecting agent utterances or farewell messages into an
    active call via Twilio's Call Modify API.

    Args:
        message: The text to speak.
        voice: Twilio/Polly voice to use.  Default: ``"Polly.Joanna"``.
        hang_up: If ``True``, append a ``<Hangup>`` verb after the ``<Say>``.

    Returns:
        str: Well-formed TwiML XML string.
    """
    response_el = Element("Response")
    say_el = SubElement(response_el, "Say")
    say_el.set("voice", voice)
    say_el.text = message

    if hang_up:
        SubElement(response_el, "Hangup")

    xml_bytes = tostring(response_el, encoding="unicode", xml_declaration=False)
    twiml = f'<?xml version="1.0" encoding="UTF-8"?>{xml_bytes}'
    logger.debug("Built say TwiML hang_up=%s message=%r", hang_up, message[:60])
    return twiml


def build_hangup_twiml() -> str:
    """Build a TwiML response that immediately hangs up the call.

    Returns:
        str: Well-formed TwiML XML string.
    """
    response_el = Element("Response")
    SubElement(response_el, "Hangup")
    xml_bytes = tostring(response_el, encoding="unicode", xml_declaration=False)
    return f'<?xml version="1.0" encoding="UTF-8"?>{xml_bytes}'


def build_reject_twiml(reason: str = "rejected") -> str:
    """Build a TwiML response that rejects an inbound call.

    Args:
        reason: Rejection reason — either ``"rejected"`` or ``"busy"``.

    Returns:
        str: Well-formed TwiML XML string.
    """
    response_el = Element("Response")
    reject_el = SubElement(response_el, "Reject")
    reject_el.set("reason", reason if reason in ("rejected", "busy") else "rejected")
    xml_bytes = tostring(response_el, encoding="unicode", xml_declaration=False)
    return f'<?xml version="1.0" encoding="UTF-8"?>{xml_bytes}'


# ---------------------------------------------------------------------------
# Twilio event type constants
# ---------------------------------------------------------------------------

_TWILIO_STATUS_MAP: dict[str, CallStatus] = {
    "queued": CallStatus.INITIATING,
    "ringing": CallStatus.INITIATING,
    "in-progress": CallStatus.IN_PROGRESS,
    "completed": CallStatus.COMPLETED,
    "busy": CallStatus.BUSY,
    "no-answer": CallStatus.NO_ANSWER,
    "canceled": CallStatus.CANCELLED,
    "failed": CallStatus.FAILED,
}


def map_twilio_status(twilio_status: str) -> Optional[CallStatus]:
    """Map a Twilio call status string to a :class:`~call_dispatch.models.CallStatus`.

    Args:
        twilio_status: The ``CallStatus`` value as provided by Twilio in a
            status callback (e.g. ``"in-progress"``, ``"completed"``).

    Returns:
        CallStatus | None: The mapped status, or ``None`` if unrecognised.
    """
    return _TWILIO_STATUS_MAP.get(twilio_status.lower().strip())


# ---------------------------------------------------------------------------
# Webhook route handlers
# ---------------------------------------------------------------------------


@router.post("/answer/{call_id}", response_class=PlainTextResponse)
async def answer_webhook(
    call_id: str,
    request: Request,
) -> str:
    """Handle the Twilio answer webhook for an outbound call.

    Called by Twilio when the called party answers.  Returns TwiML that:
    1. Opens a Media Stream WebSocket back to this server.
    2. Speaks a brief hold message while the stream is established.
    3. Pauses to keep the call alive until the stream takes over.

    Args:
        call_id: The internal call ID embedded in the webhook URL.
        request: The FastAPI request object, used to access app state.

    Returns:
        str: TwiML XML response with ``Content-Type: application/xml``.
    """
    from call_dispatch.config import get_settings

    cfg = get_settings()
    base_url = cfg.public_base_url

    # Construct the WebSocket URL for the Media Stream
    # Convert https:// to wss:// (or http:// to ws://)
    ws_base = base_url.replace("https://", "wss://").replace("http://", "ws://")
    stream_url = f"{ws_base}/ws/stream/{call_id}"

    # Update call status to IN_PROGRESS if the store is available
    store = getattr(request.app.state, "store", None)
    if store is not None:
        try:
            await store.update_status(
                call_id,
                CallStatus.IN_PROGRESS,
                started_at=datetime.utcnow(),
            )
            logger.info("Call answered: call_id=%s transitioning to IN_PROGRESS", call_id)
        except Exception as exc:
            logger.warning(
                "Failed to update call status on answer for call_id=%s: %s", call_id, exc
            )

    twiml = build_answer_twiml(
        stream_url=stream_url,
        welcome_message="Please hold for a moment.",
        timeout_seconds=cfg.call_timeout_seconds,
    )

    return Response(content=twiml, media_type="application/xml")


@router.post("/status/{call_id}", response_class=PlainTextResponse)
async def status_callback(
    call_id: str,
    request: Request,
    CallSid: Optional[str] = Form(default=None),
    CallStatus: Optional[str] = Form(default=None),
    CallDuration: Optional[str] = Form(default=None),
    ErrorCode: Optional[str] = Form(default=None),
    ErrorMessage: Optional[str] = Form(default=None),
) -> Response:
    """Handle Twilio call status callbacks.

    Twilio posts form data to this endpoint at each status transition.
    The handler updates the call record in the store accordingly.

    Args:
        call_id: The internal call ID embedded in the callback URL.
        request: The FastAPI request object, used to access app state.
        CallSid: Twilio's Call SID (form field).
        CallStatus: Twilio's call status string (form field).
        CallDuration: Duration in seconds as a string (form field, optional).
        ErrorCode: Twilio error code if the call failed (form field, optional).
        ErrorMessage: Human-readable error from Twilio (form field, optional).

    Returns:
        Response: Empty 204 No Content response.
    """
    logger.info(
        "Twilio status callback: call_id=%s CallSid=%s CallStatus=%s",
        call_id,
        CallSid,
        CallStatus,
    )

    store = getattr(request.app.state, "store", None)
    if store is None:
        logger.warning("Status callback: store not available in app state")
        return Response(status_code=204)

    if CallStatus is None:
        return Response(status_code=204)

    mapped_status = map_twilio_status(CallStatus)
    if mapped_status is None:
        logger.debug("Ignoring unrecognised Twilio status: %r", CallStatus)
        return Response(status_code=204)

    # Determine timestamps based on status
    ended_at: Optional[datetime] = None
    terminal_statuses = {
        "completed", "busy", "no-answer", "canceled", "failed"
    }
    if CallStatus.lower() in terminal_statuses:
        ended_at = datetime.utcnow()

    # Build error message if applicable
    error_msg: Optional[str] = None
    if ErrorCode or ErrorMessage:
        parts = []
        if ErrorCode:
            parts.append(f"Twilio error code: {ErrorCode}")
        if ErrorMessage:
            parts.append(ErrorMessage)
        error_msg = " — ".join(parts) if parts else None

    try:
        await store.update_status(
            call_id,
            mapped_status,
            error_message=error_msg,
            ended_at=ended_at,
        )
    except Exception as exc:
        logger.error(
            "Failed to update status from Twilio callback call_id=%s: %s", call_id, exc
        )

    # If we have a Twilio SID, store it on the record
    if CallSid:
        try:
            record = await store.get_call(call_id)
            if record is not None and record.twilio_call_sid != CallSid:
                record.twilio_call_sid = CallSid
                await store.update_call(record)
        except Exception as exc:
            logger.debug(
                "Failed to update twilio_call_sid for call_id=%s: %s", call_id, exc
            )

    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Media Stream WebSocket handler
# ---------------------------------------------------------------------------


@ws_router.websocket("/ws/stream/{call_id}")
async def media_stream_websocket(
    websocket: WebSocket,
    call_id: str,
) -> None:
    """Handle the Twilio Media Stream WebSocket connection for a call.

    This endpoint receives the raw bidirectional audio stream from Twilio
    and orchestrates real-time transcription and agent response injection.

    The WebSocket protocol follows the Twilio Media Streams specification:
    - Twilio sends JSON ``connected``, ``start``, ``media``, ``stop`` events.
    - Audio is delivered as base64-encoded mu-law PCM in ``media`` events.
    - The server can send TwiML or audio back through the WebSocket to have
      Twilio speak text on the call (via ``mark`` and ``media`` events).

    Args:
        websocket: The FastAPI WebSocket connection from Twilio.
        call_id: The internal call ID, extracted from the URL path.
    """
    await websocket.accept()
    logger.info("Media stream WebSocket opened for call_id=%s", call_id)

    store = getattr(websocket.app.state, "store", None)
    active_calls: dict[str, Any] = getattr(
        websocket.app.state, "active_calls", {}
    )

    # Look up the call record
    call_record = None
    if store is not None:
        try:
            call_record = await store.get_call(call_id)
        except Exception as exc:
            logger.warning(
                "Failed to fetch call record for call_id=%s: %s", call_id, exc
            )

    if call_record is None:
        logger.warning(
            "Media stream WebSocket: no call record found for call_id=%s; closing",
            call_id,
        )
        await websocket.close(code=1008)
        return

    # If a dispatcher is managing this call, hand off to it
    dispatcher = active_calls.get(call_id)
    if dispatcher is not None and hasattr(dispatcher, "handle_media_stream"):
        try:
            await dispatcher.handle_media_stream(websocket)
        except WebSocketDisconnect:
            logger.info(
                "Media stream WebSocket disconnected for call_id=%s", call_id
            )
        except Exception as exc:
            logger.error(
                "Error in dispatcher media stream handler for call_id=%s: %s",
                call_id,
                exc,
            )
        return

    # Fallback: basic stream handler that just processes the events
    await _handle_media_stream_basic(websocket, call_id, store)


async def _handle_media_stream_basic(
    websocket: WebSocket,
    call_id: str,
    store: Any,
) -> None:
    """Basic fallback Media Stream handler that processes Twilio stream events.

    This handler processes Twilio WebSocket events without full dispatcher
    integration.  It:
    - Acknowledges ``connected`` and ``start`` events.
    - Logs ``stop`` events and closes the WebSocket.
    - Forwards ``media`` audio payloads for transcription if a transcriber
      callback is registered on app state.

    Args:
        websocket: The accepted Twilio Media Stream WebSocket.
        call_id: Internal call identifier.
        store: The :class:`~call_dispatch.store.CallStore` instance, may be ``None``.
    """
    stream_sid: Optional[str] = None

    try:
        while True:
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=60.0)
            except asyncio.TimeoutError:
                logger.warning(
                    "Media stream WebSocket timed out waiting for events call_id=%s",
                    call_id,
                )
                break

            try:
                event: dict = json.loads(raw)
            except json.JSONDecodeError:
                logger.debug(
                    "Received non-JSON message on media stream call_id=%s", call_id
                )
                continue

            event_type = event.get("event", "")

            if event_type == "connected":
                protocol = event.get("protocol", "")
                version = event.get("version", "")
                logger.info(
                    "Twilio Media Stream connected: call_id=%s protocol=%s version=%s",
                    call_id,
                    protocol,
                    version,
                )

            elif event_type == "start":
                start_data = event.get("start", {})
                stream_sid = event.get("streamSid") or start_data.get("streamSid")
                tracks = start_data.get("tracks", [])
                media_format = start_data.get("mediaFormat", {})
                logger.info(
                    "Twilio Media Stream started: call_id=%s stream_sid=%s "
                    "tracks=%s encoding=%s sample_rate=%s",
                    call_id,
                    stream_sid,
                    tracks,
                    media_format.get("encoding"),
                    media_format.get("sampleRate"),
                )

            elif event_type == "media":
                media_data = event.get("media", {})
                payload: Optional[str] = media_data.get("payload")
                if payload:
                    logger.debug(
                        "Received media chunk call_id=%s track=%s",
                        call_id,
                        media_data.get("track"),
                    )
                    # In the basic handler, audio is not forwarded to any
                    # transcriber since the full dispatcher is not present.
                    # The dispatcher's handle_media_stream takes over in production.

            elif event_type == "stop":
                stop_data = event.get("stop", {})
                logger.info(
                    "Twilio Media Stream stopped: call_id=%s stream_sid=%s reason=%s",
                    call_id,
                    stream_sid,
                    stop_data.get("accountSid", "unknown"),
                )
                break

            elif event_type == "mark":
                mark_name = event.get("mark", {}).get("name", "")
                logger.debug(
                    "Twilio mark event: call_id=%s name=%s", call_id, mark_name
                )

            else:
                logger.debug(
                    "Unhandled Twilio stream event type=%r call_id=%s",
                    event_type,
                    call_id,
                )

    except WebSocketDisconnect:
        logger.info(
            "Media stream WebSocket disconnected (client): call_id=%s", call_id
        )
    except Exception as exc:
        logger.error(
            "Error in basic media stream handler call_id=%s: %s", call_id, exc
        )
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
        logger.info("Media stream WebSocket closed: call_id=%s", call_id)


# ---------------------------------------------------------------------------
# Twilio event parsing helpers
# ---------------------------------------------------------------------------


def parse_media_event(event: dict) -> Optional[tuple[str, bytes]]:
    """Extract the track label and decoded audio bytes from a Twilio ``media`` event.

    Args:
        event: Parsed Twilio Media Stream ``media`` event dictionary.

    Returns:
        tuple[str, bytes] | None: A ``(track, audio_bytes)`` pair, or ``None``
            if the event does not contain a valid audio payload.
    """
    media = event.get("media", {})
    payload: Optional[str] = media.get("payload")
    track: str = media.get("track", "inbound")

    if not payload:
        return None

    try:
        audio_bytes = base64.b64decode(payload)
        return track, audio_bytes
    except Exception as exc:
        logger.debug("Failed to decode media payload: %s", exc)
        return None


def extract_stream_sid(event: dict) -> Optional[str]:
    """Extract the stream SID from a Twilio ``start`` or ``media`` event.

    Args:
        event: Parsed Twilio Media Stream event dictionary.

    Returns:
        str | None: The stream SID string, or ``None`` if not present.
    """
    stream_sid = event.get("streamSid")
    if stream_sid:
        return stream_sid
    start_data = event.get("start", {})
    return start_data.get("streamSid")


def build_mark_message(stream_sid: str, mark_name: str) -> str:
    """Build a Twilio Media Stream ``mark`` message JSON string.

    Mark messages are sent from the server to Twilio to receive a
    callback when a specific point in the audio stream is reached.
    This is useful for synchronising agent speech with the stream.

    Args:
        stream_sid: The active Media Stream SID.
        mark_name: An arbitrary label for the mark event.

    Returns:
        str: JSON-encoded ``mark`` message ready to send over the WebSocket.
    """
    return json.dumps(
        {
            "event": "mark",
            "streamSid": stream_sid,
            "mark": {"name": mark_name},
        }
    )


def build_clear_message(stream_sid: str) -> str:
    """Build a Twilio Media Stream ``clear`` message JSON string.

    Sending this message instructs Twilio to discard any buffered audio
    queued for playback on the call.  Useful when the agent needs to
    interrupt the current utterance.

    Args:
        stream_sid: The active Media Stream SID.

    Returns:
        str: JSON-encoded ``clear`` message.
    """
    return json.dumps(
        {
            "event": "clear",
            "streamSid": stream_sid,
        }
    )
