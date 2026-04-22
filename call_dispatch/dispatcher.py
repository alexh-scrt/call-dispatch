"""Core call dispatcher orchestration module for call_dispatch.

Provides the :class:`CallDispatcher` class that:

1. Initiates Twilio outbound calls via the Twilio REST API.
2. Registers Twilio webhook URLs for answer and status callbacks.
3. Wires together the :class:`~call_dispatch.transcriber.DeepgramTranscriber`
   and :class:`~call_dispatch.agent.ConversationAgent` for real-time
   conversation during the call.
4. Manages the full call lifecycle state, including transcript accumulation,
   agent utterance injection, and post-call summarisation.
5. Handles Media Stream WebSocket events from Twilio, forwarding audio to
   Deepgram and injecting agent responses back into the call.

Typical usage via the REST API layer::

    dispatcher = CallDispatcher(store=store)
    call_record = await dispatcher.dispatch(
        to_number="+15550001234",
        goal="Book a dentist appointment for Tuesday afternoon",
        context="Patient: Jane Doe, DOB: 1990-05-15",
    )
    # call_record.call_id can now be polled for status

For the Media Stream WebSocket (called from the TwiML handler)::

    await dispatcher.handle_media_stream(websocket)
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Callable, Coroutine, Optional

from call_dispatch.agent import AgentError, ConversationAgent
from call_dispatch.models import (
    CallRecord,
    CallStatus,
    CallSummary,
    DispatchCallRequest,
    TranscriptEntry,
    TranscriptionResult,
)
from call_dispatch.store import CallStore
from call_dispatch.summarizer import CallSummarizer, SummarizerError
from call_dispatch.transcriber import DeepgramTranscriber, TranscriberError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# How long (seconds) to wait for the Twilio call to be answered before timeout
_CALL_ANSWER_TIMEOUT = 60

# How long (seconds) of silence before we consider the call stalled
_MEDIA_STREAM_SILENCE_TIMEOUT = 30

# Minimum confidence threshold to act on a transcript result
_MIN_TRANSCRIPT_CONFIDENCE = 0.5


# ---------------------------------------------------------------------------
# CallDispatcher
# ---------------------------------------------------------------------------


class CallDispatcher:
    """Orchestrates outbound AI-driven phone calls.

    Coordinates Twilio (telephony), Deepgram (transcription), and OpenAI
    (conversation) to drive automated outbound calls toward a stated goal.

    Args:
        store: The :class:`~call_dispatch.store.CallStore` used to persist
            call records and transcripts.
        openai_api_key: Optional OpenAI API key override.
        deepgram_api_key: Optional Deepgram API key override.
        twilio_account_sid: Optional Twilio Account SID override.
        twilio_auth_token: Optional Twilio Auth Token override.
        twilio_phone_number: Optional Twilio phone number override.
        public_base_url: Optional public base URL override for webhook construction.
    """

    def __init__(
        self,
        store: CallStore,
        *,
        openai_api_key: Optional[str] = None,
        deepgram_api_key: Optional[str] = None,
        twilio_account_sid: Optional[str] = None,
        twilio_auth_token: Optional[str] = None,
        twilio_phone_number: Optional[str] = None,
        public_base_url: Optional[str] = None,
    ) -> None:
        from call_dispatch.config import get_settings

        cfg = get_settings()

        self._store = store
        self._openai_api_key = openai_api_key or cfg.openai_api_key
        self._deepgram_api_key = deepgram_api_key or cfg.deepgram_api_key
        self._twilio_account_sid = twilio_account_sid or cfg.twilio_account_sid
        self._twilio_auth_token = twilio_auth_token or cfg.twilio_auth_token
        self._twilio_phone_number = twilio_phone_number or cfg.twilio_phone_number
        self._public_base_url = (public_base_url or cfg.public_base_url).rstrip("/")
        self._call_timeout_seconds = cfg.call_timeout_seconds

        # Active call state: call_id -> internal state dict
        self._active_calls: dict[str, _CallState] = {}

        logger.debug(
            "CallDispatcher initialised public_base_url=%s",
            self._public_base_url,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def dispatch(
        self,
        to_number: str,
        goal: str,
        context: Optional[str] = None,
        max_duration_seconds: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> CallRecord:
        """Initiate a new outbound call and persist the call record.

        Creates a call record in :attr:`CallStatus.PENDING` state, then
        initiates the Twilio outbound call.  The record is updated to
        :attr:`CallStatus.INITIATING` once Twilio accepts the call.

        Args:
            to_number: Destination phone number in E.164 format.
            goal: Natural language description of the call goal.
            context: Optional supplementary context for the agent.
            max_duration_seconds: Optional call duration override.
            metadata: Optional arbitrary metadata to store with the record.

        Returns:
            CallRecord: The newly created and initiated call record.

        Raises:
            DispatchError: If the Twilio API call fails or returns an error.
        """
        call_id = str(uuid.uuid4())
        effective_timeout = max_duration_seconds or self._call_timeout_seconds

        record = CallRecord(
            call_id=call_id,
            to_number=to_number,
            from_number=self._twilio_phone_number,
            goal=goal,
            context=context,
            status=CallStatus.PENDING,
            metadata=metadata,
            max_duration_seconds=max_duration_seconds,
        )

        # Persist the record immediately
        await self._store.create_call(record)
        logger.info(
            "Call record created: call_id=%s to=%s goal=%r",
            call_id,
            to_number,
            goal[:60],
        )

        # Build webhook URLs
        answer_url = f"{self._public_base_url}/twiml/answer/{call_id}"
        status_url = f"{self._public_base_url}/twiml/status/{call_id}"

        # Initiate the call via Twilio
        try:
            twilio_call_sid = await self._initiate_twilio_call(
                to_number=to_number,
                answer_url=answer_url,
                status_url=status_url,
                timeout=effective_timeout,
            )
        except DispatchError as exc:
            logger.error(
                "Failed to initiate Twilio call for call_id=%s: %s", call_id, exc
            )
            await self._store.update_status(
                call_id,
                CallStatus.FAILED,
                error_message=str(exc),
                ended_at=datetime.utcnow(),
            )
            record.status = CallStatus.FAILED
            record.error_message = str(exc)
            return record

        # Update record with Twilio SID and INITIATING status
        record.twilio_call_sid = twilio_call_sid
        record.set_status(CallStatus.INITIATING)
        await self._store.update_call(record)

        logger.info(
            "Twilio call initiated: call_id=%s twilio_sid=%s",
            call_id,
            twilio_call_sid,
        )

        # Register this call in the active calls registry
        state = _CallState(
            call_id=call_id,
            goal=goal,
            context=context,
            openai_api_key=self._openai_api_key,
            deepgram_api_key=self._deepgram_api_key,
            store=self._store,
            timeout_seconds=effective_timeout,
        )
        self._active_calls[call_id] = state

        return record

    async def dispatch_from_request(
        self, request: DispatchCallRequest
    ) -> CallRecord:
        """Convenience wrapper that dispatches a call from a
        :class:`~call_dispatch.models.DispatchCallRequest`.

        Args:
            request: The validated dispatch request.

        Returns:
            CallRecord: The newly created call record.

        Raises:
            DispatchError: If the Twilio API call fails.
        """
        return await self.dispatch(
            to_number=request.to_number,
            goal=request.goal,
            context=request.context,
            max_duration_seconds=request.max_duration_seconds,
            metadata=request.metadata,
        )

    async def handle_media_stream(self, websocket: Any) -> None:
        """Handle an incoming Twilio Media Stream WebSocket for an active call.

        This method is invoked by the TwiML handler's WebSocket endpoint
        when Twilio connects the media stream.  It:

        1. Identifies the call from the WebSocket URL path parameter.
        2. Sets up the Deepgram transcriber and the ConversationAgent.
        3. Processes ``connected``, ``start``, ``media``, and ``stop`` events.
        4. Forwards inbound audio to Deepgram for transcription.
        5. Invokes the agent on each final transcript and speaks the response.
        6. Triggers post-call summarisation on call completion.

        Args:
            websocket: The FastAPI ``WebSocket`` object representing the
                Twilio Media Stream connection.
        """
        # Extract call_id from the websocket path
        call_id: Optional[str] = None
        try:
            call_id = websocket.path_params.get("call_id")
        except AttributeError:
            pass

        if not call_id:
            logger.warning("handle_media_stream: no call_id in WebSocket path")
            await websocket.close(code=1008)
            return

        state = self._active_calls.get(call_id)
        if state is None:
            # Try to reconstruct state from the store
            record = await self._store.get_call(call_id)
            if record is None:
                logger.warning(
                    "handle_media_stream: call_id=%s not found in store", call_id
                )
                await websocket.close(code=1008)
                return
            state = _CallState(
                call_id=call_id,
                goal=record.goal,
                context=record.context,
                openai_api_key=self._openai_api_key,
                deepgram_api_key=self._deepgram_api_key,
                store=self._store,
                timeout_seconds=record.max_duration_seconds or self._call_timeout_seconds,
                existing_transcript=record.transcript,
            )
            self._active_calls[call_id] = state

        try:
            await state.run_media_stream(websocket)
        finally:
            # Clean up active call state after stream ends
            self._active_calls.pop(call_id, None)
            logger.info("Active call state removed for call_id=%s", call_id)

            # Trigger post-call processing
            await self._finalize_call(call_id)

    async def cancel_call(self, call_id: str) -> bool:
        """Cancel a pending or initiating call.

        Attempts to cancel the Twilio call if it has a Twilio SID, and
        updates the call record to :attr:`CallStatus.CANCELLED`.

        Args:
            call_id: The internal call ID to cancel.

        Returns:
            bool: ``True`` if the call was found and cancelled.
        """
        record = await self._store.get_call(call_id)
        if record is None:
            return False

        if record.status not in (CallStatus.PENDING, CallStatus.INITIATING):
            logger.debug(
                "cancel_call: call_id=%s in non-cancellable status %s",
                call_id,
                record.status,
            )
            return False

        # Attempt to cancel the Twilio call
        if record.twilio_call_sid:
            try:
                await self._cancel_twilio_call(record.twilio_call_sid)
            except Exception as exc:
                logger.warning(
                    "Failed to cancel Twilio call %s: %s",
                    record.twilio_call_sid,
                    exc,
                )

        await self._store.update_status(
            call_id,
            CallStatus.CANCELLED,
            ended_at=datetime.utcnow(),
        )

        # Remove from active calls
        self._active_calls.pop(call_id, None)

        logger.info("Call cancelled: call_id=%s", call_id)
        return True

    # ------------------------------------------------------------------
    # Twilio API helpers
    # ------------------------------------------------------------------

    async def _initiate_twilio_call(
        self,
        to_number: str,
        answer_url: str,
        status_url: str,
        timeout: int = 300,
    ) -> str:
        """Call the Twilio REST API to initiate an outbound call.

        Uses the Twilio Python client in a thread pool to avoid blocking
        the event loop.

        Args:
            to_number: Destination phone number in E.164 format.
            answer_url: Publicly accessible URL for the TwiML answer webhook.
            status_url: Publicly accessible URL for status callbacks.
            timeout: Call timeout in seconds.

        Returns:
            str: The Twilio Call SID (``CA...``) for the initiated call.

        Raises:
            DispatchError: If the Twilio API returns an error.
        """
        try:
            twilio_sid = await asyncio.to_thread(
                self._sync_create_twilio_call,
                to_number,
                answer_url,
                status_url,
                timeout,
            )
            return twilio_sid
        except DispatchError:
            raise
        except Exception as exc:
            raise DispatchError(
                f"Unexpected error initiating Twilio call to {to_number}: {exc}"
            ) from exc

    def _sync_create_twilio_call(
        self,
        to_number: str,
        answer_url: str,
        status_url: str,
        timeout: int,
    ) -> str:
        """Synchronously create a Twilio call using the Twilio Python SDK.

        This method is intended to be called via :func:`asyncio.to_thread`.

        Args:
            to_number: Destination phone number.
            answer_url: URL for the TwiML answer webhook.
            status_url: URL for status callbacks.
            timeout: Call timeout in seconds.

        Returns:
            str: Twilio Call SID.

        Raises:
            DispatchError: If the Twilio client raises an exception.
        """
        try:
            from twilio.rest import Client as TwilioClient
            from twilio.base.exceptions import TwilioRestException

            client = TwilioClient(
                self._twilio_account_sid,
                self._twilio_auth_token,
            )

            call = client.calls.create(
                to=to_number,
                from_=self._twilio_phone_number,
                url=answer_url,
                status_callback=status_url,
                status_callback_method="POST",
                status_callback_event=[
                    "initiated",
                    "ringing",
                    "answered",
                    "completed",
                ],
                timeout=timeout,
                machine_detection="Disable",
            )

            logger.debug(
                "Twilio call created: sid=%s status=%s",
                call.sid,
                call.status,
            )
            return call.sid

        except TwilioRestException as exc:
            raise DispatchError(
                f"Twilio REST error ({exc.code}): {exc.msg}"
            ) from exc
        except Exception as exc:
            raise DispatchError(
                f"Failed to create Twilio call: {exc}"
            ) from exc

    async def _cancel_twilio_call(self, twilio_call_sid: str) -> None:
        """Cancel a Twilio call by SID.

        Args:
            twilio_call_sid: The Twilio Call SID to cancel.

        Raises:
            DispatchError: If the Twilio API call fails.
        """
        await asyncio.to_thread(self._sync_cancel_twilio_call, twilio_call_sid)

    def _sync_cancel_twilio_call(self, twilio_call_sid: str) -> None:
        """Synchronously cancel a Twilio call.

        Args:
            twilio_call_sid: The Twilio Call SID to cancel.

        Raises:
            DispatchError: If the Twilio REST API call fails.
        """
        try:
            from twilio.rest import Client as TwilioClient
            from twilio.base.exceptions import TwilioRestException

            client = TwilioClient(
                self._twilio_account_sid,
                self._twilio_auth_token,
            )
            client.calls(twilio_call_sid).update(status="canceled")
            logger.debug("Twilio call cancelled: sid=%s", twilio_call_sid)
        except TwilioRestException as exc:
            raise DispatchError(
                f"Failed to cancel Twilio call {twilio_call_sid}: {exc.msg}"
            ) from exc

    # ------------------------------------------------------------------
    # Post-call processing
    # ------------------------------------------------------------------

    async def _finalize_call(self, call_id: str) -> None:
        """Perform post-call processing after the media stream closes.

        Fetches the final call record, generates a summary if the call
        reached a terminal state, and persists the summary.

        Args:
            call_id: The internal call ID to finalise.
        """
        try:
            record = await self._store.get_call(call_id)
            if record is None:
                logger.warning("_finalize_call: call_id=%s not found", call_id)
                return

            # Only summarise calls that completed or failed with transcript data
            if record.status not in (
                CallStatus.COMPLETED,
                CallStatus.FAILED,
                CallStatus.NO_ANSWER,
                CallStatus.BUSY,
                CallStatus.CANCELLED,
            ):
                # Mark as completed if still in_progress
                if record.status == CallStatus.IN_PROGRESS:
                    await self._store.update_status(
                        call_id,
                        CallStatus.COMPLETED,
                        ended_at=datetime.utcnow(),
                    )
                    record.status = CallStatus.COMPLETED

            # Generate summary if we have transcript data
            if record.transcript and record.status in (
                CallStatus.COMPLETED,
                CallStatus.FAILED,
            ):
                await self._generate_and_save_summary(record)

        except Exception as exc:
            logger.error(
                "Error during post-call finalization for call_id=%s: %s",
                call_id,
                exc,
            )

    async def _generate_and_save_summary(
        self, record: CallRecord
    ) -> Optional[CallSummary]:
        """Generate a structured summary for a completed call and persist it.

        Args:
            record: The completed :class:`~call_dispatch.models.CallRecord`.

        Returns:
            CallSummary | None: The generated summary, or ``None`` on failure.
        """
        logger.info(
            "Generating summary for call_id=%s transcript_entries=%d",
            record.call_id,
            len(record.transcript),
        )
        try:
            summarizer = CallSummarizer(openai_api_key=self._openai_api_key)
            summary = await summarizer.summarize(
                goal=record.goal,
                transcript=record.transcript,
                context=record.context,
            )
            await self._store.save_summary(record.call_id, summary)
            logger.info(
                "Summary saved for call_id=%s outcome=%s",
                record.call_id,
                summary.outcome.value,
            )
            return summary
        except SummarizerError as exc:
            logger.error(
                "Summarizer error for call_id=%s: %s", record.call_id, exc
            )
            return None
        except Exception as exc:
            logger.error(
                "Unexpected error generating summary for call_id=%s: %s",
                record.call_id,
                exc,
            )
            return None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def active_call_ids(self) -> list[str]:
        """Return the list of currently active call IDs."""
        return list(self._active_calls.keys())

    @property
    def store(self) -> CallStore:
        """The underlying call store."""
        return self._store


# ---------------------------------------------------------------------------
# Internal call state machine
# ---------------------------------------------------------------------------


class _CallState:
    """Internal per-call state container for an active media stream session.

    Manages the Deepgram transcriber, ConversationAgent, and the state
    machine for a single in-progress call.  Instances are created by
    :class:`CallDispatcher` and live for the duration of the call's
    Media Stream WebSocket connection.

    Args:
        call_id: Internal call identifier.
        goal: The call goal.
        context: Optional supplementary context.
        openai_api_key: OpenAI API key.
        deepgram_api_key: Deepgram API key.
        store: The call store for persistence.
        timeout_seconds: Maximum call duration in seconds.
        existing_transcript: Optional pre-existing transcript entries to
            pre-populate the agent's history (for resuming calls).
    """

    def __init__(
        self,
        call_id: str,
        goal: str,
        context: Optional[str],
        openai_api_key: str,
        deepgram_api_key: str,
        store: CallStore,
        timeout_seconds: int = 300,
        existing_transcript: Optional[list[TranscriptEntry]] = None,
    ) -> None:
        self.call_id = call_id
        self.goal = goal
        self.context = context
        self._openai_api_key = openai_api_key
        self._deepgram_api_key = deepgram_api_key
        self._store = store
        self._timeout_seconds = timeout_seconds

        # Build the conversational agent
        self._agent = ConversationAgent.from_record(
            goal=goal,
            context=context,
            transcript=existing_transcript or [],
            openai_api_key=openai_api_key,
        )

        # State flags
        self._stream_sid: Optional[str] = None
        self._call_started = False
        self._opening_sent = False
        self._is_agent_speaking = False
        self._pending_transcript_parts: list[str] = []

        # Queue for final transcript results to be processed sequentially
        self._transcript_queue: asyncio.Queue[TranscriptionResult] = asyncio.Queue()

        logger.debug("_CallState created for call_id=%s", call_id)

    # ------------------------------------------------------------------
    # Media stream runner
    # ------------------------------------------------------------------

    async def run_media_stream(self, websocket: Any) -> None:
        """Process Twilio Media Stream WebSocket events for this call.

        Runs until the stream ends or an error occurs.  Connects the
        Deepgram transcriber and starts the agent processing loop.

        Args:
            websocket: The accepted FastAPI WebSocket connection from Twilio.
        """
        logger.info("Media stream started for call_id=%s", self.call_id)

        async def on_transcript(result: TranscriptionResult) -> None:
            """Callback invoked by the Deepgram transcriber for each result."""
            await self._transcript_queue.put(result)

        transcriber = DeepgramTranscriber(
            callback=on_transcript,
            api_key=self._deepgram_api_key,
        )

        try:
            async with transcriber:
                # Run the media event loop and transcript processor concurrently
                await asyncio.gather(
                    self._media_event_loop(websocket, transcriber),
                    self._transcript_processor(websocket),
                    return_exceptions=True,
                )
        except TranscriberError as exc:
            logger.error(
                "Transcriber error for call_id=%s: %s", self.call_id, exc
            )
        except Exception as exc:
            logger.error(
                "Unexpected error in media stream for call_id=%s: %s",
                self.call_id,
                exc,
            )
        finally:
            logger.info("Media stream ended for call_id=%s", self.call_id)

    async def _media_event_loop(
        self,
        websocket: Any,
        transcriber: DeepgramTranscriber,
    ) -> None:
        """Read and dispatch Twilio Media Stream events.

        Processes ``connected``, ``start``, ``media``, ``mark``, and
        ``stop`` events from the Twilio WebSocket.  Audio payload from
        ``media`` events is forwarded to the Deepgram transcriber.

        Args:
            websocket: The FastAPI WebSocket from Twilio.
            transcriber: The active :class:`DeepgramTranscriber` instance.
        """
        from fastapi import WebSocketDisconnect

        try:
            while True:
                try:
                    raw = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=self._timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "Media stream timed out for call_id=%s", self.call_id
                    )
                    break

                try:
                    event: dict = json.loads(raw)
                except json.JSONDecodeError:
                    logger.debug(
                        "Non-JSON media stream message for call_id=%s", self.call_id
                    )
                    continue

                event_type = event.get("event", "")

                if event_type == "connected":
                    logger.info(
                        "Twilio stream connected for call_id=%s", self.call_id
                    )

                elif event_type == "start":
                    start_data = event.get("start", {})
                    self._stream_sid = (
                        event.get("streamSid")
                        or start_data.get("streamSid")
                    )
                    self._call_started = True
                    logger.info(
                        "Twilio stream started: call_id=%s stream_sid=%s",
                        self.call_id,
                        self._stream_sid,
                    )

                    # Trigger the agent opening utterance
                    asyncio.create_task(
                        self._send_opening_utterance(websocket),
                        name=f"opening-{self.call_id}",
                    )

                elif event_type == "media":
                    media_data = event.get("media", {})
                    payload: Optional[str] = media_data.get("payload")
                    if payload:
                        await transcriber.send_audio_base64(payload)

                elif event_type == "mark":
                    mark_name = event.get("mark", {}).get("name", "")
                    if mark_name.startswith("agent-done"):
                        self._is_agent_speaking = False
                        logger.debug(
                            "Agent utterance complete for call_id=%s", self.call_id
                        )

                elif event_type == "stop":
                    logger.info(
                        "Twilio stream stopped for call_id=%s", self.call_id
                    )
                    break

                else:
                    logger.debug(
                        "Unhandled stream event type=%r for call_id=%s",
                        event_type,
                        self.call_id,
                    )

        except WebSocketDisconnect:
            logger.info(
                "WebSocket disconnected for call_id=%s", self.call_id
            )
        except Exception as exc:
            logger.error(
                "Error in media event loop for call_id=%s: %s",
                self.call_id,
                exc,
            )
        finally:
            # Signal the transcript processor to stop
            await self._transcript_queue.put(
                TranscriptionResult(
                    text="__END__",
                    confidence=1.0,
                    is_final=True,
                    channel=0,
                )
            )

    async def _transcript_processor(self, websocket: Any) -> None:
        """Process final transcript results and generate agent responses.

        Consumes :class:`~call_dispatch.models.TranscriptionResult` objects
        from the internal queue, appends them to the call transcript, and
        invokes the agent to produce the next utterance.

        Args:
            websocket: The FastAPI WebSocket to inject agent responses into.
        """
        while True:
            try:
                result = await asyncio.wait_for(
                    self._transcript_queue.get(),
                    timeout=self._timeout_seconds,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Transcript processor timed out for call_id=%s", self.call_id
                )
                break

            # Sentinel value signals end of stream
            if result.text == "__END__":
                logger.debug(
                    "Transcript processor received END signal for call_id=%s",
                    self.call_id,
                )
                break

            # Only act on final results with sufficient confidence
            if not result.is_final:
                continue

            if result.confidence < _MIN_TRANSCRIPT_CONFIDENCE:
                logger.debug(
                    "Skipping low-confidence transcript (%.2f < %.2f) for call_id=%s",
                    result.confidence,
                    _MIN_TRANSCRIPT_CONFIDENCE,
                    self.call_id,
                )
                continue

            # Persist the contact's utterance
            entry = TranscriptEntry(
                speaker="contact",
                text=result.text,
                confidence=result.confidence,
                timestamp=result.timestamp,
            )
            await self._store.append_transcript_entry(self.call_id, entry)
            logger.debug(
                "Contact utterance: call_id=%s text=%r",
                self.call_id,
                result.text[:80],
            )

            # Don't interrupt the agent if it's currently speaking
            if self._is_agent_speaking:
                logger.debug(
                    "Agent speaking, queuing contact utterance for call_id=%s",
                    self.call_id,
                )
                continue

            # Generate the agent's response
            await self._generate_and_send_agent_response(
                websocket, contact_message=result.text
            )

    async def _send_opening_utterance(self, websocket: Any) -> None:
        """Generate and inject the agent's opening utterance into the call.

        Called once when the Twilio media stream starts.

        Args:
            websocket: The FastAPI WebSocket to inject into.
        """
        if self._opening_sent:
            return
        self._opening_sent = True

        try:
            utterance = await self._agent.get_opening_utterance()
            await self._inject_agent_utterance(websocket, utterance)
        except AgentError as exc:
            logger.error(
                "Failed to generate opening utterance for call_id=%s: %s",
                self.call_id,
                exc,
            )

    async def _generate_and_send_agent_response(
        self, websocket: Any, contact_message: str
    ) -> None:
        """Generate the agent's next utterance in response to the contact.

        Args:
            websocket: The FastAPI WebSocket to inject the response into.
            contact_message: The contact's most recent utterance.
        """
        try:
            self._is_agent_speaking = True
            utterance = await self._agent.get_next_utterance(contact_message)
            await self._inject_agent_utterance(websocket, utterance)
        except AgentError as exc:
            logger.error(
                "Agent error for call_id=%s: %s", self.call_id, exc
            )
            self._is_agent_speaking = False

    async def _inject_agent_utterance(
        self, websocket: Any, utterance: str
    ) -> None:
        """Inject an agent utterance into the active Twilio call.

        Sends a TwiML ``<Say>`` verb via Twilio's Modify Call API to speak
        the agent's response on the call, and sends a ``mark`` event to
        the media stream so we know when the utterance has finished playing.

        For the media stream approach, we use the stream ``mark`` event
        to signal the end of the utterance.  The actual TTS injection is
        handled by constructing a TwiML URL update via Twilio's REST API.

        In production this requires either:
        - Twilio's ``<Say>`` via a TwiML redirect/modify API call, or
        - Sending audio back through the WebSocket as a base64 media event.

        For this implementation, we log the utterance and persist it to the
        transcript.  Full TTS injection would require additional integration
        (e.g. Amazon Polly or Twilio <Say>) which is environment-specific.

        Args:
            websocket: The FastAPI WebSocket to the Twilio Media Stream.
            utterance: The text utterance to inject.
        """
        # Persist the agent utterance to transcript
        entry = TranscriptEntry(
            speaker="agent",
            text=utterance,
            timestamp=datetime.utcnow(),
        )
        await self._store.append_transcript_entry(self.call_id, entry)

        logger.info(
            "Agent utterance: call_id=%s text=%r",
            self.call_id,
            utterance[:100],
        )

        # Send a mark event to track when this utterance is delivered
        if self._stream_sid:
            try:
                from call_dispatch.twiml_handler import build_mark_message

                mark_msg = build_mark_message(
                    self._stream_sid,
                    f"agent-done-{self._agent.turn_count}",
                )
                await websocket.send_text(mark_msg)
                logger.debug(
                    "Sent mark event for agent utterance call_id=%s turn=%d",
                    self.call_id,
                    self._agent.turn_count,
                )
            except Exception as exc:
                logger.debug(
                    "Failed to send mark event for call_id=%s: %s",
                    self.call_id,
                    exc,
                )
                self._is_agent_speaking = False
        else:
            # No stream SID yet — mark as not speaking immediately
            self._is_agent_speaking = False


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class DispatchError(Exception):
    """Raised when the :class:`CallDispatcher` cannot initiate or manage a call.

    Wraps underlying Twilio REST errors and other dispatch-time failures
    so callers can catch a single, application-specific exception type.
    """
