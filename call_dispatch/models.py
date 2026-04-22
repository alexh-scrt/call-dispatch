"""Pydantic models for call_dispatch API request/response schemas and database row types.

This module defines all data structures used throughout the application:
- API request schemas for dispatching calls
- API response schemas for call status and summaries
- Internal database row types for SQLite persistence
- Enumerations for call state and outcome
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class CallStatus(str, Enum):
    """Lifecycle states for an outbound call."""

    PENDING = "pending"
    """Call has been queued but not yet initiated with Twilio."""

    INITIATING = "initiating"
    """Call request has been sent to Twilio, awaiting connection."""

    IN_PROGRESS = "in_progress"
    """Call is currently connected and the agent is active."""

    COMPLETED = "completed"
    """Call ended normally; summary may be available."""

    FAILED = "failed"
    """Call could not be connected or encountered a fatal error."""

    CANCELLED = "cancelled"
    """Call was cancelled before it could be answered."""

    NO_ANSWER = "no_answer"
    """The called party did not answer."""

    BUSY = "busy"
    """The called party's line was busy."""


class CallOutcome(str, Enum):
    """High-level outcome of a completed call as determined by the summariser."""

    SUCCESS = "success"
    """The call goal was fully achieved."""

    PARTIAL = "partial"
    """The call goal was partially achieved."""

    FAILURE = "failure"
    """The call goal was not achieved."""

    UNKNOWN = "unknown"
    """The outcome could not be determined (e.g., call did not complete)."""


# ---------------------------------------------------------------------------
# API Request Schemas
# ---------------------------------------------------------------------------


class DispatchCallRequest(BaseModel):
    """Request body for dispatching a new outbound call.

    Example::

        {
            "to_number": "+15550001234",
            "goal": "Book a dentist appointment for Tuesday afternoon",
            "context": "Patient name is Jane Doe, date of birth 1990-05-15"
        }
    """

    to_number: str = Field(
        ...,
        description="Destination phone number in E.164 format, e.g. +15550001234.",
        examples=["+15550001234"],
    )
    goal: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        description=(
            "Natural language description of what the AI agent should accomplish "
            "during the call, e.g. 'Book a dentist appointment for Tuesday afternoon'."
        ),
        examples=["Book a dentist appointment for Tuesday afternoon."],
    )
    context: Optional[str] = Field(
        default=None,
        max_length=2000,
        description=(
            "Optional additional context provided to the agent, such as the caller's "
            "name, reference numbers, or any specific constraints."
        ),
        examples=["Patient name is Jane Doe, date of birth 1990-05-15."],
    )
    max_duration_seconds: Optional[int] = Field(
        default=None,
        ge=30,
        le=3600,
        description=(
            "Maximum call duration in seconds. Overrides the server default when set. "
            "Must be between 30 and 3600 seconds."
        ),
    )
    metadata: Optional[dict[str, Any]] = Field(
        default=None,
        description="Arbitrary key-value pairs to store alongside the call record.",
    )

    @field_validator("to_number")
    @classmethod
    def validate_e164(cls, v: str) -> str:
        """Ensure the destination number is in E.164 format."""
        stripped = v.strip()
        if not stripped.startswith("+"):
            raise ValueError(
                f"to_number must be in E.164 format (start with '+'). Got: {stripped!r}"
            )
        digits = stripped[1:]
        if not digits.isdigit() or len(digits) < 7 or len(digits) > 15:
            raise ValueError(
                f"to_number contains invalid digits or length: {stripped!r}"
            )
        return stripped

    @field_validator("goal")
    @classmethod
    def validate_goal_not_blank(cls, v: str) -> str:
        """Ensure the goal is not just whitespace."""
        if not v.strip():
            raise ValueError("goal must not be blank or whitespace only.")
        return v.strip()


# ---------------------------------------------------------------------------
# Transcript Entry
# ---------------------------------------------------------------------------


class TranscriptEntry(BaseModel):
    """A single utterance in the call transcript."""

    speaker: str = Field(
        ...,
        description="Who spoke this utterance: 'agent' or 'contact'.",
        examples=["agent", "contact"],
    )
    text: str = Field(
        ...,
        description="The transcribed or generated text of the utterance.",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp when the utterance was recorded.",
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Transcription confidence score from Deepgram (0.0–1.0), if available.",
    )

    @field_validator("speaker")
    @classmethod
    def validate_speaker(cls, v: str) -> str:
        """Ensure speaker is one of the recognised roles."""
        allowed = {"agent", "contact"}
        if v.lower() not in allowed:
            raise ValueError(f"speaker must be one of {allowed}, got {v!r}")
        return v.lower()


# ---------------------------------------------------------------------------
# Call Summary
# ---------------------------------------------------------------------------


class CallSummary(BaseModel):
    """Structured summary produced by the summariser after a call completes."""

    outcome: CallOutcome = Field(
        default=CallOutcome.UNKNOWN,
        description="High-level outcome of the call.",
    )
    summary_text: str = Field(
        ...,
        description="Human-readable narrative summary of what occurred during the call.",
    )
    key_details: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Structured key-value pairs extracted from the call, e.g. "
            "{'booked_time': 'Tuesday 3pm', 'confirmation_number': 'ABC123'}."
        ),
    )
    follow_up_required: bool = Field(
        default=False,
        description="Whether a human follow-up action is recommended.",
    )
    follow_up_notes: Optional[str] = Field(
        default=None,
        description="Notes describing what follow-up is required, if any.",
    )
    generated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp when this summary was generated.",
    )


# ---------------------------------------------------------------------------
# Database Row Types (internal)
# ---------------------------------------------------------------------------


class CallRecord(BaseModel):
    """Full call record as stored in SQLite.

    This model represents a single row in the ``calls`` table, including
    all lifecycle fields, the full transcript, and the optional summary.
    """

    call_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this call record (UUID4).",
    )
    to_number: str = Field(
        ...,
        description="Destination phone number in E.164 format.",
    )
    from_number: str = Field(
        ...,
        description="Source Twilio phone number in E.164 format.",
    )
    goal: str = Field(
        ...,
        description="The natural language goal provided by the caller.",
    )
    context: Optional[str] = Field(
        default=None,
        description="Optional additional context provided at dispatch time.",
    )
    status: CallStatus = Field(
        default=CallStatus.PENDING,
        description="Current lifecycle state of the call.",
    )
    twilio_call_sid: Optional[str] = Field(
        default=None,
        description="Twilio's Call SID returned after the call is initiated.",
    )
    transcript: list[TranscriptEntry] = Field(
        default_factory=list,
        description="Ordered list of utterances from the call conversation.",
    )
    summary: Optional[CallSummary] = Field(
        default=None,
        description="Structured summary produced after call completion.",
    )
    metadata: Optional[dict[str, Any]] = Field(
        default=None,
        description="Arbitrary caller-supplied metadata stored with the record.",
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Human-readable error description if the call failed.",
    )
    max_duration_seconds: Optional[int] = Field(
        default=None,
        description="Maximum duration override, if set by the caller.",
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp when the call record was created.",
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp of the most recent update to this record.",
    )
    started_at: Optional[datetime] = Field(
        default=None,
        description="UTC timestamp when the call was answered and became in-progress.",
    )
    ended_at: Optional[datetime] = Field(
        default=None,
        description="UTC timestamp when the call ended.",
    )

    @property
    def duration_seconds(self) -> Optional[float]:
        """Computed call duration in seconds, or None if not yet ended."""
        if self.started_at is not None and self.ended_at is not None:
            return (self.ended_at - self.started_at).total_seconds()
        return None

    def add_transcript_entry(self, entry: TranscriptEntry) -> None:
        """Append a new utterance to the transcript in place.

        Args:
            entry: The :class:`TranscriptEntry` to append.
        """
        self.transcript.append(entry)
        self.updated_at = datetime.utcnow()

    def set_status(self, status: CallStatus, error_message: Optional[str] = None) -> None:
        """Update the call status and optionally set an error message.

        Args:
            status: The new :class:`CallStatus` to apply.
            error_message: Optional error description (used with FAILED status).
        """
        self.status = status
        self.updated_at = datetime.utcnow()
        if error_message is not None:
            self.error_message = error_message
        if status == CallStatus.IN_PROGRESS and self.started_at is None:
            self.started_at = datetime.utcnow()
        if status in (
            CallStatus.COMPLETED,
            CallStatus.FAILED,
            CallStatus.CANCELLED,
            CallStatus.NO_ANSWER,
            CallStatus.BUSY,
        ):
            if self.ended_at is None:
                self.ended_at = datetime.utcnow()


# ---------------------------------------------------------------------------
# API Response Schemas
# ---------------------------------------------------------------------------


class DispatchCallResponse(BaseModel):
    """Response returned after successfully dispatching a call."""

    call_id: str = Field(
        ...,
        description="Unique identifier for the dispatched call. Use this to poll status.",
    )
    status: CallStatus = Field(
        ...,
        description="Initial call status (always 'pending' immediately after dispatch).",
    )
    message: str = Field(
        default="Call successfully queued.",
        description="Human-readable confirmation message.",
    )
    created_at: datetime = Field(
        ...,
        description="UTC timestamp when the call record was created.",
    )


class CallStatusResponse(BaseModel):
    """Response schema for the call status polling endpoint.

    Returns the full live state of a call including transcript snippets
    and the final summary when available.
    """

    call_id: str = Field(..., description="Unique identifier for the call.")
    to_number: str = Field(..., description="Destination phone number.")
    goal: str = Field(..., description="The natural language goal of the call.")
    status: CallStatus = Field(..., description="Current lifecycle state of the call.")
    twilio_call_sid: Optional[str] = Field(
        default=None,
        description="Twilio's Call SID, available after the call is initiated.",
    )
    transcript: list[TranscriptEntry] = Field(
        default_factory=list,
        description="Current transcript of the call conversation.",
    )
    summary: Optional[CallSummary] = Field(
        default=None,
        description="Structured summary, available after the call completes.",
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error description if the call failed.",
    )
    duration_seconds: Optional[float] = Field(
        default=None,
        description="Total call duration in seconds, available after the call ends.",
    )
    created_at: datetime = Field(..., description="When the call record was created.")
    updated_at: datetime = Field(..., description="When the call record was last updated.")
    started_at: Optional[datetime] = Field(
        default=None, description="When the call was answered."
    )
    ended_at: Optional[datetime] = Field(
        default=None, description="When the call ended."
    )

    @classmethod
    def from_record(cls, record: CallRecord) -> "CallStatusResponse":
        """Construct a :class:`CallStatusResponse` from a :class:`CallRecord`.

        Args:
            record: The call record to convert.

        Returns:
            CallStatusResponse: The populated response schema.
        """
        return cls(
            call_id=record.call_id,
            to_number=record.to_number,
            goal=record.goal,
            status=record.status,
            twilio_call_sid=record.twilio_call_sid,
            transcript=record.transcript,
            summary=record.summary,
            error_message=record.error_message,
            duration_seconds=record.duration_seconds,
            created_at=record.created_at,
            updated_at=record.updated_at,
            started_at=record.started_at,
            ended_at=record.ended_at,
        )


class CallSummaryResponse(BaseModel):
    """Response schema for the call summary endpoint."""

    call_id: str = Field(..., description="Unique identifier for the call.")
    status: CallStatus = Field(..., description="Final lifecycle state of the call.")
    summary: Optional[CallSummary] = Field(
        default=None,
        description="Structured summary. None if the call has not yet completed.",
    )
    transcript: list[TranscriptEntry] = Field(
        default_factory=list,
        description="Full ordered transcript of the call conversation.",
    )
    duration_seconds: Optional[float] = Field(
        default=None,
        description="Total call duration in seconds.",
    )

    @classmethod
    def from_record(cls, record: CallRecord) -> "CallSummaryResponse":
        """Construct a :class:`CallSummaryResponse` from a :class:`CallRecord`.

        Args:
            record: The call record to convert.

        Returns:
            CallSummaryResponse: The populated response schema.
        """
        return cls(
            call_id=record.call_id,
            status=record.status,
            summary=record.summary,
            transcript=record.transcript,
            duration_seconds=record.duration_seconds,
        )


class ListCallsResponse(BaseModel):
    """Response schema for the list-calls endpoint."""

    total: int = Field(..., description="Total number of call records matching the query.")
    calls: list[CallStatusResponse] = Field(
        ...,
        description="List of call status summaries.",
    )


class ErrorResponse(BaseModel):
    """Standard error response body."""

    detail: str = Field(..., description="Human-readable error message.")
    error_code: Optional[str] = Field(
        default=None,
        description="Machine-readable error code for programmatic handling.",
    )


# ---------------------------------------------------------------------------
# Internal agent / transcriber helper types
# ---------------------------------------------------------------------------


class AgentMessage(BaseModel):
    """A single message in the OpenAI chat completion conversation history."""

    role: str = Field(
        ...,
        description="Message role: 'system', 'user', or 'assistant'.",
    )
    content: str = Field(
        ...,
        description="Text content of the message.",
    )

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        """Ensure the role is a valid OpenAI chat role."""
        allowed = {"system", "user", "assistant"}
        if v.lower() not in allowed:
            raise ValueError(f"role must be one of {allowed}, got {v!r}")
        return v.lower()


class TranscriptionResult(BaseModel):
    """Deepgram transcription result emitted by the transcriber."""

    text: str = Field(..., description="The transcribed text.")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score from Deepgram (0.0–1.0).",
    )
    is_final: bool = Field(
        ...,
        description="Whether this is a final (non-interim) transcript result.",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp when the transcription was received.",
    )
    channel: int = Field(
        default=0,
        description="Audio channel index (0 for mono or primary channel).",
    )
