"""Integration tests for call_dispatch REST API endpoints.

Uses FastAPI's TestClient with mocked dependencies to test all REST
endpoints without requiring real Twilio, OpenAI, or Deepgram credentials.

Test coverage includes:
- POST /calls (dispatch)
- GET /calls (list)
- GET /calls/{call_id} (status)
- GET /calls/{call_id}/summary (summary)
- DELETE /calls/{call_id} (cancel)
- GET /health (health check)
- Error paths: 404, 409, 422, 503
- Pagination and filtering for list endpoint
- Request validation (invalid phone number, short goal, etc.)
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from call_dispatch.models import (
    CallOutcome,
    CallRecord,
    CallStatus,
    CallSummary,
    TranscriptEntry,
)
from call_dispatch.routes import health_router, router as calls_router


# ---------------------------------------------------------------------------
# Test environment setup
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
    """Inject environment variables and clear settings cache."""
    for key, val in _FAKE_ENV.items():
        monkeypatch.setenv(key, val)
    from call_dispatch.config import get_settings
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# App factory helpers
# ---------------------------------------------------------------------------


def make_mock_store(
    records: Optional[list[CallRecord]] = None,
    get_call_result: Optional[CallRecord] = None,
    count_result: int = 0,
    update_status_result: bool = True,
    save_summary_result: bool = True,
) -> MagicMock:
    """Create a mock CallStore with configurable return values."""
    store = MagicMock()
    store.get_call = AsyncMock(return_value=get_call_result)
    store.list_calls = AsyncMock(return_value=records or [])
    store.count_calls = AsyncMock(return_value=count_result)
    store.create_call = AsyncMock(return_value=None)
    store.update_call = AsyncMock(return_value=None)
    store.update_status = AsyncMock(return_value=update_status_result)
    store.save_summary = AsyncMock(return_value=save_summary_result)
    store.append_transcript_entry = AsyncMock(return_value=True)
    return store


def make_mock_dispatcher(
    dispatch_result: Optional[CallRecord] = None,
    cancel_result: bool = True,
) -> MagicMock:
    """Create a mock CallDispatcher with configurable return values."""
    dispatcher = MagicMock()
    dispatcher.dispatch_from_request = AsyncMock(return_value=dispatch_result)
    dispatcher.cancel_call = AsyncMock(return_value=cancel_result)
    dispatcher.active_call_ids = []
    return dispatcher


def make_test_app(
    store: Any = None,
    dispatcher: Any = None,
) -> FastAPI:
    """Create a minimal FastAPI app with routes and injected state."""
    app = FastAPI()
    app.include_router(health_router)
    app.include_router(calls_router)

    if store is not None:
        app.state.store = store
    if dispatcher is not None:
        app.state.dispatcher = dispatcher
    app.state.active_calls = {}

    return app


def make_call_record(
    call_id: Optional[str] = None,
    to_number: str = "+15550001234",
    goal: str = "Book a dentist appointment for Tuesday afternoon",
    status: CallStatus = CallStatus.PENDING,
    twilio_call_sid: Optional[str] = None,
    context: Optional[str] = None,
    transcript: Optional[list[TranscriptEntry]] = None,
    summary: Optional[CallSummary] = None,
    error_message: Optional[str] = None,
) -> CallRecord:
    """Convenience factory for CallRecord test instances."""
    return CallRecord(
        call_id=call_id or str(uuid.uuid4()),
        to_number=to_number,
        from_number="+15559876543",
        goal=goal,
        status=status,
        twilio_call_sid=twilio_call_sid,
        context=context,
        transcript=transcript or [],
        summary=summary,
        error_message=error_message,
    )


def make_summary(
    outcome: CallOutcome = CallOutcome.SUCCESS,
    summary_text: str = "Appointment booked for Tuesday at 3pm.",
    key_details: Optional[dict] = None,
    follow_up_required: bool = False,
) -> CallSummary:
    """Convenience factory for CallSummary test instances."""
    return CallSummary(
        outcome=outcome,
        summary_text=summary_text,
        key_details=key_details or {"booked_time": "Tuesday 3pm"},
        follow_up_required=follow_up_required,
    )


# ---------------------------------------------------------------------------
# Health check tests
# ---------------------------------------------------------------------------


class TestHealthCheck:
    """Tests for GET /health."""

    def test_health_ok_with_all_deps(self) -> None:
        """Health check returns 'ok' when store and dispatcher are available."""
        store = make_mock_store(count_result=5)
        dispatcher = make_mock_dispatcher()
        dispatcher.active_call_ids = ["call-1", "call-2"]
        app = make_test_app(store=store, dispatcher=dispatcher)
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["store"] == "ok"
        assert data["dispatcher"] == "ok"
        assert data["active_calls"] == 2

    def test_health_degraded_without_store(self) -> None:
        """Health check returns 'degraded' when store is not configured."""
        dispatcher = make_mock_dispatcher()
        app = make_test_app(store=None, dispatcher=dispatcher)
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["store"] == "unavailable"

    def test_health_degraded_without_dispatcher(self) -> None:
        """Health check returns 'degraded' when dispatcher is not configured."""
        store = make_mock_store()
        app = make_test_app(store=store, dispatcher=None)
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["dispatcher"] == "unavailable"

    def test_health_includes_version(self) -> None:
        """Health check includes the package version."""
        from call_dispatch import __version__

        store = make_mock_store()
        dispatcher = make_mock_dispatcher()
        app = make_test_app(store=store, dispatcher=dispatcher)
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.json()["version"] == __version__

    def test_health_includes_total_calls(self) -> None:
        """Health check includes the total number of call records."""
        store = make_mock_store(count_result=42)
        dispatcher = make_mock_dispatcher()
        app = make_test_app(store=store, dispatcher=dispatcher)
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.json()["total_calls"] == 42


# ---------------------------------------------------------------------------
# POST /calls — dispatch tests
# ---------------------------------------------------------------------------


class TestDispatchCall:
    """Tests for POST /calls."""

    def test_dispatch_returns_202(self) -> None:
        """Successful dispatch returns HTTP 202 Accepted."""
        record = make_call_record(status=CallStatus.PENDING)
        store = make_mock_store()
        dispatcher = make_mock_dispatcher(dispatch_result=record)
        app = make_test_app(store=store, dispatcher=dispatcher)
        client = TestClient(app)

        resp = client.post(
            "/calls",
            json={
                "to_number": "+15550001234",
                "goal": "Book a dentist appointment for Tuesday afternoon.",
            },
        )
        assert resp.status_code == 202

    def test_dispatch_returns_call_id(self) -> None:
        """Dispatch response includes the call_id."""
        record = make_call_record(call_id="test-call-uuid", status=CallStatus.PENDING)
        store = make_mock_store()
        dispatcher = make_mock_dispatcher(dispatch_result=record)
        app = make_test_app(store=store, dispatcher=dispatcher)
        client = TestClient(app)

        resp = client.post(
            "/calls",
            json={
                "to_number": "+15550001234",
                "goal": "Book a dentist appointment for Tuesday afternoon.",
            },
        )
        data = resp.json()
        assert data["call_id"] == "test-call-uuid"

    def test_dispatch_returns_pending_status(self) -> None:
        """Dispatch response reports initial 'pending' status."""
        record = make_call_record(status=CallStatus.PENDING)
        store = make_mock_store()
        dispatcher = make_mock_dispatcher(dispatch_result=record)
        app = make_test_app(store=store, dispatcher=dispatcher)
        client = TestClient(app)

        resp = client.post(
            "/calls",
            json={
                "to_number": "+15550001234",
                "goal": "Check if the pharmacy is open on Sunday morning.",
            },
        )
        data = resp.json()
        assert data["status"] == "pending"

    def test_dispatch_returns_created_at(self) -> None:
        """Dispatch response includes a created_at timestamp."""
        record = make_call_record(status=CallStatus.PENDING)
        store = make_mock_store()
        dispatcher = make_mock_dispatcher(dispatch_result=record)
        app = make_test_app(store=store, dispatcher=dispatcher)
        client = TestClient(app)

        resp = client.post(
            "/calls",
            json={
                "to_number": "+15550001234",
                "goal": "Book a dentist appointment for Tuesday afternoon.",
            },
        )
        assert "created_at" in resp.json()

    def test_dispatch_with_context(self) -> None:
        """Dispatch correctly forwards context to the dispatcher."""
        record = make_call_record(
            status=CallStatus.PENDING, context="Patient: Jane Doe"
        )
        store = make_mock_store()
        dispatcher = make_mock_dispatcher(dispatch_result=record)
        app = make_test_app(store=store, dispatcher=dispatcher)
        client = TestClient(app)

        resp = client.post(
            "/calls",
            json={
                "to_number": "+15550001234",
                "goal": "Book a dentist appointment for Tuesday afternoon.",
                "context": "Patient: Jane Doe",
            },
        )
        assert resp.status_code == 202
        # Verify the dispatcher was called
        dispatcher.dispatch_from_request.assert_called_once()

    def test_dispatch_with_max_duration(self) -> None:
        """Dispatch accepts and forwards max_duration_seconds."""
        record = make_call_record(status=CallStatus.PENDING)
        store = make_mock_store()
        dispatcher = make_mock_dispatcher(dispatch_result=record)
        app = make_test_app(store=store, dispatcher=dispatcher)
        client = TestClient(app)

        resp = client.post(
            "/calls",
            json={
                "to_number": "+15550001234",
                "goal": "Book a dentist appointment for Tuesday afternoon.",
                "max_duration_seconds": 180,
            },
        )
        assert resp.status_code == 202

    def test_dispatch_with_metadata(self) -> None:
        """Dispatch accepts and forwards metadata."""
        record = make_call_record(status=CallStatus.PENDING)
        store = make_mock_store()
        dispatcher = make_mock_dispatcher(dispatch_result=record)
        app = make_test_app(store=store, dispatcher=dispatcher)
        client = TestClient(app)

        resp = client.post(
            "/calls",
            json={
                "to_number": "+15550001234",
                "goal": "Book a dentist appointment for Tuesday afternoon.",
                "metadata": {"source": "test", "priority": 1},
            },
        )
        assert resp.status_code == 202

    def test_dispatch_invalid_phone_number_returns_422(self) -> None:
        """Invalid phone number (missing +) returns HTTP 422."""
        store = make_mock_store()
        dispatcher = make_mock_dispatcher()
        app = make_test_app(store=store, dispatcher=dispatcher)
        client = TestClient(app)

        resp = client.post(
            "/calls",
            json={
                "to_number": "15550001234",  # Missing +
                "goal": "Book a dentist appointment for Tuesday afternoon.",
            },
        )
        assert resp.status_code == 422

    def test_dispatch_short_goal_returns_422(self) -> None:
        """A goal that is too short returns HTTP 422."""
        store = make_mock_store()
        dispatcher = make_mock_dispatcher()
        app = make_test_app(store=store, dispatcher=dispatcher)
        client = TestClient(app)

        resp = client.post(
            "/calls",
            json={
                "to_number": "+15550001234",
                "goal": "Hi",  # Too short (min_length=10)
            },
        )
        assert resp.status_code == 422

    def test_dispatch_missing_to_number_returns_422(self) -> None:
        """Missing to_number returns HTTP 422."""
        store = make_mock_store()
        dispatcher = make_mock_dispatcher()
        app = make_test_app(store=store, dispatcher=dispatcher)
        client = TestClient(app)

        resp = client.post(
            "/calls",
            json={"goal": "Book a dentist appointment for Tuesday afternoon."},
        )
        assert resp.status_code == 422

    def test_dispatch_missing_goal_returns_422(self) -> None:
        """Missing goal returns HTTP 422."""
        store = make_mock_store()
        dispatcher = make_mock_dispatcher()
        app = make_test_app(store=store, dispatcher=dispatcher)
        client = TestClient(app)

        resp = client.post(
            "/calls",
            json={"to_number": "+15550001234"},
        )
        assert resp.status_code == 422

    def test_dispatch_without_store_returns_503(self) -> None:
        """Dispatch without store configured returns HTTP 503."""
        dispatcher = make_mock_dispatcher()
        app = make_test_app(store=None, dispatcher=dispatcher)
        client = TestClient(app)

        resp = client.post(
            "/calls",
            json={
                "to_number": "+15550001234",
                "goal": "Book a dentist appointment for Tuesday afternoon.",
            },
        )
        assert resp.status_code == 503

    def test_dispatch_without_dispatcher_returns_503(self) -> None:
        """Dispatch without dispatcher configured returns HTTP 503."""
        store = make_mock_store()
        app = make_test_app(store=store, dispatcher=None)
        client = TestClient(app)

        resp = client.post(
            "/calls",
            json={
                "to_number": "+15550001234",
                "goal": "Book a dentist appointment for Tuesday afternoon.",
            },
        )
        assert resp.status_code == 503

    def test_dispatch_max_duration_too_short_returns_422(self) -> None:
        """max_duration_seconds below 30 returns HTTP 422."""
        store = make_mock_store()
        dispatcher = make_mock_dispatcher()
        app = make_test_app(store=store, dispatcher=dispatcher)
        client = TestClient(app)

        resp = client.post(
            "/calls",
            json={
                "to_number": "+15550001234",
                "goal": "Book a dentist appointment for Tuesday afternoon.",
                "max_duration_seconds": 10,
            },
        )
        assert resp.status_code == 422

    def test_dispatch_max_duration_too_long_returns_422(self) -> None:
        """max_duration_seconds above 3600 returns HTTP 422."""
        store = make_mock_store()
        dispatcher = make_mock_dispatcher()
        app = make_test_app(store=store, dispatcher=dispatcher)
        client = TestClient(app)

        resp = client.post(
            "/calls",
            json={
                "to_number": "+15550001234",
                "goal": "Book a dentist appointment for Tuesday afternoon.",
                "max_duration_seconds": 9999,
            },
        )
        assert resp.status_code == 422

    def test_dispatch_dispatcher_error_returns_500(self) -> None:
        """Unexpected dispatcher error returns HTTP 500."""
        store = make_mock_store()
        dispatcher = make_mock_dispatcher()
        dispatcher.dispatch_from_request = AsyncMock(
            side_effect=RuntimeError("Unexpected error")
        )
        app = make_test_app(store=store, dispatcher=dispatcher)
        client = TestClient(app)

        resp = client.post(
            "/calls",
            json={
                "to_number": "+15550001234",
                "goal": "Book a dentist appointment for Tuesday afternoon.",
            },
        )
        assert resp.status_code == 500

    def test_dispatch_calls_dispatcher_with_correct_data(self) -> None:
        """Dispatch calls dispatcher.dispatch_from_request with the request data."""
        record = make_call_record(status=CallStatus.PENDING)
        store = make_mock_store()
        dispatcher = make_mock_dispatcher(dispatch_result=record)
        app = make_test_app(store=store, dispatcher=dispatcher)
        client = TestClient(app)

        client.post(
            "/calls",
            json={
                "to_number": "+15550001234",
                "goal": "Book a dentist appointment for Tuesday afternoon.",
                "context": "Patient: Jane Doe",
            },
        )

        dispatcher.dispatch_from_request.assert_called_once()
        call_arg = dispatcher.dispatch_from_request.call_args[0][0]
        assert call_arg.to_number == "+15550001234"
        assert call_arg.goal == "Book a dentist appointment for Tuesday afternoon."
        assert call_arg.context == "Patient: Jane Doe"

    def test_dispatch_returns_message_field(self) -> None:
        """Dispatch response includes a human-readable message."""
        record = make_call_record(status=CallStatus.PENDING)
        store = make_mock_store()
        dispatcher = make_mock_dispatcher(dispatch_result=record)
        app = make_test_app(store=store, dispatcher=dispatcher)
        client = TestClient(app)

        resp = client.post(
            "/calls",
            json={
                "to_number": "+15550001234",
                "goal": "Book a dentist appointment for Tuesday afternoon.",
            },
        )
        data = resp.json()
        assert "message" in data
        assert len(data["message"]) > 0


# ---------------------------------------------------------------------------
# GET /calls — list tests
# ---------------------------------------------------------------------------


class TestListCalls:
    """Tests for GET /calls."""

    def test_list_returns_200(self) -> None:
        """List endpoint returns HTTP 200."""
        store = make_mock_store(records=[], count_result=0)
        app = make_test_app(store=store)
        client = TestClient(app)
        resp = client.get("/calls")
        assert resp.status_code == 200

    def test_list_returns_empty_when_no_calls(self) -> None:
        """List endpoint returns empty list when no calls exist."""
        store = make_mock_store(records=[], count_result=0)
        app = make_test_app(store=store)
        client = TestClient(app)
        resp = client.get("/calls")
        data = resp.json()
        assert data["total"] == 0
        assert data["calls"] == []

    def test_list_returns_call_records(self) -> None:
        """List endpoint returns call records in the response."""
        records = [
            make_call_record(call_id=f"call-{i}", status=CallStatus.COMPLETED)
            for i in range(3)
        ]
        store = make_mock_store(records=records, count_result=3)
        app = make_test_app(store=store)
        client = TestClient(app)
        resp = client.get("/calls")
        data = resp.json()
        assert data["total"] == 3
        assert len(data["calls"]) == 3

    def test_list_call_fields_present(self) -> None:
        """Each call in the list includes required fields."""
        record = make_call_record(
            call_id="test-call-id",
            status=CallStatus.COMPLETED,
            to_number="+15550001234",
        )
        store = make_mock_store(records=[record], count_result=1)
        app = make_test_app(store=store)
        client = TestClient(app)
        resp = client.get("/calls")
        call_data = resp.json()["calls"][0]
        assert "call_id" in call_data
        assert "to_number" in call_data
        assert "status" in call_data
        assert "goal" in call_data
        assert "created_at" in call_data

    def test_list_with_status_filter(self) -> None:
        """List endpoint accepts and passes along status filter."""
        records = [make_call_record(status=CallStatus.COMPLETED)]
        store = make_mock_store(records=records, count_result=1)
        app = make_test_app(store=store)
        client = TestClient(app)
        resp = client.get("/calls?status=completed")
        assert resp.status_code == 200
        # Verify the store was called with the correct filter
        store.list_calls.assert_called_once()
        call_kwargs = store.list_calls.call_args[1]
        assert call_kwargs["status"] == CallStatus.COMPLETED

    def test_list_invalid_status_filter_returns_400(self) -> None:
        """Invalid status filter returns HTTP 400."""
        store = make_mock_store()
        app = make_test_app(store=store)
        client = TestClient(app)
        resp = client.get("/calls?status=invalid_status")
        assert resp.status_code == 400

    def test_list_with_limit_parameter(self) -> None:
        """List endpoint accepts and passes along limit parameter."""
        store = make_mock_store(records=[], count_result=0)
        app = make_test_app(store=store)
        client = TestClient(app)
        resp = client.get("/calls?limit=10")
        assert resp.status_code == 200
        store.list_calls.assert_called_once()
        call_kwargs = store.list_calls.call_args[1]
        assert call_kwargs["limit"] == 10

    def test_list_with_offset_parameter(self) -> None:
        """List endpoint accepts and passes along offset parameter."""
        store = make_mock_store(records=[], count_result=0)
        app = make_test_app(store=store)
        client = TestClient(app)
        resp = client.get("/calls?offset=20")
        assert resp.status_code == 200
        store.list_calls.assert_called_once()
        call_kwargs = store.list_calls.call_args[1]
        assert call_kwargs["offset"] == 20

    def test_list_limit_too_large_returns_422(self) -> None:
        """limit > 500 returns HTTP 422."""
        store = make_mock_store()
        app = make_test_app(store=store)
        client = TestClient(app)
        resp = client.get("/calls?limit=1000")
        assert resp.status_code == 422

    def test_list_negative_limit_returns_422(self) -> None:
        """limit < 1 returns HTTP 422."""
        store = make_mock_store()
        app = make_test_app(store=store)
        client = TestClient(app)
        resp = client.get("/calls?limit=0")
        assert resp.status_code == 422

    def test_list_negative_offset_returns_422(self) -> None:
        """Negative offset returns HTTP 422."""
        store = make_mock_store()
        app = make_test_app(store=store)
        client = TestClient(app)
        resp = client.get("/calls?offset=-1")
        assert resp.status_code == 422

    def test_list_without_store_returns_503(self) -> None:
        """List without store configured returns HTTP 503."""
        app = make_test_app(store=None)
        client = TestClient(app)
        resp = client.get("/calls")
        assert resp.status_code == 503

    def test_list_all_status_values_accepted(self) -> None:
        """All valid status values are accepted by the filter."""
        valid_statuses = [
            "pending", "initiating", "in_progress", "completed",
            "failed", "cancelled", "no_answer", "busy",
        ]
        store = make_mock_store(records=[], count_result=0)
        app = make_test_app(store=store)
        client = TestClient(app)
        for s in valid_statuses:
            resp = client.get(f"/calls?status={s}")
            assert resp.status_code == 200, f"Status {s!r} should be accepted"

    def test_list_no_status_filter_passes_none(self) -> None:
        """Without a status filter, store.list_calls is called with status=None."""
        store = make_mock_store(records=[], count_result=0)
        app = make_test_app(store=store)
        client = TestClient(app)
        client.get("/calls")
        call_kwargs = store.list_calls.call_args[1]
        assert call_kwargs["status"] is None


# ---------------------------------------------------------------------------
# GET /calls/{call_id} — status tests
# ---------------------------------------------------------------------------


class TestGetCallStatus:
    """Tests for GET /calls/{call_id}."""

    def test_get_existing_call_returns_200(self) -> None:
        """Getting an existing call returns HTTP 200."""
        record = make_call_record(call_id="existing-call")
        store = make_mock_store(get_call_result=record)
        app = make_test_app(store=store)
        client = TestClient(app)
        resp = client.get("/calls/existing-call")
        assert resp.status_code == 200

    def test_get_nonexistent_call_returns_404(self) -> None:
        """Getting a non-existent call returns HTTP 404."""
        store = make_mock_store(get_call_result=None)
        app = make_test_app(store=store)
        client = TestClient(app)
        resp = client.get("/calls/nonexistent-call-id")
        assert resp.status_code == 404

    def test_get_call_returns_correct_fields(self) -> None:
        """Call status response includes all expected fields."""
        record = make_call_record(
            call_id="test-call",
            to_number="+15550001234",
            goal="Book a dentist appointment for Tuesday afternoon.",
            status=CallStatus.IN_PROGRESS,
            twilio_call_sid="CA_test_sid",
        )
        store = make_mock_store(get_call_result=record)
        app = make_test_app(store=store)
        client = TestClient(app)
        resp = client.get("/calls/test-call")
        data = resp.json()
        assert data["call_id"] == "test-call"
        assert data["to_number"] == "+15550001234"
        assert data["status"] == "in_progress"
        assert data["goal"] == "Book a dentist appointment for Tuesday afternoon."
        assert data["twilio_call_sid"] == "CA_test_sid"

    def test_get_call_includes_transcript(self) -> None:
        """Call status response includes the transcript."""
        transcript = [
            TranscriptEntry(speaker="agent", text="Hello there."),
            TranscriptEntry(speaker="contact", text="Hi, how can I help?"),
        ]
        record = make_call_record(
            call_id="transcript-call",
            transcript=transcript,
            status=CallStatus.IN_PROGRESS,
        )
        store = make_mock_store(get_call_result=record)
        app = make_test_app(store=store)
        client = TestClient(app)
        resp = client.get("/calls/transcript-call")
        data = resp.json()
        assert len(data["transcript"]) == 2
        assert data["transcript"][0]["speaker"] == "agent"
        assert data["transcript"][0]["text"] == "Hello there."

    def test_get_call_includes_summary_when_completed(self) -> None:
        """Call status response includes summary when available."""
        summary = make_summary()
        record = make_call_record(
            call_id="completed-call",
            status=CallStatus.COMPLETED,
            summary=summary,
        )
        store = make_mock_store(get_call_result=record)
        app = make_test_app(store=store)
        client = TestClient(app)
        resp = client.get("/calls/completed-call")
        data = resp.json()
        assert data["summary"] is not None
        assert data["summary"]["outcome"] == "success"

    def test_get_call_no_summary_is_null(self) -> None:
        """Call status response has null summary when not yet generated."""
        record = make_call_record(
            call_id="pending-call",
            status=CallStatus.PENDING,
        )
        store = make_mock_store(get_call_result=record)
        app = make_test_app(store=store)
        client = TestClient(app)
        resp = client.get("/calls/pending-call")
        data = resp.json()
        assert data["summary"] is None

    def test_get_call_includes_error_message_when_failed(self) -> None:
        """Failed call includes error_message in the response."""
        record = make_call_record(
            call_id="failed-call",
            status=CallStatus.FAILED,
            error_message="Twilio error: invalid number",
        )
        store = make_mock_store(get_call_result=record)
        app = make_test_app(store=store)
        client = TestClient(app)
        resp = client.get("/calls/failed-call")
        data = resp.json()
        assert data["error_message"] == "Twilio error: invalid number"

    def test_get_call_without_store_returns_503(self) -> None:
        """Get call without store configured returns HTTP 503."""
        app = make_test_app(store=None)
        client = TestClient(app)
        resp = client.get("/calls/any-call-id")
        assert resp.status_code == 503

    def test_get_call_store_called_with_correct_id(self) -> None:
        """store.get_call is called with the correct call_id."""
        record = make_call_record(call_id="specific-call-id")
        store = make_mock_store(get_call_result=record)
        app = make_test_app(store=store)
        client = TestClient(app)
        client.get("/calls/specific-call-id")
        store.get_call.assert_called_once_with("specific-call-id")

    def test_get_call_404_detail_includes_call_id(self) -> None:
        """404 response includes the call_id in the detail message."""
        store = make_mock_store(get_call_result=None)
        app = make_test_app(store=store)
        client = TestClient(app)
        resp = client.get("/calls/missing-call-xyz")
        assert "missing-call-xyz" in resp.json()["detail"]

    def test_get_call_includes_timestamps(self) -> None:
        """Call status response includes created_at and updated_at."""
        record = make_call_record(call_id="ts-call", status=CallStatus.PENDING)
        store = make_mock_store(get_call_result=record)
        app = make_test_app(store=store)
        client = TestClient(app)
        resp = client.get("/calls/ts-call")
        data = resp.json()
        assert "created_at" in data
        assert "updated_at" in data


# ---------------------------------------------------------------------------
# GET /calls/{call_id}/summary — summary tests
# ---------------------------------------------------------------------------


class TestGetCallSummary:
    """Tests for GET /calls/{call_id}/summary."""

    def test_get_summary_returns_200(self) -> None:
        """Summary endpoint returns HTTP 200 for an existing call."""
        record = make_call_record(
            call_id="summary-call",
            status=CallStatus.COMPLETED,
            summary=make_summary(),
        )
        store = make_mock_store(get_call_result=record)
        app = make_test_app(store=store)
        client = TestClient(app)
        resp = client.get("/calls/summary-call/summary")
        assert resp.status_code == 200

    def test_get_summary_nonexistent_call_returns_404(self) -> None:
        """Summary endpoint returns 404 for a non-existent call."""
        store = make_mock_store(get_call_result=None)
        app = make_test_app(store=store)
        client = TestClient(app)
        resp = client.get("/calls/nonexistent/summary")
        assert resp.status_code == 404

    def test_get_summary_includes_summary_data(self) -> None:
        """Summary endpoint includes the structured summary in response."""
        summary = make_summary(
            outcome=CallOutcome.SUCCESS,
            summary_text="Appointment booked for Tuesday at 3pm.",
            key_details={"booked_time": "Tuesday 3pm", "confirmation": "ABC123"},
        )
        record = make_call_record(
            call_id="completed-call",
            status=CallStatus.COMPLETED,
            summary=summary,
        )
        store = make_mock_store(get_call_result=record)
        app = make_test_app(store=store)
        client = TestClient(app)
        resp = client.get("/calls/completed-call/summary")
        data = resp.json()
        assert data["summary"]["outcome"] == "success"
        assert data["summary"]["summary_text"] == "Appointment booked for Tuesday at 3pm."
        assert data["summary"]["key_details"]["confirmation"] == "ABC123"

    def test_get_summary_no_summary_is_null(self) -> None:
        """Summary is null for calls that haven't completed yet."""
        record = make_call_record(
            call_id="in-progress-call",
            status=CallStatus.IN_PROGRESS,
        )
        store = make_mock_store(get_call_result=record)
        app = make_test_app(store=store)
        client = TestClient(app)
        resp = client.get("/calls/in-progress-call/summary")
        data = resp.json()
        assert data["summary"] is None

    def test_get_summary_includes_transcript(self) -> None:
        """Summary response includes the full call transcript."""
        transcript = [
            TranscriptEntry(speaker="agent", text="Hello."),
            TranscriptEntry(speaker="contact", text="Hi there."),
            TranscriptEntry(speaker="agent", text="I'd like to book an appointment."),
        ]
        record = make_call_record(
            call_id="transcript-summary-call",
            status=CallStatus.COMPLETED,
            transcript=transcript,
            summary=make_summary(),
        )
        store = make_mock_store(get_call_result=record)
        app = make_test_app(store=store)
        client = TestClient(app)
        resp = client.get("/calls/transcript-summary-call/summary")
        data = resp.json()
        assert len(data["transcript"]) == 3

    def test_get_summary_includes_status(self) -> None:
        """Summary response includes the call status."""
        record = make_call_record(
            call_id="status-call",
            status=CallStatus.COMPLETED,
            summary=make_summary(),
        )
        store = make_mock_store(get_call_result=record)
        app = make_test_app(store=store)
        client = TestClient(app)
        resp = client.get("/calls/status-call/summary")
        data = resp.json()
        assert data["status"] == "completed"

    def test_get_summary_includes_call_id(self) -> None:
        """Summary response includes the call_id."""
        record = make_call_record(call_id="my-call", status=CallStatus.COMPLETED)
        store = make_mock_store(get_call_result=record)
        app = make_test_app(store=store)
        client = TestClient(app)
        resp = client.get("/calls/my-call/summary")
        data = resp.json()
        assert data["call_id"] == "my-call"

    def test_get_summary_without_store_returns_503(self) -> None:
        """Summary endpoint without store configured returns HTTP 503."""
        app = make_test_app(store=None)
        client = TestClient(app)
        resp = client.get("/calls/any-call/summary")
        assert resp.status_code == 503

    def test_get_summary_partial_outcome(self) -> None:
        """Summary with partial outcome is correctly serialised."""
        summary = make_summary(
            outcome=CallOutcome.PARTIAL,
            summary_text="Partially achieved the goal.",
        )
        record = make_call_record(
            call_id="partial-call",
            status=CallStatus.COMPLETED,
            summary=summary,
        )
        store = make_mock_store(get_call_result=record)
        app = make_test_app(store=store)
        client = TestClient(app)
        resp = client.get("/calls/partial-call/summary")
        data = resp.json()
        assert data["summary"]["outcome"] == "partial"

    def test_get_summary_with_follow_up(self) -> None:
        """Summary with follow_up_required is correctly serialised."""
        summary = CallSummary(
            outcome=CallOutcome.PARTIAL,
            summary_text="Needs follow up.",
            follow_up_required=True,
            follow_up_notes="Call back tomorrow.",
        )
        record = make_call_record(
            call_id="followup-call",
            status=CallStatus.COMPLETED,
            summary=summary,
        )
        store = make_mock_store(get_call_result=record)
        app = make_test_app(store=store)
        client = TestClient(app)
        resp = client.get("/calls/followup-call/summary")
        data = resp.json()
        assert data["summary"]["follow_up_required"] is True
        assert data["summary"]["follow_up_notes"] == "Call back tomorrow."


# ---------------------------------------------------------------------------
# DELETE /calls/{call_id} — cancel tests
# ---------------------------------------------------------------------------


class TestCancelCall:
    """Tests for DELETE /calls/{call_id}."""

    def test_cancel_pending_call_returns_200(self) -> None:
        """Cancelling a pending call returns HTTP 200."""
        record = make_call_record(
            call_id="pending-cancel",
            status=CallStatus.PENDING,
        )
        store = make_mock_store(get_call_result=record)
        dispatcher = make_mock_dispatcher(cancel_result=True)
        app = make_test_app(store=store, dispatcher=dispatcher)
        client = TestClient(app)
        resp = client.delete("/calls/pending-cancel")
        assert resp.status_code == 200

    def test_cancel_returns_confirmation_message(self) -> None:
        """Cancel response includes a confirmation message."""
        record = make_call_record(
            call_id="to-cancel",
            status=CallStatus.PENDING,
        )
        store = make_mock_store(get_call_result=record)
        dispatcher = make_mock_dispatcher(cancel_result=True)
        app = make_test_app(store=store, dispatcher=dispatcher)
        client = TestClient(app)
        resp = client.delete("/calls/to-cancel")
        data = resp.json()
        assert "message" in data
        assert "to-cancel" in data["call_id"]

    def test_cancel_nonexistent_call_returns_404(self) -> None:
        """Cancelling a non-existent call returns HTTP 404."""
        store = make_mock_store(get_call_result=None)
        dispatcher = make_mock_dispatcher(cancel_result=False)
        app = make_test_app(store=store, dispatcher=dispatcher)
        client = TestClient(app)
        resp = client.delete("/calls/nonexistent")
        assert resp.status_code == 404

    def test_cancel_in_progress_call_returns_409(self) -> None:
        """Cancelling an in-progress call returns HTTP 409 Conflict."""
        record = make_call_record(
            call_id="active-call",
            status=CallStatus.IN_PROGRESS,
        )
        store = make_mock_store(get_call_result=record)
        dispatcher = make_mock_dispatcher(cancel_result=False)
        app = make_test_app(store=store, dispatcher=dispatcher)
        client = TestClient(app)
        resp = client.delete("/calls/active-call")
        assert resp.status_code == 409

    def test_cancel_completed_call_returns_409(self) -> None:
        """Cancelling a completed call returns HTTP 409 Conflict."""
        record = make_call_record(
            call_id="done-call",
            status=CallStatus.COMPLETED,
        )
        store = make_mock_store(get_call_result=record)
        dispatcher = make_mock_dispatcher(cancel_result=False)
        app = make_test_app(store=store, dispatcher=dispatcher)
        client = TestClient(app)
        resp = client.delete("/calls/done-call")
        assert resp.status_code == 409

    def test_cancel_calls_dispatcher_with_call_id(self) -> None:
        """Cancel calls dispatcher.cancel_call with the correct call_id."""
        record = make_call_record(
            call_id="cancel-me",
            status=CallStatus.PENDING,
        )
        store = make_mock_store(get_call_result=record)
        dispatcher = make_mock_dispatcher(cancel_result=True)
        app = make_test_app(store=store, dispatcher=dispatcher)
        client = TestClient(app)
        client.delete("/calls/cancel-me")
        dispatcher.cancel_call.assert_called_once_with("cancel-me")

    def test_cancel_without_store_returns_503(self) -> None:
        """Cancel without store configured returns HTTP 503."""
        dispatcher = make_mock_dispatcher()
        app = make_test_app(store=None, dispatcher=dispatcher)
        client = TestClient(app)
        resp = client.delete("/calls/any-call")
        assert resp.status_code == 503

    def test_cancel_without_dispatcher_returns_503(self) -> None:
        """Cancel without dispatcher configured returns HTTP 503."""
        record = make_call_record(call_id="no-dispatcher", status=CallStatus.PENDING)
        store = make_mock_store(get_call_result=record)
        app = make_test_app(store=store, dispatcher=None)
        client = TestClient(app)
        resp = client.delete("/calls/no-dispatcher")
        assert resp.status_code == 503

    def test_cancel_409_includes_current_status(self) -> None:
        """409 response detail mentions the current call status."""
        record = make_call_record(
            call_id="in-prog",
            status=CallStatus.IN_PROGRESS,
        )
        store = make_mock_store(get_call_result=record)
        dispatcher = make_mock_dispatcher(cancel_result=False)
        app = make_test_app(store=store, dispatcher=dispatcher)
        client = TestClient(app)
        resp = client.delete("/calls/in-prog")
        detail = resp.json()["detail"]
        assert "in_progress" in detail


# ---------------------------------------------------------------------------
# Response schema conformance tests
# ---------------------------------------------------------------------------


class TestResponseSchemas:
    """Tests that API responses conform to their declared schemas."""

    def test_dispatch_response_schema(self) -> None:
        """DispatchCallResponse includes all required schema fields."""
        record = make_call_record(call_id="schema-test", status=CallStatus.PENDING)
        store = make_mock_store()
        dispatcher = make_mock_dispatcher(dispatch_result=record)
        app = make_test_app(store=store, dispatcher=dispatcher)
        client = TestClient(app)

        resp = client.post(
            "/calls",
            json={
                "to_number": "+15550001234",
                "goal": "Book a dentist appointment for Tuesday afternoon.",
            },
        )
        data = resp.json()
        required_fields = {"call_id", "status", "message", "created_at"}
        assert required_fields.issubset(data.keys())

    def test_list_response_schema(self) -> None:
        """ListCallsResponse includes all required schema fields."""
        store = make_mock_store(records=[], count_result=0)
        app = make_test_app(store=store)
        client = TestClient(app)
        resp = client.get("/calls")
        data = resp.json()
        assert "total" in data
        assert "calls" in data
        assert isinstance(data["calls"], list)

    def test_status_response_schema(self) -> None:
        """CallStatusResponse includes all required schema fields."""
        record = make_call_record(call_id="schema-status")
        store = make_mock_store(get_call_result=record)
        app = make_test_app(store=store)
        client = TestClient(app)
        resp = client.get("/calls/schema-status")
        data = resp.json()
        required_fields = {
            "call_id", "to_number", "goal", "status",
            "transcript", "created_at", "updated_at",
        }
        assert required_fields.issubset(data.keys())

    def test_summary_response_schema(self) -> None:
        """CallSummaryResponse includes all required schema fields."""
        record = make_call_record(
            call_id="schema-summary",
            status=CallStatus.COMPLETED,
            summary=make_summary(),
        )
        store = make_mock_store(get_call_result=record)
        app = make_test_app(store=store)
        client = TestClient(app)
        resp = client.get("/calls/schema-summary/summary")
        data = resp.json()
        required_fields = {"call_id", "status", "summary", "transcript"}
        assert required_fields.issubset(data.keys())

    def test_health_response_schema(self) -> None:
        """Health response includes all expected fields."""
        store = make_mock_store()
        dispatcher = make_mock_dispatcher()
        app = make_test_app(store=store, dispatcher=dispatcher)
        client = TestClient(app)
        resp = client.get("/health")
        data = resp.json()
        required_fields = {
            "status", "version", "store", "dispatcher",
            "active_calls", "total_calls",
        }
        assert required_fields.issubset(data.keys())


# ---------------------------------------------------------------------------
# Root endpoint test
# ---------------------------------------------------------------------------


class TestRootEndpoint:
    """Tests for GET /."""

    def test_root_returns_200(self) -> None:
        """Root endpoint returns HTTP 200."""
        from call_dispatch.main import create_app

        # Use the full app with all routes
        full_app = create_app()

        # Override lifespan to avoid real startup
        from contextlib import asynccontextmanager
        from typing import AsyncIterator

        @asynccontextmanager
        async def mock_lifespan(app):
            app.state.store = make_mock_store()
            app.state.dispatcher = make_mock_dispatcher()
            app.state.active_calls = {}
            yield

        full_app.router.lifespan_context = mock_lifespan

        client = TestClient(full_app)
        resp = client.get("/")
        assert resp.status_code == 200

    def test_root_includes_docs_link(self) -> None:
        """Root response includes a link to /docs."""
        from call_dispatch.main import create_app
        from contextlib import asynccontextmanager

        full_app = create_app()

        @asynccontextmanager
        async def mock_lifespan(app):
            app.state.store = make_mock_store()
            app.state.dispatcher = make_mock_dispatcher()
            app.state.active_calls = {}
            yield

        full_app.router.lifespan_context = mock_lifespan

        client = TestClient(full_app)
        resp = client.get("/")
        data = resp.json()
        assert "/docs" in str(data)


# ---------------------------------------------------------------------------
# Integration: using a real in-memory store
# ---------------------------------------------------------------------------


class TestWithRealStore:
    """Integration tests that use a real in-memory CallStore."""

    @pytest.fixture
    async def real_store(self):
        """Provide a real in-memory CallStore."""
        from call_dispatch.store import CallStore
        s = CallStore(":memory:")
        await s.initialize()
        yield s
        await s.close()

    @pytest.mark.asyncio
    async def test_dispatch_and_poll_status(self, real_store) -> None:
        """Dispatch a call then poll its status using a real store."""
        # Create a call record directly in the store
        record = make_call_record(
            call_id="real-store-call",
            status=CallStatus.PENDING,
        )
        await real_store.create_call(record)

        dispatcher = make_mock_dispatcher(dispatch_result=record)
        app = make_test_app(store=real_store, dispatcher=dispatcher)
        client = TestClient(app)

        # Poll status
        resp = client.get("/calls/real-store-call")
        assert resp.status_code == 200
        data = resp.json()
        assert data["call_id"] == "real-store-call"
        assert data["status"] == "pending"

    @pytest.mark.asyncio
    async def test_list_with_real_store_returns_records(self, real_store) -> None:
        """List endpoint returns records from a real store."""
        for i in range(5):
            record = make_call_record(
                call_id=f"real-call-{i}",
                status=CallStatus.COMPLETED,
            )
            await real_store.create_call(record)

        app = make_test_app(store=real_store)
        client = TestClient(app)
        resp = client.get("/calls")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 5
        assert len(data["calls"]) == 5

    @pytest.mark.asyncio
    async def test_get_summary_from_real_store(self, real_store) -> None:
        """Summary endpoint retrieves from a real store."""
        summary = make_summary()
        record = make_call_record(
            call_id="real-summary-call",
            status=CallStatus.COMPLETED,
            summary=summary,
            transcript=[
                TranscriptEntry(speaker="agent", text="Hello."),
                TranscriptEntry(speaker="contact", text="Hi."),
            ],
        )
        await real_store.create_call(record)

        app = make_test_app(store=real_store)
        client = TestClient(app)
        resp = client.get("/calls/real-summary-call/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["summary"]["outcome"] == "success"
        assert len(data["transcript"]) == 2
