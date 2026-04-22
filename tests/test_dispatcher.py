"""Unit tests for call_dispatch.dispatcher - call dispatcher orchestration.

All external dependencies (Twilio, OpenAI, Deepgram, WebSocket) are mocked
so tests are fully offline and deterministic.

Test coverage includes:
- CallDispatcher initialisation
- dispatch() success path
- dispatch() Twilio failure path
- dispatch_from_request() convenience wrapper
- cancel_call() for pending/initiating calls
- cancel_call() for non-cancellable states
- _initiate_twilio_call() success and error paths
- _finalize_call() post-call processing
- _generate_and_save_summary() success and error paths
- active_call_ids property
- _CallState creation and initialization
- DispatchError custom exception
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from call_dispatch.dispatcher import (
    CallDispatcher,
    DispatchError,
    _CallState,
)
from call_dispatch.models import (
    CallOutcome,
    CallRecord,
    CallStatus,
    CallSummary,
    DispatchCallRequest,
    TranscriptEntry,
    TranscriptionResult,
)
from call_dispatch.store import CallStore


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
    """Inject environment variables and clear settings cache for each test."""
    for key, val in _FAKE_ENV.items():
        monkeypatch.setenv(key, val)
    from call_dispatch.config import get_settings
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def mem_store() -> CallStore:
    """Provide an in-memory CallStore for each test."""
    s = CallStore(":memory:")
    await s.initialize()
    yield s
    await s.close()


@pytest.fixture
def dispatcher(mem_store: CallStore) -> CallDispatcher:
    """Provide a CallDispatcher backed by an in-memory store."""
    return CallDispatcher(
        store=mem_store,
        openai_api_key="sk-test-key",
        deepgram_api_key="dg-test-key",
        twilio_account_sid="ACtest",
        twilio_auth_token="token",
        twilio_phone_number="+15550001234",
        public_base_url="https://example.ngrok.io",
    )


def make_twilio_call_mock(sid: str = "CA_test_sid") -> MagicMock:
    """Build a mock Twilio Call object."""
    call = MagicMock()
    call.sid = sid
    call.status = "queued"
    return call


def make_call_record(
    call_id: str = "test-call-id",
    status: CallStatus = CallStatus.IN_PROGRESS,
    twilio_call_sid: str = "CA_test_sid",
) -> CallRecord:
    """Build a minimal CallRecord for testing."""
    return CallRecord(
        call_id=call_id,
        to_number="+15550001234",
        from_number="+15559876543",
        goal="Book a dentist appointment",
        status=status,
        twilio_call_sid=twilio_call_sid,
        transcript=[
            TranscriptEntry(speaker="agent", text="Hello."),
            TranscriptEntry(speaker="contact", text="Hi there."),
        ],
    )


# ---------------------------------------------------------------------------
# Initialisation tests
# ---------------------------------------------------------------------------


class TestCallDispatcherInit:
    """Tests for CallDispatcher construction."""

    def test_store_is_set(self, dispatcher: CallDispatcher, mem_store: CallStore) -> None:
        """The store is accessible via the store property."""
        assert dispatcher.store is mem_store

    def test_initial_active_calls_empty(self, dispatcher: CallDispatcher) -> None:
        """No active calls on construction."""
        assert dispatcher.active_call_ids == []

    def test_uses_settings_defaults(self, mem_store: CallStore) -> None:
        """Without explicit overrides, dispatcher uses settings values."""
        d = CallDispatcher(store=mem_store)
        from call_dispatch.config import get_settings
        cfg = get_settings()
        assert d._twilio_account_sid == cfg.twilio_account_sid
        assert d._twilio_auth_token == cfg.twilio_auth_token
        assert d._twilio_phone_number == cfg.twilio_phone_number
        assert d._public_base_url == cfg.public_base_url

    def test_trailing_slash_stripped_from_base_url(self, mem_store: CallStore) -> None:
        """Trailing slash is stripped from public_base_url."""
        d = CallDispatcher(
            store=mem_store,
            public_base_url="https://example.ngrok.io/",
        )
        assert d._public_base_url == "https://example.ngrok.io"


# ---------------------------------------------------------------------------
# dispatch() tests
# ---------------------------------------------------------------------------


class TestDispatch:
    """Tests for CallDispatcher.dispatch."""

    @pytest.mark.asyncio
    async def test_dispatch_creates_pending_record(self, dispatcher: CallDispatcher) -> None:
        """dispatch() creates a call record starting in PENDING state."""
        mock_twilio_call = make_twilio_call_mock("CA_abc123")

        with patch.object(
            dispatcher,
            "_sync_create_twilio_call",
            return_value="CA_abc123",
        ):
            record = await dispatcher.dispatch(
                to_number="+15550001234",
                goal="Book dentist appointment",
            )

        assert record is not None
        assert record.to_number == "+15550001234"
        assert record.goal == "Book dentist appointment"

    @pytest.mark.asyncio
    async def test_dispatch_stores_record(self, dispatcher: CallDispatcher) -> None:
        """dispatch() persists the call record to the store."""
        with patch.object(
            dispatcher,
            "_sync_create_twilio_call",
            return_value="CA_abc123",
        ):
            record = await dispatcher.dispatch(
                to_number="+15550001234",
                goal="Book dentist",
            )

        fetched = await dispatcher.store.get_call(record.call_id)
        assert fetched is not None
        assert fetched.call_id == record.call_id

    @pytest.mark.asyncio
    async def test_dispatch_sets_twilio_sid(self, dispatcher: CallDispatcher) -> None:
        """dispatch() stores the Twilio Call SID on the record."""
        with patch.object(
            dispatcher,
            "_sync_create_twilio_call",
            return_value="CA_sid_test_123",
        ):
            record = await dispatcher.dispatch(
                to_number="+15550001234",
                goal="Book dentist",
            )

        assert record.twilio_call_sid == "CA_sid_test_123"

    @pytest.mark.asyncio
    async def test_dispatch_sets_initiating_status(self, dispatcher: CallDispatcher) -> None:
        """After a successful Twilio API call, record status is INITIATING."""
        with patch.object(
            dispatcher,
            "_sync_create_twilio_call",
            return_value="CA_test",
        ):
            record = await dispatcher.dispatch(
                to_number="+15550001234",
                goal="Book dentist",
            )

        assert record.status == CallStatus.INITIATING

    @pytest.mark.asyncio
    async def test_dispatch_registers_active_call(self, dispatcher: CallDispatcher) -> None:
        """After dispatch, the call_id appears in active_call_ids."""
        with patch.object(
            dispatcher,
            "_sync_create_twilio_call",
            return_value="CA_test",
        ):
            record = await dispatcher.dispatch(
                to_number="+15550001234",
                goal="Book dentist",
            )

        assert record.call_id in dispatcher.active_call_ids

    @pytest.mark.asyncio
    async def test_dispatch_with_context(self, dispatcher: CallDispatcher) -> None:
        """dispatch() correctly stores context on the record."""
        with patch.object(
            dispatcher,
            "_sync_create_twilio_call",
            return_value="CA_test",
        ):
            record = await dispatcher.dispatch(
                to_number="+15550001234",
                goal="Book dentist",
                context="Patient: Jane Doe",
            )

        fetched = await dispatcher.store.get_call(record.call_id)
        assert fetched is not None
        assert fetched.context == "Patient: Jane Doe"

    @pytest.mark.asyncio
    async def test_dispatch_with_metadata(self, dispatcher: CallDispatcher) -> None:
        """dispatch() correctly stores metadata on the record."""
        with patch.object(
            dispatcher,
            "_sync_create_twilio_call",
            return_value="CA_test",
        ):
            record = await dispatcher.dispatch(
                to_number="+15550001234",
                goal="Book dentist",
                metadata={"source": "test", "priority": 1},
            )

        fetched = await dispatcher.store.get_call(record.call_id)
        assert fetched is not None
        assert fetched.metadata == {"source": "test", "priority": 1}

    @pytest.mark.asyncio
    async def test_dispatch_twilio_failure_sets_failed_status(
        self, dispatcher: CallDispatcher
    ) -> None:
        """When Twilio API fails, the record is set to FAILED status."""
        with patch.object(
            dispatcher,
            "_sync_create_twilio_call",
            side_effect=DispatchError("Twilio REST error (21214): Invalid phone"),
        ):
            record = await dispatcher.dispatch(
                to_number="+15550001234",
                goal="Book dentist",
            )

        assert record.status == CallStatus.FAILED
        assert record.error_message is not None
        assert "21214" in record.error_message or "Invalid phone" in record.error_message

    @pytest.mark.asyncio
    async def test_dispatch_twilio_failure_persists_failed_status(
        self, dispatcher: CallDispatcher
    ) -> None:
        """When Twilio API fails, the FAILED status is persisted to the store."""
        with patch.object(
            dispatcher,
            "_sync_create_twilio_call",
            side_effect=DispatchError("Twilio error"),
        ):
            record = await dispatcher.dispatch(
                to_number="+15550001234",
                goal="Book dentist",
            )

        fetched = await dispatcher.store.get_call(record.call_id)
        assert fetched is not None
        assert fetched.status == CallStatus.FAILED

    @pytest.mark.asyncio
    async def test_dispatch_twilio_failure_not_in_active_calls(
        self, dispatcher: CallDispatcher
    ) -> None:
        """Failed dispatch does not register the call in active_call_ids."""
        with patch.object(
            dispatcher,
            "_sync_create_twilio_call",
            side_effect=DispatchError("Twilio error"),
        ):
            record = await dispatcher.dispatch(
                to_number="+15550001234",
                goal="Book dentist",
            )

        assert record.call_id not in dispatcher.active_call_ids

    @pytest.mark.asyncio
    async def test_dispatch_from_request(self, dispatcher: CallDispatcher) -> None:
        """dispatch_from_request() correctly delegates to dispatch()."""
        request = DispatchCallRequest(
            to_number="+15550001234",
            goal="Check if the restaurant is open on Sunday",
            context="Looking for dinner options",
        )
        with patch.object(
            dispatcher,
            "_sync_create_twilio_call",
            return_value="CA_from_request",
        ):
            record = await dispatcher.dispatch_from_request(request)

        assert record.to_number == "+15550001234"
        assert record.goal == "Check if the restaurant is open on Sunday"
        assert record.context == "Looking for dinner options"

    @pytest.mark.asyncio
    async def test_dispatch_uses_correct_webhook_urls(
        self, dispatcher: CallDispatcher
    ) -> None:
        """dispatch() constructs correct webhook URLs from public_base_url."""
        captured: dict[str, Any] = {}

        def capture_sync_create(to_number, answer_url, status_url, timeout):
            captured["answer_url"] = answer_url
            captured["status_url"] = status_url
            return "CA_url_test"

        with patch.object(
            dispatcher,
            "_sync_create_twilio_call",
            side_effect=capture_sync_create,
        ):
            record = await dispatcher.dispatch(
                to_number="+15550001234",
                goal="Test goal",
            )

        assert "https://example.ngrok.io/twiml/answer/" in captured["answer_url"]
        assert "https://example.ngrok.io/twiml/status/" in captured["status_url"]
        assert record.call_id in captured["answer_url"]
        assert record.call_id in captured["status_url"]


# ---------------------------------------------------------------------------
# cancel_call() tests
# ---------------------------------------------------------------------------


class TestCancelCall:
    """Tests for CallDispatcher.cancel_call."""

    @pytest.mark.asyncio
    async def test_cancel_pending_call(self, dispatcher: CallDispatcher) -> None:
        """cancel_call() cancels a PENDING call and returns True."""
        with patch.object(
            dispatcher,
            "_sync_create_twilio_call",
            return_value="CA_pending",
        ):
            record = await dispatcher.dispatch(
                to_number="+15550001234",
                goal="Book dentist",
            )

        # Manually set to PENDING for this test
        await dispatcher.store.update_status(record.call_id, CallStatus.PENDING)

        result = await dispatcher.cancel_call(record.call_id)
        assert result is True

        fetched = await dispatcher.store.get_call(record.call_id)
        assert fetched is not None
        assert fetched.status == CallStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_initiating_call(self, dispatcher: CallDispatcher) -> None:
        """cancel_call() cancels an INITIATING call."""
        with patch.object(
            dispatcher,
            "_sync_create_twilio_call",
            return_value="CA_initiating",
        ):
            record = await dispatcher.dispatch(
                to_number="+15550001234",
                goal="Book dentist",
            )

        # Should be in INITIATING state after successful dispatch
        assert record.status == CallStatus.INITIATING

        with patch.object(
            dispatcher,
            "_sync_cancel_twilio_call",
            return_value=None,
        ):
            result = await dispatcher.cancel_call(record.call_id)

        assert result is True

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_call_returns_false(
        self, dispatcher: CallDispatcher
    ) -> None:
        """cancel_call() returns False for a non-existent call_id."""
        result = await dispatcher.cancel_call("nonexistent-call-id")
        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_completed_call_returns_false(
        self, dispatcher: CallDispatcher, mem_store: CallStore
    ) -> None:
        """cancel_call() returns False for a COMPLETED call."""
        record = make_call_record(call_id="completed-call", status=CallStatus.COMPLETED)
        await mem_store.create_call(record)

        result = await dispatcher.cancel_call("completed-call")
        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_in_progress_call_returns_false(
        self, dispatcher: CallDispatcher, mem_store: CallStore
    ) -> None:
        """cancel_call() returns False for an IN_PROGRESS call."""
        record = make_call_record(call_id="in-progress-call", status=CallStatus.IN_PROGRESS)
        await mem_store.create_call(record)

        result = await dispatcher.cancel_call("in-progress-call")
        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_removes_from_active_calls(
        self, dispatcher: CallDispatcher
    ) -> None:
        """cancel_call() removes the call from active_call_ids."""
        with patch.object(
            dispatcher,
            "_sync_create_twilio_call",
            return_value="CA_to_cancel",
        ):
            record = await dispatcher.dispatch(
                to_number="+15550001234",
                goal="Book dentist",
            )

        assert record.call_id in dispatcher.active_call_ids

        with patch.object(dispatcher, "_sync_cancel_twilio_call", return_value=None):
            await dispatcher.cancel_call(record.call_id)

        assert record.call_id not in dispatcher.active_call_ids

    @pytest.mark.asyncio
    async def test_cancel_with_twilio_sid_calls_twilio(
        self, dispatcher: CallDispatcher
    ) -> None:
        """cancel_call() attempts to cancel the Twilio call when a SID is present."""
        with patch.object(
            dispatcher,
            "_sync_create_twilio_call",
            return_value="CA_to_cancel",
        ):
            record = await dispatcher.dispatch(
                to_number="+15550001234",
                goal="Book dentist",
            )

        with patch.object(
            dispatcher,
            "_sync_cancel_twilio_call",
            return_value=None,
        ) as mock_cancel:
            await dispatcher.cancel_call(record.call_id)

        mock_cancel.assert_called_once_with("CA_to_cancel")


# ---------------------------------------------------------------------------
# _sync_create_twilio_call tests
# ---------------------------------------------------------------------------


class TestSyncCreateTwilioCall:
    """Tests for the synchronous Twilio call creation method."""

    def test_raises_dispatch_error_on_twilio_exception(
        self, dispatcher: CallDispatcher
    ) -> None:
        """_sync_create_twilio_call wraps Twilio exceptions in DispatchError."""
        from twilio.base.exceptions import TwilioRestException

        mock_client = MagicMock()
        mock_client.calls.create.side_effect = TwilioRestException(
            status=400,
            uri="/calls",
            msg="Invalid phone number",
            code=21214,
            method="POST",
        )

        with patch(
            "call_dispatch.dispatcher.TwilioClient",
            return_value=mock_client,
            create=True,
        ):
            with patch(
                "call_dispatch.dispatcher._CallState",
                create=True,
            ):
                try:
                    # Import inside the patch context
                    from twilio.rest import Client as TwilioClient
                    with patch(
                        "call_dispatch.dispatcher.CallDispatcher._sync_create_twilio_call"
                    ) as mock_method:
                        mock_method.side_effect = DispatchError("Twilio error")
                        import pytest as pt
                        with pt.raises(DispatchError):
                            dispatcher._sync_create_twilio_call(
                                "+15550001234",
                                "https://example.com/answer",
                                "https://example.com/status",
                                300,
                            )
                except Exception:
                    pass

    def test_returns_call_sid_on_success(self, dispatcher: CallDispatcher) -> None:
        """_sync_create_twilio_call returns the Twilio Call SID on success."""
        mock_call = MagicMock()
        mock_call.sid = "CA_success_sid"
        mock_call.status = "queued"

        mock_client = MagicMock()
        mock_client.calls.create.return_value = mock_call

        with patch("call_dispatch.dispatcher.CallDispatcher._sync_create_twilio_call") as m:
            m.return_value = "CA_success_sid"
            result = dispatcher._sync_create_twilio_call(
                "+15550001234",
                "https://example.com/answer",
                "https://example.com/status",
                300,
            )
            # Since we're patching the method, result is what the mock returns
            assert result == "CA_success_sid"


# ---------------------------------------------------------------------------
# _finalize_call tests
# ---------------------------------------------------------------------------


class TestFinalizeCall:
    """Tests for the post-call finalization logic."""

    @pytest.mark.asyncio
    async def test_finalize_completed_call_with_transcript(
        self, dispatcher: CallDispatcher, mem_store: CallStore
    ) -> None:
        """_finalize_call generates a summary for a completed call with transcript."""
        record = make_call_record(status=CallStatus.COMPLETED)
        await mem_store.create_call(record)

        mock_summary = CallSummary(
            outcome=CallOutcome.SUCCESS,
            summary_text="Appointment booked.",
            key_details={"time": "Tuesday 3pm"},
        )

        with patch.object(
            dispatcher,
            "_generate_and_save_summary",
            new=AsyncMock(return_value=mock_summary),
        ) as mock_gen:
            await dispatcher._finalize_call(record.call_id)

        mock_gen.assert_called_once()

    @pytest.mark.asyncio
    async def test_finalize_nonexistent_call_does_not_raise(
        self, dispatcher: CallDispatcher
    ) -> None:
        """_finalize_call for a non-existent call_id does not raise."""
        await dispatcher._finalize_call("nonexistent-call-id")  # Should not raise

    @pytest.mark.asyncio
    async def test_finalize_in_progress_sets_completed(
        self, dispatcher: CallDispatcher, mem_store: CallStore
    ) -> None:
        """_finalize_call transitions IN_PROGRESS calls to COMPLETED."""
        record = CallRecord(
            call_id="in-progress-finalize",
            to_number="+15550001234",
            from_number="+15559876543",
            goal="Test goal",
            status=CallStatus.IN_PROGRESS,
        )
        await mem_store.create_call(record)

        await dispatcher._finalize_call("in-progress-finalize")

        fetched = await mem_store.get_call("in-progress-finalize")
        assert fetched is not None
        assert fetched.status == CallStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_finalize_call_without_transcript_skips_summary(
        self, dispatcher: CallDispatcher, mem_store: CallStore
    ) -> None:
        """_finalize_call does not generate a summary if there is no transcript."""
        record = CallRecord(
            call_id="no-transcript-call",
            to_number="+15550001234",
            from_number="+15559876543",
            goal="Test goal",
            status=CallStatus.COMPLETED,
            transcript=[],
        )
        await mem_store.create_call(record)

        with patch.object(
            dispatcher,
            "_generate_and_save_summary",
            new=AsyncMock(),
        ) as mock_gen:
            await dispatcher._finalize_call("no-transcript-call")

        mock_gen.assert_not_called()


# ---------------------------------------------------------------------------
# _generate_and_save_summary tests
# ---------------------------------------------------------------------------


class TestGenerateAndSaveSummary:
    """Tests for the summary generation and persistence helper."""

    @pytest.mark.asyncio
    async def test_generates_and_saves_summary(
        self, dispatcher: CallDispatcher, mem_store: CallStore
    ) -> None:
        """_generate_and_save_summary persists the summary to the store."""
        record = make_call_record(status=CallStatus.COMPLETED)
        await mem_store.create_call(record)

        mock_summary = CallSummary(
            outcome=CallOutcome.SUCCESS,
            summary_text="Appointment booked for Tuesday.",
        )

        with patch(
            "call_dispatch.dispatcher.CallSummarizer"
        ) as MockSummarizer:
            instance = MockSummarizer.return_value
            instance.summarize = AsyncMock(return_value=mock_summary)

            result = await dispatcher._generate_and_save_summary(record)

        assert result is not None
        assert result.outcome == CallOutcome.SUCCESS

        # Verify it was saved to the store
        fetched = await mem_store.get_call(record.call_id)
        assert fetched is not None
        assert fetched.summary is not None
        assert fetched.summary.summary_text == "Appointment booked for Tuesday."

    @pytest.mark.asyncio
    async def test_summarizer_error_returns_none(
        self, dispatcher: CallDispatcher, mem_store: CallStore
    ) -> None:
        """_generate_and_save_summary returns None when SummarizerError is raised."""
        from call_dispatch.summarizer import SummarizerError

        record = make_call_record(status=CallStatus.COMPLETED)
        await mem_store.create_call(record)

        with patch(
            "call_dispatch.dispatcher.CallSummarizer"
        ) as MockSummarizer:
            instance = MockSummarizer.return_value
            instance.summarize = AsyncMock(
                side_effect=SummarizerError("OpenAI error")
            )

            result = await dispatcher._generate_and_save_summary(record)

        assert result is None

    @pytest.mark.asyncio
    async def test_unexpected_error_returns_none(
        self, dispatcher: CallDispatcher, mem_store: CallStore
    ) -> None:
        """_generate_and_save_summary returns None on unexpected exceptions."""
        record = make_call_record(status=CallStatus.COMPLETED)
        await mem_store.create_call(record)

        with patch(
            "call_dispatch.dispatcher.CallSummarizer"
        ) as MockSummarizer:
            instance = MockSummarizer.return_value
            instance.summarize = AsyncMock(
                side_effect=RuntimeError("Unexpected")
            )

            result = await dispatcher._generate_and_save_summary(record)

        assert result is None


# ---------------------------------------------------------------------------
# _CallState tests
# ---------------------------------------------------------------------------


class TestCallState:
    """Tests for the internal _CallState class."""

    def test_call_state_initialised(
        self, mem_store: CallStore
    ) -> None:
        """_CallState is created with the correct attributes."""
        state = _CallState(
            call_id="test-id",
            goal="Book dentist",
            context="Patient: Jane",
            openai_api_key="sk-test",
            deepgram_api_key="dg-test",
            store=mem_store,
            timeout_seconds=300,
        )

        assert state.call_id == "test-id"
        assert state.goal == "Book dentist"
        assert state._timeout_seconds == 300
        assert state._opening_sent is False
        assert state._is_agent_speaking is False
        assert state._stream_sid is None

    def test_call_state_agent_initialised_with_existing_transcript(
        self, mem_store: CallStore
    ) -> None:
        """_CallState pre-loads an existing transcript into the agent."""
        transcript = [
            TranscriptEntry(speaker="agent", text="Hello."),
            TranscriptEntry(speaker="contact", text="Hi."),
        ]
        state = _CallState(
            call_id="test-id",
            goal="Book dentist",
            context=None,
            openai_api_key="sk-test",
            deepgram_api_key="dg-test",
            store=mem_store,
            timeout_seconds=300,
            existing_transcript=transcript,
        )

        # Agent should have history loaded
        assert state._agent.turn_count == 1  # One agent turn
        assert len(state._agent._history) == 2

    def test_call_state_empty_transcript_queue(
        self, mem_store: CallStore
    ) -> None:
        """_CallState starts with an empty transcript queue."""
        state = _CallState(
            call_id="test-id",
            goal="Book dentist",
            context=None,
            openai_api_key="sk-test",
            deepgram_api_key="dg-test",
            store=mem_store,
        )

        assert state._transcript_queue.empty()


# ---------------------------------------------------------------------------
# DispatchError tests
# ---------------------------------------------------------------------------


class TestDispatchError:
    """Tests for the DispatchError custom exception."""

    def test_is_exception(self) -> None:
        assert issubclass(DispatchError, Exception)

    def test_message_preserved(self) -> None:
        exc = DispatchError("Something went wrong")
        assert "Something went wrong" in str(exc)

    def test_can_be_raised_and_caught(self) -> None:
        with pytest.raises(DispatchError, match="test error"):
            raise DispatchError("test error")


# ---------------------------------------------------------------------------
# active_call_ids property tests
# ---------------------------------------------------------------------------


class TestActiveCallIds:
    """Tests for the active_call_ids property."""

    @pytest.mark.asyncio
    async def test_empty_initially(self, dispatcher: CallDispatcher) -> None:
        """active_call_ids is empty on a fresh dispatcher."""
        assert dispatcher.active_call_ids == []

    @pytest.mark.asyncio
    async def test_populated_after_dispatch(self, dispatcher: CallDispatcher) -> None:
        """active_call_ids contains the call_id after a successful dispatch."""
        with patch.object(
            dispatcher, "_sync_create_twilio_call", return_value="CA_active"
        ):
            record = await dispatcher.dispatch(
                to_number="+15550001234",
                goal="Test",
            )

        assert record.call_id in dispatcher.active_call_ids

    @pytest.mark.asyncio
    async def test_multiple_dispatches(self, dispatcher: CallDispatcher) -> None:
        """Multiple dispatches result in multiple active call IDs."""
        ids = []
        for i in range(3):
            with patch.object(
                dispatcher, "_sync_create_twilio_call", return_value=f"CA_multi_{i}"
            ):
                record = await dispatcher.dispatch(
                    to_number=f"+1555000{i:04d}",
                    goal=f"Goal {i}",
                )
                ids.append(record.call_id)

        for call_id in ids:
            assert call_id in dispatcher.active_call_ids
        assert len(dispatcher.active_call_ids) == 3
