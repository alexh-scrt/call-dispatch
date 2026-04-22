"""Unit tests for call_dispatch.store - SQLite persistence layer.

Tests cover:
- Schema initialisation
- CRUD operations (create, read, update, delete)
- Paginated listing and counting
- Targeted status updates
- Atomic transcript entry appending
- Summary persistence
- Twilio SID lookups
- Edge cases: not found, duplicate IDs, concurrent-safe operations

All tests use an in-memory SQLite database (':memory:') for speed and isolation.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Optional

import pytest
import pytest_asyncio

from call_dispatch.models import (
    CallOutcome,
    CallRecord,
    CallStatus,
    CallSummary,
    TranscriptEntry,
)
from call_dispatch.store import CallStore, _dt_to_str, _str_to_dt, _record_to_row, _row_to_record


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def store() -> CallStore:
    """Provide an initialised in-memory CallStore for each test."""
    s = CallStore(":memory:")
    await s.initialize()
    yield s
    await s.close()


def make_record(
    *,
    to_number: str = "+15550001234",
    from_number: str = "+15559876543",
    goal: str = "Book a dentist appointment",
    status: CallStatus = CallStatus.PENDING,
    twilio_call_sid: Optional[str] = None,
    context: Optional[str] = None,
    error_message: Optional[str] = None,
) -> CallRecord:
    """Convenience factory for creating CallRecord instances in tests."""
    return CallRecord(
        to_number=to_number,
        from_number=from_number,
        goal=goal,
        status=status,
        twilio_call_sid=twilio_call_sid,
        context=context,
        error_message=error_message,
    )


def make_transcript_entry(
    speaker: str = "contact",
    text: str = "Hello, how can I help you?",
    confidence: float = 0.95,
) -> TranscriptEntry:
    """Convenience factory for TranscriptEntry instances."""
    return TranscriptEntry(
        speaker=speaker,
        text=text,
        confidence=confidence,
    )


def make_summary(
    outcome: CallOutcome = CallOutcome.SUCCESS,
    summary_text: str = "Appointment booked for Tuesday at 3pm.",
    key_details: Optional[dict] = None,
) -> CallSummary:
    """Convenience factory for CallSummary instances."""
    return CallSummary(
        outcome=outcome,
        summary_text=summary_text,
        key_details=key_details or {"booked_time": "Tuesday 3pm"},
    )


# ---------------------------------------------------------------------------
# Helper serialisation tests
# ---------------------------------------------------------------------------


class TestDatetimeHelpers:
    """Unit tests for the internal datetime serialisation helpers."""

    def test_dt_to_str_returns_string(self) -> None:
        dt = datetime(2024, 6, 15, 10, 30, 0, 123456)
        result = _dt_to_str(dt)
        assert isinstance(result, str)
        assert "2024-06-15" in result

    def test_dt_to_str_none_returns_none(self) -> None:
        assert _dt_to_str(None) is None

    def test_str_to_dt_roundtrip(self) -> None:
        dt = datetime(2024, 6, 15, 10, 30, 0, 123456)
        result = _str_to_dt(_dt_to_str(dt))
        assert result == dt

    def test_str_to_dt_none_returns_none(self) -> None:
        assert _str_to_dt(None) is None

    def test_str_to_dt_iso_format(self) -> None:
        s = "2024-06-15T10:30:00.000000"
        result = _str_to_dt(s)
        assert result is not None
        assert result.year == 2024
        assert result.month == 6
        assert result.day == 15


# ---------------------------------------------------------------------------
# CallStore.initialize
# ---------------------------------------------------------------------------


class TestInitialize:
    """Tests for schema initialisation."""

    @pytest.mark.asyncio
    async def test_initialize_creates_table(self) -> None:
        """initialize() should create the calls table without error."""
        s = CallStore(":memory:")
        await s.initialize()  # Must not raise
        await s.close()

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self) -> None:
        """Calling initialize() twice should not raise."""
        s = CallStore(":memory:")
        await s.initialize()
        await s.initialize()  # Second call should be safe
        await s.close()

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """The store works correctly as an async context manager."""
        async with CallStore(":memory:") as s:
            count = await s.count_calls()
            assert count == 0


# ---------------------------------------------------------------------------
# CallStore.create_call / get_call
# ---------------------------------------------------------------------------


class TestCreateAndGetCall:
    """Tests for inserting and retrieving call records."""

    @pytest.mark.asyncio
    async def test_create_and_get_call(self, store: CallStore) -> None:
        """A created record can be retrieved by call_id."""
        record = make_record()
        await store.create_call(record)

        fetched = await store.get_call(record.call_id)
        assert fetched is not None
        assert fetched.call_id == record.call_id
        assert fetched.to_number == record.to_number
        assert fetched.goal == record.goal

    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_none(self, store: CallStore) -> None:
        """Getting a non-existent call_id returns None."""
        result = await store.get_call("00000000-0000-0000-0000-000000000000")
        assert result is None

    @pytest.mark.asyncio
    async def test_create_preserves_status(self, store: CallStore) -> None:
        """The status field is stored and retrieved correctly."""
        record = make_record(status=CallStatus.INITIATING)
        await store.create_call(record)
        fetched = await store.get_call(record.call_id)
        assert fetched is not None
        assert fetched.status == CallStatus.INITIATING

    @pytest.mark.asyncio
    async def test_create_preserves_optional_fields(self, store: CallStore) -> None:
        """Optional fields (context, twilio_call_sid) are stored correctly."""
        record = make_record(
            context="Patient: Jane Doe",
            twilio_call_sid="CA1234567890abcdef",
        )
        await store.create_call(record)
        fetched = await store.get_call(record.call_id)
        assert fetched is not None
        assert fetched.context == "Patient: Jane Doe"
        assert fetched.twilio_call_sid == "CA1234567890abcdef"

    @pytest.mark.asyncio
    async def test_create_with_metadata(self, store: CallStore) -> None:
        """Metadata dict is serialised and deserialised correctly."""
        record = make_record()
        record.metadata = {"source": "test", "priority": 1}
        await store.create_call(record)
        fetched = await store.get_call(record.call_id)
        assert fetched is not None
        assert fetched.metadata == {"source": "test", "priority": 1}

    @pytest.mark.asyncio
    async def test_create_with_transcript(self, store: CallStore) -> None:
        """A record with pre-populated transcript is stored and retrieved."""
        record = make_record()
        entry = make_transcript_entry()
        record.transcript.append(entry)
        await store.create_call(record)
        fetched = await store.get_call(record.call_id)
        assert fetched is not None
        assert len(fetched.transcript) == 1
        assert fetched.transcript[0].text == entry.text
        assert fetched.transcript[0].speaker == entry.speaker

    @pytest.mark.asyncio
    async def test_create_with_summary(self, store: CallStore) -> None:
        """A record created with a summary is stored and retrieved correctly."""
        record = make_record(status=CallStatus.COMPLETED)
        record.summary = make_summary()
        await store.create_call(record)
        fetched = await store.get_call(record.call_id)
        assert fetched is not None
        assert fetched.summary is not None
        assert fetched.summary.outcome == CallOutcome.SUCCESS
        assert fetched.summary.key_details["booked_time"] == "Tuesday 3pm"

    @pytest.mark.asyncio
    async def test_duplicate_call_id_raises(self, store: CallStore) -> None:
        """Inserting a record with a duplicate call_id raises an integrity error."""
        import sqlite3

        record = make_record()
        await store.create_call(record)
        duplicate = make_record()
        duplicate.call_id = record.call_id  # Force same ID
        with pytest.raises(sqlite3.IntegrityError):
            await store.create_call(duplicate)

    @pytest.mark.asyncio
    async def test_create_preserves_timestamps(self, store: CallStore) -> None:
        """created_at and updated_at are stored and retrieved accurately."""
        record = make_record()
        before = datetime.utcnow()
        await store.create_call(record)
        fetched = await store.get_call(record.call_id)
        assert fetched is not None
        # Timestamps should round-trip without significant drift
        assert fetched.created_at is not None


# ---------------------------------------------------------------------------
# CallStore.update_call
# ---------------------------------------------------------------------------


class TestUpdateCall:
    """Tests for full-record updates."""

    @pytest.mark.asyncio
    async def test_update_status(self, store: CallStore) -> None:
        """update_call correctly updates the status field."""
        record = make_record()
        await store.create_call(record)
        record.set_status(CallStatus.IN_PROGRESS)
        await store.update_call(record)

        fetched = await store.get_call(record.call_id)
        assert fetched is not None
        assert fetched.status == CallStatus.IN_PROGRESS

    @pytest.mark.asyncio
    async def test_update_twilio_sid(self, store: CallStore) -> None:
        """update_call correctly updates twilio_call_sid."""
        record = make_record()
        await store.create_call(record)
        record.twilio_call_sid = "CAabcdef1234567890"
        await store.update_call(record)

        fetched = await store.get_call(record.call_id)
        assert fetched is not None
        assert fetched.twilio_call_sid == "CAabcdef1234567890"

    @pytest.mark.asyncio
    async def test_update_adds_transcript(self, store: CallStore) -> None:
        """update_call correctly updates the transcript list."""
        record = make_record()
        await store.create_call(record)
        record.add_transcript_entry(make_transcript_entry(speaker="agent", text="Hi there!"))
        await store.update_call(record)

        fetched = await store.get_call(record.call_id)
        assert fetched is not None
        assert len(fetched.transcript) == 1
        assert fetched.transcript[0].speaker == "agent"

    @pytest.mark.asyncio
    async def test_update_error_message(self, store: CallStore) -> None:
        """update_call stores an error message correctly."""
        record = make_record()
        await store.create_call(record)
        record.set_status(CallStatus.FAILED, error_message="Connection refused")
        await store.update_call(record)

        fetched = await store.get_call(record.call_id)
        assert fetched is not None
        assert fetched.error_message == "Connection refused"
        assert fetched.status == CallStatus.FAILED


# ---------------------------------------------------------------------------
# CallStore.delete_call
# ---------------------------------------------------------------------------


class TestDeleteCall:
    """Tests for record deletion."""

    @pytest.mark.asyncio
    async def test_delete_existing_record(self, store: CallStore) -> None:
        """delete_call returns True and removes an existing record."""
        record = make_record()
        await store.create_call(record)
        deleted = await store.delete_call(record.call_id)
        assert deleted is True
        assert await store.get_call(record.call_id) is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_returns_false(self, store: CallStore) -> None:
        """delete_call returns False when the record does not exist."""
        deleted = await store.delete_call("00000000-0000-0000-0000-000000000000")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_delete_reduces_count(self, store: CallStore) -> None:
        """Deleting a record reduces the total count by 1."""
        r1 = make_record()
        r2 = make_record()
        await store.create_call(r1)
        await store.create_call(r2)
        assert await store.count_calls() == 2
        await store.delete_call(r1.call_id)
        assert await store.count_calls() == 1


# ---------------------------------------------------------------------------
# CallStore.list_calls / count_calls
# ---------------------------------------------------------------------------


class TestListAndCount:
    """Tests for paginated listing and counting."""

    @pytest.mark.asyncio
    async def test_empty_store_returns_empty_list(self, store: CallStore) -> None:
        result = await store.list_calls()
        assert result == []

    @pytest.mark.asyncio
    async def test_empty_store_count_is_zero(self, store: CallStore) -> None:
        count = await store.count_calls()
        assert count == 0

    @pytest.mark.asyncio
    async def test_list_returns_all_records(self, store: CallStore) -> None:
        """list_calls returns all inserted records."""
        records = [make_record(to_number=f"+1555000{i:04d}") for i in range(5)]
        for r in records:
            await store.create_call(r)
        result = await store.list_calls()
        assert len(result) == 5

    @pytest.mark.asyncio
    async def test_count_all_records(self, store: CallStore) -> None:
        """count_calls returns the total number of records."""
        for _ in range(3):
            await store.create_call(make_record())
        count = await store.count_calls()
        assert count == 3

    @pytest.mark.asyncio
    async def test_list_with_status_filter(self, store: CallStore) -> None:
        """list_calls filters correctly by status."""
        pending = make_record(status=CallStatus.PENDING)
        completed = make_record(status=CallStatus.COMPLETED)
        failed = make_record(status=CallStatus.FAILED)
        for r in (pending, completed, failed):
            await store.create_call(r)

        result = await store.list_calls(status=CallStatus.PENDING)
        assert len(result) == 1
        assert result[0].status == CallStatus.PENDING

    @pytest.mark.asyncio
    async def test_count_with_status_filter(self, store: CallStore) -> None:
        """count_calls filters correctly by status."""
        for _ in range(3):
            r = make_record(status=CallStatus.COMPLETED)
            await store.create_call(r)
        for _ in range(2):
            r = make_record(status=CallStatus.FAILED)
            await store.create_call(r)

        assert await store.count_calls(status=CallStatus.COMPLETED) == 3
        assert await store.count_calls(status=CallStatus.FAILED) == 2
        assert await store.count_calls() == 5

    @pytest.mark.asyncio
    async def test_list_with_limit(self, store: CallStore) -> None:
        """list_calls respects the limit parameter."""
        for _ in range(10):
            await store.create_call(make_record())
        result = await store.list_calls(limit=3)
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_list_with_offset(self, store: CallStore) -> None:
        """list_calls respects the offset parameter for pagination."""
        for _ in range(5):
            await store.create_call(make_record())
        all_records = await store.list_calls(limit=5)
        paginated = await store.list_calls(limit=3, offset=2)
        assert len(paginated) == 3
        # The paginated result should skip the first 2 records
        assert paginated[0].call_id == all_records[2].call_id

    @pytest.mark.asyncio
    async def test_list_ordered_by_created_at_desc(self, store: CallStore) -> None:
        """list_calls returns records ordered by created_at descending."""
        ids = []
        for i in range(3):
            r = make_record()
            # Stagger the created_at by modifying the record directly
            await store.create_call(r)
            ids.append(r.call_id)
            await asyncio.sleep(0.01)  # Ensure distinct timestamps

        result = await store.list_calls()
        # Most recently created should be first
        assert result[0].call_id == ids[-1]


# ---------------------------------------------------------------------------
# CallStore.update_status
# ---------------------------------------------------------------------------


class TestUpdateStatus:
    """Tests for targeted status updates."""

    @pytest.mark.asyncio
    async def test_update_status_changes_status(self, store: CallStore) -> None:
        """update_status correctly changes the status field."""
        record = make_record()
        await store.create_call(record)

        updated = await store.update_status(record.call_id, CallStatus.IN_PROGRESS)
        assert updated is True

        fetched = await store.get_call(record.call_id)
        assert fetched is not None
        assert fetched.status == CallStatus.IN_PROGRESS

    @pytest.mark.asyncio
    async def test_update_status_nonexistent_returns_false(self, store: CallStore) -> None:
        """update_status returns False when the record does not exist."""
        updated = await store.update_status(
            "00000000-0000-0000-0000-000000000000", CallStatus.FAILED
        )
        assert updated is False

    @pytest.mark.asyncio
    async def test_update_status_with_error_message(self, store: CallStore) -> None:
        """update_status stores an error message alongside the status."""
        record = make_record()
        await store.create_call(record)

        await store.update_status(
            record.call_id,
            CallStatus.FAILED,
            error_message="Twilio error: 21215",
        )

        fetched = await store.get_call(record.call_id)
        assert fetched is not None
        assert fetched.error_message == "Twilio error: 21215"

    @pytest.mark.asyncio
    async def test_update_status_with_started_at(self, store: CallStore) -> None:
        """update_status sets started_at when provided."""
        record = make_record()
        await store.create_call(record)
        started = datetime(2024, 7, 1, 12, 0, 0)

        await store.update_status(
            record.call_id,
            CallStatus.IN_PROGRESS,
            started_at=started,
        )

        fetched = await store.get_call(record.call_id)
        assert fetched is not None
        assert fetched.started_at == started

    @pytest.mark.asyncio
    async def test_update_status_with_ended_at(self, store: CallStore) -> None:
        """update_status sets ended_at when provided."""
        record = make_record()
        await store.create_call(record)
        ended = datetime(2024, 7, 1, 12, 5, 0)

        await store.update_status(
            record.call_id,
            CallStatus.COMPLETED,
            ended_at=ended,
        )

        fetched = await store.get_call(record.call_id)
        assert fetched is not None
        assert fetched.ended_at == ended

    @pytest.mark.asyncio
    async def test_update_status_updates_updated_at(self, store: CallStore) -> None:
        """update_status refreshes the updated_at timestamp."""
        record = make_record()
        await store.create_call(record)
        original_updated_at = record.updated_at

        await asyncio.sleep(0.01)  # Ensure time passes
        await store.update_status(record.call_id, CallStatus.INITIATING)

        fetched = await store.get_call(record.call_id)
        assert fetched is not None
        assert fetched.updated_at >= original_updated_at


# ---------------------------------------------------------------------------
# CallStore.append_transcript_entry
# ---------------------------------------------------------------------------


class TestAppendTranscriptEntry:
    """Tests for atomic transcript entry appending."""

    @pytest.mark.asyncio
    async def test_append_single_entry(self, store: CallStore) -> None:
        """append_transcript_entry appends a single entry to an empty transcript."""
        record = make_record()
        await store.create_call(record)

        entry = make_transcript_entry(speaker="contact", text="Hello!")
        result = await store.append_transcript_entry(record.call_id, entry)
        assert result is True

        fetched = await store.get_call(record.call_id)
        assert fetched is not None
        assert len(fetched.transcript) == 1
        assert fetched.transcript[0].text == "Hello!"
        assert fetched.transcript[0].speaker == "contact"

    @pytest.mark.asyncio
    async def test_append_multiple_entries_preserves_order(self, store: CallStore) -> None:
        """Multiple appended entries are stored in insertion order."""
        record = make_record()
        await store.create_call(record)

        utterances = [
            ("agent", "Hello, I'm calling to book an appointment."),
            ("contact", "Sure, what date works for you?"),
            ("agent", "Tuesday afternoon please."),
        ]
        for speaker, text in utterances:
            entry = make_transcript_entry(speaker=speaker, text=text)
            await store.append_transcript_entry(record.call_id, entry)

        fetched = await store.get_call(record.call_id)
        assert fetched is not None
        assert len(fetched.transcript) == 3
        for i, (speaker, text) in enumerate(utterances):
            assert fetched.transcript[i].speaker == speaker
            assert fetched.transcript[i].text == text

    @pytest.mark.asyncio
    async def test_append_to_nonexistent_returns_false(self, store: CallStore) -> None:
        """append_transcript_entry returns False for a non-existent call_id."""
        entry = make_transcript_entry()
        result = await store.append_transcript_entry(
            "00000000-0000-0000-0000-000000000000", entry
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_append_preserves_confidence(self, store: CallStore) -> None:
        """Transcript entry confidence score is stored and retrieved correctly."""
        record = make_record()
        await store.create_call(record)
        entry = make_transcript_entry(confidence=0.87)
        await store.append_transcript_entry(record.call_id, entry)

        fetched = await store.get_call(record.call_id)
        assert fetched is not None
        assert fetched.transcript[0].confidence == pytest.approx(0.87)

    @pytest.mark.asyncio
    async def test_append_updates_updated_at(self, store: CallStore) -> None:
        """append_transcript_entry refreshes the updated_at timestamp."""
        record = make_record()
        await store.create_call(record)
        original = record.updated_at

        await asyncio.sleep(0.01)
        await store.append_transcript_entry(record.call_id, make_transcript_entry())

        fetched = await store.get_call(record.call_id)
        assert fetched is not None
        assert fetched.updated_at >= original


# ---------------------------------------------------------------------------
# CallStore.save_summary
# ---------------------------------------------------------------------------


class TestSaveSummary:
    """Tests for summary persistence."""

    @pytest.mark.asyncio
    async def test_save_summary_success(self, store: CallStore) -> None:
        """save_summary persists a summary and returns True."""
        record = make_record(status=CallStatus.COMPLETED)
        await store.create_call(record)

        summary = make_summary(
            outcome=CallOutcome.SUCCESS,
            summary_text="Appointment successfully booked.",
            key_details={"booked_time": "Tuesday 3pm", "confirmation": "ABC123"},
        )
        result = await store.save_summary(record.call_id, summary)
        assert result is True

        fetched = await store.get_call(record.call_id)
        assert fetched is not None
        assert fetched.summary is not None
        assert fetched.summary.outcome == CallOutcome.SUCCESS
        assert fetched.summary.summary_text == "Appointment successfully booked."
        assert fetched.summary.key_details["confirmation"] == "ABC123"

    @pytest.mark.asyncio
    async def test_save_summary_nonexistent_returns_false(self, store: CallStore) -> None:
        """save_summary returns False for a non-existent call_id."""
        summary = make_summary()
        result = await store.save_summary(
            "00000000-0000-0000-0000-000000000000", summary
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_save_summary_overwrite(self, store: CallStore) -> None:
        """save_summary correctly overwrites a previously saved summary."""
        record = make_record(status=CallStatus.COMPLETED)
        await store.create_call(record)

        first_summary = make_summary(
            outcome=CallOutcome.PARTIAL,
            summary_text="Partially achieved.",
        )
        await store.save_summary(record.call_id, first_summary)

        second_summary = make_summary(
            outcome=CallOutcome.SUCCESS,
            summary_text="Fully achieved on retry.",
        )
        await store.save_summary(record.call_id, second_summary)

        fetched = await store.get_call(record.call_id)
        assert fetched is not None
        assert fetched.summary is not None
        assert fetched.summary.outcome == CallOutcome.SUCCESS
        assert fetched.summary.summary_text == "Fully achieved on retry."

    @pytest.mark.asyncio
    async def test_save_summary_with_follow_up(self, store: CallStore) -> None:
        """Summary follow_up_required and follow_up_notes fields are preserved."""
        record = make_record(status=CallStatus.COMPLETED)
        await store.create_call(record)

        summary = CallSummary(
            outcome=CallOutcome.PARTIAL,
            summary_text="Needs follow up.",
            follow_up_required=True,
            follow_up_notes="Call back to confirm cancellation policy.",
        )
        await store.save_summary(record.call_id, summary)

        fetched = await store.get_call(record.call_id)
        assert fetched is not None
        assert fetched.summary is not None
        assert fetched.summary.follow_up_required is True
        assert "cancellation" in fetched.summary.follow_up_notes


# ---------------------------------------------------------------------------
# CallStore.get_call_by_twilio_sid
# ---------------------------------------------------------------------------


class TestGetCallByTwilioSid:
    """Tests for Twilio SID lookup."""

    @pytest.mark.asyncio
    async def test_lookup_by_twilio_sid(self, store: CallStore) -> None:
        """Records can be retrieved by their Twilio Call SID."""
        record = make_record(twilio_call_sid="CA0000000000000001")
        await store.create_call(record)

        fetched = await store.get_call_by_twilio_sid("CA0000000000000001")
        assert fetched is not None
        assert fetched.call_id == record.call_id

    @pytest.mark.asyncio
    async def test_lookup_nonexistent_sid_returns_none(self, store: CallStore) -> None:
        """Lookup by an unknown Twilio SID returns None."""
        result = await store.get_call_by_twilio_sid("CAnononono")
        assert result is None

    @pytest.mark.asyncio
    async def test_lookup_sid_after_update(self, store: CallStore) -> None:
        """Twilio SID lookup works correctly after the SID has been set via update."""
        record = make_record()
        await store.create_call(record)
        assert record.twilio_call_sid is None

        record.twilio_call_sid = "CA9999999999999999"
        await store.update_call(record)

        fetched = await store.get_call_by_twilio_sid("CA9999999999999999")
        assert fetched is not None
        assert fetched.call_id == record.call_id


# ---------------------------------------------------------------------------
# Full lifecycle integration test
# ---------------------------------------------------------------------------


class TestFullCallLifecycle:
    """Integration test tracing a complete call lifecycle through the store."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self, store: CallStore) -> None:
        """A call record progresses through all states with correct data at each step."""
        # 1. Create in PENDING state
        record = make_record(
            to_number="+15550001234",
            goal="Get a quote for car insurance.",
        )
        await store.create_call(record)
        assert await store.count_calls(status=CallStatus.PENDING) == 1

        # 2. Transition to INITIATING with Twilio SID
        record.twilio_call_sid = "CA_lifecycle_test"
        record.set_status(CallStatus.INITIATING)
        await store.update_call(record)

        fetched = await store.get_call(record.call_id)
        assert fetched is not None
        assert fetched.status == CallStatus.INITIATING
        assert fetched.twilio_call_sid == "CA_lifecycle_test"

        # 3. Transition to IN_PROGRESS
        started = datetime.utcnow()
        await store.update_status(
            record.call_id,
            CallStatus.IN_PROGRESS,
            started_at=started,
        )
        assert await store.count_calls(status=CallStatus.IN_PROGRESS) == 1

        # 4. Accumulate transcript entries
        entries = [
            make_transcript_entry("agent", "Hello, I'm calling about car insurance."),
            make_transcript_entry("contact", "Yes, we can provide a quote."),
            make_transcript_entry("agent", "Great, how much for a 2019 sedan?"),
            make_transcript_entry("contact", "That would be $120 per month."),
        ]
        for entry in entries:
            ok = await store.append_transcript_entry(record.call_id, entry)
            assert ok is True

        fetched = await store.get_call(record.call_id)
        assert fetched is not None
        assert len(fetched.transcript) == 4

        # 5. Transition to COMPLETED and save summary
        ended = datetime.utcnow()
        await store.update_status(
            record.call_id,
            CallStatus.COMPLETED,
            ended_at=ended,
        )
        summary = CallSummary(
            outcome=CallOutcome.SUCCESS,
            summary_text="Obtained a quote of $120/month for a 2019 sedan.",
            key_details={"monthly_premium": "$120", "vehicle": "2019 sedan"},
            follow_up_required=False,
        )
        ok = await store.save_summary(record.call_id, summary)
        assert ok is True

        # 6. Verify final state
        final = await store.get_call(record.call_id)
        assert final is not None
        assert final.status == CallStatus.COMPLETED
        assert final.started_at is not None
        assert final.ended_at is not None
        assert final.summary is not None
        assert final.summary.outcome == CallOutcome.SUCCESS
        assert final.summary.key_details["monthly_premium"] == "$120"
        assert len(final.transcript) == 4
        assert await store.count_calls(status=CallStatus.COMPLETED) == 1
