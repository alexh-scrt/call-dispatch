"""SQLite persistence layer for call_dispatch.

Provides a thread-safe, async-compatible SQLite store for persisting call records,
transcripts, and summaries.  All database operations are performed synchronously
inside ``asyncio.to_thread`` wrappers so that the FastAPI event loop is never
blocked.

Schema
------
A single ``calls`` table stores every field of :class:`~call_dispatch.models.CallRecord`.
Complex nested fields (``transcript``, ``summary``, ``metadata``) are serialised
as JSON text columns.

Usage example::

    store = CallStore("sqlite:///./call_dispatch.db")
    await store.initialize()

    record = CallRecord(to_number="+15550001234", from_number="+15559876543", goal="Book dentist")
    await store.create_call(record)

    fetched = await store.get_call(record.call_id)
    await store.close()
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional

from call_dispatch.models import (
    CallOutcome,
    CallRecord,
    CallStatus,
    CallSummary,
    TranscriptEntry,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ISO-8601 helpers
# ---------------------------------------------------------------------------

_DT_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"


def _dt_to_str(dt: Optional[datetime]) -> Optional[str]:
    """Serialise a :class:`datetime` to an ISO-8601 string, or return ``None``."""
    if dt is None:
        return None
    return dt.strftime(_DT_FORMAT)


def _str_to_dt(s: Optional[str]) -> Optional[datetime]:
    """Deserialise an ISO-8601 string to a :class:`datetime`, or return ``None``."""
    if s is None:
        return None
    try:
        return datetime.strptime(s, _DT_FORMAT)
    except ValueError:
        # Fallback: fromisoformat handles several variant formats
        return datetime.fromisoformat(s)


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_CREATE_CALLS_TABLE = """
CREATE TABLE IF NOT EXISTS calls (
    call_id             TEXT PRIMARY KEY,
    to_number           TEXT NOT NULL,
    from_number         TEXT NOT NULL,
    goal                TEXT NOT NULL,
    context             TEXT,
    status              TEXT NOT NULL DEFAULT 'pending',
    twilio_call_sid     TEXT,
    transcript          TEXT NOT NULL DEFAULT '[]',
    summary             TEXT,
    metadata            TEXT,
    error_message       TEXT,
    max_duration_seconds INTEGER,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL,
    started_at          TEXT,
    ended_at            TEXT
);
"""

_CREATE_INDEX_STATUS = """
CREATE INDEX IF NOT EXISTS idx_calls_status ON calls (status);
"""

_CREATE_INDEX_CREATED = """
CREATE INDEX IF NOT EXISTS idx_calls_created_at ON calls (created_at);
"""


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _record_to_row(record: CallRecord) -> dict:
    """Convert a :class:`CallRecord` to a flat dictionary suitable for SQLite.

    All complex nested objects are JSON-encoded into strings.

    Args:
        record: The call record to serialise.

    Returns:
        dict: A flat mapping of column names to Python primitives.
    """
    transcript_json = json.dumps(
        [entry.model_dump(mode="json") for entry in record.transcript]
    )
    summary_json: Optional[str] = None
    if record.summary is not None:
        summary_json = record.summary.model_dump_json()

    metadata_json: Optional[str] = None
    if record.metadata is not None:
        metadata_json = json.dumps(record.metadata)

    return {
        "call_id": record.call_id,
        "to_number": record.to_number,
        "from_number": record.from_number,
        "goal": record.goal,
        "context": record.context,
        "status": record.status.value,
        "twilio_call_sid": record.twilio_call_sid,
        "transcript": transcript_json,
        "summary": summary_json,
        "metadata": metadata_json,
        "error_message": record.error_message,
        "max_duration_seconds": record.max_duration_seconds,
        "created_at": _dt_to_str(record.created_at),
        "updated_at": _dt_to_str(record.updated_at),
        "started_at": _dt_to_str(record.started_at),
        "ended_at": _dt_to_str(record.ended_at),
    }


def _row_to_record(row: sqlite3.Row) -> CallRecord:
    """Convert a :class:`sqlite3.Row` back into a :class:`CallRecord`.

    Args:
        row: A row fetched from the ``calls`` table with ``row_factory``
             set to :func:`sqlite3.Row`.

    Returns:
        CallRecord: The fully reconstructed call record.

    Raises:
        ValueError: If JSON decoding or model validation fails.
    """
    transcript_raw = json.loads(row["transcript"] or "[]")
    transcript = [TranscriptEntry(**entry) for entry in transcript_raw]

    summary: Optional[CallSummary] = None
    if row["summary"]:
        summary = CallSummary.model_validate_json(row["summary"])

    metadata = None
    if row["metadata"]:
        metadata = json.loads(row["metadata"])

    return CallRecord(
        call_id=row["call_id"],
        to_number=row["to_number"],
        from_number=row["from_number"],
        goal=row["goal"],
        context=row["context"],
        status=CallStatus(row["status"]),
        twilio_call_sid=row["twilio_call_sid"],
        transcript=transcript,
        summary=summary,
        metadata=metadata,
        error_message=row["error_message"],
        max_duration_seconds=row["max_duration_seconds"],
        created_at=_str_to_dt(row["created_at"]) or datetime.utcnow(),
        updated_at=_str_to_dt(row["updated_at"]) or datetime.utcnow(),
        started_at=_str_to_dt(row["started_at"]),
        ended_at=_str_to_dt(row["ended_at"]),
    )


# ---------------------------------------------------------------------------
# Store class
# ---------------------------------------------------------------------------


class CallStore:
    """Thread-safe SQLite persistence layer for call records.

    Wraps synchronous SQLite operations and exposes an async interface by
    delegating blocking calls to :func:`asyncio.to_thread`.

    Args:
        database_url: A SQLite URL string of the form
            ``"sqlite:///./path/to/db.sqlite"`` or ``"sqlite:///:memory:"``.
            The leading ``"sqlite:///"`` prefix is stripped to obtain the
            filesystem path.
    """

    def __init__(self, database_url: str = "sqlite:///./call_dispatch.db") -> None:
        self._db_path = self._parse_path(database_url)
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = asyncio.Lock()
        logger.debug("CallStore initialised with db_path=%s", self._db_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_path(database_url: str) -> str:
        """Extract the filesystem path from a sqlite URL string.

        Supports both ``sqlite:///relative/path`` and
        ``sqlite:////absolute/path`` as well as the bare path string
        ``":memory:"`` for in-memory databases used in tests.

        Args:
            database_url: The SQLite URL string.

        Returns:
            str: The filesystem path (or ``:memory:``).
        """
        if database_url.startswith("sqlite:///"):
            return database_url[len("sqlite:///"):] or ":memory:"
        if database_url.startswith("sqlite://"):
            return database_url[len("sqlite://"):] or ":memory:"
        # Treat as a raw path
        return database_url

    def _get_connection(self) -> sqlite3.Connection:
        """Return the open SQLite connection, creating it if necessary.

        The connection uses :attr:`sqlite3.Row` as the row factory so that
        columns can be accessed by name.

        Returns:
            sqlite3.Connection: The active database connection.
        """
        if self._conn is None:
            # Ensure parent directory exists for file-based databases
            if self._db_path != ":memory:":
                Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(
                self._db_path,
                check_same_thread=False,
                isolation_level=None,  # autocommit; we manage transactions manually
            )
            self._conn.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrent read/write performance
            if self._db_path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA foreign_keys=ON;")
            logger.debug("SQLite connection opened: %s", self._db_path)
        return self._conn

    @contextmanager
    def _transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager that wraps database operations in an explicit transaction.

        Commits on success; rolls back on any exception.

        Yields:
            sqlite3.Connection: The active database connection.

        Raises:
            Exception: Re-raises any exception after rolling back.
        """
        conn = self._get_connection()
        conn.execute("BEGIN")
        try:
            yield conn
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    # ------------------------------------------------------------------
    # Synchronous (private) implementations
    # ------------------------------------------------------------------

    def _sync_initialize(self) -> None:
        """Create tables and indexes synchronously."""
        conn = self._get_connection()
        conn.execute(_CREATE_CALLS_TABLE)
        conn.execute(_CREATE_INDEX_STATUS)
        conn.execute(_CREATE_INDEX_CREATED)
        logger.info("CallStore schema initialised.")

    def _sync_create_call(self, record: CallRecord) -> None:
        """Insert a new call record synchronously."""
        row = _record_to_row(record)
        placeholders = ", ".join(f":{k}" for k in row)
        columns = ", ".join(row.keys())
        sql = f"INSERT INTO calls ({columns}) VALUES ({placeholders})"
        with self._transaction() as conn:
            conn.execute(sql, row)
        logger.debug("Created call record call_id=%s", record.call_id)

    def _sync_get_call(self, call_id: str) -> Optional[CallRecord]:
        """Fetch a single call record by its ID synchronously."""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM calls WHERE call_id = ?", (call_id,)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return _row_to_record(row)

    def _sync_update_call(self, record: CallRecord) -> None:
        """Update all fields of an existing call record synchronously."""
        row = _record_to_row(record)
        call_id = row.pop("call_id")
        set_clause = ", ".join(f"{k} = :{k}" for k in row)
        row["call_id"] = call_id
        sql = f"UPDATE calls SET {set_clause} WHERE call_id = :call_id"
        with self._transaction() as conn:
            conn.execute(sql, row)
        logger.debug("Updated call record call_id=%s status=%s", record.call_id, record.status)

    def _sync_delete_call(self, call_id: str) -> bool:
        """Delete a call record by ID synchronously.  Returns True if deleted."""
        with self._transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM calls WHERE call_id = ?", (call_id,)
            )
        deleted = cursor.rowcount > 0
        logger.debug("Deleted call record call_id=%s found=%s", call_id, deleted)
        return deleted

    def _sync_list_calls(
        self,
        *,
        status: Optional[CallStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[CallRecord]:
        """Return a paginated list of call records, optionally filtered by status."""
        conn = self._get_connection()
        if status is not None:
            cursor = conn.execute(
                "SELECT * FROM calls WHERE status = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (status.value, limit, offset),
            )
        else:
            cursor = conn.execute(
                "SELECT * FROM calls ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
        rows = cursor.fetchall()
        return [_row_to_record(r) for r in rows]

    def _sync_count_calls(
        self,
        *,
        status: Optional[CallStatus] = None,
    ) -> int:
        """Return the total number of call records, optionally filtered by status."""
        conn = self._get_connection()
        if status is not None:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM calls WHERE status = ?", (status.value,)
            )
        else:
            cursor = conn.execute("SELECT COUNT(*) FROM calls")
        row = cursor.fetchone()
        return int(row[0]) if row else 0

    def _sync_get_call_by_twilio_sid(self, twilio_call_sid: str) -> Optional[CallRecord]:
        """Look up a call record by Twilio Call SID synchronously."""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM calls WHERE twilio_call_sid = ?", (twilio_call_sid,)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return _row_to_record(row)

    def _sync_update_status(
        self,
        call_id: str,
        status: CallStatus,
        error_message: Optional[str] = None,
        started_at: Optional[datetime] = None,
        ended_at: Optional[datetime] = None,
    ) -> bool:
        """Perform a targeted status update without loading the full record."""
        updated_at_str = _dt_to_str(datetime.utcnow())
        params: dict = {
            "call_id": call_id,
            "status": status.value,
            "updated_at": updated_at_str,
        }
        extra_sets: list[str] = ["status = :status", "updated_at = :updated_at"]

        if error_message is not None:
            extra_sets.append("error_message = :error_message")
            params["error_message"] = error_message

        if started_at is not None:
            extra_sets.append("started_at = :started_at")
            params["started_at"] = _dt_to_str(started_at)

        if ended_at is not None:
            extra_sets.append("ended_at = :ended_at")
            params["ended_at"] = _dt_to_str(ended_at)

        set_clause = ", ".join(extra_sets)
        sql = f"UPDATE calls SET {set_clause} WHERE call_id = :call_id"

        with self._transaction() as conn:
            cursor = conn.execute(sql, params)
        updated = cursor.rowcount > 0
        logger.debug(
            "Status update call_id=%s status=%s updated=%s", call_id, status.value, updated
        )
        return updated

    def _sync_append_transcript_entry(
        self, call_id: str, entry: TranscriptEntry
    ) -> bool:
        """Append a single transcript entry to an existing call record.

        Fetches the current transcript JSON, appends the new entry, and
        writes the result back in a single transaction.

        Args:
            call_id: The target call record ID.
            entry: The :class:`TranscriptEntry` to append.

        Returns:
            bool: True if the record was found and updated, False otherwise.
        """
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT transcript FROM calls WHERE call_id = ?", (call_id,)
        )
        row = cursor.fetchone()
        if row is None:
            logger.warning("append_transcript_entry: call_id=%s not found", call_id)
            return False

        transcript_list = json.loads(row["transcript"] or "[]")
        transcript_list.append(entry.model_dump(mode="json"))
        new_transcript_json = json.dumps(transcript_list)
        updated_at_str = _dt_to_str(datetime.utcnow())

        with self._transaction() as conn2:
            conn2.execute(
                "UPDATE calls SET transcript = ?, updated_at = ? WHERE call_id = ?",
                (new_transcript_json, updated_at_str, call_id),
            )
        logger.debug(
            "Appended transcript entry call_id=%s speaker=%s", call_id, entry.speaker
        )
        return True

    def _sync_save_summary(self, call_id: str, summary: CallSummary) -> bool:
        """Persist a :class:`CallSummary` for the given call ID.

        Args:
            call_id: The target call record ID.
            summary: The summary to persist.

        Returns:
            bool: True if the record was found and updated, False otherwise.
        """
        summary_json = summary.model_dump_json()
        updated_at_str = _dt_to_str(datetime.utcnow())
        with self._transaction() as conn:
            cursor = conn.execute(
                "UPDATE calls SET summary = ?, updated_at = ? WHERE call_id = ?",
                (summary_json, updated_at_str, call_id),
            )
        updated = cursor.rowcount > 0
        logger.debug("Saved summary call_id=%s updated=%s", call_id, updated)
        return updated

    def _sync_close(self) -> None:
        """Close the SQLite connection synchronously."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            logger.debug("SQLite connection closed: %s", self._db_path)

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Create the database schema (tables and indexes) if it does not exist.

        Must be called once before any other store operations, typically
        during application startup.

        Raises:
            sqlite3.Error: If schema creation fails.
        """
        async with self._lock:
            await asyncio.to_thread(self._sync_initialize)

    async def create_call(self, record: CallRecord) -> None:
        """Persist a new call record to the database.

        Args:
            record: The :class:`CallRecord` to insert.  Its ``call_id``
                must be unique; duplicate IDs will raise an integrity error.

        Raises:
            sqlite3.IntegrityError: If a record with the same ``call_id`` exists.
            sqlite3.Error: On other database failures.
        """
        async with self._lock:
            await asyncio.to_thread(self._sync_create_call, record)

    async def get_call(self, call_id: str) -> Optional[CallRecord]:
        """Retrieve a call record by its unique identifier.

        Args:
            call_id: The UUID string identifying the call.

        Returns:
            CallRecord | None: The record, or ``None`` if not found.

        Raises:
            sqlite3.Error: On database read failures.
        """
        return await asyncio.to_thread(self._sync_get_call, call_id)

    async def update_call(self, record: CallRecord) -> None:
        """Overwrite all fields of an existing call record.

        This is a full-record update; use :meth:`update_status` or
        :meth:`append_transcript_entry` for targeted partial updates.

        Args:
            record: The :class:`CallRecord` with updated fields.  The
                ``call_id`` is used to locate the existing row.

        Raises:
            sqlite3.Error: On database write failures.
        """
        async with self._lock:
            await asyncio.to_thread(self._sync_update_call, record)

    async def delete_call(self, call_id: str) -> bool:
        """Delete a call record from the database.

        Args:
            call_id: The UUID string identifying the call.

        Returns:
            bool: ``True`` if a record was deleted, ``False`` if not found.

        Raises:
            sqlite3.Error: On database write failures.
        """
        async with self._lock:
            return await asyncio.to_thread(self._sync_delete_call, call_id)

    async def list_calls(
        self,
        *,
        status: Optional[CallStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[CallRecord]:
        """Return a paginated list of call records.

        Args:
            status: Optional filter to restrict results to a specific
                :class:`~call_dispatch.models.CallStatus`.
            limit: Maximum number of records to return (default 100).
            offset: Number of records to skip for pagination (default 0).

        Returns:
            list[CallRecord]: Ordered by ``created_at`` descending.

        Raises:
            sqlite3.Error: On database read failures.
        """
        return await asyncio.to_thread(
            self._sync_list_calls, status=status, limit=limit, offset=offset
        )

    async def count_calls(
        self,
        *,
        status: Optional[CallStatus] = None,
    ) -> int:
        """Return the total count of call records, optionally filtered by status.

        Args:
            status: Optional :class:`~call_dispatch.models.CallStatus` filter.

        Returns:
            int: The number of matching records.

        Raises:
            sqlite3.Error: On database read failures.
        """
        return await asyncio.to_thread(self._sync_count_calls, status=status)

    async def get_call_by_twilio_sid(self, twilio_call_sid: str) -> Optional[CallRecord]:
        """Look up a call record by the Twilio Call SID.

        Useful in webhook handlers where the Twilio SID is the only
        available identifier.

        Args:
            twilio_call_sid: The ``CallSid`` value provided by Twilio.

        Returns:
            CallRecord | None: The matching record, or ``None`` if not found.

        Raises:
            sqlite3.Error: On database read failures.
        """
        return await asyncio.to_thread(
            self._sync_get_call_by_twilio_sid, twilio_call_sid
        )

    async def update_status(
        self,
        call_id: str,
        status: CallStatus,
        *,
        error_message: Optional[str] = None,
        started_at: Optional[datetime] = None,
        ended_at: Optional[datetime] = None,
    ) -> bool:
        """Perform a targeted status update on a call record.

        More efficient than loading and re-saving the full record when only
        the status field needs updating.

        Args:
            call_id: The UUID string identifying the call.
            status: The new :class:`~call_dispatch.models.CallStatus` to apply.
            error_message: Optional error description (use with FAILED status).
            started_at: Optional timestamp to set as ``started_at``.
            ended_at: Optional timestamp to set as ``ended_at``.

        Returns:
            bool: ``True`` if the record was found and updated.

        Raises:
            sqlite3.Error: On database write failures.
        """
        async with self._lock:
            return await asyncio.to_thread(
                self._sync_update_status,
                call_id,
                status,
                error_message,
                started_at,
                ended_at,
            )

    async def append_transcript_entry(
        self, call_id: str, entry: TranscriptEntry
    ) -> bool:
        """Atomically append a single transcript entry to a call's transcript.

        Avoids a full record round-trip by fetching and updating only the
        ``transcript`` JSON column.

        Args:
            call_id: The UUID string identifying the call.
            entry: The :class:`~call_dispatch.models.TranscriptEntry` to append.

        Returns:
            bool: ``True`` if the record was found and updated.

        Raises:
            sqlite3.Error: On database write failures.
        """
        async with self._lock:
            return await asyncio.to_thread(
                self._sync_append_transcript_entry, call_id, entry
            )

    async def save_summary(self, call_id: str, summary: CallSummary) -> bool:
        """Persist a structured :class:`~call_dispatch.models.CallSummary` for a call.

        Args:
            call_id: The UUID string identifying the call.
            summary: The :class:`~call_dispatch.models.CallSummary` to persist.

        Returns:
            bool: ``True`` if the record was found and updated.

        Raises:
            sqlite3.Error: On database write failures.
        """
        async with self._lock:
            return await asyncio.to_thread(self._sync_save_summary, call_id, summary)

    async def close(self) -> None:
        """Close the underlying SQLite connection.

        Safe to call even if the connection was never opened.  After calling
        this method, all subsequent operations will transparently re-open
        the connection as needed.
        """
        async with self._lock:
            await asyncio.to_thread(self._sync_close)

    # ------------------------------------------------------------------
    # Async context manager support
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "CallStore":
        """Enter the async context manager, initialising the schema."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the async context manager, closing the connection."""
        await self.close()


# ---------------------------------------------------------------------------
# Module-level convenience factory
# ---------------------------------------------------------------------------


def create_store(database_url: Optional[str] = None) -> CallStore:
    """Create a :class:`CallStore` from a database URL or the application settings.

    If ``database_url`` is ``None``, the URL is read from
    :attr:`~call_dispatch.config.Settings.database_url` via the settings
    singleton.

    Args:
        database_url: Optional SQLite URL override.  When ``None``, falls
            back to the configured default.

    Returns:
        CallStore: An uninitialised :class:`CallStore` instance.  Call
            :meth:`~CallStore.initialize` (or use it as an async context
            manager) before performing any operations.
    """
    if database_url is None:
        from call_dispatch.config import get_settings

        database_url = get_settings().database_url
    return CallStore(database_url)
