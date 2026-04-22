"""Unit tests for call_dispatch.summarizer - post-call structured summarizer.

All OpenAI API calls are mocked using unittest.mock so that tests are fully
offline and deterministic.

Test coverage includes:
- format_transcript helper function
- _parse_outcome helper function
- _parse_summary_response with valid JSON
- _parse_summary_response with markdown-fenced JSON
- _parse_summary_response with invalid JSON (fallback path)
- _fallback_summary construction
- CallSummarizer.summarize with full transcripts
- CallSummarizer.summarize with empty transcripts
- CallSummarizer.summarize_from_text
- CallSummarizer error handling (OpenAI failures, empty responses)
- Outcome extraction for all CallOutcome variants
- key_details and follow_up_required extraction
- API parameter forwarding (model, max_tokens, temperature)
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from call_dispatch.models import CallOutcome, CallSummary, TranscriptEntry
from call_dispatch.summarizer import (
    CallSummarizer,
    SummarizerError,
    _fallback_summary,
    _parse_outcome,
    _parse_summary_response,
    format_transcript,
)


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


def _make_openai_response(content: str) -> MagicMock:
    """Build a minimal mock that mimics an OpenAI ChatCompletion response."""
    message = MagicMock()
    message.content = content

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]
    return response


def _make_empty_openai_response() -> MagicMock:
    """Build a mock OpenAI response with no choices."""
    response = MagicMock()
    response.choices = []
    return response


def _make_json_response(
    outcome: str = "success",
    summary_text: str = "Call completed successfully.",
    key_details: dict | None = None,
    follow_up_required: bool = False,
    follow_up_notes: str | None = None,
) -> str:
    """Build a JSON string matching the summarizer response schema."""
    return json.dumps(
        {
            "outcome": outcome,
            "summary_text": summary_text,
            "key_details": key_details or {},
            "follow_up_required": follow_up_required,
            "follow_up_notes": follow_up_notes,
        }
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def patch_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject env vars and clear settings cache for each test."""
    for key, val in _FAKE_ENV.items():
        monkeypatch.setenv(key, val)
    from call_dispatch.config import get_settings
    get_settings.cache_clear()


@pytest.fixture
def summarizer() -> CallSummarizer:
    """Provide a CallSummarizer for testing."""
    return CallSummarizer(
        openai_api_key="sk-test-key",
        model="gpt-4o",
        max_tokens=512,
        temperature=0.2,
    )


@pytest.fixture
def sample_transcript() -> list[TranscriptEntry]:
    """Provide a sample multi-turn transcript."""
    return [
        TranscriptEntry(
            speaker="agent",
            text="Hello, I'm calling to book a dentist appointment.",
        ),
        TranscriptEntry(
            speaker="contact",
            text="Sure, we have Tuesday at 3pm available.",
        ),
        TranscriptEntry(
            speaker="agent",
            text="Tuesday at 3pm works perfectly, thank you!",
        ),
        TranscriptEntry(
            speaker="contact",
            text="Great, I've booked you in. Confirmation number is ABC123.",
        ),
        TranscriptEntry(
            speaker="agent",
            text="Wonderful, thank you. Goodbye!",
        ),
    ]


# ---------------------------------------------------------------------------
# format_transcript tests
# ---------------------------------------------------------------------------


class TestFormatTranscript:
    """Tests for the format_transcript helper function."""

    def test_empty_transcript_returns_placeholder(self) -> None:
        """An empty transcript returns the placeholder string."""
        result = format_transcript([])
        assert "No transcript available" in result

    def test_single_entry_format(self) -> None:
        """A single entry is formatted as SPEAKER: text."""
        entry = TranscriptEntry(speaker="agent", text="Hello there.")
        result = format_transcript([entry])
        assert result == "AGENT: Hello there."

    def test_speaker_labels_are_uppercased(self) -> None:
        """Speaker labels are uppercased in the output."""
        entries = [
            TranscriptEntry(speaker="agent", text="Hi."),
            TranscriptEntry(speaker="contact", text="Hello."),
        ]
        result = format_transcript(entries)
        assert "AGENT:" in result
        assert "CONTACT:" in result

    def test_multiple_entries_separated_by_newline(self) -> None:
        """Multiple entries are separated by newlines."""
        entries = [
            TranscriptEntry(speaker="agent", text="Line 1."),
            TranscriptEntry(speaker="contact", text="Line 2."),
            TranscriptEntry(speaker="agent", text="Line 3."),
        ]
        result = format_transcript(entries)
        lines = result.splitlines()
        assert len(lines) == 3

    def test_transcript_preserves_text_content(self) -> None:
        """Transcript text is preserved exactly as provided."""
        entries = [
            TranscriptEntry(speaker="agent", text="Book for Tuesday at 3pm."),
            TranscriptEntry(speaker="contact", text="Confirmation: ABC123."),
        ]
        result = format_transcript(entries)
        assert "Book for Tuesday at 3pm." in result
        assert "Confirmation: ABC123." in result

    def test_format_order_matches_input(self) -> None:
        """Entries appear in the same order as the input list."""
        entries = [
            TranscriptEntry(speaker="agent", text="First"),
            TranscriptEntry(speaker="contact", text="Second"),
            TranscriptEntry(speaker="agent", text="Third"),
        ]
        result = format_transcript(entries)
        lines = result.splitlines()
        assert "First" in lines[0]
        assert "Second" in lines[1]
        assert "Third" in lines[2]


# ---------------------------------------------------------------------------
# _parse_outcome tests
# ---------------------------------------------------------------------------


class TestParseOutcome:
    """Tests for the _parse_outcome helper function."""

    def test_success(self) -> None:
        assert _parse_outcome("success") == CallOutcome.SUCCESS

    def test_partial(self) -> None:
        assert _parse_outcome("partial") == CallOutcome.PARTIAL

    def test_failure(self) -> None:
        assert _parse_outcome("failure") == CallOutcome.FAILURE

    def test_unknown(self) -> None:
        assert _parse_outcome("unknown") == CallOutcome.UNKNOWN

    def test_case_insensitive(self) -> None:
        assert _parse_outcome("SUCCESS") == CallOutcome.SUCCESS
        assert _parse_outcome("Partial") == CallOutcome.PARTIAL

    def test_whitespace_stripped(self) -> None:
        assert _parse_outcome("  success  ") == CallOutcome.SUCCESS

    def test_unrecognised_returns_unknown(self) -> None:
        assert _parse_outcome("achieved") == CallOutcome.UNKNOWN
        assert _parse_outcome("") == CallOutcome.UNKNOWN
        assert _parse_outcome("COMPLETED") == CallOutcome.UNKNOWN


# ---------------------------------------------------------------------------
# _parse_summary_response tests
# ---------------------------------------------------------------------------


class TestParseSummaryResponse:
    """Tests for the _parse_summary_response function."""

    def test_valid_json_success(self) -> None:
        """Valid JSON with success outcome is parsed correctly."""
        raw = _make_json_response(
            outcome="success",
            summary_text="Appointment booked for Tuesday at 3pm.",
            key_details={"booked_time": "Tuesday 3pm", "confirmation": "ABC123"},
            follow_up_required=False,
            follow_up_notes=None,
        )
        result = _parse_summary_response(raw, goal="Book dentist")
        assert isinstance(result, CallSummary)
        assert result.outcome == CallOutcome.SUCCESS
        assert result.summary_text == "Appointment booked for Tuesday at 3pm."
        assert result.key_details["booked_time"] == "Tuesday 3pm"
        assert result.key_details["confirmation"] == "ABC123"
        assert result.follow_up_required is False
        assert result.follow_up_notes is None

    def test_valid_json_partial(self) -> None:
        """Partial outcome is parsed correctly."""
        raw = _make_json_response(outcome="partial", summary_text="Partially done.")
        result = _parse_summary_response(raw, goal="Test")
        assert result.outcome == CallOutcome.PARTIAL

    def test_valid_json_failure(self) -> None:
        """Failure outcome is parsed correctly."""
        raw = _make_json_response(outcome="failure", summary_text="Goal not achieved.")
        result = _parse_summary_response(raw, goal="Test")
        assert result.outcome == CallOutcome.FAILURE

    def test_valid_json_with_follow_up(self) -> None:
        """follow_up_required and follow_up_notes are extracted correctly."""
        raw = _make_json_response(
            outcome="partial",
            summary_text="Needs follow up.",
            follow_up_required=True,
            follow_up_notes="Call back to confirm.",
        )
        result = _parse_summary_response(raw, goal="Test")
        assert result.follow_up_required is True
        assert result.follow_up_notes == "Call back to confirm."

    def test_markdown_fenced_json_is_parsed(self) -> None:
        """JSON wrapped in markdown code fences is correctly unwrapped and parsed."""
        inner = _make_json_response(
            outcome="success",
            summary_text="Done.",
        )
        raw = f"```json\n{inner}\n```"
        result = _parse_summary_response(raw, goal="Test")
        assert result.outcome == CallOutcome.SUCCESS
        assert result.summary_text == "Done."

    def test_markdown_fenced_no_lang_is_parsed(self) -> None:
        """JSON wrapped in plain ``` fences is also handled."""
        inner = _make_json_response(outcome="failure", summary_text="Failed.")
        raw = f"```\n{inner}\n```"
        result = _parse_summary_response(raw, goal="Test")
        assert result.outcome == CallOutcome.FAILURE

    def test_invalid_json_returns_fallback(self) -> None:
        """Completely invalid JSON returns a fallback summary."""
        result = _parse_summary_response(
            "This is not JSON at all.", goal="Book dentist"
        )
        assert result.outcome == CallOutcome.UNKNOWN
        assert "Book dentist" in result.summary_text
        assert result.follow_up_required is True

    def test_empty_string_returns_fallback(self) -> None:
        """An empty string returns a fallback summary."""
        result = _parse_summary_response("", goal="Test goal")
        assert result.outcome == CallOutcome.UNKNOWN

    def test_missing_summary_text_uses_goal(self) -> None:
        """When summary_text is absent or blank, a fallback message is used."""
        raw = json.dumps(
            {
                "outcome": "success",
                "summary_text": "",
                "key_details": {},
                "follow_up_required": False,
                "follow_up_notes": None,
            }
        )
        result = _parse_summary_response(raw, goal="My specific goal")
        assert "My specific goal" in result.summary_text

    def test_null_follow_up_notes_becomes_none(self) -> None:
        """follow_up_notes of null/none string becomes Python None."""
        raw = json.dumps(
            {
                "outcome": "success",
                "summary_text": "Done.",
                "key_details": {},
                "follow_up_required": False,
                "follow_up_notes": "null",
            }
        )
        result = _parse_summary_response(raw, goal="Test")
        assert result.follow_up_notes is None

    def test_non_dict_key_details_becomes_empty(self) -> None:
        """Non-dict key_details is replaced with an empty dict."""
        raw = json.dumps(
            {
                "outcome": "success",
                "summary_text": "Done.",
                "key_details": "not a dict",
                "follow_up_required": False,
                "follow_up_notes": None,
            }
        )
        result = _parse_summary_response(raw, goal="Test")
        assert result.key_details == {}


# ---------------------------------------------------------------------------
# _fallback_summary tests
# ---------------------------------------------------------------------------


class TestFallbackSummary:
    """Tests for the _fallback_summary helper."""

    def test_outcome_is_unknown(self) -> None:
        result = _fallback_summary("Some goal")
        assert result.outcome == CallOutcome.UNKNOWN

    def test_follow_up_required_is_true(self) -> None:
        result = _fallback_summary("Some goal")
        assert result.follow_up_required is True

    def test_goal_in_summary_text(self) -> None:
        result = _fallback_summary("Book a table")
        assert "Book a table" in result.summary_text

    def test_raw_text_in_key_details(self) -> None:
        result = _fallback_summary("Test", raw_text="some raw output")
        assert "raw_model_output" in result.key_details

    def test_raw_text_truncated_to_500(self) -> None:
        long_text = "x" * 1000
        result = _fallback_summary("Test", raw_text=long_text)
        assert len(result.key_details["raw_model_output"]) <= 500

    def test_no_raw_text_empty_key_details(self) -> None:
        result = _fallback_summary("Test", raw_text="")
        assert "raw_model_output" not in result.key_details


# ---------------------------------------------------------------------------
# CallSummarizer.summarize tests
# ---------------------------------------------------------------------------


class TestCallSummarizerSummarize:
    """Tests for CallSummarizer.summarize."""

    @pytest.mark.asyncio
    async def test_returns_call_summary(self, summarizer: CallSummarizer) -> None:
        """summarize returns a CallSummary instance."""
        json_response = _make_json_response(
            outcome="success",
            summary_text="Appointment booked for Tuesday at 3pm.",
            key_details={"booked_time": "Tuesday 3pm"},
        )
        mock_response = _make_openai_response(json_response)

        with patch.object(
            summarizer._client.chat.completions,
            "create",
            new=AsyncMock(return_value=mock_response),
        ):
            result = await summarizer.summarize(
                goal="Book a dentist appointment",
                transcript=[
                    TranscriptEntry(speaker="agent", text="Hello."),
                    TranscriptEntry(speaker="contact", text="Hi."),
                ],
            )

        assert isinstance(result, CallSummary)
        assert result.outcome == CallOutcome.SUCCESS
        assert result.summary_text == "Appointment booked for Tuesday at 3pm."
        assert result.key_details["booked_time"] == "Tuesday 3pm"

    @pytest.mark.asyncio
    async def test_empty_transcript_succeeds(
        self, summarizer: CallSummarizer
    ) -> None:
        """summarize handles an empty transcript gracefully."""
        json_response = _make_json_response(
            outcome="unknown", summary_text="No audio was recorded."
        )
        mock_response = _make_openai_response(json_response)

        with patch.object(
            summarizer._client.chat.completions,
            "create",
            new=AsyncMock(return_value=mock_response),
        ):
            result = await summarizer.summarize(goal="Test goal", transcript=[])

        assert isinstance(result, CallSummary)

    @pytest.mark.asyncio
    async def test_context_included_in_prompt(
        self, summarizer: CallSummarizer
    ) -> None:
        """The context parameter is included in the user prompt sent to OpenAI."""
        json_response = _make_json_response()
        captured: dict[str, Any] = {}

        async def capture_create(**kwargs: Any) -> MagicMock:
            captured.update(kwargs)
            return _make_openai_response(json_response)

        with patch.object(
            summarizer._client.chat.completions, "create", new=capture_create
        ):
            await summarizer.summarize(
                goal="Test",
                transcript=[],
                context="Patient: Jane Doe",
            )

        messages = captured.get("messages", [])
        user_messages = [m for m in messages if m["role"] == "user"]
        assert any("Jane Doe" in m["content"] for m in user_messages)

    @pytest.mark.asyncio
    async def test_goal_included_in_prompt(self, summarizer: CallSummarizer) -> None:
        """The goal is included in the user prompt sent to OpenAI."""
        json_response = _make_json_response()
        captured: dict[str, Any] = {}

        async def capture_create(**kwargs: Any) -> MagicMock:
            captured.update(kwargs)
            return _make_openai_response(json_response)

        with patch.object(
            summarizer._client.chat.completions, "create", new=capture_create
        ):
            await summarizer.summarize(
                goal="Book a dentist appointment for Tuesday",
                transcript=[],
            )

        messages = captured.get("messages", [])
        user_messages = [m for m in messages if m["role"] == "user"]
        assert any(
            "Book a dentist appointment for Tuesday" in m["content"]
            for m in user_messages
        )

    @pytest.mark.asyncio
    async def test_transcript_included_in_prompt(
        self,
        summarizer: CallSummarizer,
        sample_transcript: list[TranscriptEntry],
    ) -> None:
        """The formatted transcript is included in the user prompt."""
        json_response = _make_json_response()
        captured: dict[str, Any] = {}

        async def capture_create(**kwargs: Any) -> MagicMock:
            captured.update(kwargs)
            return _make_openai_response(json_response)

        with patch.object(
            summarizer._client.chat.completions, "create", new=capture_create
        ):
            await summarizer.summarize(goal="Test", transcript=sample_transcript)

        messages = captured.get("messages", [])
        user_messages = [m for m in messages if m["role"] == "user"]
        combined = " ".join(m["content"] for m in user_messages)
        assert "AGENT:" in combined
        assert "CONTACT:" in combined

    @pytest.mark.asyncio
    async def test_openai_error_raises_summarizer_error(
        self, summarizer: CallSummarizer
    ) -> None:
        """An OpenAI API error is re-raised as SummarizerError."""
        from openai import OpenAIError

        with patch.object(
            summarizer._client.chat.completions,
            "create",
            new=AsyncMock(side_effect=OpenAIError("connection error")),
        ):
            with pytest.raises(SummarizerError, match="connection error"):
                await summarizer.summarize(goal="Test", transcript=[])

    @pytest.mark.asyncio
    async def test_empty_openai_response_returns_fallback(
        self, summarizer: CallSummarizer
    ) -> None:
        """An empty OpenAI response (no choices) returns a fallback summary."""
        mock_response = _make_empty_openai_response()
        with patch.object(
            summarizer._client.chat.completions,
            "create",
            new=AsyncMock(return_value=mock_response),
        ):
            result = await summarizer.summarize(
                goal="Book dentist", transcript=[]
            )

        assert result.outcome == CallOutcome.UNKNOWN
        assert result.follow_up_required is True

    @pytest.mark.asyncio
    async def test_invalid_json_response_returns_fallback(
        self, summarizer: CallSummarizer
    ) -> None:
        """An invalid JSON response from OpenAI returns a fallback summary."""
        mock_response = _make_openai_response("I cannot produce a JSON summary.")
        with patch.object(
            summarizer._client.chat.completions,
            "create",
            new=AsyncMock(return_value=mock_response),
        ):
            result = await summarizer.summarize(
                goal="Book dentist", transcript=[]
            )

        assert result.outcome == CallOutcome.UNKNOWN

    @pytest.mark.asyncio
    async def test_max_tokens_forwarded(self, summarizer: CallSummarizer) -> None:
        """The configured max_tokens is forwarded to the OpenAI API."""
        json_response = _make_json_response()
        captured: dict[str, Any] = {}

        async def capture_create(**kwargs: Any) -> MagicMock:
            captured.update(kwargs)
            return _make_openai_response(json_response)

        with patch.object(
            summarizer._client.chat.completions, "create", new=capture_create
        ):
            await summarizer.summarize(goal="Test", transcript=[])

        assert captured["max_tokens"] == 512

    @pytest.mark.asyncio
    async def test_temperature_forwarded(self, summarizer: CallSummarizer) -> None:
        """The configured temperature is forwarded to the OpenAI API."""
        json_response = _make_json_response()
        captured: dict[str, Any] = {}

        async def capture_create(**kwargs: Any) -> MagicMock:
            captured.update(kwargs)
            return _make_openai_response(json_response)

        with patch.object(
            summarizer._client.chat.completions, "create", new=capture_create
        ):
            await summarizer.summarize(goal="Test", transcript=[])

        assert captured["temperature"] == pytest.approx(0.2)

    @pytest.mark.asyncio
    async def test_model_forwarded(self, summarizer: CallSummarizer) -> None:
        """The configured model is forwarded to the OpenAI API."""
        json_response = _make_json_response()
        captured: dict[str, Any] = {}

        async def capture_create(**kwargs: Any) -> MagicMock:
            captured.update(kwargs)
            return _make_openai_response(json_response)

        with patch.object(
            summarizer._client.chat.completions, "create", new=capture_create
        ):
            await summarizer.summarize(goal="Test", transcript=[])

        assert captured["model"] == "gpt-4o"

    @pytest.mark.asyncio
    async def test_system_prompt_is_first_message(self, summarizer: CallSummarizer) -> None:
        """The system prompt is always the first message in the API call."""
        json_response = _make_json_response()
        captured: dict[str, Any] = {}

        async def capture_create(**kwargs: Any) -> MagicMock:
            captured.update(kwargs)
            return _make_openai_response(json_response)

        with patch.object(
            summarizer._client.chat.completions, "create", new=capture_create
        ):
            await summarizer.summarize(goal="Test", transcript=[])

        messages = captured.get("messages", [])
        assert messages[0]["role"] == "system"
        assert "JSON" in messages[0]["content"]

    @pytest.mark.asyncio
    async def test_all_outcome_types(
        self, summarizer: CallSummarizer
    ) -> None:
        """All four CallOutcome values are correctly parsed."""
        for outcome_str, expected in [
            ("success", CallOutcome.SUCCESS),
            ("partial", CallOutcome.PARTIAL),
            ("failure", CallOutcome.FAILURE),
            ("unknown", CallOutcome.UNKNOWN),
        ]:
            json_response = _make_json_response(
                outcome=outcome_str, summary_text=f"Outcome: {outcome_str}"
            )
            mock_response = _make_openai_response(json_response)
            with patch.object(
                summarizer._client.chat.completions,
                "create",
                new=AsyncMock(return_value=mock_response),
            ):
                result = await summarizer.summarize(goal="Test", transcript=[])
            assert result.outcome == expected, f"Expected {expected} for {outcome_str!r}"

    @pytest.mark.asyncio
    async def test_no_context_does_not_include_context_block(
        self, summarizer: CallSummarizer
    ) -> None:
        """When context is None, the context block is absent from the prompt."""
        json_response = _make_json_response()
        captured: dict[str, Any] = {}

        async def capture_create(**kwargs: Any) -> MagicMock:
            captured.update(kwargs)
            return _make_openai_response(json_response)

        with patch.object(
            summarizer._client.chat.completions, "create", new=capture_create
        ):
            await summarizer.summarize(goal="Test", transcript=[], context=None)

        messages = captured.get("messages", [])
        user_content = " ".join(m["content"] for m in messages if m["role"] == "user")
        assert "Additional Context" not in user_content


# ---------------------------------------------------------------------------
# CallSummarizer.summarize_from_text tests
# ---------------------------------------------------------------------------


class TestCallSummarizerSummarizeFromText:
    """Tests for CallSummarizer.summarize_from_text."""

    @pytest.mark.asyncio
    async def test_returns_call_summary(self, summarizer: CallSummarizer) -> None:
        """summarize_from_text returns a CallSummary instance."""
        json_response = _make_json_response(
            outcome="success",
            summary_text="Call completed.",
        )
        mock_response = _make_openai_response(json_response)

        with patch.object(
            summarizer._client.chat.completions,
            "create",
            new=AsyncMock(return_value=mock_response),
        ):
            result = await summarizer.summarize_from_text(
                goal="Book a table",
                transcript_text="AGENT: Hello.\nCONTACT: Hi.",
            )

        assert isinstance(result, CallSummary)
        assert result.outcome == CallOutcome.SUCCESS

    @pytest.mark.asyncio
    async def test_transcript_text_in_prompt(self, summarizer: CallSummarizer) -> None:
        """The provided transcript text appears in the API prompt."""
        json_response = _make_json_response()
        captured: dict[str, Any] = {}

        async def capture_create(**kwargs: Any) -> MagicMock:
            captured.update(kwargs)
            return _make_openai_response(json_response)

        with patch.object(
            summarizer._client.chat.completions, "create", new=capture_create
        ):
            await summarizer.summarize_from_text(
                goal="Test",
                transcript_text="AGENT: Specific line.\nCONTACT: Another line.",
            )

        messages = captured.get("messages", [])
        user_content = " ".join(m["content"] for m in messages if m["role"] == "user")
        assert "Specific line" in user_content
        assert "Another line" in user_content

    @pytest.mark.asyncio
    async def test_empty_text_uses_placeholder(self, summarizer: CallSummarizer) -> None:
        """An empty transcript_text uses the placeholder string in the prompt."""
        json_response = _make_json_response()
        captured: dict[str, Any] = {}

        async def capture_create(**kwargs: Any) -> MagicMock:
            captured.update(kwargs)
            return _make_openai_response(json_response)

        with patch.object(
            summarizer._client.chat.completions, "create", new=capture_create
        ):
            await summarizer.summarize_from_text(
                goal="Test", transcript_text=""
            )

        messages = captured.get("messages", [])
        user_content = " ".join(m["content"] for m in messages if m["role"] == "user")
        assert "No transcript available" in user_content

    @pytest.mark.asyncio
    async def test_openai_error_raises_summarizer_error(
        self, summarizer: CallSummarizer
    ) -> None:
        """An OpenAI API error is re-raised as SummarizerError."""
        from openai import OpenAIError

        with patch.object(
            summarizer._client.chat.completions,
            "create",
            new=AsyncMock(side_effect=OpenAIError("timeout")),
        ):
            with pytest.raises(SummarizerError, match="timeout"):
                await summarizer.summarize_from_text(
                    goal="Test", transcript_text="AGENT: Hi."
                )

    @pytest.mark.asyncio
    async def test_context_included_in_prompt(
        self, summarizer: CallSummarizer
    ) -> None:
        """Context is included in the prompt when provided."""
        json_response = _make_json_response()
        captured: dict[str, Any] = {}

        async def capture_create(**kwargs: Any) -> MagicMock:
            captured.update(kwargs)
            return _make_openai_response(json_response)

        with patch.object(
            summarizer._client.chat.completions, "create", new=capture_create
        ):
            await summarizer.summarize_from_text(
                goal="Test",
                transcript_text="AGENT: Hi.",
                context="Extra context here",
            )

        messages = captured.get("messages", [])
        user_content = " ".join(m["content"] for m in messages if m["role"] == "user")
        assert "Extra context here" in user_content


# ---------------------------------------------------------------------------
# Settings defaults tests
# ---------------------------------------------------------------------------


class TestCallSummarizerDefaults:
    """Verify that CallSummarizer uses settings defaults correctly."""

    def test_uses_settings_model(self) -> None:
        """Without explicit model, the settings model is used."""
        s = CallSummarizer()
        from call_dispatch.config import get_settings
        cfg = get_settings()
        assert s._model == cfg.openai_model

    def test_uses_settings_max_tokens(self) -> None:
        """Without explicit max_tokens, the settings value is used."""
        s = CallSummarizer()
        from call_dispatch.config import get_settings
        cfg = get_settings()
        assert s._max_tokens == cfg.summarizer_max_tokens

    def test_uses_settings_temperature(self) -> None:
        """Without explicit temperature, the settings value is used."""
        s = CallSummarizer()
        from call_dispatch.config import get_settings
        cfg = get_settings()
        assert s._temperature == pytest.approx(cfg.summarizer_temperature)

    def test_explicit_overrides_settings(self) -> None:
        """Explicit constructor arguments override settings defaults."""
        s = CallSummarizer(
            model="gpt-3.5-turbo",
            max_tokens=128,
            temperature=0.9,
        )
        assert s._model == "gpt-3.5-turbo"
        assert s._max_tokens == 128
        assert s._temperature == pytest.approx(0.9)
