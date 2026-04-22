"""Unit tests for call_dispatch.agent - AI conversational agent.

All OpenAI API calls are mocked using respx / unittest.mock so that tests
are fully offline and deterministic.

Test coverage includes:
- System prompt construction (goal, context, no-context variants)
- Opening utterance generation
- Next-utterance generation and history management
- History loading from transcript entries
- History reset
- from_record factory method
- Error handling: OpenAI failures, empty responses
- Turn count tracking
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from call_dispatch.agent import AgentError, ConversationAgent
from call_dispatch.models import AgentMessage, TranscriptEntry


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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def patch_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure settings can be loaded in every test by injecting env vars."""
    for key, val in _FAKE_ENV.items():
        monkeypatch.setenv(key, val)
    # Clear cached settings so each test starts fresh
    from call_dispatch.config import get_settings
    get_settings.cache_clear()


@pytest.fixture
def agent() -> ConversationAgent:
    """Provide a ConversationAgent with a simple goal for testing."""
    return ConversationAgent(
        goal="Book a dentist appointment for Tuesday afternoon",
        context="Patient name: Jane Doe, DOB: 1990-05-15",
        openai_api_key="sk-test-key",
        model="gpt-4o",
        max_tokens=256,
        temperature=0.4,
    )


@pytest.fixture
def agent_no_context() -> ConversationAgent:
    """Provide a ConversationAgent without context."""
    return ConversationAgent(
        goal="Check if the restaurant is open on Sunday",
        openai_api_key="sk-test-key",
    )


# ---------------------------------------------------------------------------
# Initialisation tests
# ---------------------------------------------------------------------------


class TestAgentInit:
    """Tests for ConversationAgent construction."""

    def test_goal_is_stored(self, agent: ConversationAgent) -> None:
        assert agent.goal == "Book a dentist appointment for Tuesday afternoon"

    def test_initial_turn_count_is_zero(self, agent: ConversationAgent) -> None:
        assert agent.turn_count == 0

    def test_initial_history_is_empty(self, agent: ConversationAgent) -> None:
        assert agent.history == []

    def test_goal_is_stripped(self) -> None:
        a = ConversationAgent(
            goal="  Book something   ",
            openai_api_key="sk-test",
        )
        assert a.goal == "Book something"

    def test_context_in_system_prompt(self, agent: ConversationAgent) -> None:
        """When context is provided, it appears in the system prompt."""
        assert "Jane Doe" in agent._system_prompt
        assert "1990-05-15" in agent._system_prompt

    def test_no_context_system_prompt_clean(self, agent_no_context: ConversationAgent) -> None:
        """When no context is given, the context block is absent."""
        assert "Additional context" not in agent_no_context._system_prompt

    def test_goal_in_system_prompt(self, agent: ConversationAgent) -> None:
        """The goal appears verbatim in the system prompt."""
        assert "Book a dentist appointment for Tuesday afternoon" in agent._system_prompt

    def test_uses_settings_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without explicit overrides, agent uses settings values."""
        from call_dispatch.config import get_settings
        get_settings.cache_clear()
        a = ConversationAgent(goal="Test goal")
        from call_dispatch.config import get_settings as gs
        cfg = gs()
        assert a._model == cfg.openai_model
        assert a._max_tokens == cfg.agent_max_tokens
        assert a._temperature == cfg.agent_temperature


# ---------------------------------------------------------------------------
# Opening utterance tests
# ---------------------------------------------------------------------------


class TestGetOpeningUtterance:
    """Tests for get_opening_utterance."""

    @pytest.mark.asyncio
    async def test_returns_utterance_string(self, agent: ConversationAgent) -> None:
        """get_opening_utterance returns a non-empty string."""
        mock_response = _make_openai_response(
            "Hello, I'm calling to schedule a dentist appointment."
        )
        with patch.object(
            agent._client.chat.completions,
            "create",
            new=AsyncMock(return_value=mock_response),
        ):
            result = await agent.get_opening_utterance()

        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_opening_increments_turn_count(self, agent: ConversationAgent) -> None:
        """get_opening_utterance increments the turn counter."""
        mock_response = _make_openai_response("Hello, I'm calling to book an appointment.")
        with patch.object(
            agent._client.chat.completions,
            "create",
            new=AsyncMock(return_value=mock_response),
        ):
            await agent.get_opening_utterance()

        assert agent.turn_count == 1

    @pytest.mark.asyncio
    async def test_opening_populates_history(self, agent: ConversationAgent) -> None:
        """After get_opening_utterance, history contains user prompt + assistant reply."""
        mock_response = _make_openai_response("Hello!")
        with patch.object(
            agent._client.chat.completions,
            "create",
            new=AsyncMock(return_value=mock_response),
        ):
            await agent.get_opening_utterance()

        # Should have 2 entries: the user prompt and the assistant response
        assert len(agent._history) == 2
        assert agent._history[-1]["role"] == "assistant"
        assert agent._history[-1]["content"] == "Hello!"

    @pytest.mark.asyncio
    async def test_response_is_stripped(self, agent: ConversationAgent) -> None:
        """Leading/trailing whitespace is stripped from the utterance."""
        mock_response = _make_openai_response("  Hello there.  ")
        with patch.object(
            agent._client.chat.completions,
            "create",
            new=AsyncMock(return_value=mock_response),
        ):
            result = await agent.get_opening_utterance()

        assert result == "Hello there."

    @pytest.mark.asyncio
    async def test_openai_error_raises_agent_error(self, agent: ConversationAgent) -> None:
        """An OpenAI API error is re-raised as AgentError."""
        from openai import OpenAIError

        with patch.object(
            agent._client.chat.completions,
            "create",
            new=AsyncMock(side_effect=OpenAIError("rate limit")),
        ):
            with pytest.raises(AgentError, match="rate limit"):
                await agent.get_opening_utterance()

    @pytest.mark.asyncio
    async def test_openai_error_does_not_corrupt_history(self, agent: ConversationAgent) -> None:
        """After an OpenAI error, history is rolled back to its pre-call state."""
        from openai import OpenAIError

        with patch.object(
            agent._client.chat.completions,
            "create",
            new=AsyncMock(side_effect=OpenAIError("server error")),
        ):
            with pytest.raises(AgentError):
                await agent.get_opening_utterance()

        assert agent._history == []
        assert agent.turn_count == 0

    @pytest.mark.asyncio
    async def test_empty_response_raises_agent_error(self, agent: ConversationAgent) -> None:
        """An empty OpenAI response raises AgentError."""
        mock_response = _make_empty_openai_response()
        with patch.object(
            agent._client.chat.completions,
            "create",
            new=AsyncMock(return_value=mock_response),
        ):
            with pytest.raises(AgentError, match="empty"):
                await agent.get_opening_utterance()

    @pytest.mark.asyncio
    async def test_none_content_raises_agent_error(self, agent: ConversationAgent) -> None:
        """A response with None content raises AgentError."""
        mock_response = _make_openai_response("")
        mock_response.choices[0].message.content = None
        with patch.object(
            agent._client.chat.completions,
            "create",
            new=AsyncMock(return_value=mock_response),
        ):
            with pytest.raises(AgentError):
                await agent.get_opening_utterance()


# ---------------------------------------------------------------------------
# Next utterance tests
# ---------------------------------------------------------------------------


class TestGetNextUtterance:
    """Tests for get_next_utterance."""

    @pytest.mark.asyncio
    async def test_returns_utterance_string(self, agent: ConversationAgent) -> None:
        """get_next_utterance returns a non-empty string."""
        mock_response = _make_openai_response("Tuesday afternoon works for me.")
        with patch.object(
            agent._client.chat.completions,
            "create",
            new=AsyncMock(return_value=mock_response),
        ):
            result = await agent.get_next_utterance("What time works for you?")

        assert result == "Tuesday afternoon works for me."

    @pytest.mark.asyncio
    async def test_increments_turn_count(self, agent: ConversationAgent) -> None:
        """Each successful call increments the turn counter."""
        mock_response = _make_openai_response("I'd like Tuesday please.")
        with patch.object(
            agent._client.chat.completions,
            "create",
            new=AsyncMock(return_value=mock_response),
        ):
            await agent.get_next_utterance("What time works?")
            await agent.get_next_utterance("Any preferences?")

        assert agent.turn_count == 2

    @pytest.mark.asyncio
    async def test_contact_message_included_in_messages(self, agent: ConversationAgent) -> None:
        """The contact's message is forwarded to the OpenAI API."""
        mock_response = _make_openai_response("3pm please.")
        captured_messages: list[Any] = []

        async def capture_create(**kwargs: Any) -> MagicMock:
            captured_messages.extend(kwargs.get("messages", []))
            return mock_response

        with patch.object(
            agent._client.chat.completions, "create", new=capture_create
        ):
            await agent.get_next_utterance("What time do you prefer?")

        # The captured messages should include the user prompt with the contact's message
        user_messages = [m for m in captured_messages if m["role"] == "user"]
        assert any("What time do you prefer?" in m["content"] for m in user_messages)

    @pytest.mark.asyncio
    async def test_history_grows_with_each_turn(self, agent: ConversationAgent) -> None:
        """Internal history accumulates entries for each exchange."""
        for i in range(3):
            mock_response = _make_openai_response(f"Response {i}")
            with patch.object(
                agent._client.chat.completions,
                "create",
                new=AsyncMock(return_value=mock_response),
            ):
                await agent.get_next_utterance(f"Contact message {i}")

        # Each turn adds 2 entries: user prompt + assistant reply
        assert len(agent._history) == 6

    @pytest.mark.asyncio
    async def test_system_prompt_always_included(self, agent: ConversationAgent) -> None:
        """The system prompt is always the first message sent to OpenAI."""
        mock_response = _make_openai_response("Sure.")
        captured_messages: list[Any] = []

        async def capture_create(**kwargs: Any) -> MagicMock:
            captured_messages.extend(kwargs.get("messages", []))
            return mock_response

        with patch.object(
            agent._client.chat.completions, "create", new=capture_create
        ):
            await agent.get_next_utterance("Hello")

        assert captured_messages[0]["role"] == "system"
        assert agent._system_prompt in captured_messages[0]["content"]

    @pytest.mark.asyncio
    async def test_openai_error_raises_agent_error(self, agent: ConversationAgent) -> None:
        """An OpenAI API error is re-raised as AgentError."""
        from openai import OpenAIError

        with patch.object(
            agent._client.chat.completions,
            "create",
            new=AsyncMock(side_effect=OpenAIError("timeout")),
        ):
            with pytest.raises(AgentError, match="timeout"):
                await agent.get_next_utterance("Are you there?")

    @pytest.mark.asyncio
    async def test_error_rolls_back_history(self, agent: ConversationAgent) -> None:
        """After an error, history is identical to its pre-call state."""
        from openai import OpenAIError

        # Add one successful turn first
        mock_ok = _make_openai_response("First reply.")
        with patch.object(
            agent._client.chat.completions,
            "create",
            new=AsyncMock(return_value=mock_ok),
        ):
            await agent.get_next_utterance("First contact message")

        history_after_success = list(agent._history)

        with patch.object(
            agent._client.chat.completions,
            "create",
            new=AsyncMock(side_effect=OpenAIError("error")),
        ):
            with pytest.raises(AgentError):
                await agent.get_next_utterance("Second contact message")

        # History should be the same as after the first successful turn
        assert agent._history == history_after_success


# ---------------------------------------------------------------------------
# History management tests
# ---------------------------------------------------------------------------


class TestHistoryManagement:
    """Tests for add_transcript_to_history and reset."""

    def test_add_transcript_to_history_loads_entries(self, agent: ConversationAgent) -> None:
        """add_transcript_to_history correctly loads transcript entries."""
        entries = [
            TranscriptEntry(speaker="agent", text="Hello, I'm calling about X."),
            TranscriptEntry(speaker="contact", text="Yes, how can I help?"),
            TranscriptEntry(speaker="agent", text="I'd like to book an appointment."),
        ]
        agent.add_transcript_to_history(entries)

        assert len(agent._history) == 3
        assert agent._history[0]["role"] == "assistant"
        assert agent._history[0]["content"] == "Hello, I'm calling about X."
        assert agent._history[1]["role"] == "user"
        assert agent._history[1]["content"] == "Yes, how can I help?"
        assert agent._history[2]["role"] == "assistant"

    def test_add_transcript_counts_agent_turns(self, agent: ConversationAgent) -> None:
        """Turn count reflects the number of agent entries in the loaded transcript."""
        entries = [
            TranscriptEntry(speaker="agent", text="Hello."),
            TranscriptEntry(speaker="contact", text="Hi."),
            TranscriptEntry(speaker="agent", text="How are you?"),
        ]
        agent.add_transcript_to_history(entries)
        assert agent.turn_count == 2  # Two agent utterances

    def test_add_transcript_clears_existing_history(self, agent: ConversationAgent) -> None:
        """Loading a new transcript replaces any existing history."""
        agent._history = [{"role": "user", "content": "old"}]
        agent._turn_count = 5

        entries = [TranscriptEntry(speaker="agent", text="Fresh start.")]
        agent.add_transcript_to_history(entries)

        assert len(agent._history) == 1
        assert agent.turn_count == 1

    def test_add_empty_transcript_clears_history(self, agent: ConversationAgent) -> None:
        """Loading an empty transcript clears existing history."""
        agent._history = [{"role": "user", "content": "old"}]
        agent._turn_count = 3

        agent.add_transcript_to_history([])

        assert agent._history == []
        assert agent.turn_count == 0

    def test_reset_clears_history_and_turn_count(self, agent: ConversationAgent) -> None:
        """reset() clears history and sets turn_count back to 0."""
        agent._history = [{"role": "user", "content": "msg"}]
        agent._turn_count = 3

        agent.reset()

        assert agent._history == []
        assert agent.turn_count == 0

    def test_history_property_returns_agent_messages(self, agent: ConversationAgent) -> None:
        """The history property returns AgentMessage objects."""
        agent._history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        history = agent.history
        assert len(history) == 2
        assert all(isinstance(m, AgentMessage) for m in history)
        assert history[0].role == "user"
        assert history[1].role == "assistant"


# ---------------------------------------------------------------------------
# from_record factory tests
# ---------------------------------------------------------------------------


class TestFromRecord:
    """Tests for the from_record class method factory."""

    def test_from_record_without_transcript(self) -> None:
        """from_record with no transcript creates an agent with empty history."""
        agent = ConversationAgent.from_record(
            goal="Get insurance quote",
            context=None,
            transcript=None,
            openai_api_key="sk-test",
        )
        assert agent.goal == "Get insurance quote"
        assert agent._history == []
        assert agent.turn_count == 0

    def test_from_record_with_transcript(self) -> None:
        """from_record with a transcript pre-populates the agent history."""
        transcript = [
            TranscriptEntry(speaker="agent", text="Hello."),
            TranscriptEntry(speaker="contact", text="Hi there."),
        ]
        agent = ConversationAgent.from_record(
            goal="Check availability",
            transcript=transcript,
            openai_api_key="sk-test",
        )
        assert len(agent._history) == 2
        assert agent.turn_count == 1

    def test_from_record_passes_context(self) -> None:
        """from_record correctly sets context in the system prompt."""
        agent = ConversationAgent.from_record(
            goal="Book table",
            context="Party of 4",
            openai_api_key="sk-test",
        )
        assert "Party of 4" in agent._system_prompt

    def test_from_record_passes_model_override(self) -> None:
        """from_record forwards model override to the agent."""
        agent = ConversationAgent.from_record(
            goal="Test goal",
            model="gpt-3.5-turbo",
            openai_api_key="sk-test",
        )
        assert agent._model == "gpt-3.5-turbo"


# ---------------------------------------------------------------------------
# API parameter forwarding tests
# ---------------------------------------------------------------------------


class TestApiParameters:
    """Verify that configuration parameters are forwarded to the OpenAI API."""

    @pytest.mark.asyncio
    async def test_max_tokens_forwarded(self, agent: ConversationAgent) -> None:
        """The configured max_tokens is passed to the OpenAI API call."""
        mock_response = _make_openai_response("Response.")
        captured: dict[str, Any] = {}

        async def capture_create(**kwargs: Any) -> MagicMock:
            captured.update(kwargs)
            return mock_response

        with patch.object(agent._client.chat.completions, "create", new=capture_create):
            await agent.get_next_utterance("Hello")

        assert captured["max_tokens"] == 256

    @pytest.mark.asyncio
    async def test_temperature_forwarded(self, agent: ConversationAgent) -> None:
        """The configured temperature is passed to the OpenAI API call."""
        mock_response = _make_openai_response("Response.")
        captured: dict[str, Any] = {}

        async def capture_create(**kwargs: Any) -> MagicMock:
            captured.update(kwargs)
            return mock_response

        with patch.object(agent._client.chat.completions, "create", new=capture_create):
            await agent.get_next_utterance("Hello")

        assert captured["temperature"] == pytest.approx(0.4)

    @pytest.mark.asyncio
    async def test_model_forwarded(self, agent: ConversationAgent) -> None:
        """The configured model is passed to the OpenAI API call."""
        mock_response = _make_openai_response("Response.")
        captured: dict[str, Any] = {}

        async def capture_create(**kwargs: Any) -> MagicMock:
            captured.update(kwargs)
            return mock_response

        with patch.object(agent._client.chat.completions, "create", new=capture_create):
            await agent.get_next_utterance("Hello")

        assert captured["model"] == "gpt-4o"


# ---------------------------------------------------------------------------
# Multi-turn conversation tests
# ---------------------------------------------------------------------------


class TestMultiTurnConversation:
    """Integration-style tests for multi-turn conversation flow."""

    @pytest.mark.asyncio
    async def test_conversation_history_accumulates(self, agent: ConversationAgent) -> None:
        """History grows correctly over multiple turns."""
        responses = [
            "Hello, I'm calling to book a dentist appointment.",
            "Tuesday at 3pm would be perfect.",
            "Jane Doe, date of birth May 15th 1990.",
            "Thank you so much, goodbye!",
        ]
        contacts = [
            "Hello, how can I help?",
            "What time works for you?",
            "Can I get your name and date of birth?",
            "Great, you're booked in!",
        ]

        for contact_msg, agent_reply in zip(contacts, responses):
            mock_response = _make_openai_response(agent_reply)
            with patch.object(
                agent._client.chat.completions,
                "create",
                new=AsyncMock(return_value=mock_response),
            ):
                result = await agent.get_next_utterance(contact_msg)
                assert result == agent_reply

        assert agent.turn_count == 4
        # History: 4 turns × 2 entries each (user prompt + assistant reply)
        assert len(agent._history) == 8

    @pytest.mark.asyncio
    async def test_history_sent_to_api_grows(self, agent: ConversationAgent) -> None:
        """The messages list sent to OpenAI grows with each turn."""
        message_counts: list[int] = []

        async def capture_create(**kwargs: Any) -> MagicMock:
            message_counts.append(len(kwargs.get("messages", [])))
            return _make_openai_response("reply")

        for i in range(3):
            with patch.object(
                agent._client.chat.completions, "create", new=capture_create
            ):
                await agent.get_next_utterance(f"message {i}")

        # First call: system + 1 user = 2
        # Second call: system + 2 user + 1 assistant + 1 user = 5 ... growing by 2 each time
        assert message_counts[0] < message_counts[1] < message_counts[2]
