"""AI agent module for call_dispatch.

Provides the :class:`ConversationAgent` class that uses OpenAI's GPT-4 to
generate contextually appropriate utterances for the AI caller based on the
call goal, optional context, and the running transcript history.

The agent maintains an internal conversation history in OpenAI chat format
and produces short, natural-sounding utterances designed to advance the call
toward the configured goal.

Usage example::

    agent = ConversationAgent(
        goal="Book a dentist appointment for Tuesday afternoon",
        context="Patient name: Jane Doe, DOB: 1990-05-15",
        settings=get_settings(),
    )
    utterance = await agent.get_next_utterance(
        last_contact_message="Sure, what time works for you?"
    )
    print(utterance)  # "Tuesday afternoon would be ideal, around 3pm if possible."
"""

from __future__ import annotations

import logging
from typing import Optional

from openai import AsyncOpenAI, OpenAIError

from call_dispatch.models import AgentMessage, TranscriptEntry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt constants
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
You are a professional, polite AI phone agent making an outbound call on behalf of a user.

Your goal for this call:
{goal}

{context_block}\
Guidelines:
- Be concise and natural — speak as a human caller would on the phone.
- Keep each utterance SHORT (1-3 sentences maximum).
- Stay focused on achieving the stated goal.
- Be polite and professional at all times.
- If you have achieved the goal, say so clearly and wrap up the call gracefully.
- If the contact cannot help or the goal is unachievable, politely thank them and end the call.
- Never reveal that you are an AI unless directly and explicitly asked.
- Do not add conversational filler like "Certainly!" or "Absolutely!" at the start of every line.
- Respond ONLY with what you would say aloud — no stage directions, no JSON, no markdown.
"""

_CONTEXT_BLOCK_TEMPLATE = """Additional context:
{context}

"""

_OPENING_PROMPT = """\
The phone has just been answered. Generate your opening statement to begin \
working toward the goal. Start the conversation naturally.
"""

_CONTINUATION_PROMPT_TEMPLATE = """\
The contact just said: "{contact_message}"

Generate your next response to continue progressing toward the goal.
"""


# ---------------------------------------------------------------------------
# ConversationAgent
# ---------------------------------------------------------------------------


class ConversationAgent:
    """GPT-4 powered conversational agent for outbound phone calls.

    Maintains an internal OpenAI chat history and generates short, natural
    utterances to advance the call toward the configured goal.

    Args:
        goal: Natural language description of what the agent should accomplish.
        context: Optional supplementary context (caller name, reference numbers, etc.).
        openai_api_key: OpenAI API key.  If not provided, the value from
            :func:`~call_dispatch.config.get_settings` is used.
        model: OpenAI model identifier.  Defaults to the configured model.
        max_tokens: Maximum tokens per response.  Defaults to the configured value.
        temperature: Sampling temperature.  Defaults to the configured value.
    """

    def __init__(
        self,
        goal: str,
        context: Optional[str] = None,
        *,
        openai_api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> None:
        from call_dispatch.config import get_settings

        cfg = get_settings()

        self._goal = goal.strip()
        self._context = context.strip() if context else None
        self._model = model or cfg.openai_model
        self._max_tokens = max_tokens if max_tokens is not None else cfg.agent_max_tokens
        self._temperature = temperature if temperature is not None else cfg.agent_temperature

        self._client = AsyncOpenAI(api_key=openai_api_key or cfg.openai_api_key)

        # Build the system prompt once at construction time
        context_block = (
            _CONTEXT_BLOCK_TEMPLATE.format(context=self._context)
            if self._context
            else ""
        )
        self._system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            goal=self._goal,
            context_block=context_block,
        )

        # OpenAI chat history (list of AgentMessage dicts for the API call)
        self._history: list[dict[str, str]] = []
        self._turn_count: int = 0

        logger.debug(
            "ConversationAgent initialised goal=%r model=%s",
            self._goal[:60],
            self._model,
        )

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def goal(self) -> str:
        """The call goal this agent is working toward."""
        return self._goal

    @property
    def turn_count(self) -> int:
        """Number of agent utterances generated so far."""
        return self._turn_count

    @property
    def history(self) -> list[AgentMessage]:
        """Read-only view of the current conversation history as AgentMessage objects."""
        return [AgentMessage(role=m["role"], content=m["content"]) for m in self._history]

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def get_opening_utterance(self) -> str:
        """Generate the agent's opening statement when the call is first answered.

        This is a convenience wrapper around :meth:`get_next_utterance` that
        uses an internal opening prompt instead of a contact message.

        Returns:
            str: The opening utterance to speak to the contact.

        Raises:
            AgentError: If the OpenAI API call fails.
        """
        logger.debug("Generating opening utterance")
        return await self._generate(_OPENING_PROMPT, speaker="user")

    async def get_next_utterance(self, last_contact_message: str) -> str:
        """Generate the agent's next utterance in response to the contact's message.

        Appends the contact's message to the history, calls the OpenAI API,
        and appends the agent's response before returning it.

        Args:
            last_contact_message: The most recent utterance from the contact
                (human on the other end of the call).

        Returns:
            str: The agent's next utterance.

        Raises:
            AgentError: If the OpenAI API call fails or returns an empty response.
        """
        prompt = _CONTINUATION_PROMPT_TEMPLATE.format(
            contact_message=last_contact_message.strip()
        )
        logger.debug(
            "Generating utterance (turn %d) in response to: %r",
            self._turn_count + 1,
            last_contact_message[:80],
        )
        return await self._generate(prompt, speaker="user")

    def add_transcript_to_history(self, entries: list[TranscriptEntry]) -> None:
        """Populate the internal history from an existing transcript.

        Useful when reconstructing an agent's state after a process restart or
        when replaying a partial conversation.

        Args:
            entries: Ordered list of :class:`~call_dispatch.models.TranscriptEntry`
                objects representing the conversation so far.
        """
        self._history.clear()
        self._turn_count = 0
        for entry in entries:
            role = "assistant" if entry.speaker == "agent" else "user"
            self._history.append({"role": role, "content": entry.text})
            if entry.speaker == "agent":
                self._turn_count += 1
        logger.debug(
            "Loaded %d transcript entries into agent history", len(entries)
        )

    def reset(self) -> None:
        """Clear the conversation history and reset the turn counter.

        After calling this, the agent behaves as if it has just been created
        with the same goal and context.
        """
        self._history.clear()
        self._turn_count = 0
        logger.debug("ConversationAgent history reset")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _generate(self, user_prompt: str, speaker: str = "user") -> str:
        """Append *user_prompt* to history and call the OpenAI chat completions API.

        Args:
            user_prompt: The user-role message to send to the model.
            speaker: Role label for history bookkeeping (always "user" here).

        Returns:
            str: The stripped text content of the model's response.

        Raises:
            AgentError: On API failure or empty response content.
        """
        self._history.append({"role": speaker, "content": user_prompt})

        messages = [
            {"role": "system", "content": self._system_prompt},
            *self._history,
        ]

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
        except OpenAIError as exc:
            logger.error("OpenAI API error during agent generation: %s", exc)
            # Remove the prompt we just appended so history remains consistent
            self._history.pop()
            raise AgentError(f"OpenAI API call failed: {exc}") from exc

        choice = response.choices[0] if response.choices else None
        if choice is None or not choice.message.content:
            self._history.pop()
            raise AgentError("OpenAI returned an empty response for agent utterance.")

        utterance = choice.message.content.strip()

        # Append the assistant's response to history
        self._history.append({"role": "assistant", "content": utterance})
        self._turn_count += 1

        logger.debug("Agent utterance (turn %d): %r", self._turn_count, utterance[:100])
        return utterance

    # ------------------------------------------------------------------
    # Class-level factory
    # ------------------------------------------------------------------

    @classmethod
    def from_record(
        cls,
        goal: str,
        context: Optional[str] = None,
        transcript: Optional[list[TranscriptEntry]] = None,
        *,
        openai_api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> "ConversationAgent":
        """Create a :class:`ConversationAgent` and pre-populate its history.

        Convenience factory that builds an agent and loads an existing
        transcript so the agent has full context when resuming a call.

        Args:
            goal: The call goal.
            context: Optional supplementary context.
            transcript: Optional ordered list of transcript entries to load.
            openai_api_key: Optional API key override.
            model: Optional model override.
            max_tokens: Optional max tokens override.
            temperature: Optional temperature override.

        Returns:
            ConversationAgent: Fully initialised agent with history loaded.
        """
        agent = cls(
            goal=goal,
            context=context,
            openai_api_key=openai_api_key,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if transcript:
            agent.add_transcript_to_history(transcript)
        return agent


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class AgentError(Exception):
    """Raised when the :class:`ConversationAgent` cannot generate an utterance.

    This wraps underlying :class:`openai.OpenAIError` instances and other
    failure conditions so callers can catch a single, application-specific
    exception type.
    """
