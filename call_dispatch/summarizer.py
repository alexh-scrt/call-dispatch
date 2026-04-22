"""Post-call summarizer module for call_dispatch.

Provides the :class:`CallSummarizer` class that uses OpenAI's GPT-4 to
generate a structured :class:`~call_dispatch.models.CallSummary` from a
completed call's transcript and goal.

The summarizer extracts:
- A high-level outcome (success / partial / failure / unknown)
- A human-readable narrative summary
- Structured key details as a JSON dictionary (e.g. booked time, confirmation number)
- Whether a follow-up action is required and any relevant notes

Usage example::

    summarizer = CallSummarizer(settings=get_settings())
    transcript = [
        TranscriptEntry(speaker="agent", text="Hello, I'm calling to book an appointment."),
        TranscriptEntry(speaker="contact", text="Sure, how about Tuesday at 3pm?"),
        TranscriptEntry(speaker="agent", text="Tuesday at 3pm works perfectly, thank you!"),
    ]
    summary = await summarizer.summarize(
        goal="Book a dentist appointment for Tuesday afternoon",
        transcript=transcript,
    )
    print(summary.outcome)        # CallOutcome.SUCCESS
    print(summary.summary_text)   # "Appointment booked for Tuesday at 3pm."
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from openai import AsyncOpenAI, OpenAIError

from call_dispatch.models import CallOutcome, CallSummary, TranscriptEntry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert call analyst. You will be given the goal of an outbound AI phone call
and its full transcript. Your job is to produce a structured JSON summary of the call.

You MUST respond with valid JSON and nothing else — no markdown fences, no commentary.

The JSON object must have exactly these fields:
{
  "outcome": "success" | "partial" | "failure" | "unknown",
  "summary_text": "<concise narrative of what happened>",
  "key_details": { <key-value pairs of important extracted information> },
  "follow_up_required": true | false,
  "follow_up_notes": "<notes if follow_up_required is true, else null>"
}

Definitions:
- "success": The call goal was fully achieved.
- "partial": The call goal was partially achieved.
- "failure": The call goal was not achieved.
- "unknown": The outcome cannot be determined from the transcript.

For key_details, extract any concrete facts that were established during the call,
such as: appointment times, prices, confirmation numbers, names, addresses, etc.
If no concrete details were extracted, use an empty object {}.

Be factual and concise. Do not invent information not present in the transcript.
"""

_USER_PROMPT_TEMPLATE = """\
Call Goal:
{goal}

{context_block}\
Call Transcript:
{transcript_text}

Provide the structured JSON summary for this call.
"""

_CONTEXT_BLOCK_TEMPLATE = """Additional Context:
{context}

"""

_EMPTY_TRANSCRIPT_PLACEHOLDER = "[No transcript available — the call did not connect or no audio was recorded.]"


# ---------------------------------------------------------------------------
# Transcript formatter
# ---------------------------------------------------------------------------


def format_transcript(transcript: list[TranscriptEntry]) -> str:
    """Convert a list of :class:`~call_dispatch.models.TranscriptEntry` objects
    into a readable plain-text transcript.

    Each line is formatted as ``SPEAKER: text``, where speaker labels are
    uppercased for readability (``AGENT`` or ``CONTACT``).

    Args:
        transcript: Ordered list of transcript entries.

    Returns:
        str: Plain-text representation of the conversation, or a placeholder
            string if the transcript is empty.
    """
    if not transcript:
        return _EMPTY_TRANSCRIPT_PLACEHOLDER

    lines: list[str] = []
    for entry in transcript:
        speaker_label = entry.speaker.upper()
        lines.append(f"{speaker_label}: {entry.text}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------


def _parse_outcome(raw: str) -> CallOutcome:
    """Parse an outcome string into a :class:`~call_dispatch.models.CallOutcome`.

    Args:
        raw: The raw string value from the JSON response.

    Returns:
        CallOutcome: The matching enum member, or :attr:`CallOutcome.UNKNOWN`
            if the value is not recognised.
    """
    mapping = {
        "success": CallOutcome.SUCCESS,
        "partial": CallOutcome.PARTIAL,
        "failure": CallOutcome.FAILURE,
        "unknown": CallOutcome.UNKNOWN,
    }
    return mapping.get(str(raw).lower().strip(), CallOutcome.UNKNOWN)


def _parse_summary_response(raw_content: str, goal: str) -> CallSummary:
    """Parse the raw JSON string returned by the model into a :class:`CallSummary`.

    Attempts strict JSON parsing first; falls back to a degraded summary
    on any parse error rather than propagating an exception.

    Args:
        raw_content: The raw text content from the OpenAI completion.
        goal: The call goal, used to construct a fallback summary.

    Returns:
        CallSummary: The parsed or fallback summary object.
    """
    content = raw_content.strip()

    # Strip potential markdown code fences the model may have added despite instructions
    if content.startswith("```"):
        lines = content.splitlines()
        # Drop the first line (```json or ```) and any trailing ```
        inner_lines = []
        in_block = False
        for line in lines:
            if line.startswith("```") and not in_block:
                in_block = True
                continue
            if line.startswith("```") and in_block:
                break
            if in_block:
                inner_lines.append(line)
        content = "\n".join(inner_lines).strip()

    try:
        data: dict[str, Any] = json.loads(content)
    except json.JSONDecodeError as exc:
        logger.warning(
            "Failed to parse summarizer JSON response: %s. Raw: %r", exc, raw_content[:200]
        )
        return _fallback_summary(goal, raw_text=raw_content)

    try:
        outcome = _parse_outcome(data.get("outcome", "unknown"))
        summary_text = str(data.get("summary_text", "")).strip()
        key_details = data.get("key_details", {})
        if not isinstance(key_details, dict):
            key_details = {}
        follow_up_required = bool(data.get("follow_up_required", False))
        follow_up_notes_raw = data.get("follow_up_notes")
        follow_up_notes: Optional[str] = None
        if follow_up_notes_raw and str(follow_up_notes_raw).strip().lower() not in (
            "null",
            "none",
            "",
        ):
            follow_up_notes = str(follow_up_notes_raw).strip()

        if not summary_text:
            summary_text = f"Call completed for goal: {goal}"

        return CallSummary(
            outcome=outcome,
            summary_text=summary_text,
            key_details=key_details,
            follow_up_required=follow_up_required,
            follow_up_notes=follow_up_notes,
        )
    except Exception as exc:
        logger.warning("Error constructing CallSummary from parsed data: %s", exc)
        return _fallback_summary(goal, raw_text=raw_content)


def _fallback_summary(goal: str, raw_text: str = "") -> CallSummary:
    """Construct a minimal fallback :class:`CallSummary` when parsing fails.

    Args:
        goal: The call goal to include in the fallback text.
        raw_text: The raw model output, included in key_details for debugging.

    Returns:
        CallSummary: A summary with ``outcome=UNKNOWN`` and a generic description.
    """
    details: dict[str, Any] = {}
    if raw_text:
        details["raw_model_output"] = raw_text[:500]
    return CallSummary(
        outcome=CallOutcome.UNKNOWN,
        summary_text=f"Summary generation failed. Call goal was: {goal}",
        key_details=details,
        follow_up_required=True,
        follow_up_notes="Manual review required — automated summary could not be parsed.",
    )


# ---------------------------------------------------------------------------
# CallSummarizer
# ---------------------------------------------------------------------------


class CallSummarizer:
    """Generates structured post-call summaries using OpenAI GPT-4.

    Accepts the call goal, optional context, and full transcript and produces
    a :class:`~call_dispatch.models.CallSummary` containing the outcome,
    narrative summary, and extracted key details.

    Args:
        openai_api_key: OpenAI API key.  Falls back to the configured value.
        model: OpenAI model identifier.  Falls back to the configured model.
        max_tokens: Maximum tokens for the summary response.
        temperature: Sampling temperature for the summary (lower = more consistent).
    """

    def __init__(
        self,
        *,
        openai_api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> None:
        from call_dispatch.config import get_settings

        cfg = get_settings()

        self._model = model or cfg.openai_model
        self._max_tokens = max_tokens if max_tokens is not None else cfg.summarizer_max_tokens
        self._temperature = (
            temperature if temperature is not None else cfg.summarizer_temperature
        )
        self._client = AsyncOpenAI(api_key=openai_api_key or cfg.openai_api_key)

        logger.debug(
            "CallSummarizer initialised model=%s max_tokens=%d temperature=%.2f",
            self._model,
            self._max_tokens,
            self._temperature,
        )

    async def summarize(
        self,
        goal: str,
        transcript: list[TranscriptEntry],
        context: Optional[str] = None,
    ) -> CallSummary:
        """Generate a structured summary for a completed call.

        Args:
            goal: The natural language call goal.
            transcript: Ordered list of :class:`~call_dispatch.models.TranscriptEntry`
                objects from the call.
            context: Optional supplementary context that was provided at dispatch time.

        Returns:
            CallSummary: The structured summary with outcome and key details.

        Raises:
            SummarizerError: If the OpenAI API call fails unrecoverably.
        """
        transcript_text = format_transcript(transcript)
        context_block = (
            _CONTEXT_BLOCK_TEMPLATE.format(context=context.strip())
            if context and context.strip()
            else ""
        )
        user_prompt = _USER_PROMPT_TEMPLATE.format(
            goal=goal.strip(),
            context_block=context_block,
            transcript_text=transcript_text,
        )

        logger.debug(
            "Requesting call summary for goal=%r transcript_entries=%d",
            goal[:60],
            len(transcript),
        )

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                response_format={"type": "json_object"},
            )
        except OpenAIError as exc:
            logger.error("OpenAI API error during summarization: %s", exc)
            raise SummarizerError(f"OpenAI API call failed: {exc}") from exc

        choice = response.choices[0] if response.choices else None
        if choice is None or not choice.message.content:
            logger.warning("OpenAI returned empty response for summarization")
            return _fallback_summary(goal)

        raw_content = choice.message.content
        logger.debug("Raw summarizer response: %r", raw_content[:300])

        summary = _parse_summary_response(raw_content, goal=goal)
        logger.info(
            "Summary generated outcome=%s follow_up=%s",
            summary.outcome.value,
            summary.follow_up_required,
        )
        return summary

    async def summarize_from_text(
        self,
        goal: str,
        transcript_text: str,
        context: Optional[str] = None,
    ) -> CallSummary:
        """Generate a summary from a pre-formatted transcript string.

        This variant accepts a plain-text transcript instead of a list of
        :class:`~call_dispatch.models.TranscriptEntry` objects, useful when
        the transcript has already been formatted or comes from an external source.

        Args:
            goal: The natural language call goal.
            transcript_text: Pre-formatted transcript text (e.g. "AGENT: ...\nCONTACT: ...").
            context: Optional supplementary context.

        Returns:
            CallSummary: The structured summary.

        Raises:
            SummarizerError: If the OpenAI API call fails unrecoverably.
        """
        # Create a minimal proxy entry so we can reuse the main summarize path
        context_block = (
            _CONTEXT_BLOCK_TEMPLATE.format(context=context.strip())
            if context and context.strip()
            else ""
        )
        user_prompt = _USER_PROMPT_TEMPLATE.format(
            goal=goal.strip(),
            context_block=context_block,
            transcript_text=transcript_text or _EMPTY_TRANSCRIPT_PLACEHOLDER,
        )

        logger.debug("Requesting call summary (text mode) for goal=%r", goal[:60])

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                response_format={"type": "json_object"},
            )
        except OpenAIError as exc:
            logger.error("OpenAI API error during text summarization: %s", exc)
            raise SummarizerError(f"OpenAI API call failed: {exc}") from exc

        choice = response.choices[0] if response.choices else None
        if choice is None or not choice.message.content:
            return _fallback_summary(goal)

        return _parse_summary_response(choice.message.content, goal=goal)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class SummarizerError(Exception):
    """Raised when the :class:`CallSummarizer` cannot produce a summary.

    Wraps underlying :class:`openai.OpenAIError` instances so callers can
    catch a single, application-specific exception type.
    """
