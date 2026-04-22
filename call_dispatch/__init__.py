"""call_dispatch - AI agent phone call dispatcher.

A lightweight, locally-runnable system for automated outbound calls using
Twilio for telephony, OpenAI GPT-4 for conversation logic, and Deepgram
for real-time transcription.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("call_dispatch")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["__version__"]
