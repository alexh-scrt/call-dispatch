# call_dispatch

A lightweight, locally-runnable AI agent phone call dispatcher for automated outbound calls. Built with FastAPI, Twilio, OpenAI GPT-4, and Deepgram — an open-source alternative to commercial AI calling services.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

`call_dispatch` enables you to fire off automated outbound phone calls driven by a GPT-4 conversational agent. You describe a goal in plain English (e.g. *"Book a dentist appointment for Tuesday afternoon"*), and the agent will have the conversation on your behalf — transcribing the other party in real time, generating contextually appropriate responses, and producing a structured outcome summary when the call ends.

### Key Features

- **Simple REST API** — one `POST` request dispatches a call; poll for live status with `GET`
- **Real-time transcription** — Deepgram WebSocket streaming integrated with Twilio Media Streams
- **GPT-4 conversational agent** — generates natural, goal-directed utterances turn by turn
- **Structured outcome extraction** — automatic JSON summary (success/partial/failure, booked time, confirmation numbers, etc.) saved to SQLite
- **Full call lifecycle management** — pending → initiating → in-progress → completed, with error states for busy/no-answer/failed
- **Locally runnable** — SQLite storage, no external database required

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    REST API (FastAPI)                     │
│  POST /calls  GET /calls/{id}  DELETE /calls/{id}         │
└───────────────────────┬──────────────────────────────────┘
                        │
           ┌────────────▼────────────┐
           │     CallDispatcher      │
           │  (orchestration core)   │
           └──┬──────────┬──────────┘
              │          │
    ┌─────────▼──┐  ┌────▼──────────────┐
    │   Twilio   │  │  ConversationAgent │
    │  REST API  │  │    (OpenAI GPT-4)  │
    └─────────┬──┘  └────────────────────┘
              │
    ┌─────────▼──────────────────────────┐
    │      Twilio Media Stream           │
    │  (bidirectional audio WebSocket)   │
    └─────────┬──────────────────────────┘
              │
    ┌─────────▼──────────┐    ┌──────────────────┐
    │ DeepgramTranscriber │    │  CallSummarizer   │
    │ (real-time STT)     │    │  (post-call GPT-4)│
    └────────────────────┘    └──────────────────┘
              │                         │
    ┌─────────▼─────────────────────────▼──┐
    │           SQLite Store                │
    │  (call records, transcripts, summaries)│
    └───────────────────────────────────────┘
```

**Call flow:**
1. Client `POST /calls` with a phone number and goal
2. Dispatcher creates a call record (SQLite) and initiates Twilio outbound call
3. Twilio calls the destination; on answer, Twilio POSTs to `/twiml/answer/{call_id}`
4. TwiML response opens a bidirectional Media Stream WebSocket
5. Audio streams to Deepgram; transcripts feed the GPT-4 agent
6. Agent utterances are injected back into the call
7. On call end, GPT-4 generates a structured summary saved to SQLite
8. Client polls `GET /calls/{call_id}` for live status and final summary

---

## Requirements

- Python 3.10 or higher
- A publicly accessible HTTPS URL for Twilio webhooks (use [ngrok](https://ngrok.com/) locally)
- Accounts and API keys for:
  - [Twilio](https://www.twilio.com/) (Account SID, Auth Token, phone number)
  - [OpenAI](https://platform.openai.com/) (API key, GPT-4 access)
  - [Deepgram](https://deepgram.com/) (API key)

---

## Installation

### From source (recommended for local development)

```bash
# Clone the repository
git clone https://github.com/your-org/call_dispatch.git
cd call_dispatch

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate          # Windows

# Install the package with all dependencies
pip install -e .
```

### From PyPI (when published)

```bash
pip install call_dispatch
```

---

## Configuration

All configuration is via environment variables. Copy the template and fill in your credentials:

```bash
cp .env.example .env
```

Then edit `.env`:

```dotenv
# Twilio — https://console.twilio.com/
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_twilio_auth_token_here
TWILIO_PHONE_NUMBER=+15551234567

# OpenAI — https://platform.openai.com/api-keys
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxx
OPENAI_MODEL=gpt-4o

# Deepgram — https://console.deepgram.com/
DEEPGRAM_API_KEY=your_deepgram_api_key_here

# Public URL for Twilio webhooks (REQUIRED)
# Use ngrok for local development: ngrok http 8000
PUBLIC_BASE_URL=https://your-tunnel.ngrok.io

# Optional settings (defaults shown)
DATABASE_URL=sqlite:///./call_dispatch.db
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO
CALL_TIMEOUT_SECONDS=300
```

### Required Variables

| Variable | Description |
|---|---|
| `TWILIO_ACCOUNT_SID` | Twilio Account SID (starts with `AC`) |
| `TWILIO_AUTH_TOKEN` | Twilio Auth Token |
| `TWILIO_PHONE_NUMBER` | Your Twilio outbound number (E.164 format, e.g. `+15551234567`) |
| `OPENAI_API_KEY` | OpenAI API key |
| `DEEPGRAM_API_KEY` | Deepgram API key |
| `PUBLIC_BASE_URL` | Publicly accessible HTTPS base URL for Twilio webhooks |

### Optional Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_MODEL` | `gpt-4o` | OpenAI model for agent and summarizer |
| `DATABASE_URL` | `sqlite:///./call_dispatch.db` | SQLite database file path |
| `HOST` | `0.0.0.0` | Server bind host |
| `PORT` | `8000` | Server bind port |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `CALL_TIMEOUT_SECONDS` | `300` | Maximum call duration in seconds |
| `AGENT_MAX_TOKENS` | `150` | Max tokens per agent utterance |
| `AGENT_TEMPERATURE` | `0.7` | Sampling temperature for agent responses |
| `SUMMARIZER_MAX_TOKENS` | `500` | Max tokens for call summary |
| `SUMMARIZER_TEMPERATURE` | `0.2` | Sampling temperature for summaries |

---

## Running the Server

### Local development with ngrok

1. **Start ngrok** to create a public tunnel:
   ```bash
   ngrok http 8000
   ```
   Copy the `https://...ngrok.io` URL and set it as `PUBLIC_BASE_URL` in your `.env`.

2. **Start the server:**
   ```bash
   # Using the installed CLI command
   call-dispatch

   # Or via uvicorn directly
   uvicorn call_dispatch.main:app --host 0.0.0.0 --port 8000 --reload
   ```

3. **Verify the server is running:**
   ```bash
   curl http://localhost:8000/health
   ```
   Expected response:
   ```json
   {
     "status": "ok",
     "version": "0.1.0",
     "store": "ok",
     "dispatcher": "ok",
     "active_calls": 0,
     "total_calls": 0
   }
   ```

### Production deployment

For production, deploy behind a reverse proxy (e.g. nginx) with a real TLS certificate. Set `PUBLIC_BASE_URL` to your domain:

```bash
export PUBLIC_BASE_URL=https://calls.yourdomain.com
call-dispatch
```

Or use Docker:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -e .
CMD ["call-dispatch"]
```

```bash
docker build -t call_dispatch .
docker run -p 8000:8000 --env-file .env call_dispatch
```

---

## API Reference

Interactive API documentation is available at:
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`
- **OpenAPI JSON:** `http://localhost:8000/openapi.json`

### Base URL

```
http://localhost:8000
```

---

### `POST /calls` — Dispatch a Call

Initiates a new AI-driven outbound call. Returns immediately with a `call_id` for polling.

**Request body:**

```json
{
  "to_number": "+15550001234",
  "goal": "Book a dentist appointment for Tuesday afternoon",
  "context": "Patient name: Jane Doe, date of birth: May 15 1990",
  "max_duration_seconds": 180,
  "metadata": {
    "source": "webapp",
    "user_id": "usr_abc123"
  }
}
```

| Field | Type | Required | Constraints | Description |
|---|---|---|---|---|
| `to_number` | string | ✅ | E.164 format (e.g. `+15551234567`) | Destination phone number |
| `goal` | string | ✅ | 10–2000 characters | Natural language description of the call goal |
| `context` | string | ❌ | max 5000 characters | Additional context for the agent (e.g. patient info, account details) |
| `max_duration_seconds` | integer | ❌ | 30–3600, default 300 | Maximum call duration |
| `metadata` | object | ❌ | Any JSON object | Arbitrary metadata stored with the call record |

**Response: `202 Accepted`**

```json
{
  "call_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "status": "pending",
  "message": "Call successfully queued.",
  "created_at": "2024-01-15T14:30:00.000000"
}
```

**Error responses:**

| Status | Condition |
|---|---|
| `422` | Invalid request body (bad phone number, goal too short, etc.) |
| `503` | Server not ready (starting up) |

**Example:**

```bash
curl -X POST http://localhost:8000/calls \
  -H "Content-Type: application/json" \
  -d '{
    "to_number": "+15550001234",
    "goal": "Check if you have availability for a teeth cleaning on Tuesday or Wednesday afternoon next week",
    "context": "Patient: Jane Doe, DOB: 1990-05-15, Insurance: BlueCross"
  }'
```

---

### `GET /calls/{call_id}` — Get Call Status

Returns the current status of a call including the live transcript and final summary.

**Path parameters:**

| Parameter | Type | Description |
|---|---|---|
| `call_id` | string (UUID) | The call ID returned by `POST /calls` |

**Response: `200 OK`**

```json
{
  "call_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "to_number": "+15550001234",
  "from_number": "+15559876543",
  "goal": "Book a dentist appointment for Tuesday afternoon",
  "context": "Patient: Jane Doe",
  "status": "completed",
  "twilio_call_sid": "CA1234567890abcdef1234567890abcdef",
  "transcript": [
    {
      "speaker": "agent",
      "text": "Hello, I'm calling to book a dentist appointment for Jane Doe.",
      "confidence": null,
      "timestamp": "2024-01-15T14:30:05.123456"
    },
    {
      "speaker": "contact",
      "text": "Sure, we have Tuesday at 3pm available.",
      "confidence": 0.97,
      "timestamp": "2024-01-15T14:30:12.456789"
    },
    {
      "speaker": "agent",
      "text": "Tuesday at 3pm works perfectly, thank you!",
      "confidence": null,
      "timestamp": "2024-01-15T14:30:15.789012"
    }
  ],
  "summary": {
    "outcome": "success",
    "summary_text": "Successfully booked a teeth cleaning appointment for Jane Doe on Tuesday at 3pm. Confirmation number ABC123.",
    "key_details": {
      "appointment_time": "Tuesday 3pm",
      "confirmation_number": "ABC123",
      "patient_name": "Jane Doe"
    },
    "follow_up_required": false,
    "follow_up_notes": null
  },
  "error_message": null,
  "created_at": "2024-01-15T14:30:00.000000",
  "updated_at": "2024-01-15T14:32:30.000000",
  "started_at": "2024-01-15T14:30:05.000000",
  "ended_at": "2024-01-15T14:32:30.000000"
}
```

**Call status values:**

| Status | Description |
|---|---|
| `pending` | Call queued, not yet sent to Twilio |
| `initiating` | Twilio call created, waiting for answer |
| `in_progress` | Call answered, conversation underway |
| `completed` | Call ended normally |
| `failed` | Call failed (error in Twilio or dispatch) |
| `cancelled` | Call cancelled before completion |
| `no_answer` | Destination did not answer |
| `busy` | Destination line was busy |

**Summary outcome values:**

| Outcome | Description |
|---|---|
| `success` | The call goal was fully achieved |
| `partial` | The call goal was partially achieved |
| `failure` | The call goal was not achieved |
| `unknown` | Outcome cannot be determined from the transcript |

**Error responses:**

| Status | Condition |
|---|---|
| `404` | Call not found |
| `503` | Server not ready |

**Example:**

```bash
curl http://localhost:8000/calls/3fa85f64-5717-4562-b3fc-2c963f66afa6
```

---

### `GET /calls` — List Calls

Returns a paginated list of all call records.

**Query parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `status` | string | (all) | Filter by status (see status values above) |
| `limit` | integer | `50` | Max records to return (1–500) |
| `offset` | integer | `0` | Pagination offset |

**Response: `200 OK`**

```json
{
  "total": 42,
  "calls": [
    {
      "call_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
      "to_number": "+15550001234",
      "from_number": "+15559876543",
      "goal": "Book a dentist appointment for Tuesday afternoon",
      "status": "completed",
      "created_at": "2024-01-15T14:30:00.000000",
      "updated_at": "2024-01-15T14:32:30.000000"
    }
  ]
}
```

**Examples:**

```bash
# List all calls
curl http://localhost:8000/calls

# List completed calls only
curl "http://localhost:8000/calls?status=completed"

# Paginate: second page of 10
curl "http://localhost:8000/calls?limit=10&offset=10"

# Failed calls only
curl "http://localhost:8000/calls?status=failed"
```

---

### `GET /calls/{call_id}/summary` — Get Call Summary

Returns the structured summary and full transcript for a call. The summary is `null` until the call completes and the summarizer finishes.

**Response: `200 OK`**

```json
{
  "call_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "status": "completed",
  "goal": "Book a dentist appointment for Tuesday afternoon",
  "summary": {
    "outcome": "success",
    "summary_text": "Successfully booked appointment for Jane Doe on Tuesday at 3pm. Confirmation ABC123.",
    "key_details": {
      "appointment_time": "Tuesday 3pm",
      "confirmation_number": "ABC123"
    },
    "follow_up_required": false,
    "follow_up_notes": null
  },
  "transcript": [
    {
      "speaker": "agent",
      "text": "Hello, I'm calling to book a dentist appointment.",
      "confidence": null,
      "timestamp": "2024-01-15T14:30:05.123456"
    }
  ]
}
```

**Example:**

```bash
curl http://localhost:8000/calls/3fa85f64-5717-4562-b3fc-2c963f66afa6/summary
```

---

### `DELETE /calls/{call_id}` — Cancel a Call

Cancels a `pending` or `initiating` call. Returns `409 Conflict` for calls that are already in progress or completed.

**Response: `200 OK`**

```json
{
  "message": "Call 3fa85f64-5717-4562-b3fc-2c963f66afa6 cancelled successfully.",
  "call_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6"
}
```

**Error responses:**

| Status | Condition |
|---|---|
| `404` | Call not found |
| `409` | Call is in a non-cancellable state (in_progress, completed, etc.) |
| `503` | Server not ready |

**Example:**

```bash
curl -X DELETE http://localhost:8000/calls/3fa85f64-5717-4562-b3fc-2c963f66afa6
```

---

### `GET /health` — Health Check

Returns the health status of the API server.

**Response: `200 OK`**

```json
{
  "status": "ok",
  "version": "0.1.0",
  "store": "ok",
  "dispatcher": "ok",
  "active_calls": 2,
  "total_calls": 147
}
```

`status` is `"ok"` when all components are operational, `"degraded"` otherwise.

---

## Usage Examples

### Example 1: Book a dental appointment

```bash
# 1. Dispatch the call
CALL_ID=$(curl -s -X POST http://localhost:8000/calls \
  -H "Content-Type: application/json" \
  -d '{
    "to_number": "+15550001234",
    "goal": "Book a teeth cleaning appointment for next Tuesday or Wednesday afternoon",
    "context": "Patient: Jane Doe, DOB: May 15 1990, Insurance: BlueCross PPO"
  }' | python3 -c "import sys,json; print(json.load(sys.stdin)['call_id'])")

echo "Call ID: $CALL_ID"

# 2. Poll for status (every 5 seconds)
watch -n 5 curl -s http://localhost:8000/calls/$CALL_ID | python3 -m json.tool

# 3. Get the final summary
curl http://localhost:8000/calls/$CALL_ID/summary | python3 -m json.tool
```

### Example 2: Get a quote

```bash
curl -X POST http://localhost:8000/calls \
  -H "Content-Type: application/json" \
  -d '{
    "to_number": "+15559876543",
    "goal": "Get a quote for replacing a roof on a 2000 square foot house",
    "context": "Property: 123 Main St, Springfield. Looking for a quote for asphalt shingles.",
    "max_duration_seconds": 240
  }'
```

### Example 3: Check availability

```bash
curl -X POST http://localhost:8000/calls \
  -H "Content-Type: application/json" \
  -d '{
    "to_number": "+15550005678",
    "goal": "Check if the restaurant has availability for 4 people this Saturday evening around 7pm",
    "metadata": {"reservation_type": "dinner", "party_size": 4}
  }'
```

### Python client example

```python
import httpx
import time

BASE_URL = "http://localhost:8000"

def dispatch_call(to_number: str, goal: str, context: str = None) -> str:
    """Dispatch a call and return the call_id."""
    payload = {"to_number": to_number, "goal": goal}
    if context:
        payload["context"] = context

    resp = httpx.post(f"{BASE_URL}/calls", json=payload)
    resp.raise_for_status()
    return resp.json()["call_id"]


def wait_for_completion(call_id: str, timeout: int = 360, poll_interval: int = 5) -> dict:
    """Poll until the call completes or times out."""
    terminal_statuses = {"completed", "failed", "cancelled", "no_answer", "busy"}
    deadline = time.time() + timeout

    while time.time() < deadline:
        resp = httpx.get(f"{BASE_URL}/calls/{call_id}")
        resp.raise_for_status()
        data = resp.json()

        print(f"Status: {data['status']}")

        if data["status"] in terminal_statuses:
            return data

        time.sleep(poll_interval)

    raise TimeoutError(f"Call {call_id} did not complete within {timeout} seconds")


def get_summary(call_id: str) -> dict:
    """Retrieve the structured summary for a completed call."""
    resp = httpx.get(f"{BASE_URL}/calls/{call_id}/summary")
    resp.raise_for_status()
    return resp.json()


# Usage
if __name__ == "__main__":
    call_id = dispatch_call(
        to_number="+15550001234",
        goal="Book a dentist appointment for Tuesday afternoon",
        context="Patient: Jane Doe, DOB: 1990-05-15",
    )
    print(f"Dispatched call: {call_id}")

    result = wait_for_completion(call_id)
    print(f"Call ended with status: {result['status']}")

    summary = get_summary(call_id)
    if summary["summary"]:
        print(f"Outcome: {summary['summary']['outcome']}")
        print(f"Summary: {summary['summary']['summary_text']}")
        print(f"Key details: {summary['summary']['key_details']}")
```

---

## Development

### Running tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/test_agent.py -v

# Run with coverage
pytest --cov=call_dispatch --cov-report=html
```

### Project structure

```
call_dispatch/
├── __init__.py          # Package init, version
├── main.py              # FastAPI app, startup/shutdown lifecycle
├── config.py            # Settings via pydantic-settings
├── models.py            # Pydantic request/response/DB models
├── dispatcher.py        # Core orchestration (Twilio + agent + transcriber)
├── agent.py             # GPT-4 conversational agent
├── transcriber.py       # Deepgram WebSocket transcription client
├── summarizer.py        # Post-call GPT-4 structured summarizer
├── store.py             # SQLite persistence layer
├── routes.py            # FastAPI REST route definitions
└── twiml_handler.py     # Twilio webhooks and TwiML builders
tests/
├── test_agent.py
├── test_summarizer.py
├── test_transcriber.py
├── test_twiml_handler.py
├── test_dispatcher.py
├── test_routes.py
├── test_store.py
└── test_config.py
```

### Code style

This project follows PEP 8 with type hints on all public functions. Docstrings use Google style.

```bash
# Lint
pip install ruff
ruff check call_dispatch/

# Type check
pip install mypy
mypy call_dispatch/
```

---

## Twilio Setup

### 1. Get a Twilio phone number

1. Sign up at [twilio.com](https://www.twilio.com)
2. Navigate to **Phone Numbers → Manage → Buy a Number**
3. Purchase a number with **Voice** capability
4. Note your Account SID and Auth Token from the [Console Dashboard](https://console.twilio.com/)

### 2. Configure webhook URLs

`call_dispatch` automatically registers webhook URLs when it initiates a call via the Twilio REST API — you don't need to configure them in the Twilio console. The URLs are constructed from your `PUBLIC_BASE_URL`:

- **Answer webhook:** `{PUBLIC_BASE_URL}/twiml/answer/{call_id}`
- **Status callback:** `{PUBLIC_BASE_URL}/twiml/status/{call_id}`
- **Media stream:** `{PUBLIC_BASE_URL}/ws/stream/{call_id}` (WSS)

### 3. ngrok for local development

```bash
# Install ngrok: https://ngrok.com/download
ngrok http 8000

# Copy the https:// URL, e.g. https://abc123.ngrok.io
# Set in .env:
PUBLIC_BASE_URL=https://abc123.ngrok.io
```

> **Note:** Free ngrok URLs change on restart. Use a paid ngrok plan or a fixed domain for persistent development.

---

## OpenAI Setup

1. Create an API key at [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Ensure your account has access to the model you configure (`gpt-4o` by default)
3. Set spending limits to avoid unexpected charges during testing

The agent uses `gpt-4o` by default. You can use any chat model:

```dotenv
# More capable (higher cost)
OPENAI_MODEL=gpt-4o

# Faster and cheaper for simple tasks
OPENAI_MODEL=gpt-4o-mini

# Legacy
OPENAI_MODEL=gpt-3.5-turbo
```

---

## Deepgram Setup

1. Sign up at [deepgram.com](https://deepgram.com)
2. Create an API key in the [Deepgram Console](https://console.deepgram.com/)
3. The free tier includes credits for getting started

`call_dispatch` uses the `nova-2-phonecall` model optimised for telephone audio (8kHz mu-law). This model is well-tuned for the audio quality typical of phone calls.

---

## Troubleshooting

### "Failed to load settings" on startup

Ensure all required environment variables are set. Check `.env.example` for the full list.

```bash
# Verify your .env file loads correctly
python3 -c "from call_dispatch.config import get_settings; s = get_settings(); print('OK:', s.public_base_url)"
```

### Twilio cannot reach webhooks

- Ensure `PUBLIC_BASE_URL` is set to your public ngrok URL (not `localhost`)
- Verify ngrok is running: `curl https://your-tunnel.ngrok.io/health`
- Check Twilio's call logs in the [Twilio Console](https://console.twilio.com/) for webhook errors

### No audio transcription / agent not responding

- Check that Deepgram credentials are correct
- Verify the Media Stream WebSocket URL is accessible (`wss://your-tunnel.ngrok.io/ws/stream/{call_id}`)
- Enable `LOG_LEVEL=DEBUG` for detailed transcription logs
- Check Twilio's Media Stream debugger in the console

### Calls going straight to voicemail

Twilio's `machine_detection` is set to `"Disable"` by default (the agent will speak even if a machine answers). To handle voicemail detection, you would need to implement AMD (Answering Machine Detection) in the dispatcher.

### OpenAI rate limits

If you encounter OpenAI rate limit errors (`429`), consider:
- Reducing `AGENT_MAX_TOKENS` to shorten responses
- Using `gpt-4o-mini` instead of `gpt-4o`
- Adding retry logic via the `tenacity` library

### High latency between agent responses

Agent response latency is the sum of:
1. Deepgram transcription time (~200-500ms for final results)
2. OpenAI API call (~500-2000ms depending on model)
3. TTS synthesis (if using a TTS service)

Use `gpt-4o-mini` with `AGENT_TEMPERATURE=0.3` and `AGENT_MAX_TOKENS=100` for faster, shorter responses.

---

## Security Considerations

### Twilio signature validation

For production deployments, validate Twilio webhook signatures to ensure requests originate from Twilio:

```python
from twilio.request_validator import RequestValidator

validator = RequestValidator(settings.twilio_auth_token)
if not validator.validate(url, params, signature):
    raise HTTPException(status_code=403, detail="Invalid Twilio signature")
```

This is not included by default to keep the setup simple, but is strongly recommended for production.

### API authentication

The REST API has no authentication by default. For production:
- Add API key authentication middleware
- Use a reverse proxy with HTTP Basic Auth
- Restrict access by IP if possible

### Environment variables

- Never commit `.env` to version control
- Use a secrets manager (AWS Secrets Manager, HashiCorp Vault, etc.) in production
- Rotate API keys regularly

---

## Cost Estimates

Approximate costs per minute of call time (as of 2024):

| Service | Cost |
|---|---|
| Twilio voice (outbound) | ~$0.014/min |
| Deepgram (nova-2) | ~$0.0043/min |
| OpenAI GPT-4o (~5 turns) | ~$0.005/call |
| **Total** | **~$0.023/min** |

A 3-minute appointment booking call costs approximately $0.07.

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes with tests
4. Ensure all tests pass: `pytest`
5. Submit a pull request

### Running the full test suite

```bash
# All tests (no API keys required — all mocked)
pytest tests/ -v

# With coverage report
pytest tests/ --cov=call_dispatch --cov-report=term-missing
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [Twilio](https://www.twilio.com/) for telephony infrastructure
- [OpenAI](https://openai.com/) for GPT-4 conversation and summarization
- [Deepgram](https://deepgram.com/) for real-time speech-to-text
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [Pydantic](https://docs.pydantic.dev/) for data validation
