# call_dispatch

> Dispatch AI-powered phone calls with a single API request.

`call_dispatch` is a lightweight, locally-runnable server that makes automated outbound phone calls on your behalf. You describe a goal in plain English — *"Book a dentist appointment for Tuesday afternoon"* — and a GPT-4 agent handles the conversation, transcribes responses in real time, and delivers a structured outcome summary when the call ends. It's a self-hosted, open-source alternative to commercial AI calling services.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Quick Start

**1. Install**

```bash
pip install call_dispatch
```

Or clone and install locally:

```bash
git clone https://github.com/your-org/call_dispatch.git
cd call_dispatch
pip install -e .
```

**2. Configure**

```bash
cp .env.example .env
# Edit .env with your Twilio, OpenAI, and Deepgram credentials
```

**3. Run the server**

```bash
call-dispatch
# or
uvicorn call_dispatch.main:app --host 0.0.0.0 --port 8000 --reload
```

**4. Dispatch your first call**

```bash
curl -X POST http://localhost:8000/calls \
  -H "Content-Type: application/json" \
  -d '{
    "to_number": "+15550001234",
    "goal": "Book a dentist appointment for Tuesday afternoon",
    "context": "Patient name: Jane Doe, DOB: 1990-05-15"
  }'
```

You'll receive a `call_id`. Use it to poll for results:

```bash
curl http://localhost:8000/calls/{call_id}
```

---

## Features

- **One-request dispatch** — `POST /calls` with a phone number and a plain-English goal kicks off the entire call
- **Real-time transcription** — Deepgram WebSocket streaming integrated with Twilio Media Streams for live, low-latency transcripts
- **GPT-4 conversational agent** — generates contextually appropriate, natural-sounding utterances turn by turn to advance toward your goal
- **Structured outcome extraction** — automatic JSON summary (success/partial/failure, booked time, confirmation numbers, key details) persisted to SQLite after every call
- **Full lifecycle status API** — poll any call for its live transcript, current state, and final summary

---

## Usage Examples

### Dispatch a call

```bash
curl -X POST http://localhost:8000/calls \
  -H "Content-Type: application/json" \
  -d '{
    "to_number": "+15550001234",
    "goal": "Get a price quote for a 2-bedroom apartment cleaning",
    "context": "Apartment is 950 sq ft, located in Austin TX, available weekday mornings"
  }'
```

```json
{
  "call_id": "3f7a2b1c-...",
  "status": "initiating",
  "to_number": "+15550001234",
  "created_at": "2024-01-15T10:23:45Z"
}
```

### Poll call status (live transcript included)

```bash
curl http://localhost:8000/calls/3f7a2b1c-...
```

```json
{
  "call_id": "3f7a2b1c-...",
  "status": "in_progress",
  "goal": "Get a price quote for a 2-bedroom apartment cleaning",
  "transcript": [
    { "speaker": "agent",   "text": "Hi, I'm calling to get a quote for an apartment cleaning." },
    { "speaker": "contact", "text": "Sure! What's the square footage?" },
    { "speaker": "agent",   "text": "It's about 950 square feet, 2 bedrooms." }
  ],
  "duration_seconds": 42
}
```

### Fetch the final summary

```bash
curl http://localhost:8000/calls/3f7a2b1c-.../summary
```

```json
{
  "outcome": "success",
  "summary": "Received a quote of $180 for a standard 2-bedroom cleaning. Available Tuesday and Thursday mornings.",
  "key_details": {
    "quote_amount": "$180",
    "available_slots": ["Tuesday morning", "Thursday morning"]
  },
  "follow_up_required": true,
  "follow_up_notes": "Call back to confirm booking once date is decided."
}
```

### List all calls

```bash
curl "http://localhost:8000/calls?limit=20&offset=0"
```

### Cancel a pending call

```bash
curl -X DELETE http://localhost:8000/calls/3f7a2b1c-...
```

### Health check

```bash
curl http://localhost:8000/health
# { "status": "ok", "version": "0.1.0" }
```

---

## Project Structure

```
call_dispatch/
├── call_dispatch/
│   ├── __init__.py        # Package init, version export
│   ├── main.py            # FastAPI app entry point, lifecycle hooks
│   ├── config.py          # Environment variable loading & validation
│   ├── models.py          # Pydantic request/response schemas, DB row types
│   ├── routes.py          # REST API endpoint definitions
│   ├── dispatcher.py      # Core orchestration: Twilio call lifecycle
│   ├── agent.py           # GPT-4 conversational agent logic
│   ├── transcriber.py     # Deepgram WebSocket transcription client
│   ├── summarizer.py      # Post-call summary & outcome extraction
│   ├── store.py           # SQLite persistence layer
│   └── twiml_handler.py   # Twilio webhook handlers & TwiML generation
├── tests/
│   ├── test_agent.py
│   ├── test_dispatcher.py
│   ├── test_routes.py
│   ├── test_store.py
│   ├── test_summarizer.py
│   ├── test_transcriber.py
│   └── test_twiml_handler.py
├── .env.example
├── pyproject.toml
└── README.md
```

---

## Configuration

Copy `.env.example` to `.env` and fill in your credentials. All variables are required unless marked optional.

```dotenv
# Twilio — https://console.twilio.com/
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_PHONE_NUMBER=+15551234567

# OpenAI — https://platform.openai.com/
OPENAI_API_KEY=sk-...

# Deepgram — https://console.deepgram.com/
DEEPGRAM_API_KEY=dg-...

# Public URL for Twilio webhooks (use ngrok or similar for local dev)
PUBLIC_BASE_URL=https://your-subdomain.ngrok.io

# Optional
OPENAI_MODEL=gpt-4o          # default: gpt-4o
SERVER_PORT=8000             # default: 8000
DATABASE_URL=sqlite:///./call_dispatch.db
LOG_LEVEL=INFO
```

> **Local development tip:** Twilio needs to reach your server to deliver webhooks. Use [ngrok](https://ngrok.com/) to expose your local port:
> ```bash
> ngrok http 8000
> # Copy the https URL into PUBLIC_BASE_URL in your .env
> ```

### Configuration Reference

| Variable | Required | Description |
|---|---|---|
| `TWILIO_ACCOUNT_SID` | ✅ | Twilio Account SID |
| `TWILIO_AUTH_TOKEN` | ✅ | Twilio Auth Token |
| `TWILIO_PHONE_NUMBER` | ✅ | Outbound caller ID (E.164 format) |
| `OPENAI_API_KEY` | ✅ | OpenAI API key |
| `DEEPGRAM_API_KEY` | ✅ | Deepgram API key |
| `PUBLIC_BASE_URL` | ✅ | Publicly reachable base URL for Twilio webhooks |
| `OPENAI_MODEL` | ❌ | GPT model to use (default: `gpt-4o`) |
| `SERVER_PORT` | ❌ | Port to bind the server (default: `8000`) |
| `DATABASE_URL` | ❌ | SQLite connection string (default: `sqlite:///./call_dispatch.db`) |
| `LOG_LEVEL` | ❌ | Logging verbosity (default: `INFO`) |

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest
```

All tests are fully offline — external APIs (Twilio, OpenAI, Deepgram) are mocked.

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

*Built with [Jitter](https://github.com/jitter-ai) - an AI agent that ships code daily.*
