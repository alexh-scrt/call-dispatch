"""FastAPI application entry point for call_dispatch.

Creates and configures the FastAPI application, registers all routers,
sets up startup and shutdown lifecycle events, and provides the
``run()`` entry point used by the ``call-dispatch`` CLI command.

Application lifecycle:
1. On startup: initialise the SQLite store, create the CallDispatcher,
   and attach both to ``app.state`` for dependency injection.
2. On shutdown: close the SQLite store connection gracefully.

Usage::

    # Via uvicorn directly:
    uvicorn call_dispatch.main:app --host 0.0.0.0 --port 8000 --reload

    # Via the installed CLI entry point:
    call-dispatch

    # Programmatically:
    from call_dispatch.main import app
"""

from __future__ import annotations

import logging
import logging.config
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from call_dispatch import __version__

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------


def _configure_logging(log_level: str = "INFO") -> None:
    """Configure the root logger with a structured format.

    Args:
        log_level: The logging level string (e.g. ``"DEBUG"``, ``"INFO"``)
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application startup and shutdown lifecycle.

    On startup:
    - Loads and validates settings.
    - Configures logging.
    - Initialises the SQLite store schema.
    - Creates the CallDispatcher.
    - Attaches store, dispatcher, and active_calls dict to ``app.state``.

    On shutdown:
    - Closes the SQLite store connection.

    Args:
        app: The FastAPI application instance.

    Yields:
        None: Control is yielded to the application during the lifespan.
    """
    # ------------------------------------------------------------------ #
    # Startup
    # ------------------------------------------------------------------ #
    from call_dispatch.config import get_settings
    from call_dispatch.dispatcher import CallDispatcher
    from call_dispatch.store import CallStore

    try:
        cfg = get_settings()
    except Exception as exc:
        # Log a clear error message before re-raising — the server will not start
        logging.basicConfig(level=logging.ERROR)
        logging.error(
            "Failed to load settings: %s. "
            "Check your .env file or environment variables.",
            exc,
        )
        raise

    _configure_logging(cfg.log_level)
    logger.info("call_dispatch v%s starting up", __version__)
    logger.info(
        "Configuration: host=%s port=%d public_base_url=%s",
        cfg.host,
        cfg.port,
        cfg.public_base_url,
    )

    # Initialise the SQLite store
    store = CallStore(cfg.database_url)
    await store.initialize()
    logger.info("CallStore initialised: %s", cfg.database_url)

    # Create the call dispatcher
    dispatcher = CallDispatcher(store=store)
    logger.info("CallDispatcher initialised")

    # Attach to app state for dependency injection
    app.state.store = store
    app.state.dispatcher = dispatcher
    app.state.active_calls = dispatcher._active_calls  # shared reference

    logger.info("call_dispatch startup complete — ready to accept requests")

    yield

    # ------------------------------------------------------------------ #
    # Shutdown
    # ------------------------------------------------------------------ #
    logger.info("call_dispatch shutting down")
    try:
        await store.close()
        logger.info("CallStore closed")
    except Exception as exc:
        logger.warning("Error closing store during shutdown: %s", exc)

    logger.info("call_dispatch shutdown complete")


# ---------------------------------------------------------------------------
# FastAPI application factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Registers all routers, middleware, and exception handlers.
    Does not start the server — use :func:`run` or ``uvicorn`` for that.

    Returns:
        FastAPI: The configured application instance.
    """
    application = FastAPI(
        title="call_dispatch",
        description=(
            "A lightweight, locally-runnable AI agent phone call dispatcher "
            "for automated outbound calls. Integrates Twilio, OpenAI GPT-4, "
            "and Deepgram for real-time AI-powered phone conversations."
        ),
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ------------------------------------------------------------------ #
    # CORS middleware
    # ------------------------------------------------------------------ #
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Restrict in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------ #
    # Register routers
    # ------------------------------------------------------------------ #
    from call_dispatch.routes import health_router, router as calls_router
    from call_dispatch.twiml_handler import router as twiml_router
    from call_dispatch.twiml_handler import ws_router

    application.include_router(health_router)
    application.include_router(calls_router)
    application.include_router(twiml_router)
    application.include_router(ws_router)

    # ------------------------------------------------------------------ #
    # Global exception handlers
    # ------------------------------------------------------------------ #

    @application.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Catch-all handler for unhandled exceptions.

        Logs the error and returns a generic 500 response to avoid
        leaking internal implementation details to clients.

        Args:
            request: The incoming request that triggered the exception.
            exc: The unhandled exception.

        Returns:
            JSONResponse: A generic 500 error response.
        """
        logger.error(
            "Unhandled exception for %s %s: %s",
            request.method,
            request.url.path,
            exc,
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={
                "detail": "An unexpected internal error occurred.",
                "error_code": "INTERNAL_SERVER_ERROR",
            },
        )

    # ------------------------------------------------------------------ #
    # Root endpoint
    # ------------------------------------------------------------------ #

    @application.get(
        "/",
        include_in_schema=False,
        summary="Root",
    )
    async def root() -> dict:
        """Return a brief welcome message and version info."""
        return {
            "name": "call_dispatch",
            "version": __version__,
            "docs": "/docs",
            "health": "/health",
        }

    return application


# ---------------------------------------------------------------------------
# Application singleton
# ---------------------------------------------------------------------------

# Module-level application instance used by uvicorn and imports.
app: FastAPI = create_app()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def run() -> None:
    """Start the Uvicorn server using settings from the environment.

    This is the entry point for the ``call-dispatch`` CLI command defined
    in ``pyproject.toml``.  Reads host, port, and log level from the
    application settings.

    Raises:
        SystemExit: If settings cannot be loaded or the server fails to start.
    """
    import sys

    try:
        import uvicorn
    except ImportError:
        print("uvicorn is required to run call-dispatch. Install it with: pip install uvicorn[standard]")
        sys.exit(1)

    try:
        from call_dispatch.config import get_settings

        cfg = get_settings()
        host = cfg.host
        port = cfg.port
        log_level = cfg.log_level.lower()
    except Exception as exc:
        print(f"Failed to load settings: {exc}")
        print("Ensure all required environment variables are set (see .env.example).")
        sys.exit(1)

    uvicorn.run(
        "call_dispatch.main:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=False,
        access_log=True,
    )
