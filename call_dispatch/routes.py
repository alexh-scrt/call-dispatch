"""REST API routes for call_dispatch.

Defines all FastAPI endpoints for:
- Dispatching new outbound calls (POST /calls)
- Polling call status (GET /calls/{call_id})
- Fetching call summaries (GET /calls/{call_id}/summary)
- Listing all calls (GET /calls)
- Cancelling a pending call (DELETE /calls/{call_id})
- Health check (GET /health)

All routes are registered on a FastAPI router and expect the application
state to have ``store`` (CallStore) and ``dispatcher`` (CallDispatcher)
attributes set during startup.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse

from call_dispatch.models import (
    CallStatus,
    CallStatusResponse,
    CallSummaryResponse,
    DispatchCallRequest,
    DispatchCallResponse,
    ErrorResponse,
    ListCallsResponse,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/calls", tags=["calls"])
health_router = APIRouter(tags=["health"])


# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------


def get_store(request: Request):
    """Dependency that extracts the CallStore from application state.

    Args:
        request: The incoming FastAPI request.

    Returns:
        CallStore: The active call store.

    Raises:
        HTTPException: 503 if the store is not initialised.
    """
    store = getattr(request.app.state, "store", None)
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Store not initialised. Server may still be starting.",
        )
    return store


def get_dispatcher(request: Request):
    """Dependency that extracts the CallDispatcher from application state.

    Args:
        request: The incoming FastAPI request.

    Returns:
        CallDispatcher: The active call dispatcher.

    Raises:
        HTTPException: 503 if the dispatcher is not initialised.
    """
    dispatcher = getattr(request.app.state, "dispatcher", None)
    if dispatcher is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Dispatcher not initialised. Server may still be starting.",
        )
    return dispatcher


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@health_router.get(
    "/health",
    summary="Health check",
    description="Returns the health status of the API server.",
    response_description="Health status and version information.",
    tags=["health"],
)
async def health_check(request: Request) -> dict:
    """Return the health status of the API server.

    Checks whether the store and dispatcher are initialised and reports
    the number of currently active calls.

    Args:
        request: The incoming FastAPI request.

    Returns:
        dict: Health status payload including store and dispatcher status.
    """
    from call_dispatch import __version__

    store = getattr(request.app.state, "store", None)
    dispatcher = getattr(request.app.state, "dispatcher", None)

    store_ok = store is not None
    dispatcher_ok = dispatcher is not None

    active_calls = 0
    if dispatcher is not None:
        active_calls = len(dispatcher.active_call_ids)

    total_calls = 0
    if store is not None:
        try:
            total_calls = await store.count_calls()
        except Exception:
            pass

    overall_status = "ok" if (store_ok and dispatcher_ok) else "degraded"

    return {
        "status": overall_status,
        "version": __version__,
        "store": "ok" if store_ok else "unavailable",
        "dispatcher": "ok" if dispatcher_ok else "unavailable",
        "active_calls": active_calls,
        "total_calls": total_calls,
    }


# ---------------------------------------------------------------------------
# Dispatch a new call
# ---------------------------------------------------------------------------


@router.post(
    "",
    response_model=DispatchCallResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Dispatch a new outbound call",
    description=(
        "Initiates a new AI-driven outbound call to the specified number. "
        "The call is queued immediately and the response includes a `call_id` "
        "that can be used to poll the call status."
    ),
    responses={
        202: {"description": "Call successfully queued."},
        422: {"description": "Validation error in request body."},
        503: {"description": "Server not ready."},
    },
)
async def dispatch_call(
    payload: DispatchCallRequest,
    store=Depends(get_store),
    dispatcher=Depends(get_dispatcher),
) -> DispatchCallResponse:
    """Dispatch a new outbound AI call.

    Creates a call record and initiates an outbound Twilio call toward the
    specified goal.  Returns immediately with a ``call_id`` for status polling.

    Args:
        payload: The validated dispatch request body.
        store: Injected CallStore dependency.
        dispatcher: Injected CallDispatcher dependency.

    Returns:
        DispatchCallResponse: The created call ID and initial status.

    Raises:
        HTTPException: 503 if dependencies are not available.
        HTTPException: 500 if an unexpected error occurs during dispatch.
    """
    logger.info(
        "Dispatch request received: to=%s goal=%r",
        payload.to_number,
        payload.goal[:60],
    )

    try:
        record = await dispatcher.dispatch_from_request(payload)
    except Exception as exc:
        logger.error("Unexpected error dispatching call: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to dispatch call: {exc}",
        )

    logger.info(
        "Call dispatched: call_id=%s status=%s",
        record.call_id,
        record.status.value,
    )

    return DispatchCallResponse(
        call_id=record.call_id,
        status=record.status,
        message="Call successfully queued.",
        created_at=record.created_at,
    )


# ---------------------------------------------------------------------------
# List all calls
# ---------------------------------------------------------------------------


@router.get(
    "",
    response_model=ListCallsResponse,
    status_code=status.HTTP_200_OK,
    summary="List calls",
    description="Returns a paginated list of all call records, optionally filtered by status.",
    responses={
        200: {"description": "List of call records."},
        503: {"description": "Server not ready."},
    },
)
async def list_calls(
    status_filter: Optional[str] = Query(
        default=None,
        alias="status",
        description=(
            "Filter by call status: pending, initiating, in_progress, "
            "completed, failed, cancelled, no_answer, busy."
        ),
    ),
    limit: int = Query(
        default=50,
        ge=1,
        le=500,
        description="Maximum number of records to return (1–500).",
    ),
    offset: int = Query(
        default=0,
        ge=0,
        description="Number of records to skip for pagination.",
    ),
    store=Depends(get_store),
) -> ListCallsResponse:
    """Return a paginated list of call records.

    Args:
        status_filter: Optional status string to filter results.
        limit: Max records to return.
        offset: Pagination offset.
        store: Injected CallStore dependency.

    Returns:
        ListCallsResponse: Paginated list of call status summaries.

    Raises:
        HTTPException: 400 if the status filter is invalid.
        HTTPException: 503 if the store is not available.
    """
    # Parse optional status filter
    parsed_status: Optional[CallStatus] = None
    if status_filter is not None:
        try:
            parsed_status = CallStatus(status_filter.lower())
        except ValueError:
            valid = [s.value for s in CallStatus]
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status filter {status_filter!r}. Valid values: {valid}",
            )

    try:
        records = await store.list_calls(
            status=parsed_status,
            limit=limit,
            offset=offset,
        )
        total = await store.count_calls(status=parsed_status)
    except Exception as exc:
        logger.error("Error listing calls: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list calls: {exc}",
        )

    return ListCallsResponse(
        total=total,
        calls=[CallStatusResponse.from_record(r) for r in records],
    )


# ---------------------------------------------------------------------------
# Get call status
# ---------------------------------------------------------------------------


@router.get(
    "/{call_id}",
    response_model=CallStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Get call status",
    description=(
        "Returns the current status of a call including the live transcript "
        "and final summary when available."
    ),
    responses={
        200: {"description": "Call status retrieved successfully."},
        404: {"description": "Call not found."},
        503: {"description": "Server not ready."},
    },
)
async def get_call_status(
    call_id: str,
    store=Depends(get_store),
) -> CallStatusResponse:
    """Return the current status of a specific call.

    Args:
        call_id: The UUID of the call to query.
        store: Injected CallStore dependency.

    Returns:
        CallStatusResponse: The full call status including transcript and summary.

    Raises:
        HTTPException: 404 if the call is not found.
        HTTPException: 503 if the store is not available.
    """
    try:
        record = await store.get_call(call_id)
    except Exception as exc:
        logger.error("Error fetching call %s: %s", call_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch call: {exc}",
        )

    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Call not found: {call_id}",
        )

    return CallStatusResponse.from_record(record)


# ---------------------------------------------------------------------------
# Get call summary
# ---------------------------------------------------------------------------


@router.get(
    "/{call_id}/summary",
    response_model=CallSummaryResponse,
    status_code=status.HTTP_200_OK,
    summary="Get call summary",
    description=(
        "Returns the structured summary and full transcript for a completed call. "
        "The summary is generated automatically after the call ends."
    ),
    responses={
        200: {"description": "Call summary retrieved successfully."},
        404: {"description": "Call not found."},
        503: {"description": "Server not ready."},
    },
)
async def get_call_summary(
    call_id: str,
    store=Depends(get_store),
) -> CallSummaryResponse:
    """Return the structured summary and transcript for a completed call.

    Args:
        call_id: The UUID of the call to query.
        store: Injected CallStore dependency.

    Returns:
        CallSummaryResponse: The call summary, transcript, and final status.

    Raises:
        HTTPException: 404 if the call is not found.
        HTTPException: 503 if the store is not available.
    """
    try:
        record = await store.get_call(call_id)
    except Exception as exc:
        logger.error("Error fetching summary for call %s: %s", call_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch call summary: {exc}",
        )

    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Call not found: {call_id}",
        )

    return CallSummaryResponse.from_record(record)


# ---------------------------------------------------------------------------
# Cancel a call
# ---------------------------------------------------------------------------


@router.delete(
    "/{call_id}",
    status_code=status.HTTP_200_OK,
    summary="Cancel a call",
    description=(
        "Cancels a pending or initiating call. Returns 409 if the call "
        "is already in progress or has completed."
    ),
    responses={
        200: {"description": "Call cancelled successfully."},
        404: {"description": "Call not found."},
        409: {"description": "Call cannot be cancelled in its current state."},
        503: {"description": "Server not ready."},
    },
)
async def cancel_call(
    call_id: str,
    store=Depends(get_store),
    dispatcher=Depends(get_dispatcher),
) -> dict:
    """Cancel a pending or initiating call.

    Args:
        call_id: The UUID of the call to cancel.
        store: Injected CallStore dependency.
        dispatcher: Injected CallDispatcher dependency.

    Returns:
        dict: Confirmation message.

    Raises:
        HTTPException: 404 if the call is not found.
        HTTPException: 409 if the call cannot be cancelled in its current state.
        HTTPException: 503 if dependencies are not available.
    """
    # First check if the call exists
    try:
        record = await store.get_call(call_id)
    except Exception as exc:
        logger.error("Error fetching call %s for cancellation: %s", call_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch call: {exc}",
        )

    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Call not found: {call_id}",
        )

    # Attempt cancellation
    try:
        cancelled = await dispatcher.cancel_call(call_id)
    except Exception as exc:
        logger.error("Error cancelling call %s: %s", call_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel call: {exc}",
        )

    if not cancelled:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Call {call_id} cannot be cancelled in status '{record.status.value}'. "
                "Only 'pending' or 'initiating' calls can be cancelled."
            ),
        )

    logger.info("Call cancelled via API: call_id=%s", call_id)
    return {"message": f"Call {call_id} cancelled successfully.", "call_id": call_id}
