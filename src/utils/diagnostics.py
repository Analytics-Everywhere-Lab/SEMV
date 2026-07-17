from __future__ import annotations

from contextvars import ContextVar
from typing import Any


_case_id: ContextVar[str | None] = ContextVar("diagnostic_case_id", default=None)
_events: ContextVar[list[dict[str, Any]] | None] = ContextVar("diagnostic_events", default=None)


def start_diagnostics(case_id: str) -> None:
    _case_id.set(case_id)
    _events.set([])


def record_fallback(
    component: str,
    exception: BaseException,
    fallback_used: str,
    *,
    case_id: str | None = None,
    claim_id: str | None = None,
    uncertainty: bool = True,
) -> dict[str, Any]:
    event = {
        "component": component,
        "case_id": case_id or _case_id.get(),
        "claim_id": claim_id,
        "exception_category": type(exception).__name__,
        "fallback_used": fallback_used,
        "uncertainty": uncertainty,
    }
    events = _events.get()
    if events is not None:
        events.append(event)
    return event


def diagnostic_events() -> list[dict[str, Any]]:
    return list(_events.get() or [])


def diagnostic_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in diagnostic_events():
        component = str(event["component"])
        counts[component] = counts.get(component, 0) + 1
    return counts
