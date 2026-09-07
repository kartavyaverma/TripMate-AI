"""
app/monitoring/guardrail_monitor.py

Single Responsibility: track guardrail allow/block/fallback events and
raise rolling-window alerts. Nothing here knows about LangGraph, FastAPI,
or the LLM — it is a plain, thread-safe in-memory metrics store that any
caller can record events into and read metrics from.
"""

from __future__ import annotations

import datetime
import logging
import threading
from collections import deque
from typing import Any

logger = logging.getLogger("tripmate.guardrail")

class GuardrailMonitor:
    """
    Production-grade guardrail monitoring and fallback tracker.

    Tracks allowed, blocked, and fallback events in real-time with a
    rolling window, computes fallback rates, and triggers alerts when
    error/fallback rates spike above defined thresholds.
    """

    def __init__(
        self,
        window_size: int = 100,
        alert_threshold: float = 0.20,
        min_sample_size: int = 5,
    ):
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        self.min_sample_size = min_sample_size
        self.lock = threading.Lock()
        self.total_requests = 0
        self.allowed_count = 0
        self.blocked_count = 0
        self.fallback_count = 0
        self.recent_events: deque = deque(maxlen=window_size)
        self.alert_history: deque = deque(maxlen=50)
        self.current_alert: dict[str, Any] | None = None

    def record_event(
        self,
        status: str,
        latency_ms: float,
        query: str,
        reason: str = "",
        error: str | None = None,
    ) -> dict[str, Any]:
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        event = {
            "timestamp": timestamp,
            "status": status,
            "latency_ms": round(latency_ms, 2),
            "query_snippet": query[:120],
            "reason": reason,
            "error": error,
        }

        with self.lock:
            self.total_requests += 1
            if status == "allowed":
                self.allowed_count += 1
            elif status == "blocked":
                self.blocked_count += 1
            elif status == "fallback":
                self.fallback_count += 1

            self.recent_events.append(event)
            self._check_alerts()

        if status == "fallback":
            logger.warning(
                "Guardrail fallback event triggered! Error: %s | Query: %s",
                error,
                query[:80],
            )

        return event

    def _check_alerts(self) -> None:
        if len(self.recent_events) < self.min_sample_size:
            self.current_alert = None
            return

        window_fallbacks = sum(
            1 for e in self.recent_events if e["status"] == "fallback"
        )
        window_total = len(self.recent_events)
        rate = window_fallbacks / window_total

        if rate >= self.alert_threshold:
            alert = {
                "triggered_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "level": "CRITICAL" if rate >= 0.40 else "WARNING",
                "fallback_rate": round(rate * 100, 1),
                "threshold": round(self.alert_threshold * 100, 1),
                "sample_size": window_total,
                "fallback_count": window_fallbacks,
                "message": (
                    f"Guardrail fallback rate spike detected: {rate * 100:.1f}% "
                    f"(threshold: {self.alert_threshold * 100:.1f}%) across last "
                    f"{window_total} requests."
                ),
            }
            self.current_alert = alert
            self.alert_history.append(alert)
            logger.error("GUARDRAIL ALERT TRIGGERED: %s", alert["message"])
        else:
            self.current_alert = None

    def get_metrics(self) -> dict[str, Any]:
        with self.lock:
            window_total = len(self.recent_events)
            window_fallbacks = sum(
                1 for e in self.recent_events if e["status"] == "fallback"
            )
            window_fallback_rate = (
                (window_fallbacks / window_total) if window_total > 0 else 0.0
            )

            return {
                "total_requests": self.total_requests,
                "allowed_count": self.allowed_count,
                "blocked_count": self.blocked_count,
                "fallback_count": self.fallback_count,
                "lifetime_fallback_rate": round(
                    (self.fallback_count / self.total_requests * 100)
                    if self.total_requests > 0
                    else 0.0,
                    1,
                ),
                "window_size": self.window_size,
                "window_requests": window_total,
                "window_fallback_count": window_fallbacks,
                "window_fallback_rate": round(window_fallback_rate * 100, 1),
                "alert_threshold_percent": round(self.alert_threshold * 100, 1),
                "is_alerting": self.current_alert is not None,
                "current_alert": self.current_alert,
                "recent_events": list(self.recent_events)[-10:],
                "recent_alerts": list(self.alert_history)[-5:],
            }

    def reset(self) -> None:
        with self.lock:
            self.total_requests = 0
            self.allowed_count = 0
            self.blocked_count = 0
            self.fallback_count = 0
            self.recent_events.clear()
            self.alert_history.clear()
            self.current_alert = None

guardrail_monitor = GuardrailMonitor()

def get_guardrail_metrics() -> dict[str, Any]:
    return guardrail_monitor.get_metrics()
