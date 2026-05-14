# =============================================================================
# simulation/logger.py — Event Logger (Phase 15)
# =============================================================================
# Stores every simulation event with step number, type, message, and timestamp.
# Used by the UI event log panel and the final summary report.
# =============================================================================

from typing import List
from dataclasses import dataclass, field
from datetime import datetime


# Color hints for the UI event log (mapped by event type)
EVENT_COLORS = {
    "LAYOUT":    "gray",
    "ROADS":     "gray",
    "RISK":      "orange",
    "AMBULANCE": "green",
    "FLOOD":     "red",
    "ROUTE":     "blue",
    "DRIFT":     "orange",
    "RETRAIN":   "orange",
    "GA_RERUN":  "green",
    "QUIET":     "lightgray",
    "STATE":     "darkgray",
    "ERROR":     "crimson",
}


@dataclass
class LogEntry:
    step:       int
    event_type: str
    message:    str
    timestamp:  str = field(
        default_factory=lambda: datetime.now().strftime("%H:%M:%S")
    )

    def to_dict(self) -> dict:
        """Serialise to dict for JSON export / server.py API response."""
        return {
            "step":      self.step,
            "type":      self.event_type,
            "message":   self.message,
            "timestamp": self.timestamp,
            "color":     EVENT_COLORS.get(self.event_type, "white"),
        }


class EventLogger:
    """
    Stores and formats all simulation events.

    Used by simulation/loop.py to record every step's events.
    Used by server.py to return events as JSON to the dashboard.
    Used by UI to populate the scrollable event log panel.
    """

    def __init__(self):
        self.entries: List[LogEntry] = []

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log(self, step: int, event_type: str, message: str) -> LogEntry:
        """
        Record one event. Prints to console immediately.
        Returns the created LogEntry (useful for tests).
        """
        entry = LogEntry(step=step, event_type=event_type, message=message)
        self.entries.append(entry)
        color_tag = EVENT_COLORS.get(event_type, "")
        print(f"  [{entry.timestamp}][Step {step:2d}][{event_type:9s}] {message}")
        return entry

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_recent(self, n: int = 20) -> List[LogEntry]:
        """Returns the n most recent entries (newest last)."""
        return self.entries[-n:]

    def get_by_type(self, event_type: str) -> List[LogEntry]:
        """Returns all entries matching the given event type."""
        return [e for e in self.entries if e.event_type == event_type]

    def get_by_step(self, step: int) -> List[LogEntry]:
        """Returns all entries for a given simulation step."""
        return [e for e in self.entries if e.step == step]

    def to_json_list(self) -> List[dict]:
        """Serialise all entries to a list of dicts for JSON export."""
        return [e.to_dict() for e in self.entries]

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def print_summary(self) -> None:
        """Prints a count-by-type summary to the console."""
        print("\n── EVENT SUMMARY " + "─" * 38)
        for etype in ["FLOOD", "ROUTE", "DRIFT", "RETRAIN", "GA_RERUN", "AMBULANCE"]:
            count = len(self.get_by_type(etype))
            if count:
                print(f"  {etype:10s}: {count} event(s)")
        print(f"  {'─'*20}")
        print(f"  TOTAL LOG ENTRIES : {len(self.entries)}")
        print("─" * 55 + "\n")
