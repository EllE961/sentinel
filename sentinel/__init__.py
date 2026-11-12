"""sentinel: production ML monitoring for a real, already-trained recommender.

Tracks two signals that are not the same thing and don't move together: input drift (does the
traffic look different from what the model was trained on) and output quality (is the model's
ranking actually getting worse). Built on `compass-recsys` -- the model being monitored, imported
as a tool, not retrained or modified.
"""

from __future__ import annotations

__all__ = ["drift", "quality", "pipeline"]
