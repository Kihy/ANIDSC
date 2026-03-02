"""Base class for all plot types.

To add a new plot:
1. Create a new file in plots/ (e.g. plots/my_plot.py).
2. Subclass BasePlot, implement label, description, file_type, sidebar_widgets, and render.
3. Set file_type to a list of expected filenames (e.g., ["results.csv"], ["summary.json"], ["input_graph_features.ndjson"]).
4. That's it! The class will be auto-discovered if it ends with 'Plot'.

The dashboard automatically discovers plot classes and validates file names against file_type.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import panel as pn

if TYPE_CHECKING:
    from dashboard.data_manager import DataManager
    from dashboard.widget_manager import WidgetManager


class BasePlot(ABC):
    """Contract every plot mode must satisfy."""

    #: Short name shown in the mode selector button group.
    label: str = ""

    #: Optional one-liner shown as a tooltip or sub-heading.
    description: str = ""

    #: List of expected filenames this plot can process (e.g., ["results.csv"], ["summary.json"])
    file_type: list = []

    def __init__(self, data_mgr: "DataManager", widgets: "WidgetManager"):
        self.data_mgr = data_mgr
        self.w = widgets

    # ── Override these ────────────────────────────────────────────────────

    @abstractmethod
    def sidebar_widgets(self) -> pn.viewable.Viewable:
        """Return a Panel component containing this mode's sidebar controls."""

    @abstractmethod
    def render(self, selected_path: Path) -> pn.viewable.Viewable:
        """Build and return the plot pane(s) for the selected path."""

    # ── Shared helpers ────────────────────────────────────────────────────

    def _warn(self, msg: str) -> pn.pane.Markdown:
        return pn.pane.Markdown(f"⚠️ {msg}")
