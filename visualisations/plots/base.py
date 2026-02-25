"""Base class for all plot types.

To add a new plot:
1. Create a new file in plots/ (e.g. plots/my_plot.py).
2. Subclass BasePlot, implement label, description, file_type, sidebar_widgets, and render.
3. Set file_type to "graph" for JSON graph files or "csv" for CSV files.
4. That's it! The class will be auto-discovered if it ends with 'Plot'.

The dashboard automatically discovers plot classes and loads the correct file type.
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

    #: File type this plot expects: "graph" (JSON graph files) or "csv" (CSV files)
    file_type: str = "graph"

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
