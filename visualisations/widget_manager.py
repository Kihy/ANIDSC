"""Widget manager: path navigation and shared sidebar controls.

The manager owns navigation widgets (path breadcrumbs), action buttons
(generate, clear, download), and progress/status indicators. 
Plot-specific widgets are defined in each BasePlot subclass.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

import panel as pn

if TYPE_CHECKING:
    from data_manager import DataManager


@dataclass
class WidgetManager:
    data_mgr: "DataManager"
    current_plot_callback: Optional[callable] = None  # Returns current BasePlot instance

    def __post_init__(self):
        # ── Path breadcrumb navigator ────────────────────────────────────
        self._selectors: List[dict] = []          # [{widget, path}, ...]
        self.path_container = pn.Column(sizing_mode="stretch_width")
        self._add_selector(self.data_mgr.root, level=0)

        # ── Action buttons and progress indicators ──────────────────────────
        self.generate_btn = pn.widgets.Button(
            name="Generate", icon="caret-right",
            button_type="primary", sizing_mode="stretch_width",
        )
        self.clear_btn = pn.widgets.Button(
            name="Clear", icon="trash",
            button_type="warning", sizing_mode="stretch_width",
        )
        self.download_csv = pn.widgets.FileDownload(
            filename="visible_traces.csv",
            callback=lambda: "No plot available",
            button_type="success",
            label="⬇️ Download Visible Traces (CSV)",
            sizing_mode="stretch_width",
        )
        self.progress = pn.widgets.Progress(
            name="Loading", sizing_mode="stretch_width", visible=False,
        )
        self.status = pn.pane.Markdown("", sizing_mode="stretch_width", visible=False)

        # ── Widgets to disable during loading ────────────────────────────
        self._disable_targets = [
            self.generate_btn, self.clear_btn, self.download_csv,
        ]

    # ── Path breadcrumb ──────────────────────────────────────────────────────

    def _add_selector(self, current_path: Path, level: int):
        items = self.data_mgr.list_dir(current_path)
        label = current_path.name if level > 0 else "Root"
        widget = pn.widgets.Select(
            name=f"{'  ' * level}📁 {label}",
            options=[""] + items,
            value="",
            sizing_mode="stretch_width",
        )
        self._selectors = self._selectors[:level]
        self._selectors.append({"widget": widget, "path": current_path})
        widget.param.watch(lambda e, lvl=level: self._on_select(e, lvl), "value")
        self._refresh_path_container()

    def _refresh_path_container(self):
        self.path_container.clear()
        # Show a compact breadcrumb string above the selectors
        crumb = self._breadcrumb_str()
        if crumb:
            self.path_container.append(
                pn.pane.Markdown(f"📂 `{crumb}`", sizing_mode="stretch_width")
            )
        for info in self._selectors:
            self.path_container.append(info["widget"])

    def _breadcrumb_str(self) -> str:
        parts = []
        for info in self._selectors:
            val = info["widget"].value
            if val:
                parts.append(val)
        return " / ".join(parts)

    def _on_select(self, event, level: int):
        if not event.new:
            self._selectors = self._selectors[: level + 1]
            self._refresh_path_container()
            return

        current_path = self._selectors[level]["path"] / event.new
        self._selectors = self._selectors[: level + 1]

        if current_path.is_file():
            self._refresh_path_container()
            self._on_file_selected(current_path)
        elif current_path.is_dir():
            self._add_selector(current_path, level + 1)

    def _on_file_selected(self, file_path: Path):
        """Handle file selection (loading is deferred to Generate button click)."""
        # Just update the breadcrumb; actual data loading happens on Generate
        self._refresh_path_container()

    # ── Path accessors ───────────────────────────────────────────────────────

    def get_selected_path(self) -> Optional[Path]:
        path = self.data_mgr.root
        for info in self._selectors:
            val = info["widget"].value
            if val:
                path = path / val
            else:
                break
        return path if path != self.data_mgr.root else None

    def get_selected_path_level(self, n: int) -> str:
        """Return the selected value at selector index *n* (supports negative indexing)."""
        try:
            return self._selectors[n]["widget"].value
        except IndexError:
            return ""

    # ── Enable / disable ────────────────────────────────────────────────────

    def disable(self, state: bool):
        for w in self._disable_targets:
            w.disabled = state
        for info in self._selectors:
            info["widget"].disabled = state

    # ── Sidebar layout (assembled by PlotManager) ────────────────────────────

    def path_section(self) -> pn.Column:
        return pn.Column(
            pn.pane.Markdown("### 📁 Data Path", sizing_mode="stretch_width"),
            self.path_container,
            sizing_mode="stretch_width",
        )

    def action_row(self) -> pn.Row:
        return pn.Row(self.generate_btn, self.clear_btn, sizing_mode="stretch_width")
