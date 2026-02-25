"""Widget manager: path navigation and shared sidebar controls.

The manager owns *shared* widgets (progress, max-frames, etc.) and the
path-breadcrumb navigator.  Plot-specific widgets live in each BasePlot
subclass and are injected into the sidebar by PlotManager.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

import networkx as nx
import panel as pn

from config import MAC_TO_DEVICE

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

        # ── Shared controls used by multiple plots ───────────────────────
        self.max_frames_input = pn.widgets.IntInput(
            name="Max Frames", value=10_000, step=100, start=1,
            sizing_mode="stretch_width",
        )
        self.range_start = pn.widgets.IntInput(
            name="Range Start", step=100, value=0, start=0,
            sizing_mode="stretch_width",
        )
        self.range_end = pn.widgets.IntInput(
            name="Range End", step=100, value=10_000, start=1,
            sizing_mode="stretch_width",
        )

        # Node plot
        self.node_input = pn.widgets.Select(
            name="Node", options=["None"], sizing_mode="stretch_width",
        )
        # Network plot
        self.graph_slider = pn.widgets.EditableIntSlider(
            name="Graph Index", start=0, end=0, sizing_mode="stretch_width",
        )
        # Summary plot
        self.metric_input = pn.widgets.Select(
            name="Metric",
            options=[
                "mAP", "f1", "accuracy_binary", "accuracy_device",
                "accuracy_attack", "accuracy_full", "hmean_acc",
                "average time", "total positive",
            ],
            sizing_mode="stretch_width",
        )
        self.pipeline_regex = pn.widgets.TextInput(
            name="Pipeline Regex", placeholder="Regex to filter pipeline names",
            sizing_mode="stretch_width",
        )
        self.plot_height = pn.widgets.IntInput(
            name="Plot Height", step=1, value=2, sizing_mode="stretch_width",
        )

        # ── Progress / status ────────────────────────────────────────────
        self.progress = pn.widgets.Progress(
            name="Loading", sizing_mode="stretch_width", visible=False,
        )
        self.status = pn.pane.Markdown("", sizing_mode="stretch_width", visible=False)

        # ── Action buttons ───────────────────────────────────────────────
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
        """Populate widgets based on the selected file and current plot type."""
        self.disable(True)
        try:
            # Get current plot's file type preference
            current_plot = self.current_plot_callback() if self.current_plot_callback else None
            file_type = current_plot.file_type if current_plot else "graph"
            
            if file_type == "graph":
                # Load graph files and populate node/graph widgets
                frames = list(self.data_mgr.load_frames(file_path, self))
                if frames:
                    ids: set = set()
                    for fr in frames:
                        ids.update(nx.get_node_attributes(fr, "id").values())
                    self.node_input.options = [MAC_TO_DEVICE.get(h, h) for h in ids]
                    self.graph_slider.start = 0
                    self.graph_slider.end = max(0, len(frames) - 1)
            elif file_type == "csv":
                # For CSV files, just validate the file can be read
                # The actual loading happens in the plot's render() method
                try:
                    df = self.data_mgr.read_csv(file_path, self)
                    # CSV loaded successfully - no specific widgets to populate
                except Exception as e:
                    print(f"Warning: Could not load CSV file {file_path}: {e}")
        finally:
            self.disable(False)

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
