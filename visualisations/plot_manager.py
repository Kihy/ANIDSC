"""Plot manager: wires the plot registry into Panel widgets and handles events."""
from __future__ import annotations

import io
import traceback
from pathlib import Path
from typing import Dict, List

import pandas as pd
import panel as pn

from data_manager import DataManager
from widget_manager import WidgetManager
from plots import REGISTRY
from plots.base import BasePlot


class PlotManager:
    """
    Owns the plot container and the mode-selector.
    Instantiates every registered BasePlot and routes generate/clear clicks.
    """

    def __init__(self, data_mgr: DataManager, widgets: WidgetManager, container: pn.Column):
        self.data_mgr = data_mgr
        self.w = widgets
        self.plot_container = container

        # Instantiate each registered plot class
        self._plots: Dict[str, BasePlot] = {
            cls.label: cls(data_mgr, widgets) for cls in REGISTRY
        }
        self._labels: List[str] = [cls.label for cls in REGISTRY]

        # Mode selector (dropdown driven by the registry)
        self.mode_selector = pn.widgets.Select(
            name="View Mode",
            options=self._labels,
            value=self._labels[0] if self._labels else None,
            sizing_mode="stretch_width",
        )

        # Provide widget_manager with access to the current plot
        self.w.current_plot_callback = self._current_plot

        # Dynamic sidebar section: reacts to mode changes
        self._sidebar_widgets_pane = pn.Column(
            self._current_plot().sidebar_widgets(),
            sizing_mode="stretch_width",
        )
        self.mode_selector.param.watch(self._on_mode_change, "value")

        # Wire buttons
        self.w.generate_btn.on_click(self._generate)
        self.w.clear_btn.on_click(self._clear)
        self.w.download_csv.callback = self._csv_export

    # ── Mode switching ───────────────────────────────────────────────────────

    def _current_plot(self) -> BasePlot:

        return self._plots[self.mode_selector.value]

    def _on_mode_change(self, _):
        self._sidebar_widgets_pane.clear()
        self._sidebar_widgets_pane.append(self._current_plot().sidebar_widgets())

    # ── Generate / clear ─────────────────────────────────────────────────────

    def _clear(self, _=None):
        self.plot_container.clear()

    def _generate(self, _=None):
        selected_path = self.w.get_selected_path()
        if selected_path is None:
            self.plot_container.append(pn.pane.Markdown("⚠️ No path selected"))
            return

        self.w.disable(True)
        try:
            result = self._current_plot().render(selected_path)
            self.plot_container.append(result)
        except Exception:
            self.plot_container.append(
                pn.pane.HTML(
                    "<div style='padding:20px;color:#e74c3c;'>"
                    f"❌ <pre>{traceback.format_exc()}</pre></div>"
                )
            )
        finally:
            self.w.disable(False)

    # ── CSV export ───────────────────────────────────────────────────────────

    def _csv_export(self) -> io.StringIO | str:
        plotly_pane = next(
            (o for o in self.plot_container.objects if isinstance(o, pn.pane.Plotly)),
            None,
        )
        if plotly_pane is None:
            return "No plot available"
        fig = plotly_pane.object
        out_df = pd.DataFrame()
        for trace in fig.data:
            if trace.visible is None or trace.visible is True:
                if "x" not in out_df:
                    out_df["x"] = trace.x
                out_df[trace.name] = trace.y
        sio = io.StringIO()
        out_df.to_csv(sio, index=False)
        sio.seek(0)
        return sio

    # ── Full sidebar ─────────────────────────────────────────────────────────

    def sidebar(self) -> pn.Column:
        return pn.Column(
            pn.pane.Markdown("## 🔍 View Mode"),
            self.mode_selector,
            pn.layout.Divider(),
            self.w.path_section(),
            pn.layout.Divider(),
            pn.pane.Markdown("### ⚙️ Parameters"),
            self._sidebar_widgets_pane,
            pn.layout.Divider(),
            self.w.action_row(),
            self.w.download_csv,
            self.w.progress,
            self.w.status,
            sizing_mode="stretch_width",
        )
