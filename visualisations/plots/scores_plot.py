"""Score distribution time-series plot (median, IQR, range, threshold)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import panel as pn
import plotly.graph_objects as go

from plots.base import BasePlot


class ScoresPlot(BasePlot):
    label = "Scores"
    description = "Score distribution (median / IQR / range) over time"
    file_type = ["results.csv"]

    def __init__(self, data_mgr, widgets):
        super().__init__(data_mgr, widgets)
        
        # Plot-specific widgets
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

    # ── Sidebar ──────────────────────────────────────────────────────────────

    def sidebar_widgets(self) -> pn.viewable.Viewable:
        return pn.WidgetBox(
            "Parameters",
            self.max_frames_input,
            self.range_start,
            self.range_end,
        )

    # ── Render ───────────────────────────────────────────────────────────────

    def render(self, selected_path: Path) -> pn.viewable.Viewable:
        if not selected_path.exists():
            return self._warn(f"Score CSV not found: {selected_path}")
        
        if not any(selected_path.name.startswith(f) for f in self.file_type):
            return self._warn(f"Expected file starting with {self.file_type[0]}, got {selected_path.name}")

        df = self.data_mgr.read_csv(selected_path, self)
        times = pd.to_datetime(df["timestamp"], unit="s")

        fig = go.Figure()

        # Min-max band
        fig.add_trace(go.Scatter(x=times, y=df["soft_max_score"], mode="lines",
                                  line=dict(width=0), name="Max", showlegend=False))
        fig.add_trace(go.Scatter(x=times, y=df["soft_min_score"], mode="lines",
                                  line=dict(width=0), fill="tonexty",
                                  fillcolor="rgba(200,200,200,0.3)", name="Range (Min–Max)"))
        # IQR band
        fig.add_trace(go.Scatter(x=times, y=df["upper_quartile_score"], mode="lines",
                                  line=dict(width=0), name="Q3", showlegend=False))
        fig.add_trace(go.Scatter(x=times, y=df["lower_quartile_score"], mode="lines",
                                  line=dict(width=0), fill="tonexty",
                                  fillcolor="rgba(0,100,200,0.3)", name="IQR (Q1–Q3)"))
        # Median & threshold
        fig.add_trace(go.Scatter(x=times, y=df["median_score"], mode="lines",
                                  line=dict(color="blue", width=1), name="Median"))
        fig.add_trace(go.Scatter(x=times, y=df["median_threshold"], mode="lines",
                                  line=dict(color="red", width=1), name="Threshold"))

        fig.update_layout(
            title=str(selected_path.name),
            xaxis_title="Time", yaxis_title="Score",
            hovermode="x unified", template="plotly_white",
            updatemenus=[dict(
                buttons=[
                    dict(label="Linear", method="relayout", args=[{"yaxis.type": "linear"}]),
                    dict(label="Log",    method="relayout", args=[{"yaxis.type": "log"}]),
                ],
                direction="down", x=1.1, xanchor="left", y=1.15, yanchor="top",
            )],
        )
        return pn.pane.Plotly(fig, config={"responsive": True})
