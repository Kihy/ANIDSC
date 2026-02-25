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
    file_type = "csv"

    # ── Sidebar ──────────────────────────────────────────────────────────────

    def sidebar_widgets(self) -> pn.viewable.Viewable:
        return pn.WidgetBox(
            "Parameters",
            self.w.max_frames_input,
            self.w.range_start,
            self.w.range_end,
        )

    # ── Render ───────────────────────────────────────────────────────────────

    def render(self, selected_path: Path) -> pn.viewable.Viewable:
        if not selected_path.exists():
            return self._warn(f"Score CSV not found: {selected_path}")

        df = self.data_mgr.read_csv(selected_path, self.w)
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
