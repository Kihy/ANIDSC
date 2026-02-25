"""Dashboard entry point.

Run with:
    panel serve main.py --show
or:
    python main.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import panel as pn
import holoviews as hv
import scienceplots  # noqa: F401

plt.style.use(["science", "ieee"])
pn.extension("tabulator", "plotly", "ipywidgets")
hv.extension("bokeh")

from config import ROOT
from data_manager import DataManager
from widget_manager import WidgetManager
from plot_manager import PlotManager


def build_dashboard() -> pn.template.FastListTemplate:
    data_mgr = DataManager(ROOT)
    widget_mgr = WidgetManager(data_mgr)

    plot_container = pn.Column(
        pn.pane.Markdown(
            "📊 Select a data path, choose a view mode, configure parameters, then click **Generate**."
        ),
        sizing_mode="stretch_width",
    )

    plot_mgr = PlotManager(data_mgr, widget_mgr, plot_container)

    # Info card for the header
    info_card = pn.Card(
        pn.pane.Markdown("""
        **Quick Start:** Navigate data folders → Select plot type → Configure parameters → Generate visualization
        
        **Plot Types:** Node (time series) | Network (graph) | Scores (distributions) | Summary (metrics)
        """),
        title="📖 Guide",
        collapsed=True,
        header_background="#e8f4f8",
        sizing_mode="stretch_width",
    )


    # Main layout with plot area and info panel side by side
    main_layout = pn.Row(
        plot_container,
        sizing_mode="stretch_both",
    )

    template = pn.template.MaterialTemplate(
        site="Network Analysis",
        title="Intrusion Detection Dashboard",
        sidebar=[plot_mgr.sidebar()],
        main=[main_layout],
        sidebar_width=350,

    )
    template.servable()
    return template


if __name__.startswith("bokeh") or __name__ == "__main__":
    build_dashboard()
