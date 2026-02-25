"""Interactive multilayer network (Bokeh) plot."""
from __future__ import annotations

from pathlib import Path

import panel as pn
from bokeh.models import (
    ArrowHead, Circle, ColumnDataSource, CustomJS,
    HoverTool, Label, MultiLine, PointDrawTool, VeeHead,
)
from bokeh.plotting import figure

from config import LAYER_COLORS
from plots.base import BasePlot
from utils import horizontal_band_layout


class NetworkPlot(BasePlot):
    label = "Network"
    description = "Multilayer graph snapshot — drag nodes to rearrange"
    file_type = "graph"

    # ── Sidebar ──────────────────────────────────────────────────────────────

    def sidebar_widgets(self) -> pn.viewable.Viewable:
        return pn.WidgetBox(
            "Parameters",
            self.w.max_frames_input,
            self.w.range_start,
            self.w.range_end,
            self.w.graph_slider,
        )

    # ── Render ───────────────────────────────────────────────────────────────

    def render(self, selected_path: Path) -> pn.viewable.Viewable:
        graph_idx = self.w.graph_slider.value
        graph = self.data_mgr.get_frame(selected_path, graph_idx, self.w)
        if graph is None:
            return self._warn("Graph not found")

        pos = horizontal_band_layout(graph)

        # ── Node data ────────────────────────────────────────────────────
        node_x, node_y, node_colors, node_labels, node_layers = [], [], [], [], []
        node_indices: dict = {}

        for idx, node in enumerate(graph.nodes()):
            layer = graph.nodes[node].get("layer", "Unknown")
            x, y = pos[node]
            node_x.append(x * 500)
            node_y.append(y * 500)
            node_colors.append(LAYER_COLORS.get(layer, "#CCCCCC"))
            node_labels.append(str(node))
            node_layers.append(layer)
            node_indices[node] = idx

        # ── Edge data ────────────────────────────────────────────────────
        edge_xs, edge_ys, edge_colors, edge_widths, edge_dashes = [], [], [], [], []
        edge_sources, edge_targets, edge_sizes, edge_counts = [], [], [], []

        for u, v, data in graph.edges(data=True):
            lu = graph.nodes[u].get("layer")
            lv = graph.nodes[v].get("layer")
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_xs.append([x0 * 500, x1 * 500])
            edge_ys.append([y0 * 500, y1 * 500])
            edge_sources.append(node_indices[u])
            edge_targets.append(node_indices[v])
            edge_sizes.append(data.get("size", "N/A"))
            edge_counts.append(data.get("count", "N/A"))
            same_layer = lu == lv
            edge_colors.append("#666666" if same_layer else "#999999")
            edge_dashes.append("solid" if same_layer else "dashed")
            edge_widths.append(2)

        # ── Bokeh figure ─────────────────────────────────────────────────
        plot = figure(
            title="Multilayer Network (drag nodes to rearrange)",
            width=1000, height=600,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            toolbar_location="above",
        )

        edge_src = ColumnDataSource(dict(
            xs=edge_xs, ys=edge_ys, colors=edge_colors,
            widths=edge_widths, line_dash=edge_dashes,
            sources=edge_sources, targets=edge_targets,
            sizes=edge_sizes, counts=edge_counts,
        ))
        node_src = ColumnDataSource(dict(
            x=node_x, y=node_y, colors=node_colors,
            labels=node_labels, layers=node_layers,
        ))

        edge_r = plot.multi_line(
            "xs", "ys", source=edge_src,
            line_width="widths", color="colors",
            line_dash="line_dash", alpha=0.6,
        )
        node_r = plot.scatter(
            "x", "y", size=20, source=node_src,
            fill_color="colors", line_color="black",
            line_width=2, alpha=0.9,
        )
        plot.text(
            "x", "y", text="labels", source=node_src,
            text_align="center", text_baseline="top",
            y_offset=-25, text_font_size="10pt",
        )

        plot.add_tools(HoverTool(renderers=[node_r],
                                  tooltips=[("Node", "@labels"), ("Layer", "@layers")]))
        plot.add_tools(HoverTool(renderers=[edge_r],
                                  tooltips=[("Size", "@sizes"), ("Count", "@counts")],
                                  line_policy="interp"))

        draw_tool = PointDrawTool(renderers=[node_r], add=False)
        plot.add_tools(draw_tool)
        plot.toolbar.active_tap = draw_tool
        plot.axis.visible = False
        plot.grid.visible = False
        plot.outline_line_color = None

        # JS callback to keep edges connected when nodes are dragged
        callback = CustomJS(
            args=dict(node_source=node_src, edge_source=edge_src),
            code="""
                const nd = node_source.data, ed = edge_source.data;
                for (let i = 0; i < ed.sources.length; i++) {
                    const s = ed.sources[i], t = ed.targets[i];
                    ed.xs[i] = [nd.x[s], nd.x[t]];
                    ed.ys[i] = [nd.y[s], nd.y[t]];
                }
                edge_source.change.emit();
            """,
        )
        node_src.js_on_change("data", callback)

        return pn.pane.Bokeh(plot, sizing_mode="stretch_height")
