from __future__ import annotations
import re
from collections import defaultdict
from dataclasses import dataclass
import io
from pathlib import Path
import json
import os
import random
import re
import time
import traceback
from typing import Dict, Iterable, List, Optional
import networkx as nx
import numpy as np
import pandas as pd
import panel as pn
import plotly.graph_objects as go
import seaborn as sns
import holoviews as hv
import matplotlib.pyplot as plt
import scienceplots #leave this in
from sklearn.metrics import average_precision_score
import fsspec
from bokeh.plotting import figure
from bokeh.models import (HoverTool, Circle, MultiLine, CustomJS, VeeHead, ArrowHead,
                          Label, ColumnDataSource, PointDrawTool)


# keep your plotting style lines (you used scienceplots earlier)
plt.style.use(["science", "ieee"])

pn.extension("tabulator", "plotly", "ipywidgets")
hv.extension("bokeh")


def horizontal_band_layout(G):
    """
    For each layer, compute a force-directed layout on the subgraph containing
    only nodes of that layer, then place the resulting mini-layout onto a fixed
    horizontal band.
    """

    # Group nodes by layer
    groups = {}
    for node, data in G.nodes(data=True):
        key = data.get("layer")
        groups.setdefault(key, []).append(node)

    # Fixed y-positions per layer
    layer_y = {
        'Physical': 0,
        'Internet': 200,
        'Transport': 400
    }

    pos = {}
    


    for layer, nodes in groups.items():
        # Extract subgraph for this layer
        subG = G.subgraph(nodes)

        # Force-directed positions *within the layer*
        sub_pos = nx.kamada_kawai_layout(subG)   # deterministic layout

        # Normalize/shift x positions so nodes spread out nicely
        xs = [p[0] for p in sub_pos.values()]
        min_x, max_x = min(xs), max(xs)
        span = max(max_x - min_x, 1e-6)
        
        # Normalize/shift y positions so nodes spread out nicely
        ys = [p[1] for p in sub_pos.values()]
        min_y, max_y = min(ys), max(ys)
        span_y = max(max_y - min_y, 1e-6)

        
        for node in nodes:
            x = (sub_pos[node][0] - min_x) / span * 800   # scale to width ~800
            y = (sub_pos[node][1] - min_y) / span_y * 100   # scale to width ~100
            # vertically shift y
            y = y+layer_y[layer]
            
            pos[node] = (x, y)
    return pos

# ------------------------------
# Config / constants
# ------------------------------
ROOT = Path("/workspace/intrusion_detection/datasets")

MAC_TO_DEVICE: Dict[str, str] = {
    "52:8e:44:e1:da:9a": "Cam",
    "be:7b:f6:f2:1b:5f": "Attacker",
    "5e:ea:b2:63:fc:aa": "Google-Nest",
    "52:a8:55:3e:34:46": "Lenovo_Bulb",
    "42:7f:83:17:55:c0": "Raspberry Pi",
    "a2:bd:fa:b5:89:92": "Smart_Clock",
    "22:c9:ca:f6:da:60": "Smartphone_1",
    "d2:19:d4:e9:94:86": "Smartphone_2",
    "7e:d1:9d:c4:d1:73": "SmartTV",
}


# ------------------------------
# Utilities & Data I/O
# ------------------------------
class DataManager:
    """Encapsulates file-system operations and frame iteration."""

    def __init__(self, root: Path = ROOT, seed: int = 42):
        self.root = Path(root)
        self.frames = {}
        self.csvs = {}
        random.seed(seed)

    def list_subdirectories(self, path: Optional[Path]) -> List[str]:
        """Return ['None'] or a list of entries inside path (quick fail if invalid)."""
        if path is None or not path.exists() or not path.is_dir():
            return [""]
        time.sleep(0.1)
        return [p.name for p in path.iterdir()]

    def list_files(self, folder: Path) -> List[str]:
        """Return leaf directories (relative paths) under folder."""
        time.sleep(0.2)
        folder = Path(folder)
        leaves = []
        for p in folder.rglob("*"):
            
            if p.is_dir() and not any(child.is_dir() for child in p.iterdir()):
                leaves.append(str(p.relative_to(folder)))
        return leaves

    def _manage_progress_widget(
        self, widget_mgr, visible: bool, active: bool = False, status: str = ""
    ):
        """Helper to manage progress widget visibility and status."""
        if widget_mgr:
            widget_mgr.progress.visible = visible
            widget_mgr.progress.active = active
            widget_mgr.status.visible = visible
            if status:
                widget_mgr.status.object = status

    def _reservoir_sample(
        self, file_handle, max_frames: int, sample_range: tuple = None, widget_mgr=None
    ):
        """Perform reservoir sampling on file lines with optional progress updates."""
        sample = []
        start_idx, end_idx = sample_range if sample_range else (0, float("inf"))

        for i, line in enumerate(file_handle, start=1):
            if i < start_idx:
                continue
            if i > end_idx:
                break

            if len(sample) < max_frames:
                sample.append(line)
            else:
                j = random.randint(1, i)
                if j <= max_frames:
                    sample[j - 1] = line

            if widget_mgr:
                widget_mgr.status.object = f"**{i:,} lines processed...**"

        return sample

    def load_frames(
        self,
        file_path: Path,
        widget_mgr: Optional["WidgetManager"] = None,
    ) -> Iterable[dict]:
        """Yield JSON frames from a file and parse it to networkx graph, sampling by prob, with optional progress updates."""
        file_path = Path(file_path)
        if not file_path.exists():
            return

        max_frames = widget_mgr.max_frames_input.value
        sample_range = (widget_mgr.range_start.value, widget_mgr.range_end.value)
        # Check cache
        cache_key = (file_path, max_frames, sample_range)
        if cache_key in self.frames:
            yield from self.frames[cache_key]
            return

        self._manage_progress_widget(
            widget_mgr, visible=True, active=True, status="**0 lines read...**"
        )

        try:
            with fsspec.open(file_path, "rt", compression="zstd") as f:
                sample = self._reservoir_sample(f, max_frames, sample_range, widget_mgr)

            frames = [nx.readwrite.json_graph.node_link_graph(json.loads(line),edges="links") for line in sample]
            self.frames[cache_key] = frames
            yield from frames

        finally:
            self._manage_progress_widget(widget_mgr, visible=False)

    def read_csv(
        self,
        path: Path,
        widget_mgr: Optional["WidgetManager"] = None,
        max_frames: int = None,
        sample_range: tuple = None,
    ) -> pd.DataFrame:
        """Read CSV with optional Panel progress bar from widget manager (indeterminate)."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        max_frames = widget_mgr.max_frames_input.value
        sample_range = (widget_mgr.range_start.value, widget_mgr.range_end.value)

        # Check cache
        cache_key = (path, max_frames, sample_range)
        if cache_key in self.csvs:
            header = self.csvs[cache_key][0]
            sample = self.csvs[cache_key][1]
        else:

            self._manage_progress_widget(
                widget_mgr, visible=True, active=True, status="**0 rows read...**"
            )
            try:
                with fsspec.open(path, "rt", compression="zstd") as f:
                    header = next(f)
                    sample = self._reservoir_sample(
                        f, max_frames, sample_range, widget_mgr
                    )

                self.csvs[cache_key] = (header, sample)

            finally:
                self._manage_progress_widget(widget_mgr, visible=False)

        df = pd.read_csv(io.StringIO(header + "".join(sample)))
        return df.sort_values(by="timestamp")

    def get_frame(
        self,
        file_path: Path,
        idx: int,
        widget_mgr: Optional["WidgetManager"] = None,
    ) -> Optional[dict]:
        """Return the frame at index idx."""
        for i, frame in enumerate(self.load_frames(file_path, widget_mgr)):
          

            if i == idx:
                return frame
        return None

    def count_frames(
        self,
        file_path: Path,
        widget_mgr: Optional["WidgetManager"] = None,
    ) -> int:
        """Count frames available (after sampling)."""
        return sum(1 for _ in self.load_frames(file_path, widget_mgr))


def simple_pipeline_name(name: str) -> Optional[str]:
    """Extract model name."""

    extracted_name = []

    def find_comp(comp_regex):
        m = re.search(comp_regex, name)
        if m:
            extracted_name.extend(m.groups())

    find_comp(r"GraphProcessor\(([^)]+)\)")

    find_comp(r"BaseNodeEmbedder\(([^)]+)\)")
    find_comp(r"OnlineOD\(([^)]+)\)")

    if "LivePercentile" in name:
        extracted_name.append("scaler")

    return "+".join(extracted_name)


# ------------------------------
# Stats helpers
# ------------------------------
def f_beta(tp: float, fn: float, fp: float, beta: float = 1) -> float:
    """Compute F-beta score."""
    beta2 = beta**2
    denom = (1 + beta2) * tp + beta2 * fn + fp
    return (1 + beta2) * tp / denom if denom else 0.0


def weighted_hmean(precision: float, recall: float, beta: float = 1) -> float:
    """Weighted harmonic mean (used for combined accuracy metric)."""
    beta2 = beta**2
    denom = beta2 * precision + recall
    return (1 + beta2) * precision * recall / denom if denom else 0.0


def calc_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate accuracy and F-measure statistics for a pipeline (same logic as original)."""
    stats = {}

    # benign
    mask_b = df["file"].str.startswith("benign")
    stats["pos_benign"] = df.loc[mask_b, "pos_count"].sum()
    stats["total_benign"] = df.loc[mask_b, "batch_size"].sum()

    # malicious
    mask_m = df["file"].str.startswith("attack")
    stats["pos_malicious"] = df.loc[mask_m, "pos_count"].sum()
    stats["total_malicious"] = df.loc[mask_m, "batch_size"].sum()

    stats["f1"] = f_beta(
        stats["pos_malicious"],
        stats["total_malicious"] - stats["pos_malicious"],
        stats["pos_benign"],
    )

    df["accuracy"] = df["pos_count"] / df["batch_size"]
    df.loc[mask_b, "accuracy"] = 1 - df.loc[mask_b, "accuracy"]

    stats["acc_benign"] = df.loc[mask_b, "accuracy"].mean()
    stats["acc_malicious"] = df.loc[mask_m, "accuracy"].mean()

    stats["hmean_acc"] = weighted_hmean(
        stats["acc_benign"], stats["acc_malicious"], beta=1
    )
    return pd.DataFrame([stats])

# ------------------------------
# Widget manager
# ------------------------------
@dataclass
class WidgetManager:
    data_mgr: DataManager

    def __post_init__(self):
        # Dynamic path selectors
        self.path_selectors = []
        self.path_container = pn.Column(sizing_mode="stretch_width")
        
        # Initialize with root selector
        self._add_path_selector(self.data_mgr.root, level=0)

        # numeric and mode controls
        self.mode_selector = pn.widgets.RadioButtonGroup(
            name="Mode",
            options=["Node", "Network", "Scores", "Summary"],
            value="Node",
            button_type="primary",
            button_style="outline",
        )

        # per-mode controls
        self.node_input = pn.widgets.Select(name="Node", options=["None"])
        self.graph_slider = pn.widgets.EditableIntSlider(name="Graph Index", start=0, end=0)
        self.metric_input = pn.widgets.Select(
            name="Metric",
            options=[
                "mAP",
                "f1",
                "accuracy_binary",
                "accuracy_device",
                "accuracy_attack",
                "accuracy_full",
                "hmean_acc",
                "average time",
                "total positive",
            ],
        )

        # action buttons (bound externally)
        self.generate_btn = pn.widgets.Button(
            name="Generate", icon="caret-right", button_type="primary"
        )
        self.clear_btn = pn.widgets.Button(
            name="Clear Plots", icon="trash", button_type="primary"
        )
        self.download_csv = pn.widgets.FileDownload(
            filename="visible_traces.csv",
            callback=lambda: "No plot available",
            button_type="success",
            label="⬇️ Download Visible Traces (CSV)",
        )

        # pack the dynamic widget placeholder
        self.dynamic_widget = pn.bind(self._switch_widget, self.mode_selector)

        # Progress bar (indeterminate)
        self.progress = pn.widgets.Progress(
            name="Loading Frames",
            sizing_mode="stretch_width",
            visible=False,  # hidden by default
        )
        # Status text showing number of lines read
        self.status = pn.pane.Markdown("", sizing_mode="stretch_width", visible=False)

        # Max frames widget
        self.max_frames_input = pn.widgets.IntInput(
            name="Max Frames",
            value=10000,
            step=100,
            start=1,
            sizing_mode="stretch_width",
        )

        # Sample range widgets
        self.range_start = pn.widgets.IntInput(
            name="Range Start", step=100, value=0, sizing_mode="stretch_width"
        )

        self.range_end = pn.widgets.IntInput(
            name="Range End", step=100, value=10000, sizing_mode="stretch_width"
        )

        self.pipeline_regex = pn.widgets.TextInput(
            name="Pipeline Regex", placeholder="Regex to filter pipeline names",
            sizing_mode="stretch_width"
        )
        
        self.plot_height = pn.widgets.IntInput(
            name="Plot Height", step=1, value=2, sizing_mode="stretch_width"
        )

        # watch wiring
        self.mode_selector.param.watch(self._on_mode_change, "value")

        # small state to allow disabling widgets during long ops
        self._disable_targets = [
            self.metric_input,
            self.node_input,
            self.generate_btn,
            self.download_csv,
            self.mode_selector,
        ]

        for w in self._disable_targets:
            w.sizing_mode = "stretch_width"

    def _add_path_selector(self, current_path, level=0):
        """Add a new path selector at the given level"""
        # Get items at current path
        items = []
        if current_path.exists() and current_path.is_dir():
            items = sorted([
                item.name for item in current_path.iterdir()
                if not item.name.startswith('.')
            ])
        
        # Create selector
        selector = pn.widgets.Select(
            name=f"Level {level}: {current_path.name if level > 0 else 'Root'}",
            options=[""] + items,
            value="",
            sizing_mode="stretch_width"
        )
        
        # Store selector info
        selector_info = {
            'widget': selector,
            'level': level,
            'path': current_path
        }
        
        # Trim selectors beyond this level
        self.path_selectors = self.path_selectors[:level]
        self.path_selectors.append(selector_info)
        
        # Watch for changes
        selector.param.watch(
            lambda event, lvl=level: self._on_path_change(event, lvl),
            'value'
        )
        
        # Update container
        self._update_path_container()
        
        return selector

    def _update_path_container(self):
        """Update the visual container with current selectors"""
        self.path_container.clear()
        for selector_info in self.path_selectors:
            self.path_container.append(selector_info['widget'])

    def _on_path_change(self, event, level):
        """Handle path selector change"""
        if not event.new:
            # Empty selection - remove subsequent selectors
            self.path_selectors = self.path_selectors[:level + 1]
            self._update_path_container()
            return
        
        # Build current path
        current_path = self.path_selectors[level]['path'] / event.new
        
        # Remove selectors after this level
        self.path_selectors = self.path_selectors[:level + 1]
        
        # Check if this is a file or directory
        if current_path.is_file():
            # It's a file - we're done, update dependent widgets
            self._update_path_container()
            self._on_file_selected()
        elif current_path.is_dir():
            # It's a directory - add another selector
            self._add_path_selector(current_path, level + 1)
            
    def _on_file_selected(self):
        """Called when a complete file path is selected"""
        file_path = self.get_selected_path()
        if file_path is None or not file_path.is_file():
            return
            
        self._disable(True)
        try:
            # Update mode-specific widgets based on the selected file
            if self.mode_selector.value in ["Node", "Network"]:
                # Load frames and populate node/graph controls
                frames = list(self.data_mgr.load_frames(file_path, self))
                if frames:

                    # Collect unique node ids
                    node_ids = {nx.get_node_attributes(fr, "id").values() for fr in frames}
                    self.node_input.options = [MAC_TO_DEVICE.get(h, h) for h in node_ids]
                    self.graph_slider.start = 0
                    self.graph_slider.end = max(0, len(frames) - 1)
        finally:
            self._disable(False)

    def get_selected_path(self):
        """Get the currently selected full path"""
        if not self.path_selectors:
            return None
            
        path = self.data_mgr.root
        for selector_info in self.path_selectors:
            value = selector_info['widget'].value
            if value:
                path = path / value
            else:
                break
                
        return path if path != self.data_mgr.root else None
    
    def get_selected_path_level(self, n):
        return self.path_selectors[n]['widget'].value

    def _disable(self, state: bool):
        for w in self._disable_targets:
            w.disabled = state
        for selector_info in self.path_selectors:
            selector_info['widget'].disabled = state

    # --------- widget layout switching ----------
    def _switch_widget(self, mode: str):
        if mode == "Node":
            return pn.WidgetBox(
                "Parameters",
                self.max_frames_input,
                self.range_start,
                self.range_end,
                self.node_input,
            )
        elif mode == "Network":
            return pn.WidgetBox(
                "Parameters",
                self.max_frames_input,
                self.range_start,
                self.range_end,
                self.graph_slider,
            )
        elif mode == "Scores":
            return pn.WidgetBox(
                "Parameters",
                self.max_frames_input,
                self.range_start,
                self.range_end,
            )
        elif mode == "Summary":
            return pn.WidgetBox("Parameters", self.metric_input, self.pipeline_regex, self.plot_height)
        return pn.pane.Markdown("Unknown mode")

    def _on_mode_change(self, _):
        # Mode change might affect which files are relevant
        # Could reset path selectors or filter differently
        pass

    # expose a simple layout helper
    def sidebar(self):
        return pn.Column(
            self.mode_selector,
            pn.pane.Markdown("**Select Path:**"),
            self.path_container,
            self.dynamic_widget,
            pn.Row(
                self.generate_btn,
                self.clear_btn,
            ),
            self.download_csv,
            self.progress,
            self.status,
            sizing_mode="stretch_width",
        )


# ------------------------------
# Plot manager: all plot logic collected here
# ------------------------------
class PlotManager:

    def __init__(
        self, data_mgr: DataManager, widgets: WidgetManager, container: pn.Column
    ):
        self.data_mgr = data_mgr
        self.w = widgets
        self.plot_container = container

        # mode dispatch table
        self.mode_map = {
            "Node": self.node_plot,
            "Network": self.network_plot,
            "Scores": self.scores_plot,
            "Summary": self.summary_plot,
        }

        # bind generate/clear
        self.w.generate_btn.on_click(self.generate_graph)
        self.w.clear_btn.on_click(self.clear)

    def clear(self, _=None):
        """Clear the plot container."""
        self.plot_container.clear()

    def generate_graph(self, _=None):
        """Dispatch to the selected plotting routine with safety wrapper."""
        self.w._disable(True)
        try:
            mode = self.w.mode_selector.value
            plot_fn = self.mode_map.get(mode)
            if plot_fn is None:
                self.plot_container.append(pn.pane.Markdown("Unknown mode"))
                return
            plot_fn()
        except Exception:
            self.plot_container.append(
                pn.pane.HTML(
                    f"<div style='text-align: center; padding: 20px; color: #e74c3c;'>❌ Error generating plot: {traceback.format_exc()}</div>"
                )
            )
        finally:
            self.w._disable(False)

    # ---------- Node plot ----------
    def node_plot(self):
        node_label = self.w.node_input.value
        if not node_label or node_label == "None":
            self.plot_container.append(pn.pane.Markdown("⚠️ Select a file and node"))
            return

        # map displayed device name back to mac address if needed
        actual_node_id = next(
            (mac for mac, dev in MAC_TO_DEVICE.items() if dev == node_label), node_label
        )

        file_path = (
            self.data_mgr.root
            / self.w.dataset_input.value
            / "features"
            / self.w.fe_input.value
            / self.w.file_input.value
            / self.w.pipeline_input.value
        )
        if not file_path.exists():
            self.plot_container.append(
                pn.pane.Markdown(f"⚠️ File not found: {file_path}")
            )
            return

        data_dict = defaultdict(list)
        prev_concept = None
        concept_change_indices = []
        concept_id_order = []
        unique_concepts = set()
        time_val = None

        # iterate frames
        for fr in self.data_mgr.load_frames(file_path, self.w):
            # Find node for this frame
            node = next(
                (
                    n
                    for n in fr["elements"]["nodes"]
                    if n["data"]["id"] == actual_node_id
                ),
                None,
            )
            # extract main frame-level keys (time_stamp, threshold)
            if node:
                for key, value in fr.get("data", []):

                    if key == "time_stamp":
                        time_val = pd.to_datetime(
                            float(value), unit="s", origin="unix"
                        ).tz_localize("Pacific/Auckland")
                        data_dict["times"].append(time_val)
                    elif key == "threshold":
                        data_dict["threshold"].append(value)

                concept_id = str(node["data"].get("concept_id"))

                if concept_id != "-1":
                    if prev_concept != concept_id or prev_concept is None:
                        prev_concept = concept_id
                        concept_id_order.append(concept_id)
                        concept_change_indices.append(time_val)
                        unique_concepts.add(concept_id)

                # collect node fields (safe-get)
                data_dict["node_as_vals"].append(node["data"].get("node_as"))
                data_dict["count_vals"].append(node["data"].get("count"))
                data_dict["size_vals"].append(node["data"].get("size"))
                data_dict["original_count_vals"].append(
                    node["data"].get("original_count")
                )
                data_dict["original_size_vals"].append(
                    node["data"].get("original_size")
                )

        if not data_dict["times"]:
            self.plot_container.append(
                pn.pane.Markdown(f"⚠️ No data found for node {node_label}")
            )
            return

        if time_val is not None:
            concept_change_indices.append(time_val)

        # color mapping for concepts
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        color_map = {
            concept: colors[i % len(colors)]
            for i, concept in enumerate(sorted(unique_concepts))
        }

        df = pd.DataFrame(data_dict).sort_values("times")

        # Build the plotly figure (keeps the same traces & names as original)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df["times"],
                y=df["node_as_vals"],
                mode="markers",
                name="Node Anomaly Score",
                line=dict(color="black"),
            )
        )
        if "threshold" in df:
            fig.add_trace(
                go.Scatter(
                    x=df["times"],
                    y=df["threshold"],
                    mode="markers",
                    name="Threshold",
                    line=dict(color="gray", dash="dash"),
                )
            )
        fig.add_trace(
            go.Scatter(
                x=df["times"],
                y=df["count_vals"],
                mode="markers",
                name="Count",
                line=dict(color="red"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["times"],
                y=df["size_vals"],
                mode="markers",
                name="Size",
                line=dict(color="blue"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["times"],
                y=df["original_count_vals"],
                mode="markers",
                name="Original Count",
                line=dict(color="orange", dash="dot"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["times"],
                y=df["original_size_vals"],
                mode="markers",
                name="Original Size",
                line=dict(color="green", dash="dot"),
            )
        )

        print(len(concept_change_indices))
        print(concept_id_order)

        if len(concept_id_order) < 20:
            print("plotting concepts")
            for i, (prev_idx, cur_idx) in enumerate(
                zip(concept_change_indices[:-1], concept_change_indices[1:])
            ):
                concept = concept_id_order[i]
                fig.add_vrect(
                    x0=prev_idx,
                    x1=cur_idx,
                    fillcolor=color_map.get(concept, "#cccccc"),
                    opacity=0.1,
                    annotation_text=f"{concept}",
                    annotation_position="top left",
                    layer="below",
                    line_width=0,
                )
        else:
            print("concepts greater than 20, skipping")

        fig.update_layout(
            title=f"Node {node_label}: Values Over Time",
            xaxis_title="Time",
            yaxis_title="Value",
            width=1200,
            height=700,
            legend=dict(x=1.02, y=1),
            hovermode="x unified",
        )

        self.plot_container.append(pn.pane.Plotly(fig, config={"responsive": True}))

    # ---------- Network (graph) plot ----------
    def network_plot(self):
        graph_idx = self.w.graph_slider.value
        file_path = file_path = self.w.get_selected_path()
        
        graph = self.data_mgr.get_frame(file_path, graph_idx, self.w)
        if graph is None:
            self.plot_container.append(pn.pane.Markdown("⚠️ Graph not found"))
            return
                
        # Layer colors
        layer_colors = {'Physical': '#FF8C42', 'Internet': '#4ECDC4', 'Transport': '#95E06C'}
        
        # Calculate node positions for each layer
        pos = horizontal_band_layout(graph)
        # pos=nx.multipartite_layout(self.graph, subset_key="layer")
        
        # Prepare node data
        node_x, node_y, node_colors, node_labels, node_layers = [], [], [], [], []
        node_indices = {}
        
        for idx, node in enumerate(graph.nodes()):
            layer = graph.nodes[node].get("layer", "Unknown")
            x, y = pos[node]
            
            node_x.append(x * 500)
            node_y.append(y * 500)
            node_colors.append(layer_colors.get(layer, '#CCCCCC'))
            node_labels.append(str(node))
            node_layers.append(layer)
            node_indices[node] = idx
    
        
        # Prepare edge data
        edge_xs, edge_ys, edge_colors, edge_widths, edge_dashes = [], [], [], [], []
        edge_sources, edge_targets = [], []
        edge_sizes, edge_counts = [], []

        for u, v, data in graph.edges(data=True):
            lu = graph.nodes[u].get("layer", None)
            lv = graph.nodes[v].get("layer", None)
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            
            # Draw straight edge
            edge_xs.append([x0 * 500, x1 * 500])
            edge_ys.append([y0 * 500, y1 * 500])
            
            edge_sources.append(node_indices[u])
            edge_targets.append(node_indices[v])
            
            # Extract edge attributes
            edge_sizes.append(data.get('size', 'N/A'))
            edge_counts.append(data.get('count', 'N/A'))
            
            if lu == lv:
                edge_colors.append('#666666')
                edge_dashes.append('solid')
            else:
                edge_colors.append('#999999')
                edge_dashes.append('dashed')
            edge_widths.append(2)
            
        # Create figure
        plot = figure(
            title="Multilayer Network Visualization (Drag nodes to move)",
            width=1000,
            height=600,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            toolbar_location="above"
        )
        
        # Create data sources
        edge_source = ColumnDataSource(data=dict(
            xs=edge_xs, ys=edge_ys, colors=edge_colors,
            widths=edge_widths, line_dash=edge_dashes,
            sources=edge_sources, targets=edge_targets,
            sizes=edge_sizes, counts=edge_counts
        ))
        
        node_source = ColumnDataSource(data=dict(
            x=node_x,
            y=node_y,
            colors=node_colors,
            labels=node_labels,
            layers=node_layers
        ))
        
        # Draw edges first (so they're behind nodes)
        edge_renderer = plot.multi_line('xs', 'ys', source=edge_source,
                    line_width='widths', color='colors',
                    line_dash='line_dash', alpha=0.6)
                
        # Draw nodes - 
        node_renderer = plot.scatter('x', 'y', size=20, source=node_source,
                        fill_color='colors', line_color='black',
                        line_width=2, alpha=0.9)
        
        # Add labels to nodes
        labels = plot.text('x', 'y', text='labels', source=node_source,
                        text_align='center', text_baseline='top',
                        y_offset=-25, text_font_size='10pt')
        
        # Add hover tool for nodes
        node_hover = HoverTool(renderers=[node_renderer], tooltips=[
            ("Node", "@labels"),
            ("Layer", "@layers")
        ])
        plot.add_tools(node_hover)
        
        # Add hover tool for edges
        edge_hover = HoverTool(renderers=[edge_renderer], tooltips=[
            ("Size", "@sizes"),
            ("Count", "@counts")
        ], line_policy='interp')
        plot.add_tools(edge_hover)
        
        # Add PointDrawTool for dragging nodes
        draw_tool = PointDrawTool(renderers=[node_renderer], add=False)
        plot.add_tools(draw_tool)
        plot.toolbar.active_tap = draw_tool
        
        # Style the plot
        plot.axis.visible = False
        plot.grid.visible = False
        plot.outline_line_color = None        

        # Updated callback to handle arrows
        callback_args = dict(node_source=node_source, edge_source=edge_source)

        callback_code = """
            const node_data = node_source.data;
            const edge_data = edge_source.data;
            
            // Update edge positions based on new node positions
            for (let i = 0; i < edge_data.sources.length; i++) {
                const src_idx = edge_data.sources[i];
                const tgt_idx = edge_data.targets[i];
                
                const x0 = node_data.x[src_idx];
                const y0 = node_data.y[src_idx];
                const x1 = node_data.x[tgt_idx];
                const y1 = node_data.y[tgt_idx];
                
                // Draw straight edge
                edge_data.xs[i] = [x0, x1];
                edge_data.ys[i] = [y0, y1];
                
            }
            
            edge_source.change.emit();
        """
        
        callback = CustomJS(args=callback_args, code=callback_code)
        
        # Attach callback to node source changes
        node_source.js_on_change('data', callback)
        
        self.plot_container.append(
            pn.pane.Bokeh(plot, sizing_mode="stretch_height")
        )
        
        # threshold = G.graph.get("threshold", "N/A")
        # # compute divergent values
        # divergent_values = np.log10(
        #     [
        #         max(n.get("node_as", 1e-12), 1e-12)
        #         / (threshold if threshold not in [0, "N/A"] else 1)
        #         for n in G.nodes.values()
        #     ]
        # )
        # max_abs = np.max(np.abs(divergent_values)) if divergent_values.size > 0 else 1.0
        # clim = (-max_abs, max_abs)

        # for n, val in zip(G.nodes(), divergent_values):
        #     G.nodes[n]["divergent"] = val

        # hv_graph = hv.Graph.from_networkx(G, nx.layout.circular_layout).opts(
        #     node_color="divergent",
        #     node_size=50,
        #     edge_color="gray",
        #     arrowhead_length=0.1,
        #     directed=True,
        #     cmap="RdBu",
        #     tools=["hover"],
        #     width=900,
        #     colorbar=True,
        #     clim=clim,
        #     xaxis=None,
        #     yaxis=None,
        #     xlabel="",
        #     ylabel="",
        #     title=f"{graph_idx}",
        # )

        # nodes_data = hv_graph.nodes.data.copy()
        # nodes_data["name"] = (
        #     nodes_data["name"].map(MAC_TO_DEVICE).fillna(nodes_data["name"])
        # )

        # labels = hv.Labels(nodes_data, ["x", "y"], "name")
        # labelled = hv_graph * labels.opts(text_font_size="8pt", text_color="black")
        # self.plot_container.append(
        #     pn.pane.HoloViews(labelled, sizing_mode="stretch_height")
        # )

    # ---------- Scores plot ----------
    def scores_plot(self):
        file_path = file_path = self.w.get_selected_path()
        
        if not file_path.exists():
            self.plot_container.append(pn.pane.Markdown("⚠️ Score CSV not found"))
            return

        df = self.data_mgr.read_csv(file_path, self.w)

        fig = go.Figure()

        times = pd.to_datetime(df["timestamp"], unit="s")

        # Full range shading: max / min
        fig.add_trace(
            go.Scatter(
                x=times,
                y=df["soft_max_score"],
                mode="lines",
                line=dict(width=0),
                name="Max",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=times,
                y=df["soft_min_score"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(200,200,200,0.3)",
                name="Range (Min-Max)",
            )
        )
        # IQR shading
        fig.add_trace(
            go.Scatter(
                x=times,
                y=df["upper_quartile_score"],
                mode="lines",
                line=dict(width=0),
                name="Upper Quartile",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=times,
                y=df["lower_quartile_score"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(0,100,200,0.3)",
                name="IQR (Q1–Q3)",
            )
        )
        # median
        fig.add_trace(
            go.Scatter(
                x=times,
                y=df["median_score"],
                mode="lines",
                line=dict(color="blue", width=1),
                name="Median",
            )
        )

        # median
        fig.add_trace(
            go.Scatter(
                x=times,
                y=df["median_threshold"],
                mode="lines",
                line=dict(color="red", width=1),
                name="Threshold",
            )
        )

        fig.update_layout(
            title=f"{self.w.get_selected_path_level(-1)}",
            xaxis_title="X",
            yaxis_title="Score",
            hovermode="x unified",
            template="plotly_white",
        )
        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=list(
                        [
                            dict(
                                label="Linear",
                                method="relayout",
                                args=[{"yaxis.type": "linear"}],
                            ),
                            dict(
                                label="Log",
                                method="relayout",
                                args=[{"yaxis.type": "log"}],
                            ),
                        ]
                    ),
                    direction="down",
                    x=1.1,
                    xanchor="left",
                    y=1.15,
                    yanchor="top",
                )
            ]
        )

        self.plot_container.append(pn.pane.Plotly(fig, config={"responsive": True}))

    def compute_map(self, scores_dict):
        ap_results = {}

        for model_name, file_dict in scores_dict.items():

            # Identify benign and attack files
            benign_file = [f for f in file_dict if f.startswith("benign")][0]
            attack_files = [f for f in file_dict if f.startswith("attack")]

            benign_scores = file_dict[benign_file].values

            model_ap = {}
            for attack_file in attack_files:
                attack_scores = file_dict[attack_file].values

                # Combine scores and labels
                scores = np.concatenate([benign_scores, attack_scores])
                labels = np.concatenate(
                    [
                        np.zeros_like(benign_scores, dtype=int),
                        np.ones_like(attack_scores, dtype=int),
                    ]
                )

                # Compute AP
                ap = average_precision_score(labels, scores)
                model_ap[attack_file] = ap

            ap_results[model_name] = model_ap

        ap_df = pd.DataFrame(ap_results).T
        df_map = ap_df.mean(axis=1).reset_index()
        df_map.columns = ["model", "MAP"]
        return df_map

    # ---------- Summary plot ----------
    def summary_plot(self):
        base_dir = self.w.get_selected_path()
        
        if not base_dir.exists():
            self.plot_container.append(
                pn.pane.Markdown("⚠️ Results directory not found")
            )
            return

        AGG_FUNCS = {
            "process_time": "mean",
            "detection_rate": "mean",
            "median_score": "mean",
            "median_threshold": "mean",
            "pos_count": "sum",
            "batch_size": "sum",
        }

        summary_list = []
        scores_dict = defaultdict(dict)
        # iterate leaf files (re-implementation of original)
        for file in [
            str(p.relative_to(base_dir))
            for p in base_dir.rglob("*")
            if p.is_dir() and not any(child.is_dir() for child in p.iterdir())
        ]:
            file_dir = base_dir / file
            for pipeline_file in [
                str(p.relative_to(file_dir)) for p in file_dir.rglob("*") if p.is_file()
            ]:

                regex = self.w.pipeline_regex.value

                if regex != "":
                    match = re.search(regex, pipeline_file)
                    if not match:
                        continue

                csv_path = file_dir / pipeline_file
                if os.path.getsize(csv_path) > 0:
                    df = pd.read_csv(csv_path)
                else:
                    print(f"Skipping {csv_path}: file is empty")
                if len(df) < 5:
                    continue

                df = df.assign(
                    dataset=self.w.get_selected_path_level(0),
                    fe_name=self.w.get_selected_path_level(-1),
                    file=file,
                    pipeline=simple_pipeline_name(pipeline_file),
                )

                # filter
                df = df.replace(np.inf, np.finfo(np.float64).max).dropna()

                if len(df) == 0:
                    print(f"skipping {csv_path} as it contains no finite value")
                else:
                    scores_dict[simple_pipeline_name(pipeline_file)].update(
                        {file: df["median_score"]}
                    )

                summary = (
                    df.groupby(["dataset", "fe_name", "file", "pipeline"])
                    .agg(AGG_FUNCS)
                    .reset_index()
                )
                summary_list.append(summary)

        if not summary_list:
            self.plot_container.append(pn.pane.Markdown("⚠️ No summary data"))
            return

        summary_df = pd.concat(summary_list, ignore_index=True)

        metric = self.w.metric_input.value
        if metric == "average time":
            fig = sns.catplot(
                data=summary_df,
                y="pipeline",
                x="process_time",
                kind="bar",
                aspect=1.6,
                height=self.w.plot_height.value,
            )
            for ax in fig.axes.flat:
                ax.tick_params(axis="x", rotation=45)
            plt.close(fig.fig)
            pane = pn.pane.Matplotlib(fig.fig, interactive=False)
            self.plot_container.append(pane)
            return

        if metric == "mAP":
            ap_df = self.compute_map(scores_dict)

            fig = sns.catplot(
                data=ap_df,
                y="model",
                x="MAP",
                kind="bar",
                aspect=1.6,
                height=self.w.plot_height.value,
                order=sorted(ap_df["model"].unique()),
            )
            for ax in fig.axes.flat:
                ax.tick_params(axis="x", rotation=45)
                ax.set_xlim(0, 1)
                
            plt.close(fig.fig)
            pane = pn.pane.Matplotlib(fig.fig, interactive=False)
            self.plot_container.append(pane)
            return

        if metric == "total positive":
            fig = sns.catplot(
                data=summary_df,
                y="pipeline",
                x="pos_count",
                kind="bar",
                aspect=1.6,
                estimator=sum,
                errorbar=None,
                height=2,
            )
            for ax in fig.axes.flat:
                ax.tick_params(axis="x", rotation=45)
            plt.close(fig.fig)
            pane = pn.pane.Matplotlib(fig.fig, interactive=False)
            self.plot_container.append(pane)
            return

        if metric.startswith("accuracy"):
            summary_df[["label", "device", "attack"]] = summary_df["file"].str.split(
                "/", expand=True
            )
            summary_df["attack"] = summary_df["attack"].astype(str)

            # Define metric-specific configuration
            metric_config = {
                "binary": ("label", lambda x: x.startswith("benign")),
                "device": ("device", lambda x: x.startswith("whole_week")),
                "attack": ("attack", lambda x: x.startswith("None")),
                "full": ("file", lambda x: x.startswith("benign")),
            }

            # Get the appropriate grouping column and negative condition
            for suffix, (group_col, is_negative) in metric_config.items():
                if metric.endswith(suffix):
                    groupby_cols = ["dataset", "fe_name", "pipeline", group_col]
                    break

            # Process groups
            results = []
            for group_keys, group in summary_df.groupby(groupby_cols):
                dataset, fe_name, pipeline, label = group_keys

                # Calculate accuracy
                ratio = group["pos_count"].sum() / group["batch_size"].sum()
                accuracy = 1 - ratio if is_negative(label) else ratio

                # Create result
                results.append(
                    {
                        "accuracy": accuracy,
                        "dataset": dataset,
                        "fe_name": fe_name,
                        "pipeline": pipeline,
                        "label": label,
                    }
                )

            results_df = pd.concat(
                [pd.DataFrame([r]) for r in results], ignore_index=True
            )

            # Plot
            fig = sns.catplot(
                data=results_df,  # Fixed: was df_stats, should be results_df
                y="label",
                x="accuracy",
                hue="pipeline",
                aspect=1.6,
                height=self.w.plot_height.value,
                kind="bar",
            )
            pane = pn.pane.Matplotlib(fig.figure, interactive=False)
            self.plot_container.append(pane)
            fig.set(xlim=(0, 1))
            return

        # otherwise compute derived metrics using calc_stats
        results = []
        for (dataset, fe_name, pipeline), group in summary_df.groupby(
            ["dataset", "fe_name", "pipeline"]
        ):
            df_stats = calc_stats(group)
            df_stats["dataset"] = dataset
            df_stats["fe_name"] = fe_name
            df_stats["pipeline"] = pipeline
            results.append(df_stats)

        results_df = pd.concat(results, ignore_index=True)
        fig = sns.catplot(
            data=results_df, y="pipeline", x=metric, kind="bar", aspect=1.6, height=2
        )
        # for ax in fig.axes.flat:
        #     ax.tick_params(axis="x", rotation=90)
        plt.close(fig.figure)
        pane = pn.pane.Matplotlib(fig.figure, interactive=False)
        self.plot_container.append(pane)

    # ---------- helper for CSV export of visible traces ----------
    def get_visible_traces_csv(self) -> str:
        sio = io.StringIO()
        out_df = pd.DataFrame()
        plotly_pane = next(
            (o for o in self.plot_container.objects if isinstance(o, pn.pane.Plotly)),
            None,
        )
        if plotly_pane is None:
            return "No plot available"
        fig = plotly_pane.object
        for trace in fig.data:
            if trace.visible is None or trace.visible is True:
                if "x" not in out_df:
                    out_df["x"] = trace.x
                out_df[trace.name] = trace.y
        out_df.to_csv(sio, index=False)
        sio.seek(0)
        return sio


# ------------------------------
# Wire up the app
# ------------------------------
def serve_dashboard():
    data_mgr = DataManager(ROOT)
    widget_mgr = WidgetManager(data_mgr)
    plot_container = pn.Column(
        pn.pane.Markdown(
            "📊 Select dataset, feature extractor, pipeline, and node to generate visualization"
        ),
        sizing_mode="stretch_width",
    )
    plot_mgr = PlotManager(data_mgr, widget_mgr, plot_container)

    # patch download callback to call instance method which returns StringIO
    widget_mgr.download_csv.callback = lambda: plot_mgr.get_visible_traces_csv()

    template = pn.template.MaterialTemplate(
        site="Network Analysis",
        title="Interactive Dashboard",
        sidebar=[widget_mgr.sidebar()],
        main=[plot_container],
    )
    template.servable()
    return template


if __name__.startswith("bokeh") or __name__ == "__main__":
    serve_dashboard()
