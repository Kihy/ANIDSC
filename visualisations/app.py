# refactor_dashboard.py
"""
Refactored interactive dashboard for network/node/score/summary plotting.

Structure:
 - DataManager: file IO, frame iteration, small parsing helpers
 - WidgetManager: create and keep widgets, wire watchers
 - PlotManager: all plotting routines; mode dispatch
"""
from __future__ import annotations
import re
from collections import defaultdict
from dataclasses import dataclass
import io
import csv
from pathlib import Path
import json
import os
import random
import re
import time
import traceback
from typing import Dict, Iterable, List, Optional
from panel.widgets import Tqdm
import networkx as nx
import numpy as np
import pandas as pd
import panel as pn
import plotly.graph_objects as go
import seaborn as sns
import holoviews as hv
from networkx.readwrite import json_graph
import matplotlib.pyplot as plt
import scienceplots

from sklearn.metrics import average_precision_score
import fsspec

# keep your plotting style lines (you used scienceplots earlier)
plt.style.use(["science", "ieee"])

pn.extension("tabulator", "plotly", "ipywidgets")
hv.extension("bokeh")


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
        """Yield JSON frames from a file, sampling by prob, with optional progress updates."""
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

            frames = [json.loads(line) for line in sample]
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
            print(i, idx, frame)

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
        # main controls
        self.dataset_input = pn.widgets.AutocompleteInput(
            name="Dataset",
            options=self.data_mgr.list_subdirectories(self.data_mgr.root),
            case_sensitive=False,
            search_strategy="includes",
            min_characters=0,
            placeholder="Search",
        )
        self.fe_input = pn.widgets.AutocompleteInput(
            name="Feature Extractor",
            options=self.data_mgr.list_subdirectories(None),
            case_sensitive=False,
            search_strategy="includes",
            min_characters=0,
            placeholder="Search",
        )
        self.file_input = pn.widgets.AutocompleteInput(
            name="File",
            options=self.data_mgr.list_subdirectories(None),
            case_sensitive=False,
            search_strategy="includes",
            min_characters=0,
            placeholder="Search",
        )
        self.pipeline_input = pn.widgets.AutocompleteInput(
            name="Pipeline",
            options=self.data_mgr.list_subdirectories(None),
            case_sensitive=False,
            search_strategy="includes",
            min_characters=0,
            placeholder="Search",
        )

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
        self.graph_slider = pn.widgets.IntSlider(name="Graph Index", start=0, end=0)
        self.metric_input = pn.widgets.Select(
            name="Metric",
            options=[
                "mAP",
                "f1",
                "accuracy",
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

        self.file_filter = pn.WidgetBox(
            "File Sampling", self.max_frames_input, self.range_start, self.range_end
        )

        self.pipeline_regex = pn.widgets.TextInput(
            name="Pipeline Regex", placeholder="Regex to filter pipeline names"
        )

        # watch wiring
        self.dataset_input.param.watch(self._on_dataset_change, "value")
        self.fe_input.param.watch(self._on_fe_change, "value")
        self.file_input.param.watch(self._on_file_change, "value")
        self.pipeline_input.param.watch(self._on_pipeline_change, "value")
        self.mode_selector.param.watch(self._on_mode_change, "value")

        # small state to allow disabling widgets during long ops
        self._disable_targets = [
            self.dataset_input,
            self.fe_input,
            self.file_input,
            self.pipeline_input,
            self.metric_input,
            self.node_input,
            self.generate_btn,
            self.download_csv,
            self.mode_selector,
            self.file_filter,
        ]

        for w in self._disable_targets:
            w.sizing_mode = "stretch_width"

    def _disable(self, state: bool):
        for w in self._disable_targets:
            w.disabled = state

    # --------- widget layout switching ----------
    def _switch_widget(self, mode: str):
        if mode == "Node":
            return pn.WidgetBox(
                "Parameters",
                self.file_input,
                self.pipeline_input,
                self.file_filter,
                self.node_input,
                
            )
        elif mode == "Network":
            return pn.WidgetBox(
                "Parameters",
                self.file_input,
                self.pipeline_input,
                self.file_filter,
                self.graph_slider,
            )
        elif mode == "Scores":
            return pn.WidgetBox(
                "Parameters", self.file_input, self.pipeline_input, self.file_filter
            )
        elif mode == "Summary":
            return pn.WidgetBox("Parameters", self.metric_input, self.pipeline_regex)
        return pn.pane.Markdown("Unknown mode")

    # --------- watchers (populate dependent options) ----------
    def _on_dataset_change(self, _):
        dataset = self.dataset_input.value
        if dataset is None:
            return
        self._disable(True)
        try:
            self.fe_input.options = self.data_mgr.list_subdirectories(
                self.data_mgr.root / dataset
            )
        finally:
            self._disable(False)

    def _on_fe_change(self, _):
        """Select a feature extractor - update file listing depending on mode"""
        dataset = self.dataset_input.value
        fe = self.fe_input.value
        if dataset is None or fe is None:
            return

        self._disable(True)
        try:
            folder = (
                "graphs"
                if self.mode_selector.value in ["Node", "Network"]
                else "results"
            )
            file_path = self.data_mgr.root / dataset / fe / folder
            if file_path.exists():
                self.file_input.options = self.data_mgr.list_files(file_path)
        finally:
            self._disable(False)

    def _on_file_change(self, _):
        """When a file (leaf) is selected, update pipeline dropdown"""
        dataset = self.dataset_input.value
        fe = self.fe_input.value
        file_val = self.file_input.value

        if dataset is None or fe is None or file_val is None:
            return

        folder = (
            "graphs" if self.mode_selector.value in ["Node", "Network"] else "results"
        )

        self._disable(True)
        try:
            pipelines = self.data_mgr.list_subdirectories(
                self.data_mgr.root / dataset / fe / folder / file_val
            )

            self.pipeline_input.options = {
                simple_pipeline_name(p): p for p in pipelines
            }
        finally:
            self._disable(False)

    def _on_pipeline_change(self, _):
        """Populate node options and graph slider when pipeline is chosen."""
        dataset = self.dataset_input.value
        fe = self.fe_input.value
        file_val = self.file_input.value
        pipeline = self.pipeline_input.value

        if dataset is None or fe is None or file_val is None or pipeline is None:
            return

        self._disable(True)
        try:
            file_path = (
                self.data_mgr.root / dataset / fe / "graphs" / file_val / pipeline
            )

            frames = list(self.data_mgr.load_frames(file_path, self))

            # collect unique node ids from frames
            node_ids = sorted(
                {n["data"]["id"] for fr in frames for n in fr["elements"]["nodes"]}
            )
            self.node_input.options = [MAC_TO_DEVICE.get(h, h) for h in node_ids]
            self.graph_slider.start = 0
            self.graph_slider.end = max(0, len(frames) - 1)

        finally:
            self._disable(False)

    def _on_mode_change(self, _):
        # The mode affects the folders listed under fe->file
        self._on_fe_change(None)

    def _on_percent_change(self, _):
        # percent changed — refresh node options if pipeline is set
        self._on_pipeline_change(None)

    # expose a simple layout helper
    def sidebar(self):
        return pn.Column(
            self.dataset_input,
            self.fe_input,
            self.mode_selector,
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
            / self.w.fe_input.value
            / "graphs"
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
        concept_idx_order = []
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

                concept_idx = str(node["data"].get("concept_idx"))

                if prev_concept != concept_idx or prev_concept is None:
                    prev_concept = concept_idx
                    concept_idx_order.append(concept_idx)
                    concept_change_indices.append(time_val)
                    unique_concepts.add(concept_idx)

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

        print("plotting")

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
                name="Scaled Count",
                line=dict(color="orange", dash="dot"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["times"],
                y=df["original_size_vals"],
                mode="markers",
                name="Scaled Size",
                line=dict(color="green", dash="dot"),
            )
        )

        print(len(concept_change_indices))
        print(concept_idx_order)

        if len(concept_idx_order) < 20:
            print("plotting concepts")
            for i, (prev_idx, cur_idx) in enumerate(
                zip(concept_change_indices[:-1], concept_change_indices[1:])
            ):
                concept = concept_idx_order[i]
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
        file_path = (
            self.data_mgr.root
            / self.w.dataset_input.value
            / self.w.fe_input.value
            / "graphs"
            / self.w.file_input.value
            / self.w.pipeline_input.value
        )
        graph = self.data_mgr.get_frame(file_path, graph_idx, self.w)
        if graph is None:
            self.plot_container.append(pn.pane.Markdown("⚠️ Graph not found"))
            return

        # convert to networkx (same approach)
        G = nx.cytoscape_graph(graph)
        threshold = G.graph.get("threshold", "N/A")
        # compute divergent values
        divergent_values = np.log10(
            [
                max(n.get("node_as", 1e-12), 1e-12)
                / (threshold if threshold not in [0, "N/A"] else 1)
                for n in G.nodes.values()
            ]
        )
        max_abs = np.max(np.abs(divergent_values)) if divergent_values.size > 0 else 1.0
        clim = (-max_abs, max_abs)

        for n, val in zip(G.nodes(), divergent_values):
            G.nodes[n]["divergent"] = val

        hv_graph = hv.Graph.from_networkx(G, nx.layout.circular_layout).opts(
            node_color="divergent",
            node_size=50,
            edge_color="gray",
            arrowhead_length=0.1,
            directed=True,
            cmap="RdBu",
            tools=["hover"],
            width=900,
            colorbar=True,
            clim=clim,
            xaxis=None,
            yaxis=None,
            xlabel="",
            ylabel="",
            title=f"{graph_idx}",
        )
        labels = hv.Labels(hv_graph.nodes, ["x", "y"], "name")
        labelled = hv_graph * labels.opts(text_font_size="8pt", text_color="black")
        self.plot_container.append(
            pn.pane.HoloViews(labelled, sizing_mode="stretch_height")
        )

    # ---------- Scores plot ----------
    def scores_plot(self):
        file_path = (
            self.data_mgr.root
            / self.w.dataset_input.value
            / self.w.fe_input.value
            / "results"
            / self.w.file_input.value
            / self.w.pipeline_input.value
        )
        if not file_path.exists():
            self.plot_container.append(pn.pane.Markdown("⚠️ Score CSV not found"))
            return

        df = self.data_mgr.read_csv(file_path, self.w)
        print(df.max(axis=0))

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
            title=f"{self.w.pipeline_input.value}",
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

            benign_scores = file_dict[benign_file].dropna().values

            model_ap = {}
            for attack_file in attack_files:
                attack_scores = file_dict[attack_file].dropna().values

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
        base_dir = (
            self.data_mgr.root
            / self.w.dataset_input.value
            / self.w.fe_input.value
            / "results"
        )
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
                df = pd.read_csv(csv_path)
                if len(df)<5:
                    continue
                df = df.replace([np.inf, -np.inf], np.nan).assign(
                    dataset=self.w.dataset_input.value,
                    fe_name=self.w.fe_input.value,
                    file=file,
                    pipeline=simple_pipeline_name(pipeline_file),
                )
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
                height=2,
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
                height=0.2 * len(ap_df["model"].unique()),
                order=sorted(ap_df["model"].unique()),
            )
            for ax in fig.axes.flat:
                ax.tick_params(axis="x", rotation=45)
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

        if metric == "accuracy":
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

            df_long = results_df.melt(
                id_vars="pipeline",
                value_vars=["acc_benign", "acc_malicious"],
                var_name="metric",
                value_name="accuracy",
            )

            # Plot
            fig = sns.catplot(
                data=df_long,
                y="pipeline",
                x="accuracy",
                hue="metric",
                aspect=1.6,
                height=0.2 * len(df_long["pipeline"].unique()),
                kind="bar",  # or "point", "box", etc
            )
            pane = pn.pane.Matplotlib(fig.fig, interactive=False)
            self.plot_container.append(pane)
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
        plt.close(fig.fig)
        pane = pn.pane.Matplotlib(fig.fig, interactive=False)
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
