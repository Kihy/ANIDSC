"""Aggregate summary plot across multiple pipeline runs."""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import panel as pn
import re
import seaborn as sns
from sklearn.metrics import average_precision_score

from plots.base import BasePlot
from utils import calc_stats, simple_pipeline_name

import scienceplots  # noqa: F401
plt.style.use(["science", "ieee"])
matplotlib.use('agg')


class SummaryPlot(BasePlot):
    label = "Summary"
    description = "Cross-pipeline accuracy / F1 / mAP summary charts"

    def __init__(self, data_mgr, widgets):
        super().__init__(data_mgr, widgets)
        
        self.metric_options=[
                'process_time',
                'detection_rate', 'median_score', 'median_threshold', 'pos_count',
                'batch_size', 'pooled_ap', 'macro_ap', 'weighted_ap'
            ]
        self.file_options=[
            'file_accuracy']
        
        # Plot-specific widgets
        self.metric_input = pn.widgets.Select(
            name="Metric",
            options=self.metric_options+self.file_options,
            sizing_mode="stretch_width",
        )
        self.pipeline_regex = pn.widgets.TextInput(
            name="Pipeline Regex", placeholder="Regex to filter pipeline names",
            sizing_mode="stretch_width",
        )
        self.plot_height = pn.widgets.IntInput(
            name="Plot Height", step=1, value=2, sizing_mode="stretch_width",
        )

    # ── Sidebar ──────────────────────────────────────────────────────────────

    def sidebar_widgets(self) -> pn.viewable.Viewable:
        return pn.WidgetBox(
            "Parameters",
            self.metric_input,
            self.pipeline_regex,
            self.plot_height,
        )

    # ── Render ───────────────────────────────────────────────────────────────

    def render(self, selected_path: Path) -> pn.viewable.Viewable:
        if not selected_path.exists():
            return self._warn("Results directory not found")
        
      
        summary_df, file_accuracy_df = self._load_summaries(selected_path)
        
        
        metric = self.metric_input.value
        height = self.plot_height.value
        
        summary_df['full_name']=summary_df["pipeline_name"] + " / " + summary_df["run_identifier"]

        if metric in self.metric_options:
            return self._bar_chart(summary_df, x=metric, y="full_name", height=height)
        elif metric in self.file_options:
            return self._bar_chart(file_accuracy_df, x="accuracy", y="file_name", height=height)
        

        plt.close(fig.figure)
        return pn.pane.Matplotlib(fig.figure, interactive=False)

    # ── Private helpers ──────────────────────────────────────────────────────
    def _bar_chart(self, df: pd.DataFrame, x: str, y: str, height: int):
        fig = sns.catplot(data=df, y=y, x=x, kind="bar", aspect=1.6, height=height)
        for ax in fig.axes.flat:
            ax.tick_params(axis="x", rotation=45)
        plt.close(fig.fig)
        return pn.pane.Matplotlib(fig.fig, interactive=False)


    def _load_summaries(self, base_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Find and combine all summary.yaml files from pipeline configurations."""
        summary_data = []
        file_accuracy_data = []
        regex = self.pipeline_regex.value
        
        # Find all summary.yaml files recursively
        summary_files = list(base_dir.rglob("summary.yaml"))
        
        for summary_path in summary_files:
            # Skip if doesn't match regex filter
            if regex and not re.search(regex, str(summary_path)):
                continue
            
            try:
                with open(summary_path, 'r') as f:
                    data = json.load(f)
                
                # Extract run_summary data
                if 'run_summary' not in data:
                    continue
                
                run_summary = data['run_summary']
                
                # Create a record with both run_summary metrics and metadata
                record = {
                    'dataset_name': data.get('dataset_name', 'unknown'),
                    'pipeline_name': data.get('pipeline_name', 'unknown'),
                    'run_identifier': data.get('run_identifier', 'unknown'),
                    **run_summary  # Unpack all run_summary metrics
                }
                
                # Add file path for context
                relative_path = summary_path.relative_to(base_dir)
                record['file_path'] = str(relative_path.parent)
                
                summary_data.append(record)
                
                # Extract file_accuracy data
                if 'file_accuracy' in data:
                    file_accuracy = data['file_accuracy']
                    for file_name, accuracy in file_accuracy.items():
                        file_record = {
                            'dataset_name': data.get('dataset_name', 'unknown'),
                            'pipeline_name': data.get('pipeline_name', 'unknown'),
                            'run_identifier': data.get('run_identifier', 'unknown'),
                            'file_name': file_name,
                            'accuracy': accuracy,
                            'file_path': str(relative_path.parent)
                        }
                        file_accuracy_data.append(file_record)
                
            except Exception as e:
                print(f"Error reading {summary_path}: {e}")
                continue
        
        if not summary_data:
            return pd.DataFrame(), pd.DataFrame()
        
        return pd.DataFrame(summary_data), pd.DataFrame(file_accuracy_data)
