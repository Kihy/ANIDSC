"""Aggregate summary plot across multiple pipeline runs."""
from __future__ import annotations

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

_AGG_FUNCS = {
    "process_time": "mean",
    "detection_rate": "mean",
    "median_score": "mean",
    "median_threshold": "mean",
    "pos_count": "sum",
    "batch_size": "sum",
}


class SummaryPlot(BasePlot):
    label = "Summary"
    description = "Cross-pipeline accuracy / F1 / mAP summary charts"
    file_type = "csv"

    # ── Sidebar ──────────────────────────────────────────────────────────────

    def sidebar_widgets(self) -> pn.viewable.Viewable:
        return pn.WidgetBox(
            "Parameters",
            self.w.metric_input,
            self.w.pipeline_regex,
            self.w.plot_height,
        )

    # ── Render ───────────────────────────────────────────────────────────────

    def render(self, selected_path: Path) -> pn.viewable.Viewable:
        if not selected_path.exists():
            return self._warn("Results directory not found")

        summary_list, scores_dict = self._load_summaries(selected_path)
        if not summary_list:
            return self._warn("No summary data found")

        summary_df = pd.concat(summary_list, ignore_index=True)
        metric = self.w.metric_input.value
        height = self.w.plot_height.value

        if metric == "average time":
            return self._bar_chart(summary_df, x="process_time", y="pipeline", height=height)

        if metric == "mAP":
            ap_df = self._compute_map(scores_dict)
            fig = sns.catplot(data=ap_df, y="model", x="MAP", kind="bar",
                              aspect=1.6, height=height,
                              order=sorted(ap_df["model"].unique()))
            for ax in fig.axes.flat:
                ax.tick_params(axis="x", rotation=45)
                ax.set_xlim(0, 1)
            plt.close(fig.fig)
            return pn.pane.Matplotlib(fig.fig, interactive=False)

        if metric == "total positive":
            fig = sns.catplot(data=summary_df, y="pipeline", x="pos_count", kind="bar",
                              aspect=1.6, estimator=sum, errorbar=None, height=height)
            for ax in fig.axes.flat:
                ax.tick_params(axis="x", rotation=45)
            plt.close(fig.fig)
            return pn.pane.Matplotlib(fig.fig, interactive=False)

        if metric.startswith("accuracy"):
            return self._accuracy_chart(summary_df, metric, height)

        # Derived metrics (f1, hmean_acc, …)
        results = []
        for (dataset, fe_name, pipeline), group in summary_df.groupby(
            ["dataset", "fe_name", "pipeline"]
        ):
            df_stats = calc_stats(group)
            df_stats[["dataset", "fe_name", "pipeline"]] = dataset, fe_name, pipeline
            results.append(df_stats)

        results_df = pd.concat(results, ignore_index=True)
        fig = sns.catplot(data=results_df, y="pipeline", x=metric, kind="bar",
                          aspect=1.6, height=height)
        plt.close(fig.figure)
        return pn.pane.Matplotlib(fig.figure, interactive=False)

    # ── Private helpers ──────────────────────────────────────────────────────

    def _load_summaries(self, base_dir: Path):
        summary_list = []
        scores_dict: Dict[str, dict] = defaultdict(dict)
        regex = self.w.pipeline_regex.value

  
        # Recurse to find all subdirectories with results files
        # Exclude any path containing "trial_" to avoid processing individual trials
        dirs_to_process = [
            p for p in base_dir.rglob("*")
            if p.is_dir() 
            and "trial_" not in str(p.relative_to(base_dir))
            and any(f.name.startswith("results.") for f in p.iterdir() if f.is_file())
        ]

        
        for file_dir in dirs_to_process:
            file = str(file_dir.relative_to(base_dir)) if file_dir != base_dir else "."
            
            print(file)
            
            # Use glob to only get files directly in this directory (non-recursive)
            for pipeline_file in [
                str(p.relative_to(file_dir)) for p in file_dir.glob("*") if p.is_file() and p.name.startswith("results.")
            ]:
                if regex and not re.search(regex, pipeline_file):
                    continue

                csv_path = file_dir / pipeline_file
                if os.path.getsize(csv_path) == 0:
                    continue

                print("csv_path:", csv_path)
                print("pipeline_file:", pipeline_file)
                df = pd.read_csv(csv_path)
                if len(df) < 5:
                    continue

                pipeline_name = simple_pipeline_name(pipeline_file)
                df = df.assign(
                    dataset=self.w.get_selected_path_level(0),
                    fe_name=self.w.get_selected_path_level(-1),
                    file=file,
                    pipeline=pipeline_name,
                )
                df = df.replace(np.inf, np.finfo(np.float64).max).dropna()
                if len(df) == 0:
                    continue

                scores_dict[pipeline_name][file] = df["median_score"]

                summary = (
                    df.groupby(["dataset", "fe_name", "file", "pipeline"])
                    .agg(_AGG_FUNCS)
                    .reset_index()
                )
                summary_list.append(summary)

        return summary_list, scores_dict

    def _compute_map(self, scores_dict: dict) -> pd.DataFrame:
        ap_results: dict = {}
        for model_name, file_dict in scores_dict.items():
            benign_file = next(f for f in file_dict if f.startswith("benign"))
            attack_files = [f for f in file_dict if f.startswith("attack")]
            benign_scores = file_dict[benign_file].values
            model_ap: dict = {}
            for af in attack_files:
                attack_scores = file_dict[af].values
                scores = np.concatenate([benign_scores, attack_scores])
                labels = np.concatenate([
                    np.zeros_like(benign_scores, dtype=int),
                    np.ones_like(attack_scores, dtype=int),
                ])
                model_ap[af] = average_precision_score(labels, scores)
            ap_results[model_name] = model_ap

        ap_df = pd.DataFrame(ap_results).T
        df_map = ap_df.mean(axis=1).reset_index()
        df_map.columns = ["model", "MAP"]
        return df_map

    def _bar_chart(self, df: pd.DataFrame, x: str, y: str, height: int):
        fig = sns.catplot(data=df, y=y, x=x, kind="bar", aspect=1.6, height=height)
        for ax in fig.axes.flat:
            ax.tick_params(axis="x", rotation=45)
        plt.close(fig.fig)
        return pn.pane.Matplotlib(fig.fig, interactive=False)

    def _accuracy_chart(self, summary_df: pd.DataFrame, metric: str, height: int):
        summary_df = summary_df.copy()
        summary_df[["label", "device", "attack"]] = summary_df["file"].str.split(
            "/", expand=True
        )
        summary_df["attack"] = summary_df["attack"].astype(str)

        metric_config = {
            "binary": ("label",  lambda x: x.startswith("benign")),
            "device": ("device", lambda x: x.startswith("whole_week")),
            "attack": ("attack", lambda x: x.startswith("None")),
            "full":   ("file",   lambda x: x.startswith("benign")),
        }
        for suffix, (group_col, is_negative) in metric_config.items():
            if metric.endswith(suffix):
                groupby_cols = ["dataset", "fe_name", "pipeline", group_col]
                break

        results = []
        for (dataset, fe_name, pipeline, label), group in summary_df.groupby(groupby_cols):
            ratio = group["pos_count"].sum() / group["batch_size"].sum()
            accuracy = 1 - ratio if is_negative(label) else ratio
            results.append(dict(accuracy=accuracy, dataset=dataset,
                                fe_name=fe_name, pipeline=pipeline, label=label))

        results_df = pd.concat([pd.DataFrame([r]) for r in results], ignore_index=True)
        fig = sns.catplot(data=results_df, y="label", x="accuracy",
                          hue="pipeline", aspect=1.6, height=height, kind="bar")
        fig.set(xlim=(0, 1))
        return pn.pane.Matplotlib(fig.figure, interactive=False)
