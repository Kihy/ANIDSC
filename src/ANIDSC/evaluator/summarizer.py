
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt


import re

import scienceplots
# Configure plot style
plt.style.use(["science", "ieee"])


def f_beta(tp: float, fn: float, fp: float, beta: float = 1) -> float:
    """Compute the F-beta score."""
    beta2 = beta ** 2
    denom = (1 + beta2) * tp + beta2 * fn + fp
    return (1 + beta2) * tp / denom if denom else 0.0


def weighted_hmean(precision: float, recall: float, beta: float = 1) -> float:
    """Compute weighted harmonic mean of precision and recall."""
    beta2 = beta ** 2
    denom = beta2 * precision + recall
    return (1 + beta2) * precision * recall / denom if denom else 0.0


class BasicSummarizer:
    """Orchestrates reading results, computing statistics, and plotting."""

    AGG_FUNCS = {
        "time": "mean",
        "detection_rate": "mean",
        "median_score": "mean",
        "median_threshold": "mean",
        "pos_count": "sum",
        "batch_size": "sum",
    }

    def __init__(
        self,
        datasets: List[str],
        fe_names: List[str] = None,
        files: List[str] = None,
        pipelines: List[str] = None,
        fig_ext: str = "png",
        max_samples: int = 500,
    ):
        self.datasets = [Path(d) for d in datasets]
        self.fe_names = fe_names
        self.files = files
        self.pipelines = pipelines
        self.fig_ext = fig_ext
        self.max_samples=max_samples
        
        self.benign_prefix=["benign"]
        self.malicious_prefix=["malicious","attack"]

    def _list_dirs(self, path: Path, overrides: List[str]) -> List[str]:
        return overrides or [p.name for p in path.iterdir()]
    
    def _list_pipeline(self, path: Path, overrides: List[str]) -> List[str]:
        return overrides or [str(p.relative_to(path)) for p in path.rglob('*') if p.is_file()]

    def _extract_model_name(self, name):
        m = re.search(r'OnlineOD\(([^)]+)\)', name)
        if m:
            model = m.group(1)
            return model

    def read_csv(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Read raw CSV results and return all_data and aggregated summary."""
        entries = []
        summary_entries=[]
        for dataset in self.datasets:
            for fe in self._list_dirs(dataset, self.fe_names):
                res_dir = dataset / fe / "results"
                for file_name in self._list_dirs(res_dir, self.files):
                    file_dir = res_dir / file_name
                    for pipeline in self._list_pipeline(file_dir, self.pipelines):
                        csv_path = file_dir / pipeline
                        df = pd.read_csv(csv_path)
                        
                        if len(df)==0:
                            continue
                        
                        df = df.replace([np.inf, -np.inf], np.nan)
                        
                        df = df.assign(
                            dataset=dataset.name,
                            fe_name=fe,
                            file=file_name,
                            pipeline=self._extract_model_name(pipeline),
                        )
                        summary=df.groupby(["dataset", "fe_name", "file", "pipeline"]).agg(self.AGG_FUNCS).reset_index()
                        
                        
                        # limit number of records
                        if len(df)>self.max_samples:
                            df=df.sample(self.max_samples).sort_index()
                        
                        entries.append(df)
                        summary_entries.append(summary)
        all_df = pd.concat(entries, ignore_index=True)
        summary_df = pd.concat(summary_entries, ignore_index=True)
        return all_df, summary_df

    def _save_fig(self, fig: plt.Figure, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        plt.close(fig)

    def _rotate_xticks(self, grid: sns.FacetGrid) -> None:
        for ax in grid.axes.flat:
            for lbl in ax.get_xticklabels():
                lbl.set_rotation(45)

    def plot_batch_wise_metric(self, df: pd.DataFrame, metrics: List[str]) -> None:
        """Plot time-series metrics across batches for each dataset/feature extractor."""
        for (dataset, fe_name), group in df.groupby(["dataset", "fe_name"]):
            base_path = Path(dataset) / fe_name / "summary"/"plots"
            for metric in metrics:
                grid = sns.relplot(
                    data=group,
                    x="batch_num", y=metric,
                    col="file", row="pipeline",
                    color="#7FB7BE",
                    kind="line", aspect=1.6, height=2,
                    facet_kws={"sharex": False, "sharey": False},
                )
                if metric == "median_score":
                    def extras(data, **kwargs):
                        ax = plt.gca()
                        sns.lineplot(data=data, x="batch_num", y="median_threshold", color="#F25F5C", ax=ax)
                        ax.fill_between(data["batch_num"], data["lower_quartile_score"], data["upper_quartile_score"],color="#7FB7BE", alpha=0.5)
                        ax.fill_between(data["batch_num"], data["soft_min_score"], data["soft_max_score"], color="#7FB7BE", alpha=0.2)
                        ax.set_yscale("log")
                        for child in ax.get_children(): child.set_rasterized(True)
                    grid.map_dataframe(extras)
                out_file = base_path / f"{metric}.{self.fig_ext}"
                self._save_fig(grid.figure, out_file)

    def plot_summary(self, summary_df: pd.DataFrame, metrics: List[str]) -> None:
        """Plot aggregated summary bar charts and save CSV summaries."""
        for (dataset, fe_name), group in summary_df.groupby(["dataset", "fe_name"]):
            base_plot = Path(dataset) / fe_name / "summary"/"plots"
            base_csv = Path(dataset) / fe_name / "summary"/"csv"

            melted = group.melt(
                id_vars=["file", "pipeline"],
                value_vars=metrics,
                var_name="metric",
                value_name="value",
            )
            grid = sns.catplot(
                data=melted, x="pipeline", y="value",
                col="file", row="metric", kind="bar",
                aspect=1.6, height=2, sharey=False
            )
            self._rotate_xticks(grid)
            self._save_fig(grid.figure, base_plot / f"summary_metrics.{self.fig_ext}")

            base_csv.mkdir(parents=True, exist_ok=True)
            group.to_csv(base_csv / "summary_metrics.csv", index=False)

    def calc_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate accuracy and F-measure statistics for each pipeline."""
        stats = {}
        for label in self.benign_prefix:
            mask = df["file"].str.startswith(label)
            stats[f"pos_benign"] = df.loc[mask, "pos_count"].sum()
            stats[f"total_benign"] = df.loc[mask, "batch_size"].sum()
            
            
        for label in self.malicious_prefix:
            mask = df["file"].str.startswith(label)
            stats[f"pos_malicious"] = df.loc[mask, "pos_count"].sum()
            stats[f"total_malicious"] = df.loc[mask, "batch_size"].sum()
        
            
            
        if stats["total_benign"]==0:
            stats["acc_benign"] = 0
        else:
            stats["acc_benign"] = 1 - stats["pos_benign"] / stats["total_benign"]
        
        if stats["total_malicious"]==0:
            stats["acc_malicious"]=0
        else:
            stats["acc_malicious"] = stats["pos_malicious"] / stats["total_malicious"]
        stats["f1"] = f_beta(stats["pos_malicious"], stats["total_malicious"] - stats["pos_malicious"], stats["pos_benign"])
        for b in (0.5, 1, 2):
            stats[f"hmean_{b}_acc"] = weighted_hmean(stats["acc_benign"], stats["acc_malicious"], beta=b)
        return pd.DataFrame([stats])

    def gen_accuracy(self, summary_df: pd.DataFrame) -> pd.DataFrame:
        """Generate accuracy stats for all dataset/fe/pipeline combinations."""
        results = []
        for (dataset, fe_name, pipeline), group in summary_df.groupby(["dataset", "fe_name", "pipeline"]):
            df_stats = self.calc_stats(group)
            df_stats["dataset"] = dataset
            df_stats["fe_name"] = fe_name
            df_stats["pipeline"] = pipeline
            results.append(df_stats)
        return pd.concat(results, ignore_index=True)

    def plot_accuracy(self, df: pd.DataFrame, metrics: List[str]) -> None:
        """Plot accuracy and F-measure bar charts for each dataset/feature extractor."""
        for (dataset, fe_name), group in df.groupby(["dataset", "fe_name"]):
            melted = group.melt(
                id_vars=["pipeline"],
                value_vars=metrics,
                var_name="metric",
                value_name="value"
            )
            grid = sns.catplot(
                data=melted, x="pipeline", y="value",
                row="metric", kind="bar", aspect=1.6, height=2,
                sharey=False
            )
            self._rotate_xticks(grid)
            out_plot = Path(dataset) / fe_name / "summary"/ "plots" / f"accuracy_metrics.{self.fig_ext}"
            self._save_fig(grid.figure, out_plot)
            out_csv = Path(dataset) / fe_name / "summary"/ "csv" / "accuracy_metrics.csv"
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            group.to_csv(out_csv, index=False)

    def run(self) -> pd.DataFrame:
        """Full run: read data, plot metrics, summary, and accuracy."""
        all_df, summary_df = self.read_csv()
        accuracy_df = self.gen_accuracy(summary_df)

        # Plot per-batch metrics
        self.plot_batch_wise_metric(all_df, ["median_score", "detection_rate"])

        # Plot summary metrics
        self.plot_summary(summary_df, ["time", "detection_rate", "pos_count"])

        # Plot accuracy metrics
        self.plot_accuracy(accuracy_df, ["f1", "acc_benign", "acc_malicious", "hmean_0.5_acc", "hmean_1_acc", "hmean_2_acc"])

        return all_df
