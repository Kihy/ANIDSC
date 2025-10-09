from abc import abstractmethod
import json
import os
from typing import Any, Dict, List

from ..save_mixin.pickle import PickleSaveMixin
import networkx as nx
import numpy as np

from ..component.feature_buffer import OutputWriter

from ..save_mixin.null import NullSaveMixin
from ..component.pipeline_component import PipelineComponent
from . import od_metrics
from pathlib import Path
import time
from torch.utils.tensorboard import SummaryWriter


METRICS = [
    "detection_rate",
    "lower_quartile_score",
    "upper_quartile_score",
    "soft_min_score",
    "soft_max_score",
    "median_score",
    "median_threshold",
    "pos_count",
    "batch_size",
]


class BaseEvaluator(NullSaveMixin, PipelineComponent):
    def __init__(
        self,
        log_to_tensorboard: bool = True,
        graph_period=False,
    ):
        """base evaluator that evaluates the output of a single model
        Args:
            metric_list (List[str]): list of metric names in string format
            log_to_tensorboard (bool, optional): whether to write to tensorboard. if there is a collate evaluatr, it is better to delegate it to collate evaluator. Defaults to True.
            save_results (bool, optional): whether to save results in CSV file. if there is a collate evaluatr, it is better to delegate it to collate evaluator. Defaults to True.
            draw_graph_rep_interval (bool, optional): whether to draw the graph representation. only available if pipeline contains graph representation. Defaults to False.
        """
        super().__init__()
        self.metrics = [getattr(od_metrics, m) for m in METRICS]
        self.metric_list = METRICS
        self.log_to_tensorboard = log_to_tensorboard
        self.graph_period = graph_period
        self.last_anomaly = 0

    def setup(self):
        super().setup()

        dataset_name = self.request_attr("dataset_name")
        fe_name = self.request_attr("fe_name")
        file_name = self.request_attr("file_name")

        pipeline_name = str(self.parent_pipeline)

        self.file_template = (
            f"{dataset_name}/{fe_name}/{{}}/{file_name}/{pipeline_name}"
        )

        if self.graph_period:
            graph_plot_path = Path(self.file_template.format("graphs") + ".ndjson")
            graph_plot_path.parent.mkdir(parents=True, exist_ok=True)
            self.graph_file = open(str(graph_plot_path), "a")

        # file to store outputs
        scores_and_thresholds_path = Path(self.file_template.format("results") + ".csv")
        scores_and_thresholds_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_file = open(str(scores_and_thresholds_path), "a")
        if os.stat(scores_and_thresholds_path).st_size == 0:
            self.output_file.write(
                ",".join(["process_time", "batch_num", "timestamp"] + self.metric_list)
                + "\n"
            )

        if self.log_to_tensorboard:
            log_dir = self.file_template.format("runs")
            self.writer = SummaryWriter(log_dir=log_dir)
            print("tensorboard logging to", log_dir)

        self.prev_timestamp = time.time()

    def save(self):

        self.output_file.close()
        print("results file saved at", self.output_file.name)
        if self.graph_period:
            self.graph_file.close()
            print("graph file saved at", self.output_file.name)

        self.write_header = True

    def process(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """processes results and log them accordingly

        Args:
            results (Dict[str, Any]): gets metric values in metric_list based on results

        Returns:
            Dict[str, Any]: dictionary of metric, value pair
        """
        # records time

        current_time = time.time()
        duration = current_time - self.parent_pipeline.start_time

        result_dict = {
            "process_time": duration,
            "batch_num": self.request_attr("model", "batch_evaluated"),
            "timestamp": self.request_attr("data_source", "timestamp"),
        }

        for metric_name, metric in zip(self.metric_list, self.metrics):
            result_dict[metric_name] = metric(results)

            if self.log_to_tensorboard:
                self.writer.add_scalar(
                    metric_name,
                    result_dict[metric_name],
                    result_dict["batch_num"],
                )

        self.output_file.write(",".join(map(str, result_dict.values())) + "\n")

        if self.graph_period:
            # plot graphs periodically
            if result_dict["batch_num"] % self.graph_period == 0:
                self.save_graph(results)

            # plot if there is an anomaly
            if results["score"] is not None:
                if (
                    result_dict["batch_num"] - self.last_anomaly
                ) > self.graph_period and np.median(results["score"]) > results[
                    "threshold"
                ]:
                    self.save_graph(results)
                    self.last_anomaly = result_dict["batch_num"]

    def save_graph(self, results):
        G = self.request_action("graph_rep", "get_networkx")

        # get threshold
        G.graph["threshold"] = results["threshold"]

        # add anomaly score
        if results["score"] is None:
            nx.set_node_attributes(G, float("inf"), "node_as")
        else:
            # assign edge nodes
            score_map = {i: float(v) for i, v in zip(G.nodes, results["score"])}
            nx.set_node_attributes(G, score_map, "node_as")

        json.dump(nx.cytoscape_data(G), self.graph_file)
        self.graph_file.write("\n")


class ResultWriter(OutputWriter):
    def setup(self):
        # setup files
        dataset_name = self.request_attr("dataset_name")
        file_name = self.request_attr("file_name")  
        fe_name = self.request_attr("fe_name")
        pipeline_name=self.parent_pipeline.name
        

        self.feature_path = Path(
            f"{dataset_name}/{fe_name}/{self.folder_name}/{file_name}/{pipeline_name}.{self.file_type}"
        )
        self.feature_path.parent.mkdir(parents=True, exist_ok=True)

        self.save_file = open(self.feature_path, "a")

    def teardown(self):
        self.save_file.close()

class CSVResultWriter(PickleSaveMixin, ResultWriter):
    @property
    def folder_name(self):
        return "results"

    @property
    def file_type(self):
        return "csv"

    def setup(self):
        super().setup()

        self.metrics = [getattr(od_metrics, m) for m in METRICS]
        if os.stat(self.feature_path).st_size == 0:
            self.save_file.write(
                ",".join(["process_time", "batch_num", "timestamp"] + METRICS) + "\n"
            )

    def process(self, data):
        current_time = time.time()
        duration = current_time - self.request_attr("start_time")

        result_dict = {
            "process_time": duration,
            "batch_num": self.request_attr("batch_evaluated"),
            "timestamp": self.request_attr("timestamp"),
        }

        for metric_name, metric in zip(METRICS, self.metrics):

            result_dict[metric_name] = metric(data)

        self.save_file.write(",".join(map(str, result_dict.values())) + "\n")

        return result_dict


class GraphResultWriter(PickleSaveMixin,ResultWriter):

    @property
    def folder_name(self):
        return "graphs"

    @property
    def file_type(self):
        return "ndjson"

    def __init__(self, graph_period=1):
        super().__init__()
        self.graph_period = graph_period
        self.last_timestamp = None

    def process(self, data):
        cur_time = self.request_attr("timestamp")

        if self.last_timestamp is None:
            self.last_timestamp = self.request_attr("timestamp")

        if self.last_timestamp - cur_time > self.graph_period:
            self.save_graph(data)

            self.last_timestamp = cur_time
            
        return data 

    def save_graph(self, data):
        G = self.request_attr("networkx")

        # get threshold
        G.graph["threshold"] = data["threshold"]

        # add anomaly score
        if data["score"] is None:
            nx.set_node_attributes(G, float("inf"), "node_as")
        else:
            # assign edge nodes
            score_map = {i: float(v) for i, v in zip(G.nodes, data["score"])}
            nx.set_node_attributes(G, score_map, "node_as")

        json.dump(nx.cytoscape_data(G), self.save_file)
        self.save_file.write("\n")
