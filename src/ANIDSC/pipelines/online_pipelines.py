# from ..models.base_model import *
from .base_pipeline import BasePipeline
from ..utils import get_node_map, draw_graph, JSONEncoder
import json
from ..datasets.base_dataset import load_dataset
from .. import metrics
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from matplotlib.figure import Figure
import numpy as np
from pathlib import Path
from typing import List, Tuple
from .. import feature_extractors
from utils import fig_to_array

class LiveODPipeline(BasePipeline):
    def __init__(
        self,
        steps: List[str] = ["process"],
        metric_list: List[str] = [],
        write_to_tensorboard: bool = True,
        **kwargs,
    ):
       
        super().__init__(
            allowed=["model", "fe", "dataset_name","file_name"], **kwargs
        )
        self.steps = steps
        self.metrics = [getattr(metrics, m) for m in metric_list]
        self.write_to_tensorboard = write_to_tensorboard
        self.buffer_size=256

    def process(self) -> None:
        """processes each dataset"""
        if self.write_to_tensorboard:
            # create tensorboard
            writer = SummaryWriter(
                log_dir=f"runs/{self.model.model_name}/{self.dataset_name}/{self.fe.name}/{self.file_name}"
            )

            # custom layout for score and threshold in each layer
            layout = {
                "decision": {
                    p: ["Multiline", [f"score/{p}", f"threshold/{p}"]]
                    for p in self.model.protocol_inv.keys()
                },
            }

            writer.add_custom_scalars(layout)

        # initialize file
        model_results = self.results[self.model.model_name]
        model_results[self.file_name] = {}

        # file to store outputs
        scores_and_thresholds_path = Path(
            f"../datasets/{self.dataset_name}/{self.fe.name}/outputs/{self.file_name}/{self.model.model_name}.csv"
        )
        scores_and_thresholds_path.parent.mkdir(parents=True, exist_ok=True)
        output_file = open(str(scores_and_thresholds_path), "w")

        write_header = True

        start_time = time.time()

        count = 0
        batch_count = 0

        self.fe.setup()
        
        buffer=[]
        for packet in tqdm(self.fe.input_pcap, desc=f"{self.model.model_name} on {self.dataset_name}-{self.fe.name}-{self.file_name}"):
            traffic_vector = self.fe.get_traffic_vector(packet)
            
            if traffic_vector is None:
                self.fe.skipped += 1
                continue

            if self.fe.offset_time is None and self.fe.offset_timestamp:
                self.fe.offset_time = traffic_vector["timestamp"] - self.fe.state["last_timestamp"]
            else:
                self.fe.offset_time = 0
                
            traffic_vector["timestamp"] -= self.fe.offset_time

            feature = self.fe.update(traffic_vector)
            assert feature.dtype=="float64"
            buffer.append(feature)
            if len(buffer)==self.buffer_size:
                features=np.vstack(buffer)
                buffer=[]
            else:
                continue
            
            # get index to MAC address mapping
            idx_to_mac_map = self.fe.state["node_map"]
            
            prediction_results = self.model.process(features)
            
            count += feature.shape[0]
            batch_count+=1

            if prediction_results is None:
                continue

            # if results is dictionary, only a single layer
            if isinstance(prediction_results, dict):
                prediction_results["protocol"] = "all"
                prediction_results = [prediction_results]

            # iterate over results for each layer
            for pred_res in prediction_results:
                if hasattr(self.model, "save_graph") and (
                    np.median(pred_res["score"]) > pred_res["threshold"]
                    or batch_count % 100 == 0
                ):  # save draw graph to tensorboard every 100 epochs or anomaly is found

                    # get graph of model
                    G = self.model.model_pool[
                        self.model.protocol_inv[pred_res["protocol"]]
                    ].graph_state.get_graph_state()
                    fig, ax = plt.subplots()

                    # draw graph
                    draw_graph(
                        G, fig, ax, False, pred_res["protocol"], {}, idx_to_mac_map
                    )

                    image = fig_to_array(fig)
                    plt.close(fig)
                    writer.add_image(
                        pred_res["protocol"], image, global_step=batch_count
                    )
                    
                    #save graph dotfile
                    self.model.save_graph(
                        self.dataset_name,
                        self.fe.name,
                        self.file_name,
                        pred_res["protocol"],
                    )

                # gather metrics for each batch
                for metric in self.metrics:
                    metric_value = metric(pred_res)
                    metric_name = f"{pred_res['protocol']}_{metric.__name__}"
                    if metric_name not in model_results[self.file_name]:
                        model_results[self.file_name][metric_name] = []
                    model_results[self.file_name][metric_name].append(
                        metric_value
                    )

                    if self.write_to_tensorboard:
                        writer.add_scalar(
                            f'{metric.__name__}/{pred_res["protocol"]}',
                            metric_value,
                            batch_count,
                        )

                pred_res["score"] = np.nan_to_num(
                    pred_res["score"], nan=0.0, posinf=0, neginf=0
                )
                pred_res["threshold"] = np.nan_to_num(
                    pred_res["threshold"], nan=0.0, posinf=0, neginf=0
                )
                
                #visualize histogram of scores
                if self.write_to_tensorboard:
                    writer.add_histogram(
                        f"{pred_res['protocol']}_score",
                        pred_res["score"],
                        batch_count,
                    )

                # find median score and threshold over batch
                pred_res["score"] = np.median(pred_res["score"])
                pred_res["threshold"] = np.median(pred_res["threshold"])


                #write these scalars
                if self.write_to_tensorboard:
                    writer.add_scalar(
                        f'score/{pred_res["protocol"]}',
                        pred_res["score"],
                        batch_count,
                    )
                    writer.add_scalar(
                        f'threshold/{pred_res["protocol"]}',
                        pred_res["threshold"],
                        batch_count,
                    )

                    writer.add_scalar(
                        f'num_model/{pred_res["protocol"]}',
                        pred_res.get("num_model", 1),
                        batch_count,
                    )

                # write results to file
                if write_header:
                    output_file.write(",".join(list(pred_res.keys())) + "\n")
                    write_header = False
                output_file.write(
                    ",".join(list(map(str, list(pred_res.values())))) + "\n"
                )

        duration = time.time() - start_time

        # record time and count
        model_results[self.file_name]["time"] = duration
        model_results[self.file_name]["count"] = count

        # measure average
        for metric_name, metric_values in model_results[self.file_name].items():
            model_results[self.file_name][metric_name] = np.nanmean(metric_values)

        with open(str(self.results_path), "w") as f:
            json.dump(self.results, f, indent=4, cls=JSONEncoder)

    def save_model(self):
        """save model
        """        
        self.model.save(self.dataset_name)

    def setup(self):
        """
        set up the pipeline
        """
        
        self.results_path = Path(
            f"../datasets/{self.dataset_name}/{self.fe.name}/results.json"
        )

        if self.results_path.is_file():
            with open((str(self.results_path))) as f:
                self.results = json.load(open((str(self.results_path))))
                if self.model.model_name not in self.results.keys():
                    self.results[self.model.model_name] = {}
        else:
            self.results_path.parent.mkdir(parents=True, exist_ok=True)
            self.results = {self.model.model_name: {}}

    def teardown(self):
        pass



class OnlineODPipeline(BasePipeline):
    def __init__(
        self,
        batch_size: int = 1024,
        profile: bool = False,
        steps: List[str] = ["process"],
        metric_list: List[str] = [],
        percentage: Tuple[float, float] = [0, 1],
        epochs: int = 1,
        write_to_tensorboard: bool = True,
        **kwargs,
    ):
        """initialize online OD evaluator

        Args:
            batch_size (int, optional): batch size of model. Defaults to 1024.
            profile (bool, optional): whether to profile the model with pytorch profiler. Defaults to False.
            steps (List[str], optional): name of the steps to use. Defaults to ["process"].
            metric_list (List[str], optional): name of metric function for evaluation. Defaults to [].
            percentage (Tuple[float, float], optional): start and end percentage of feature. Defaults to [0, 1].
            epochs (int, optional): number of epochs. Defaults to 1.
            write_to_tensorboard (bool, optional): whether to write result to tensorboard. Defaults to True.
        """
        super().__init__(
            allowed=["model", "dataset_name", "fe_name", "files"], **kwargs
        )
        self.steps = steps
        self.batch_size = batch_size
        self.training_size = 0
        self.profile = profile
        self.metrics = [getattr(metrics, m) for m in metric_list]
        self.percentage = percentage
        self.epochs = epochs
        self.write_to_tensorboard = write_to_tensorboard

    def process(self) -> None:
        """processes each dataset"""
        for loader in self.datasets:
            # get index to MAC address mapping
            idx_to_mac_map = get_node_map(
                loader.dataset.dataset_name,
                loader.dataset.fe_name,
                loader.dataset.file_name,
            )

            if self.write_to_tensorboard:
                # create tensorboard
                writer = SummaryWriter(
                    log_dir=f"runs/{self.model.model_name}/{loader.dataset.dataset_name}/{loader.dataset.fe_name}/{loader.dataset.file_name}"
                )

                # custom layout for score and threshold in each layer
                layout = {
                    "decision": {
                        p: ["Multiline", [f"score/{p}", f"threshold/{p}"]]
                        for p in self.model.protocol_inv.keys()
                    },
                }

                writer.add_custom_scalars(layout)

            # initialize file
            model_results = self.results[self.model.model_name]
            model_results[loader.dataset.file_name] = {}

            # file to store outputs
            scores_and_thresholds_path = Path(
                f"../datasets/{loader.dataset.dataset_name}/{loader.dataset.fe_name}/outputs/{loader.dataset.file_name}/{self.model.model_name}.csv"
            )
            scores_and_thresholds_path.parent.mkdir(parents=True, exist_ok=True)
            output_file = open(str(scores_and_thresholds_path), "w")

            write_header = True

            start_time = time.time()

            count = 0
            batch_count = 0

            base_desc = f"processing {self.model.model_name} on {loader.dataset.name}"
            t_bar = tqdm(loader, desc=base_desc, leave=False)
            for feature, label in t_bar:
                prediction_results = self.model.process(feature)
                t_bar.set_description(
                    f"processing {self.model.model_name} on {loader.dataset.name}",
                    refresh=True,
                )

                count += feature.shape[0]
                batch_count += 1

                if prediction_results is None:
                    continue

                # if results is dictionary, only a single layer
                if isinstance(prediction_results, dict):
                    prediction_results["protocol"] = "all"
                    prediction_results = [prediction_results]

                # iterate over results for each layer
                for pred_res in prediction_results:
                    if hasattr(self.model, "save_graph") and (
                        np.median(pred_res["score"]) > pred_res["threshold"]
                        or batch_count % 100 == 0
                    ):  # save draw graph to tensorboard every 100 epochs or anomaly is found

                        # get graph of model
                        G = self.model.model_pool[
                            self.model.protocol_inv[pred_res["protocol"]]
                        ].graph_state.get_graph_state()
                        fig, ax = plt.subplots()

                        # draw graph
                        draw_graph(
                            G, fig, ax, False, pred_res["protocol"], {}, idx_to_mac_map
                        )

                        image = fig_to_array(fig)
                        plt.close(fig)
                        writer.add_image(
                            pred_res["protocol"], image, global_step=batch_count
                        )
                        
                        #save graph dotfile
                        self.model.save_graph(
                            loader.dataset.dataset_name,
                            loader.dataset.fe_name,
                            loader.dataset.file_name,
                            pred_res["protocol"],
                        )

                    # gather metrics for each batch
                    for metric in self.metrics:
                        metric_value = metric(pred_res)
                        metric_name = f"{pred_res['protocol']}_{metric.__name__}"
                        if metric_name not in model_results[loader.dataset.file_name]:
                            model_results[loader.dataset.file_name][metric_name] = []
                        model_results[loader.dataset.file_name][metric_name].append(
                            metric_value
                        )

                        if self.write_to_tensorboard:
                            writer.add_scalar(
                                f'{metric.__name__}/{pred_res["protocol"]}',
                                metric_value,
                                batch_count,
                            )

                    pred_res["score"] = np.nan_to_num(
                        pred_res["score"], nan=0.0, posinf=0, neginf=0
                    )
                    pred_res["threshold"] = np.nan_to_num(
                        pred_res["threshold"], nan=0.0, posinf=0, neginf=0
                    )
                    
                    #visualize histogram of scores
                    if self.write_to_tensorboard:
                        writer.add_histogram(
                            f"{pred_res['protocol']}_score",
                            pred_res["score"],
                            batch_count,
                        )

                    # find median score and threshold over batch
                    pred_res["score"] = np.median(pred_res["score"])
                    pred_res["threshold"] = np.median(pred_res["threshold"])

    
                    #write these scalars
                    if self.write_to_tensorboard:
                        writer.add_scalar(
                            f'score/{pred_res["protocol"]}',
                            pred_res["score"],
                            batch_count,
                        )
                        writer.add_scalar(
                            f'threshold/{pred_res["protocol"]}',
                            pred_res["threshold"],
                            batch_count,
                        )

                        writer.add_scalar(
                            f'num_model/{pred_res["protocol"]}',
                            pred_res.get("num_model", 1),
                            batch_count,
                        )

                    # write results to file
                    if write_header:
                        output_file.write(",".join(list(pred_res.keys())) + "\n")
                        write_header = False
                    output_file.write(
                        ",".join(list(map(str, list(pred_res.values())))) + "\n"
                    )

            duration = time.time() - start_time

            # record time and count
            model_results[loader.dataset.file_name]["time"] = duration
            model_results[loader.dataset.file_name]["count"] = count

            # measure average
            for metric_name, metric_values in model_results[loader.dataset.file_name].items():
                model_results[loader.dataset.file_name][metric_name] = np.nanmean(metric_values)

            loader.dataset.reset()

        with open(str(self.results_path), "w") as f:
            json.dump(self.results, f, indent=4, cls=JSONEncoder)

    def save_model(self):
        """save model
        """        
        self.model.save(self.dataset_name)

    def setup(self):
        """set up the pipeline
        """        
        self.datasets = []

        for f in self.files:
            dataset = load_dataset(
                dataset_name=self.dataset_name,
                fe_name=self.fe_name,
                file_name=f["file_name"],
                batch_size=self.batch_size,
                percentage=self.percentage,
                epochs=self.epochs,
            )
            self.datasets.append(dataset)

        self.results_path = Path(
            f"../datasets/{self.dataset_name}/{self.fe_name}/results.json"
        )

        if self.results_path.is_file():
            with open((str(self.results_path))) as f:
                self.results = json.load(open((str(self.results_path))))
                if self.model.model_name not in self.results.keys():
                    self.results[self.model.model_name] = {}
        else:
            self.results_path.parent.mkdir(parents=True, exist_ok=True)
            self.results = {self.model.model_name: {}}

    def teardown(self):
        pass
