from abc import ABC, abstractmethod
from pathlib import Path
import pickle
import torch
from collections import deque
import numpy as np
import models
from utils import *
import matplotlib.pyplot as plt


def load_model(dataset_id, model_cls, model_config, indent=""):
    save_type = model_config["save_type"]
    model_name = model_config["model_name"]
    print(f"{indent}loading {model_name}")
    if save_type == "ensemble":
        with open(f"../../models/{dataset_id}/{model_name}.pkl", "rb") as f:
            model = pickle.load(f)

        for model_name in model.model_names:
            model_config["base_model_config"]["model_name"] = model_name
            model.model_pool.append(
                load_model(
                    dataset_id,
                    model_config["base_model_cls"],
                    model_config["base_model_config"],
                    indent=indent + "----",
                )
            )

        return model

    elif save_type == "pkl":
        with open(f"../../models/{dataset_id}/{model_name}.pkl", "rb") as f:
            model = pickle.load(f)
        return model

    elif save_type == "dict":
        model = getattr(models, model_name)(**model_config)
        with open(f"../../models/{dataset_id}/{model_name}.pkl", "rb") as f:
            model_dict = pickle.load(f)

        model.load_dict(model_dict)
        return model

    elif save_type == "pth":
        model = getattr(models, model_cls)(**model_config)
        checkpoint = torch.load(f"../../models/{dataset_id}/{model_name}.pth")

        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint.keys():
            model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"][0])
        return model

    elif save_type == "mix":
        model = getattr(models, model_cls)(**model_config)
        checkpoint = torch.load(f"../../models/{dataset_id}/{model_name}.pth")

        model.load_state_dict(checkpoint["model_state_dict"])
        model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        for module_name, save_file_name in zip(
            model.save_modules, model.save_module_names
        ):
            module_cls = model_config[f"{module_name}_cls"]
            module_config = model_config[f"{module_name}_kwargs"]
            module_config["model_name"] = save_file_name
            setattr(
                model,
                module_name,
                load_model(
                    dataset_id, module_cls, module_config, indent=indent + "----"
                ),
            )
        return model
    else:
        raise ValueError(f"unknow save type {save_type}")


class EnsembleSaveMixin:
    def save(self, dataset_name, suffix=""):
        for model in self.model_pool:
            model.save(dataset_name)

        save_path = Path(
            f"../../models/{dataset_name}/{self.model_name}{f'-{suffix}' if suffix !='' else ''}.pkl"
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "wb") as f:
            pickle.dump(self, f)

        print(f"saved at {save_path}")


class MixtureSaveMixin:
    def save(self, dataset_name, suffix=""):
        for module_name in self.save_modules:
            getattr(self, module_name).save(dataset_name)

        checkpoint = {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_state_dict": self.state_dict(),
        }

        ckpt_path = Path(
            f"../../models/{dataset_name}/{self.model_name}{f'-{suffix}' if suffix !='' else ''}.pth"
        )
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, str(ckpt_path))
        print(f"saved at {ckpt_path}")


class DictSaveMixin:
    def save(self, dataset_name, suffix=""):
        save_path = Path(
            f"../../models/{dataset_name}/{self.model_name}{f'-{suffix}' if suffix !='' else ''}.pkl"
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "wb") as f:
            pickle.dump(self.to_dict(), f)

        print(f"saved at {save_path}")


class PickleSaveMixin:
    def save(self, dataset_name, suffix=""):
        save_path = Path(
            f"../../models/{dataset_name}/{self.model_name}{f'-{suffix}' if suffix !='' else ''}.pkl"
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "wb") as f:
            pickle.dump(self, f)

        print(f"saved at {save_path}")


class TorchSaveMixin:
    def save(self, dataset_name, suffix=""):
        checkpoint = {
            "model_state_dict": self.state_dict(),
        }
        if hasattr(self, "optimizer"):
            checkpoint["optimizer_state_dict"] = (self.optimizer.state_dict(),)
        ckpt_path = Path(
            f"../../models/{dataset_name}/{self.model_name}{f'-{suffix}' if suffix !='' else ''}.pth"
        )
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, str(ckpt_path))
        print(f"saved at {ckpt_path}")

    @abstractmethod
    def state_dict(self):
        pass

    @abstractmethod
    def load_state_dict(self, state_dict):
        pass


class BaseModel(ABC):
    def __init__(self, **kwargs):

        for k, v in kwargs.items():
            setattr(self, k, v)

    def preprocess(self, X):
        if len(self.preprocessors) > 0:
            for p in self.preprocessors:
                X = getattr(self, p)(X)
        return X


class GNNOCDModel(torch.nn.Module, MixtureSaveMixin):
    def __init__(self, model_name, gnn_cls, gnn_kwargs, od_cls, od_kwargs, **kwargs):
        torch.nn.Module.__init__(self)

        gnn_conf = dict(gnn_kwargs)
        od_conf = dict(od_kwargs)
        gnn_conf["model_name"] = f"{model_name}/{gnn_conf['model_name']}"
        od_conf["model_name"] = f"{model_name}/{od_conf['model_name']}"

        self.gnn = getattr(models, gnn_cls)(**gnn_conf)
        self.od = getattr(models, od_cls)(**od_conf)

        self.gnn_cls = gnn_cls
        self.gnn_kwargs = gnn_conf
        self.od_cls = od_cls
        self.od_kwargs = od_conf
        self.model_name = f"{model_name}"
        # add gnn to od optimizer
        if isinstance(self.od, torch.nn.Module):
            self.optimizer = torch.optim.Adam(
                list(self.gnn.parameters()) + list(self.od.parameters())
            )
        else:
            self.optimizer = torch.optim.Adam(self.gnn.parameters())

        self.save_modules = ["od", "gnn"]
        self.save_module_names = [od_conf["model_name"], gnn_conf["model_name"]]

        # add od attributes
        include_attrs = ["converged", "loss_queue"]
        for i in include_attrs:
            setattr(self, i, getattr(self.od, i))

    def state_dict(self):
        return {"save_module_names": self.save_module_names}

    def load_state_dict(self, state_dict):
        self.save_module_names = state_dict["save_module_names"]

    def predict_scores(self, data):
        x, edge_index, edge_attr = data
        node_embed = self.gnn(x, edge_index, edge_attr)
        score, threshold = self.od.predict_scores(node_embed)
        return score, threshold

    def train_step(self, data):
        x, edge_index, edge_attr = data
        self.optimizer.zero_grad()
        node_embed = self.gnn(x, edge_index, edge_attr)
        if isinstance(self.od, torch.nn.Module):
            loss = self.od.get_loss(node_embed) + self.gnn.sw_loss(node_embed)
        else:
            self.od.train_step(node_embed)  # train od seperately
            loss = self.gnn.sw_loss(node_embed)
        loss.backward()
        self.optimizer.step()

    def update_scaler(self, data):
        x, edge_index, edge_attr = data
        node_embed = self.gnn(x, edge_index, edge_attr)

        self.gnn.node_stats.add(x)
        self.gnn.edge_stats.add(edge_attr)
        self.od.update_scaler(node_embed.clone().detach())


class MultiLayerOCDModel(EnsembleSaveMixin):
    def __init__(
        self,
        base_model_cls,
        base_model_config,
        model_name="MultiLayerOCDModel",
        **kwargs,
    ):
        self.model_name = f"{model_name}"
        self.protocols = {0: "UDP", 1: "TCP", 2: "ARP", 3: "ICMP", 4: "Other"}
        self.base_model_name = base_model_config["model_name"]

        self.processed = 0
        self.batch_num = 0

        self.model_pool = []
        self.model_names = []
        for i, p in self.protocols.items():
            model_config = dict(base_model_config)
            model_config["model_name"] = f"{self.model_name}/{p}/{self.base_model_name}"
            self.model_names.append(model_config["model_name"])
            new_model = getattr(models, base_model_cls)(**model_config)
            new_model.protocol = p
            new_model.proto_id = i
            new_model.draw_graph = False

            self.model_pool.append(new_model)

    def process(self, x):
        all_results = []
        for model in self.model_pool:
            idx = torch.where(x[:, 3].long() == model.proto_id)
            data = x[idx]
            if data.nelement() > 0:
                results = model.process(data)
                if results is not None:
                    model.draw_graph = True

                    results["protocol"] = model.protocol

                    all_results.append(results)

        self.batch_num += 1
        self.processed += x.size(0)

        if len(all_results) == 0:
            return None

        return all_results

    def visualize_graph(self, dataset_name, fe_name, file_name):
        n_plots = np.sum([m.draw_graph for m in self.model_pool])
        if n_plots == 0:
            return

        fig, ax = plt.subplots(figsize=(3, n_plots * 2), nrows=n_plots, squeeze=False)

        node_map = get_node_map(fe_name, dataset_name)
        if node_map is not None:
            node_map = {v: k for k, v in node_map.items()}

        ax_idx = 0
        for m in self.model_pool:
            if m.draw_graph:
                ax[ax_idx][0].set_title(
                    f"{m.protocol} after {m.graph_state.processed} packets"
                )
                
                m.graph_state.visualize_graph(fig, ax[ax_idx][0], node_map)
                ax_idx += 1
                m.draw_graph = False

        fig.tight_layout()
        path = Path(
            f"../../datasets/{dataset_name}/{fe_name}/graph_plots/{file_name}/{self.model_name}_{self.processed}.png"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, format="png")
        # Close the figure to release resources
        plt.close(fig)


class OnlineCDModel(EnsembleSaveMixin):
    def __init__(
        self,
        base_model_cls,
        base_model_config,
        patience,
        confidence,
        model_name="OnlineCDModel",
        **kwargs,
    ):
        self.model_name = model_name
        self.patience = patience
        self.loss_queue = deque(maxlen=patience)
        self.potential_queue = deque(maxlen=patience)
        self.potential_x_queue = deque(maxlen=patience)
        self.base_model_name = str(base_model_config["model_name"])
        self.confidence = confidence
        self.base_model_cls = base_model_cls
        self.base_model_config = base_model_config
        self.model_idx = 0

        self.model_pool = []
        self.model_names = []
        self.model_pool.append(self.create_model())

        self.num_batch = 0
        self.ensemble_size = 1

        if self.base_model_cls == "GNNOCDModel":
            self.graph_state = models.HomoGNN()

    def create_model(self):
        model_config = dict(self.base_model_config)
        model_config["model_name"] = (
            f"{self.model_name}/{self.base_model_name}/{len(self.model_pool)}"
        )
        self.model_names.append(model_config["model_name"])
        return getattr(models, self.base_model_cls)(**model_config)

    def no_cd(self, score):
        # add average AS to queue
        self.loss_queue.append(np.mean(score))
        # empty potential queue
        self.clear_potential_queue()

    def cd(self, score, x):
        self.potential_queue.append(np.mean(score))
        self.potential_x_queue.append(x)

    def clear_potential_queue(self):
        self.potential_queue.clear()
        self.potential_x_queue.clear()

    def process(self, x):
        if self.base_model_cls == "GNNOCDModel":
            x = self.graph_state.update_graph(x)

        if x is None:
            return None
        
        score, threshold = self.model_pool[self.model_idx].predict_scores(x)
        
        
        loss_diff = np.mean(score) - threshold

        # if model has converged and loss is high, find alternative model in model pool
        if loss_diff > 0 and self.model_pool[self.model_idx].converged:
            for i, model in enumerate(self.model_pool):
                tmp_score, tmp_threshold = model.predict_scores(x)
                tmp_loss_diff = np.mean(tmp_score) - tmp_threshold
                if tmp_loss_diff < loss_diff:
                    loss_diff = tmp_loss_diff
                    score = tmp_score
                    threshold = tmp_threshold
                    self.model_idx = i

        # update anomaly score
        if self.base_model_cls == "GNNOCDModel":
            self.graph_state.update_node_score(score)

        
        # if still greater than 0, it indicates concept drift
        if loss_diff > 0 and self.model_pool[self.model_idx].converged:
            self.cd(score, x)
        else:
            # other wise, train the model
            self.model_pool[self.model_idx].train_step(x)

            self.model_pool[self.model_idx].loss_queue.extend(score)
            if not self.model_pool[self.model_idx].converged and is_stable(
                self.model_pool[self.model_idx].loss_queue
            ):
                self.model_pool[self.model_idx].converged = True

            self.no_cd(score)
            self.model_pool[self.model_idx].update_scaler(x)

        if (
            len(self.loss_queue) == self.patience
            and len(self.potential_queue) == self.patience
        ):
            difference = np.median(self.potential_queue) - np.median(self.loss_queue)
            range = max(np.max(self.potential_queue), np.max(self.loss_queue)) - min(
                np.min(self.potential_queue), np.min(self.loss_queue)
            )
            diff_magnitude = (self.patience * difference**2) / (range**2)

            if diff_magnitude > self.confidence:
                drift_level = "malicious"
                self.clear_potential_queue()
            else:
                drift_level = "benign"
                new_model = self.create_model()
                # train model on existing features
                for potential_x in self.potential_x_queue:

                    new_model.train_step(potential_x)
                    new_model.update_scaler(potential_x)

                score, threshold = new_model.predict_scores(x)
                self.model_idx = len(self.model_pool)
                self.model_pool.append(new_model)

                self.ensemble_size += 1
                self.no_cd(score)

        elif len(self.loss_queue) != self.patience:
            drift_level = "unfull loss queue"
        else:
            drift_level = "no drift"

        self.num_batch += 1
        
        return {
            "drift_level": drift_level,
            "threshold": threshold,
            "score": score,
            "index": self.num_batch,
            "count":self.graph_state.processed,
            "model_idx": self.model_idx,
            "num_model": len(self.model_pool),
        }


class BaseOnlineODModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_queue = deque(maxlen=self.n)

        self.converged = False
        self.scaler = LivePercentile(ndim=self.n_features)
        self.additional_params = ["preprocessors", "loss_queue", "scaler", "converged"]

    @abstractmethod
    def process(self, X):
        pass

    def get_threshold(self):
        if len(self.loss_queue) == 0:
            threshold = 0
        else:
            threshold=calc_quantile(self.loss_queue, 0.95)
        return threshold

    def standardize(self, x):
        percentiles = self.scaler.quantiles([0.25, 0.50, 0.75])

        if percentiles is None:
            percentiles = torch.quantile(
                x.float(), torch.tensor([0.25, 0.50, 0.75]).to(x.device), dim=0
            )
        percentiles = percentiles.to(x.device)
        scaled_features = (x - percentiles[1]) / (percentiles[2] - percentiles[0])
        scaled_features = torch.nan_to_num(
            scaled_features, nan=0.0, posinf=0.0, neginf=0.0
        )
        return scaled_features.float()

    def update_scaler(self, x):
        self.scaler.add(x)


class BaseODModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._threshold = None

    @abstractmethod
    def train_step(self, X):
        pass

    @abstractmethod
    def predict_scores(self, X):
        pass

    @abstractmethod
    def predict_labels(self, X):
        pass

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        self._threshold = value

    def predict_labels(self, X):
        if self.threshold is None:
            raise ValueError(
                "predict_labels only works after threshold is calculated. Call calc_threshold() first"
            )
        return self.predict_scores(X) > self.threshold
