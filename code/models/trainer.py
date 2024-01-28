from models.sklearn_models import *
from datasets.custom_dataset import *
import pickle
from abc import ABC, abstractmethod
from tqdm import tqdm
from utils import LazyInitializationMixin

class BasePipeline(ABC):
    
    def run_pipeline(self):
        self.setup_dataset()
        results = {}
        for step in self.steps:
            func = getattr(self, step)
            results[step] = func()
        return results
    
    def setup_dataset(self):
        for key, file_name in self.files.items():
            if key == "benign":
                train_val_test = self.train_val_test
            else:
                train_val_test = False
            self.files[key] = [
                load_dataset(
                    name, train_val_test=train_val_test, batch_size=self.batch_size
                )
                for name in file_name
            ]
            
    def save(self):
        self.model.save(self.dataset_id)

    def eval(self):
        model_output = {
            "benign": {},
            "malicious": {},
            "adversarial": {},
            "threshold": self.model.threshold,
        }

        for dataset in self.files["benign"]:
            scores = []
            for feature, label in tqdm(
                dataset["test"], desc=f"evaluating {dataset.name}"
            ):
                scores.append(self.model.predict_scores(feature))
            model_output["benign"][dataset["test"].name] = np.hstack(scores)

        for dataset in self.files["malicious"]:
            scores = []
            for feature, label in tqdm(dataset, desc=f"evaluating {dataset.name}"):
                scores.append(self.model.predict_scores(feature))
            model_output["malicious"][dataset.name] = np.hstack(scores)

        for dataset in self.files["adversarial"]:
            scores = []
            for feature, label in tqdm(dataset, desc=f"evaluating {dataset.name}"):
                scores.append(self.model.predict_scores(feature))
            model_output["adversarial"][dataset.name] = np.hstack(scores)

        results = {}
        for metric in self.metrics:
            results[metric.__name__] = metric(model_output)
        return results

    def train(self):
        for i in range(self.epochs):
            for dataset in self.files["benign"]:
                for feature, label in tqdm(
                    dataset["train"], desc=f"training {dataset.name}"
                ):
                    self.model.train(feature)


class OutlierDetectionPipeline(BasePipeline, LazyInitializationMixin):
    def __init__(
        self,
        batch_size=1024,
        train_val_test=[0.8, 0.1, 0.1],
        steps=["train", "calc_threshold", "eval", "save"],
        **kwargs,
    ):
        

        self.steps = steps
        self.batch_size = batch_size
        self.train_val_test = train_val_test
        
        self.allowed = ("metrics", "model", "files")
        self.lazy_init(**kwargs)
        self.entry=self.run_pipeline

    def calc_threshold(self, func=lambda x: np.percentile(x, 99.9)):
        scores = []
        for feature, label in tqdm(self.benign_dataset["val"]):
            scores.append(self.model.predict_scores(feature))
        threshold = func(np.hstack(scores))
        self.model.threshold = threshold
        return threshold
