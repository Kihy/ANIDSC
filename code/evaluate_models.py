
from datasets.custom_dataset import *

from models.sklearn_models import *
from pipelines.od_pipelines import *
from metrics.od_metrics import *
from models.kitsune import KitNET
from models.base_model import *
from models.goad import GOAD
from models.icl import ICL
from models.misc_models import *
import torch
import sys

def to_numpy(x):
    return x.numpy()


def to_tensor(x):
    return torch.tensor(x)


def train_uq_models(devices, fe_name):

    for device in devices:
        dataset_name = f"uq/{device}"

        train_file = [
            {
                "dataset_name": dataset_name,
                "fe_name": fe_name,
                "file_name": f"benign/{device}",
            }
        ]

        # train scaler for normalization
        trainer = ScalerTrainer(
            batch_size=1024,
            dataset_name=dataset_name,
            files=train_file,
        )

        scaler = SklearnOutlierDetector("MinMaxScaler", "sklearn.preprocessing")
        {"model": scaler} >> trainer

        # scaler = load_pkl_model(dataset_name, "MinMaxScaler")

        models = [
            (ICL, {"preprocessors": [scaler.model.transform, to_tensor]}),
            (GOAD,{"preprocessors": [scaler.model.transform, to_tensor]}),
            (KitNET, {"preprocessors": [to_numpy]}),
        ]

        trainer = OutlierDetectionTrainer(
            batch_size=128,
            dataset_name=dataset_name,
            files=train_file,
            val_epochs=5,
            epochs=50,
        )

        for cls, conf in models:
            model = cls(**conf)
            {"model": model} >> trainer


def evaluate_uq_models(device, fe_name, attacks):
    
    for device in devices:
        dataset_name = f"uq/{device}"

        test_file = [{
                    "dataset_name": dataset_name,
                    "fe_name": fe_name,
                    "file_name": f"benign/{device}",
                }]
        for a in attacks:
            test_file.append(
                {
                    "dataset_name": dataset_name,
                    "fe_name": fe_name,
                    "file_name": f"malicious/{a}",
                }
            )

        evaluator = OutlierDetectionEvaluator(
            batch_size=128,
            files=test_file,
            dataset_name=dataset_name,
            test_split=0.9,
            metrics=[count, detection_rate],
        )

        # eval models
        for model_name in models:
            if model_name in ["ICL","GOAD"]:
                model = getattr(sys.modules[__name__], model_name)()
                load_torch_model(model, dataset_name, f"{model_name}-0")
            else:
                model = load_pkl_model(dataset_name, f"{model_name}-0")
            {"model": model} >> evaluator



if __name__ == "__main__":
    devices = [
        "Lenovo_Bulb_1","Cam_1","Smart_TV","Raspberry_Pi_telnet","Smart_Clock_1","Google_Nest_Mini_1","Smartphone_1"
    ] 
    
    fe_name="after_image"
    attacks = ["Port_Scanning", "Service_Detection"]
    models = ["ICL","GOAD","Kitsune"]
    
    train_uq_models(devices,fe_name)
    evaluate_uq_models(devices,fe_name,attacks, models)
