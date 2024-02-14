
from datasets.custom_dataset import *

from pipelines.od_pipelines import *
from metrics.od_metrics import *
from models import *
import torch
import sys




def train_uq_models(fe_name, dataset_name):
    

    
    train_file = [
        {
            "dataset_name": dataset_name,
            "fe_name": fe_name,
            "file_name": f"benign/whole_week",
        }
    ]

    models = [
        (GOAD, {}),
        (ICL, {}),
        (SLAD, {}),
        (KitNET, {}),
    ]

    trainer = OnlineODEvaluator(
        batch_size=128,
        dataset_name=dataset_name,
        files=train_file,
        metrics=[detection_rate,average_score, average_threshold],
        steps=["process","save_model"]
    )

    for cls, conf in models:
        model = cls(**conf)
        {"model": model} >> trainer


def evaluate_uq_models(devices, fe_name, attacks, models, dataset_name):
    test_file = []    
    for device in devices:
        for attack in attacks:
            test_file.append(
                {
                    "dataset_name": dataset_name,
                    "fe_name": fe_name,
                    "file_name": f"malicious/{device}/{attack}",
                }
            )

        evaluator = OnlineODEvaluator(
            batch_size=128,
            dataset_name=dataset_name,
            files=test_file,
            metrics=[detection_rate,average_score, average_threshold],
            steps=["process"]
    )

        # eval models
        for model_name in models:
            if model_name in ["GOAD","ICL","SLAD"]:
                model = getattr(sys.modules[__name__], model_name)()
                load_torch_model(model, dataset_name, f"{model_name}")
            else:
                model = load_pkl_model(dataset_name, f"{model_name}")
            {"model": model} >> evaluator



if __name__ == "__main__":
    devices = [
        "Smart_TV","Raspberry_Pi_telnet","Smart_Clock_1","Google_Nest_Mini_1","Smartphone_1","Lenovo_Bulb_1","Cam_1",
    ] 
    dataset_name = f"UQ_IoT_IDS21"
    fe_name="AfterImage"
    attacks = ["Port_Scanning", "Service_Detection","ACK_Flooding","SYN_Flooding","UDP_Flooding"]
    models = ["SLAD","ICL","KitNET"] #"GOAD",
    
    # train_uq_models(fe_name, dataset_name)
    evaluate_uq_models(devices, fe_name, attacks, models, dataset_name)
