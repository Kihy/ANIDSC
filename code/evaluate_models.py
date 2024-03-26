from datasets.custom_dataset import *

from pipelines.od_pipelines import *
from metrics.od_metrics import *
from models import *
import models


def train_uq_models(fe_name, dataset_name, model_list):
    train_file = [
        {
            "dataset_name": dataset_name,
            "fe_name": fe_name,
            "file_name": f"benign/whole_week",
        }
    ]
    trainer = OnlineODEvaluator(
        batch_size=256,
        dataset_name=dataset_name,
        files=train_file,
        # percentage=[0,0.0001],
        metrics=[average_metric("edge_loss"),average_metric("node_loss"), average_metric("structural_loss")],
        steps=["process", "save_model"],
    )

    for cls, conf, _ in model_list:
        model = getattr(models, cls)(**conf)
        {"model": model} >> trainer


def evaluate_uq_models(devices, fe_name, attacks, models, dataset_name):
    for device in devices:
        test_file = []
        for attack in attacks:
            test_file.append(
                {
                    "dataset_name": dataset_name,
                    "fe_name": fe_name,
                    "file_name": f"malicious/{device}/{attack}",
                }
            )

        evaluator = OnlineODEvaluator(
            batch_size=256,
            dataset_name=dataset_name,
            files=test_file,
            metrics=[average_metric("node_energy"), average_metric("edge_energy"), average_metric("node_reliability"),
                 average_metric("edge_loss"),average_metric("node_loss"), average_metric("structural_loss")],
            steps=["process"],
        )

        # eval models
        for model_name, model_config, save_type in models:
            model=load_model(dataset_name, model_name, save_type, model_config)
            
            {"model": model} >> evaluator


if __name__ == "__main__":
    devices = [
        "Smart_TV",
        "Raspberry_Pi_telnet",
        "Smart_Clock_1",
        "Google_Nest_Mini_1",
        "Smartphone_1",
        "Lenovo_Bulb_1",
        "Cam_1",
    ]  # ,
    dataset_name = "UQ_IoT_IDS21"
    fe_name = "AfterImageGraph_homo"
    attacks = [
        "ACK_Flooding",
        "Port_Scanning",
        "Service_Detection",
        "SYN_Flooding",
        "UDP_Flooding",
    ]  #
    model_list = [
        # (ICL, {}),
        # ("KitNET", {},"pkl"),
        # (SLAD, {}),
        # (GOAD, {}),
        # ("RRCF", {},"dict"),
        # ("ARCUS",{"seed" :10,
        # "_model_type" :"RAPP",
        # "_inf_type" :"ADP",
        # "_batch_size" :256,
        # "_min_batch_size":32,
        # "_init_epoch" :5,
        # "_intm_epoch" :1,
        # "hidden_dim":24, 
        # "layer_num":3,
        # "learning_rate":1e-4,
        # "_reliability_thred" :0.95,
        # "_similarity_thred" :0.80}, "pth"),
        ("HomoGNNIDS",{},"pth")
        
    ]

    train_uq_models(fe_name, dataset_name,model_list)
    evaluate_uq_models(devices, fe_name, attacks, model_list, dataset_name)
