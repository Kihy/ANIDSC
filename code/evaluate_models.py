from datasets.custom_dataset import *
from generate_report import plot_concept_drift
from pipelines.od_pipelines import *
from metrics.od_metrics import *
import argparse
import models

def train_models(fe_name, dataset_name, model_dict, file_name, metrics, batch_size):
    train_file = [
        {
            "file_name": file_name,
        }
    ]
    trainer = OnlineODEvaluator(
        batch_size=batch_size,
        dataset_name=dataset_name,
        fe_name=fe_name,
        files=train_file,
        epochs=1,
        # percentage=[0, 0.01],
        metrics=metrics,
        steps=["process", "save_model"],
    )

    torch.cuda.empty_cache()
    model = getattr(models, model_dict["cls"])(**model_dict["conf"])
    {"model": model} >> trainer

    plot_concept_drift(dataset_name, fe_name, file_name, model.model_name)


def evaluate_uq_models(
    devices, fe_name, attacks, model_dict, dataset_name, metrics, batch_size
):
    for device in devices:

        for attack in attacks:
            test_file = []
            test_file.append(
                {
                    "file_name": f"malicious/{device}/{attack}",
                }
            )

            evaluator = OnlineODEvaluator(
                batch_size=batch_size,
                dataset_name=dataset_name,
                fe_name=fe_name,
                files=test_file,
                metrics=metrics,
                steps=["process"],
                write_to_tensorboard=False
            )

            # eval models
            
            torch.cuda.empty_cache()
            model = load_model(dataset_name, model_dict["cls"], model_dict["conf"])

            {"model": model} >> evaluator
            
            # plot_concept_drift(
            #     dataset_name, fe_name, test_file[0]["file_name"], model.model_name
            # )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate models')
    parser.add_argument('model_idx', metavar='N', type=int,
                        help='index of model')
    parser.add_argument('--graph_based',action=argparse.BooleanOptionalAction,
                        help='whether to use graph based features')
   
    args = parser.parse_args()

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
    fe_name = "AfterImageGraph_multi_layer"
    # fe_name = "AfterImage"
    file_name = "benign/whole_week"

    # fe_name="SyntheticFeatureExtractor"
    # file_name="benign/mean_std_drift_4"
    # dataset_name = "FakeGraphData"

    attacks = [
        "Port_Scanning",
        "Service_Detection",
        "ACK_Flooding",
        "SYN_Flooding",
        "UDP_Flooding",
    ]  #

    batch_size = 256
    
    
    base_configs = {"n": 10000,
                    "n_features":100,
                    "device":"cuda",
                    "preprocessors":["to_device"]}
    
    if not args.graph_based:
        base_configs["preprocessors"].insert(0,"standardize")
    
    model_list = [
        ("AE", "AE", base_configs, "pth"),
        ("VAE", "VAE", base_configs, "pth"),
        ("ICL", "ICL", base_configs, "pth"),
        ("SLAD","SLAD", base_configs, "pth"),
        ("GOAD", "GOAD", base_configs, "pth"),
        ("KitNET", "KitNET", base_configs, "pkl"),
        ("ARCUS", "ARCUS", dict(base_configs, **{"seed" :10,
        "_model_type" :"RAPP",
        "_inf_type" :"ADP",
        "_batch_size" :256,
        "_min_batch_size":32,
        "_init_epoch" :5,
        "_intm_epoch" :1,
        "hidden_dim":24,
        "layer_num":3,
        "learning_rate":1e-4,
        "_reliability_thred" :0.95,
        "_similarity_thred" :0.80}), "pth")
    ]

    gnn_kwargs = {
        "l_features": 35,
        "n_features": 15,
        "embedding_dist": "gaussian",
        "model_name":"GCNNodeEncoder-gaussian",
        "node_latent_dim": 15,
        "save_type":"pth"
    }
    patience = 1000
    confidence = 50
    metrics = [detection_rate]

    
    name, cls, conf, save_type = model_list[args.model_idx]
    conf["model_name"] = name
    conf["save_type"] = save_type
    
    if args.graph_based:
        model_dict = {
                "cls":"MultiLayerOCDModel",
                "conf":{
                    "model_name":f"MultiLayerOCDModel-{patience}-{confidence}-{gnn_kwargs['model_name']}-{name}",
                    "base_model_cls": "OnlineCDModel",
                    "save_type":"ensemble",
                    "base_model_config": {
                        "model_name": f"OnlineCDModel",
                        "base_model_cls": "GNNOCDModel",
                        "save_type": "ensemble",
                        "base_model_config": {
                            "model_name": "GNNOCDModel",
                            "save_type": "mix",
                            "gnn_cls": "GCNNodeEncoder",
                            "gnn_kwargs": gnn_kwargs,
                            "od_cls": cls,
                            "od_kwargs": conf,
                        },
                        "patience": patience,
                        "confidence": confidence,
                    }
                }
                }
        
        conf["n_features"] = gnn_kwargs["node_latent_dim"]
    else:
        model_dict = {
                "cls": cls,
                "conf": conf
            }
        
    
    # torch.autograd.set_detect_anomaly(True)
    train_models(fe_name, dataset_name, model_dict, file_name, metrics, batch_size)
    
    
    evaluate_uq_models(
        devices, fe_name, attacks, model_dict, dataset_name, metrics, batch_size
    )
    
    