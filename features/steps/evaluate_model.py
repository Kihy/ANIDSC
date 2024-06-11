from behave import *
from src.ANIDSC.templates import train_models



@given(
    "A MultiLayerOCDModel with {model} as detection engine and {GNN} with {prior} as node encoder"
)
def step_impl(context, model, GNN, prior):
    gnn_kwargs = {
        "l_features": 15,
        "n_features": 15,
        "embedding_dist": prior,
        "model_name": f"{GNN}-{prior}",
        "node_latent_dim": 15,
        "save_type": "pth",
    }

    base_configs = {
        "n": 10000,
        "device": "cuda",
        "preprocessors": ["to_device"],
        "model_name": model,
        "save_type": "pth",
        "n_features": gnn_kwargs["node_latent_dim"],
    }

    context.model_dict = {
        "cls": "MultiLayerOCDModel",
        "conf": {
            "model_name": f"MultiLayerOCDModel-1000-50-{gnn_kwargs['model_name']}-{model}",
            "base_model_cls": "OnlineCDModel",
            "protocols": ["TCP", "DNS", "SSH", "FTP", "HTTP", "UDP", "HTTPS"],
            "save_type": "ensemble",
            "base_model_config": {
                "model_name": "OnlineCDModel",
                "base_model_cls": "GNNOCDModel",
                "save_type": "ensemble",
                "base_model_config": {
                    "model_name": "GNNOCDModel",
                    "save_type": "mix",
                    "gnn_cls": GNN,
                    "gnn_kwargs": gnn_kwargs,
                    "od_cls": model,
                    "od_kwargs": base_configs,
                },
                "patience": 1000,
                "confidence": 50,
            },
        },
    }


@when(
    "we want to train with AfterImageGraph features from {filename} in {dataset} dataset with the following metrics"
)
def step_impl(context, filename, dataset):
    metrics = [row["metric_name"] for row in context.table]

    train_models("AfterImageGraph", dataset, context.model_dict, filename, metrics, 256)
