import pickle
from . import feature_extractors
from . import models
from .pipelines.online_pipelines import OnlineODPipeline, LiveODPipeline
from typing import Dict, Any, List, Union

def live_pipeline(dataset_name:str, model_dict:Dict[str, Any], fe_name:str, fe_config:Dict[str, Any],
    file_name: str, metrics:List[str]
    )->None:
    
    model = getattr(models, model_dict["cls"])(**model_dict["conf"])
    fe = getattr(feature_extractors, fe_name)(**fe_config)

    pipeline = LiveODPipeline(
        dataset_name=dataset_name,
        model=model, 
        fe=fe,
        metric_list=metrics,
        steps=["process", "save_model"],
    )
    
    {"file_name":file_name}>>pipeline


def extract_network_features(
    dataset_name: str,
    fe_name: str,
    fe_config: Dict[str, Any],
    file_name: str,
    state: str = None,
) -> None:
    """template for extracting network feature from pcap files

    Args:
        dataset_name (str): name of dataset
        fe_name (str): name of feature extractor
        fe_config (Dict[str, Any]): configurations of feature extractor
        file_name (str): name of file to be extracted
        state (str, optional): name of state used to initialize feature extractor. Defaults to None.
    """    
    if state is not None:
        with open(
            f"../datasets/{dataset_name}/{fe_name}/state/{state}.pkl", "rb"
        ) as pf:
            state = pickle.load(pf)

    f = getattr(feature_extractors, fe_name)(**fe_config)

    {
        "dataset_name": dataset_name,
        "file_name": file_name,
        "state": state,
    } >> f


def extract_synthetic_features(dataset_name: str, file_name: str) -> None:
    """extract synthetic features

    Args:
        dataset_name (str): name of dataset
        file_name (str): name of extracted file
    """    
    fe = getattr(feature_extractors, "SyntheticFeatureExtractor")(
        dataset_name, file_name
    )
    fe.generate_features(1)


def train_models(
    fe_name: str,
    dataset_name: str,
    model_dict: Dict[str, Any],
    file_name: Union[str,List[str]],
    metrics: List[str],
    batch_size: int,
) -> None:
    """train model with extracted features

    Args:
        fe_name (str): name of feature extractor
        dataset_name (str): name of dataset
        model_dict (Dict[str, Any]): configuration of model
        file_name (Union[str,List[str]]): name of feature file
        metrics (List[str]): list of metric names to evaluate the model
        batch_size (int): batch size for training
    """    
    if isinstance(file_name, list):
        train_file = [
            {
                "file_name": f,
            }
            for f in file_name
        ]
    else:
        train_file = [{"file_name": file_name}]

    trainer = OnlineODPipeline(
        batch_size=batch_size,
        dataset_name=dataset_name,
        fe_name=fe_name,
        files=train_file,
        epochs=1,
        # percentage=[0, 0.01],
        metric_list=metrics,
        steps=["process", "save_model"],
    )

    model = getattr(models, model_dict["cls"])(**model_dict["conf"])
    {"model": model} >> trainer
