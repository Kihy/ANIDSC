import pickle
from . import feature_extractors
from . import models
from .pipelines.od_pipelines import OnlineODEvaluator



def extract_network_features(dataset_name, fe_name, fe_config, file_name, state=None):
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
        "save_state": True,
    } >> f
    


def synthetic_features(dataset_name, fe_name):
    fe = getattr(feature_extractors, fe_name)(
        dataset_name, "benign/feature_correlation_test"
    )
    fe.generate_features(1)    

def train_models(fe_name, dataset_name, model_dict, file_name, metrics, batch_size):
    if isinstance(file_name,list):
        train_file = [
            {
                "file_name": f,
            }
            for f in file_name
        ]
    else:
        train_file=[{"file_name":file_name}]
        
    trainer = OnlineODEvaluator(
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