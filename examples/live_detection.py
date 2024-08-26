import itertools

import torch


from ANIDSC import cdd_frameworks, models
from ANIDSC.data_source import LiveSniffer, PacketReader, CSVReader
from ANIDSC.base_files import Processor, FeatureBuffer
from ANIDSC.feature_extractors import AfterImageGraph, AfterImage, FrequencyExtractor
from ANIDSC.base_files import BaseEvaluator

import warnings
from ANIDSC.templates import get_pipeline

def uq_feature_extraction(fe_type="vanilla"):
    # benign
    dataset_name = "../datasets/UQ_IoT_IDS21"
    benign_file = "benign/whole_week"
    
    offline_reader = PacketReader(dataset_name, benign_file)
    
    pipeline=get_pipeline(None, ["fe", fe_type])
    offline_reader >> pipeline
    offline_reader.start()

    # malicious
    attacks = [
        "ACK_Flooding",
        "UDP_Flooding",
        "SYN_Flooding",
        "Port_Scanning",
        "Service_Detection",
    ]

    devices = [
        "Cam_1",
        "Google_Nest_Mini_1",
        "Lenovo_Bulb_1",
        "Raspberry_Pi_telnet",
        "Smart_Clock_1",
        "Smart_TV",
        "Smartphone_1",
        "Smartphone_2",
    ]

    for d, a in itertools.product(devices, attacks):
        file_name = f"malicious/{d}/{a}"
        
        offline_reader = PacketReader(dataset_name, file_name)

        if fe_type=="vanilla":
            fe_name="AfterImage"
        else:
            fe_name = "AfterImageGraph(TCP,UDP,ARP,ICMP,Other)"
        
        pipeline=get_pipeline(None, ["fe", fe_type], load_existing=[dataset_name, fe_name, benign_file])
        offline_reader >> pipeline
        offline_reader.start()


def uq_benign(
    
    pipeline_desc,
    pipeline_componenets=["detection"],
):
    dataset_name = "../datasets/UQ_IoT_IDS21"
    file_name = "benign/whole_week"

    pipeline = get_pipeline(
        pipeline_components=pipeline_componenets,
        pipeline_desc=pipeline_desc,
        load_existing=False
    )

    if pipeline_desc["fe_cls"]=="AfterImage":
        offline_reader = CSVReader(dataset_name, "AfterImage", "AfterImage", file_name)
    else:
        offline_reader = CSVReader(
                dataset_name, "AfterImageGraph", "AfterImageGraph(TCP,UDP,ARP,ICMP,Other)", file_name
            )
    offline_reader >> pipeline
    offline_reader.start()
    
    


def uq_malicious(
    pipeline_desc,
    pipeline_componenets=["detection"],
):
    dataset_name = "../datasets/UQ_IoT_IDS21"
    benign_file = "benign/whole_week"
    attacks = [
        "ACK_Flooding",
        "UDP_Flooding",
        "SYN_Flooding",
        "Port_Scanning",
        "Service_Detection",
    ]

    devices = [
        "Cam_1",
        "Google_Nest_Mini_1",
        "Lenovo_Bulb_1",
        "Raspberry_Pi_telnet",
        "Smart_Clock_1",
        "Smart_TV",
        "Smartphone_1",
        "Smartphone_2",
    ]

    for d, a in itertools.product(devices, attacks):
        file_name = f"malicious/{d}/{a}"

        if pipeline_desc["fe_cls"]=="AfterImage":
            offline_reader = CSVReader(dataset_name, "AfterImage", "AfterImage", file_name)
            fe_name="AfterImage"
        else:
            offline_reader = CSVReader(
                    dataset_name, "AfterImageGraph", "AfterImageGraph(TCP,UDP,ARP,ICMP,Other)", file_name
                )
            fe_name="AfterImageGraph(TCP,UDP,ARP,ICMP,Other)"
        

        pipeline = get_pipeline(
            pipeline_components=pipeline_componenets,
            pipeline_desc=pipeline_desc,
            load_existing=[dataset_name, fe_name, benign_file]
        )
        
        offline_reader >> pipeline
        offline_reader.start()



def uq_benign_boxplot():
    feature_extractor = FrequencyExtractor()
    feature_buffer = FeatureBuffer(buffer_size=256)

    model = getattr(models, "BoxPlot")()

    evaluator = BaseEvaluator(
        [
            "detection_rate",
            "median_score",
            "median_threshold",
            "pos_count",
            "batch_size",
        ],
        log_to_tensorboard=True,
        save_results=True,
        draw_graph_rep_interval=100,
    )

    dataset_name = "../datasets/UQ_IoT_IDS21"
    benign_file = "benign/whole_week"
    pipeline = feature_extractor | feature_buffer | model | evaluator
    offline_reader = PacketReader(dataset_name, benign_file)

    offline_reader >> pipeline
    offline_reader.start()


def uq_malicious_boxplot():
    dataset_name = "../datasets/UQ_IoT_IDS21"
    benign_file = "benign/whole_week"

    feature_buffer = FeatureBuffer(buffer_size=256)

    model = getattr(models, "BoxPlot").load_pickle(
        "models", dataset_name, "FrequencyExtractor", benign_file, "BoxPlot"
    )

    evaluator = BaseEvaluator(
        [
            "detection_rate",
            "median_score",
            "median_threshold",
            "pos_count",
            "batch_size",
        ],
        log_to_tensorboard=True,
        save_results=True,
        draw_graph_rep_interval=100,
    )

    attacks = [
        "ACK_Flooding",
        "UDP_Flooding",
        "SYN_Flooding",
        "Port_Scanning",
        "Service_Detection",
    ]

    devices = [
        "Cam_1",
        "Google_Nest_Mini_1",
        "Lenovo_Bulb_1",
        "Raspberry_Pi_telnet",
        "Smart_Clock_1",
        "Smart_TV",
        "Smartphone_1",
        "Smartphone_2",
    ]

    for d, a in itertools.product(devices, attacks):
        file_name = f"malicious/{d}/{a}"

        feature_extractor = FrequencyExtractor.load_pickle(
            "feature_extractors",
            dataset_name,
            "FrequencyExtractor",
            benign_file,
            "FrequencyExtractor",
        )
        pipeline = feature_extractor | feature_buffer | model | evaluator

        offline_reader = PacketReader(dataset_name, file_name)
        offline_reader >> pipeline
        offline_reader.start()


if __name__ == "__main__":

    # basic boxplot
    # uq_benign_boxplot()
    # uq_malicious_boxplot()


    # evaluate models
    model_names = ["AE", "VAE", "GOAD", "ICL", "Kitsune", "SLAD"] 
    
    # #vanilla models 
    # with warnings.catch_warnings():
    #     warnings.filterwarnings("error", category=RuntimeWarning)
    #     for model_name in model_names:
            
    #         pipeline_desc={"fe_cls": "AfterImage", "model_name": model_name}
    #         uq_benign(pipeline_desc, pipeline_componenets=["detection"])
    #         uq_malicious(pipeline_desc,pipeline_componenets=["detection"])

    # # graph models
    node_encoders = ["LinearNodeEncoder","GCNNodeEncoder", "GATNodeEncoder"] #"LinearNodeEncoder",
    distribution = "log-normal"

    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=RuntimeWarning)
        for node_encoder, model_name in itertools.product(node_encoders, model_names):
            if not (node_encoder=="LinearNodeEncoder" and model_name != "SLAD"):
                continue 
            pipeline_desc={"fe_cls": "AfterImageGraph", 
                            "model_name": model_name,
                       "node_encoder":node_encoder,
                       "distribution":distribution}
            
            uq_benign(pipeline_desc)
            uq_malicious(pipeline_desc)

    # # concept drift models
    # cdd=["DriftSense"] #"ARCUS",
    # with warnings.catch_warnings():
    #     warnings.filterwarnings("error", category=RuntimeWarning)
    #     for cdd_framework, model_name in itertools.product(cdd, model_names):
            
    #         # if cdd_framework=="ARCUS" and model_name in ["AE","VAE","GOAD","ICL","Kitsune"]: # goad consumes too much memory, ICL and SLAD always creates too many model, Kitsune cannot merge models since feature mapper is different
    #         #     continue 
            
    #         if cdd_framework=="DriftSense" and model_name in ["AE","VAE","GOAD"]: # goad consumes too much memory,
    #             continue 
            
    #         pipeline_desc={"fe_cls": "AfterImage", "model_name": model_name,
    #                     "cdd_framework":cdd_framework}
    #         uq_benign(pipeline_desc, pipeline_componenets=["detection","cdd"])
    #         uq_malicious(pipeline_desc,pipeline_componenets=["detection","cdd"])
    
    # graph cdd models 
    
    # node_encoder = "GCNNodeEncoder"
    # distribution = "gaussian"
    # cdd_framework="DriftSense"
    # with warnings.catch_warnings():
    #     warnings.filterwarnings("error", category=RuntimeWarning)
    #     for model_name in model_names:
            
    #         pipeline_desc={"fe_cls": "AfterImageGraph", 
    #                         "model_name": model_name,
    #                         "cdd_framework":cdd_framework,
    #                    "node_encoder":node_encoder,
    #                    "distribution":distribution}
            
    #         uq_benign(pipeline_desc,pipeline_componenets=["detection","cdd"])
    #         uq_malicious(pipeline_desc,pipeline_componenets=["detection","cdd"])
    
    
    