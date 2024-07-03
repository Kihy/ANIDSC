import itertools
from ANIDSC import models
from ANIDSC.data_source import LiveSniffer, PacketReader, CSVReader
from ANIDSC.base_files import Processor, FeatureBuffer
from ANIDSC.feature_extractors import AfterImageGraph, AfterImage
from ANIDSC.models import AE, GCNNodeEncoder, HomoGraphRepresentation
from ANIDSC.base_files import (
    BaseEvaluator,
    MultilayerSplitter,
    CollateEvaluator,
    ConceptDriftWrapper,
)
from ANIDSC.models.gnnids import NodeEncoderWrapper
from ANIDSC.normalizer import LivePercentile

def get_vanilla_pipeline(model_name, from_pcap=True):
    model = getattr(models, model_name)(preprocessors=["to_float_tensor", "to_device"], profile=False)
    standardizer = LivePercentile()
    evaluator = BaseEvaluator(
        ["detection_rate", "median_score", "median_threshold","pos_count","batch_size"],
        log_to_tensorboard=True,
        save_results=True,
        draw_graph_rep_interval=0,
    )
    
    if from_pcap:
        feature_extractor = AfterImage()
        feature_buffer = FeatureBuffer(buffer_size=256)
        pipeline = feature_extractor | feature_buffer | standardizer | model | evaluator
    else:
        pipeline = standardizer | model | evaluator
    
    return pipeline

def load_vanilla_pipeline(dataset_name, fe_name, benign_file, model_name, from_pcap=True):
    evaluator = BaseEvaluator(
        ["detection_rate", "median_score", "median_threshold","pos_count","batch_size"],
        log_to_tensorboard=True,
        save_results=True,
        draw_graph_rep_interval=0,
    )

    standardizer = LivePercentile.load_pickle(
        "scalers", dataset_name, fe_name, benign_file, "LivePercentile"
    )
    model = getattr(models, model_name).load_pickle("models",dataset_name, fe_name, benign_file, model_name)
    
    if from_pcap:
        feature_buffer = FeatureBuffer(buffer_size=256)
        
        feature_extractor = AfterImage.load_pickle(
                "feature_extractors", dataset_name, fe_name, benign_file, "AfterImage"
            )
        pipeline = feature_extractor | feature_buffer | standardizer | model | evaluator
    else:
        pipeline = standardizer | model | evaluator
    return pipeline 

def vanilla_feature_extraction(dataset_name, file_name, benign_file=''):
    offline_reader = PacketReader(dataset_name, file_name)
    feature_buffer = FeatureBuffer(buffer_size=256)
    
    if benign_file != '':
        feature_extractor = AfterImage.load_pickle(
                "feature_extractors", dataset_name, "AfterImage", benign_file, "AfterImage"
        )
    else:
        feature_extractor = AfterImage()
    pipeline = feature_extractor | feature_buffer
    
    offline_reader>>pipeline 
    offline_reader.start()
    
def uq_vanilla_feature_extraction():
    #benign
    dataset_name = "../datasets/UQ_IoT_IDS21"
    benign_file = "benign/whole_week"
    vanilla_feature_extraction(dataset_name, benign_file)
    
    #malicious
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
        vanilla_feature_extraction(dataset_name, file_name, benign_file)

def uq_benign_vanilla(model_name, from_pcap=True):
    dataset_name = "../datasets/UQ_IoT_IDS21"
    file_name = "benign/whole_week"

    pipeline=get_vanilla_pipeline(model_name, from_pcap)

    if from_pcap:
        # data sources
        offline_reader = PacketReader(dataset_name, file_name)  #
    else:
        offline_reader=CSVReader(dataset_name, "AfterImage",file_name)

    offline_reader >> pipeline
    offline_reader.start()

def uq_malicious_vanilla(model_name, from_pcap=True):
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
        
        if from_pcap:
            # data sources
            offline_reader = PacketReader(dataset_name, file_name)  #
        else:
            offline_reader=CSVReader(dataset_name, "AfterImage",file_name)

        pipeline=load_vanilla_pipeline(dataset_name, "AfterImage", benign_file, model_name, from_pcap)
        offline_reader >> pipeline
        offline_reader.start()

    

def uq_benign_cdd():
    dataset_name = "../datasets/UQ_IoT_IDS21"
    file_name = "benign/whole_week"

    # data sources
    csv_reader = CSVReader(dataset_name, "AfterImage", file_name)
    model = AE(preprocessors=["to_float_tensor", "to_device"], profile=False)
    cd_model=ConceptDriftWrapper(model, 1000, 50)
    standardizer = LivePercentile()
    evaluator = BaseEvaluator(
        ["detection_rate", "median_score", "median_threshold"],
        log_to_tensorboard=True,
        save_results=True,
        draw_graph_rep_interval=0,
    )
    pipeline = standardizer | cd_model | evaluator

    csv_reader >> pipeline
    csv_reader.start()
    

def uq_malicious_cdd():
    dataset_name = "../datasets/UQ_IoT_IDS21"
    fe_name = "AfterImage"

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
    evaluator = BaseEvaluator(
        ["detection_rate", "median_score", "median_threshold"],
        log_to_tensorboard=True,
        save_results=True,
        draw_graph_rep_interval=0,
    )

    for d, a in itertools.product(devices, attacks):
        file_name = f"malicious/{d}/{a}"
        csv_reader = CSVReader(dataset_name, "AfterImage", file_name)
        
        standardizer = LivePercentile.load_pickle(
            "scalers", dataset_name, fe_name, benign_file, "LivePercentile"
        )

        model =ConceptDriftWrapper.load_pickle("models", dataset_name, fe_name, benign_file, "ConceptDriftWrapper(AE)")
        pipeline = standardizer | model | evaluator

        csv_reader >> pipeline
        csv_reader.start()



def uq_benign_graph_cdd():
    dataset_name = "../datasets/UQ_IoT_IDS21"
    file_name = "benign/whole_week"
    protocols = ["TCP", "UDP", "ARP"]

    # data sources
    offline_reader = PacketReader(dataset_name, file_name)  #
    feature_buffer = FeatureBuffer(buffer_size=256)
    model = AE(preprocessors=[], profile=False)
    node_encoder = GCNNodeEncoder(10, "gaussian")
    encoder_model = NodeEncoderWrapper(node_encoder, model)
    standardizer = LivePercentile()
    feature_extractor = AfterImageGraph(protocols)
    graph_rep = HomoGraphRepresentation(
        preprocessors=["to_float_tensor", "to_device"]
    )
    evaluator = BaseEvaluator(
        ["detection_rate", "median_score", "median_threshold"],
        log_to_tensorboard=False,
        save_results=False,
        draw_graph_rep_interval=10,
    )
    collate_evaluator = CollateEvaluator(log_to_tensorboard=True, save_results=True)
    feature_extraction = feature_extractor | feature_buffer
    cd_model = ConceptDriftWrapper(encoder_model, 1000, 50)
    protocol_splitter = MultilayerSplitter(
        pipeline=(standardizer | graph_rep | cd_model | evaluator)
    )
    pipeline = feature_extraction | protocol_splitter | collate_evaluator

    offline_reader >> pipeline
    offline_reader.start()


def uq_malicious_graph_cdd():
    dataset_name = "../datasets/UQ_IoT_IDS21"
    fe_name = "AfterImageGraph"
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

    feature_buffer = FeatureBuffer(buffer_size=256)
    collate_evaluator = CollateEvaluator(log_to_tensorboard=True, save_results=True)

    for d, a in itertools.product(devices, attacks):
        file_name = f"malicious/{d}/{a}"
        offline_reader = PacketReader(dataset_name, file_name)

        feature_extractor = AfterImageGraph.load_pickle(
            "feature_extractors", dataset_name, fe_name, benign_file, "AfterImageGraph"
        )

        feature_extraction = feature_extractor | feature_buffer

        protocol_splitter = MultilayerSplitter.load_pickle(
            "models",
            dataset_name,
            fe_name,
            benign_file,
            "MultlayerSplitter(LivePercentile-HomoGraphRepresentation-ConceptDriftWrapper(NodeEncoderWrapper(GCNNodeEncoder-AE))-BaseEvaluator)",
        )

        pipeline = feature_extraction | protocol_splitter | collate_evaluator

        offline_reader >> pipeline
        offline_reader.start()
def cic_vanilla():
    model = AE(preprocessors=["to_float_tensor", "to_device"], profile=False)
    standardizer = LivePercentile()
    feature_extractor = AfterImage()
    evaluator = BaseEvaluator(
        ["detection_rate", "median_score", "median_threshold"],
        log_to_tensorboard=True,
        save_results=True,
        draw_graph_rep_interval=0,
    )
    feature_buffer = FeatureBuffer(buffer_size=256)
    pipeline = feature_extractor | feature_buffer | standardizer | model | evaluator

    dataset_name = "../datasets/CIC_IDS_2017"
    file_names = ["Monday","Tuesday","Wednesday","Thursday", "Friday"]
    for file_name in file_names:
        
        # data sources
        offline_reader = PacketReader(dataset_name, f"{file_name}-WorkingHours")  #
        
        offline_reader >> pipeline
        offline_reader.start()

def cic_cdd():
    model = AE(preprocessors=["to_float_tensor", "to_device"], profile=False)
    cd_model=ConceptDriftWrapper(model, 1000, 50)
    standardizer = LivePercentile()
    evaluator = BaseEvaluator(
        ["detection_rate", "median_score", "median_threshold"],
        log_to_tensorboard=True,
        save_results=True,
        draw_graph_rep_interval=0,
    )
    pipeline = standardizer | cd_model | evaluator
    
    dataset_name = "../datasets/CIC_IDS_2017"
    file_names = ["Monday","Tuesday","Wednesday","Thursday", "Friday"]
    for file_name in file_names:
        
        # data sources
        offline_reader = CSVReader(dataset_name, "AfterImage", f"{file_name}-WorkingHours")  #
        
        offline_reader >> pipeline
        offline_reader.start()


def cic_graph_cdd():
    dataset_name = "../datasets/CIC_IDS_2017"
    file_names = ["Monday","Tuesday","Wednesday","Thursday", "Friday"]
    
    protocols=["TCP","UDP","HTTP","HTTPS","SSH","FTP"]
    feature_buffer = FeatureBuffer(buffer_size=256)
    model = AE(preprocessors=[], profile=False)
    node_encoder = GCNNodeEncoder(10, "gaussian")
    encoder_model = NodeEncoderWrapper(node_encoder, model)
    standardizer = LivePercentile()
    feature_extractor = AfterImageGraph(protocols)
    graph_rep = HomoGraphRepresentation(
        preprocessors=["to_float_tensor", "to_device"]
    )
    evaluator = BaseEvaluator(
        ["detection_rate", "median_score", "median_threshold"],
        log_to_tensorboard=False,
        save_results=False,
        draw_graph_rep_interval=10,
    )
    collate_evaluator = CollateEvaluator(log_to_tensorboard=True, save_results=True)
    feature_extraction = feature_extractor | feature_buffer
    cd_model = ConceptDriftWrapper(encoder_model, 1000, 50)
    protocol_splitter = MultilayerSplitter(
        pipeline=(standardizer | graph_rep | cd_model | evaluator)
    )
    pipeline = feature_extraction | protocol_splitter | collate_evaluator
    
    for file_name in file_names:
        
        # data sources
        offline_reader = PacketReader(dataset_name, f"{file_name}-WorkingHours")  #
        
        offline_reader >> pipeline
        offline_reader.start()
    

if __name__ == "__main__":
    # feature extraction
    uq_vanilla_feature_extraction()
    
    # evaluate models
    model_names=["AE", "GOAD", "VAE", "ICL", "SLAD", "KitNET"] 
    for model_name in model_names:
        uq_benign_vanilla(model_name,False)
        uq_malicious_vanilla(model_name,False)
        
