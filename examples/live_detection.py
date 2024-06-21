from ANIDSC.data_source import LiveSniffer, PacketReader, CSVReader
from ANIDSC.base_files import Processor, FeatureBuffer
from ANIDSC.feature_extractors import AfterImageGraph, AfterImage
from ANIDSC.models import AE, GCNNodeEncoder, HomoGraphRepresentation
from ANIDSC.base_files import BaseEvaluator, MultilayerSplitter, CollateEvaluator, ConceptDriftWrapper
from ANIDSC.models.gnnids import NodeEncoderWrapper
from ANIDSC.normalizer import LivePercentile


def print_data(data):
    print(data)
    return data


if __name__ == "__main__":

    dataset_name = "../datasets/UQ_IoT_IDS21"
    fe_name = "AfterImageGraph"
    file_name = "benign/whole_week"

    printer = Processor(print_data)

    # data sources
    live_sniffer = LiveSniffer("lo")
    offline_reader = PacketReader(dataset_name, file_name)  #
    csv_reader = CSVReader(dataset_name, fe_name, file_name, fe_features=100)

    protocols = ["TCP", "UDP", "ARP"]

    feature_buffer = FeatureBuffer(buffer_size=256)
    
    model = AE(
        preprocessors=[],
        profile=False,
        load_existing=False
    )

    node_encoder=GCNNodeEncoder(10, "gaussian")

    encoder_model=NodeEncoderWrapper(node_encoder, model)
    
    # standardizer = LivePercentile.load("scalers",dataset_name, fe_name, file_name, "LivePercentile")  # skip first 4 elements that are indices
    standardizer=LivePercentile()
    
    feature_extractor = AfterImageGraph(protocols)
    # feature_extractor=AfterImageGraph.load("feature_extractors", dataset_name, fe_name, file_name, fe_name)
    
    graph_rep = HomoGraphRepresentation(preprocessors=["to_float_tensor", "to_device"], load_existing=False)
    
    evaluator = BaseEvaluator(
        ["detection_rate", "average_score", "average_threshold"],
        log_to_tensorboard=False,
        save_results=False,
        draw_graph_rep_interval=10,
    )
    
    collate_evaluator = CollateEvaluator(log_to_tensorboard=True, save_results=True)


    # pipeline =  model | evaluator
    # csv_reader >> pipeline
    # csv_reader.start()

    feature_extraction = feature_extractor | feature_buffer

    cd_model=ConceptDriftWrapper(encoder_model, 1000, 50)

    protocol_splitter = MultilayerSplitter(
        pipeline=(standardizer | graph_rep | cd_model | evaluator)
    )
    
    pipeline = feature_extraction | protocol_splitter | collate_evaluator

    
    offline_reader >> pipeline
    offline_reader.start()
