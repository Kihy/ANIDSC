from ANIDSC.data_source import LiveSniffer, PacketReader, CSVReader
from ANIDSC.base_files import Processor, FeatureBuffer
from ANIDSC.feature_extractors import AfterImageGraph, AfterImage
from ANIDSC.models import AE, GCNNodeEncoder, HomoGraphRepresentation
from ANIDSC.base_files import BaseEvaluator, MultilayerSplitter, CollateEvaluator, ConceptDriftWrapper
from ANIDSC.normalizer import LivePercentile


def print_data(data):
    print(data)
    return data


if __name__ == "__main__":

    dataset_name = "../datasets/UQ_IoT_IDS21"
    fe_name = "AfterImage"
    file_name = "benign/whole_week"

    printer = Processor(print_data)

    # data sources
    live_sniffer = LiveSniffer("lo")
    offline_reader = PacketReader(dataset_name, file_name, max_pkts=1e4)  #
    csv_reader = CSVReader(dataset_name, fe_name, file_name, n_features=100)

    protocols = ["TCP", "UDP", "ARP"]

    feature_buffer = FeatureBuffer(buffer_size=256)
    feature_extractor = AfterImageGraph(protocols)
    evaluator = BaseEvaluator(
        ["detection_rate", "average_score", "average_threshold"],
        log_to_tensorboard=False,
        save_results=False,
        draw_graph_rep_interval=10,
    )
    
    
    collate_evaluator = CollateEvaluator(log_to_tensorboard=True, save_results=True)

    model = AE(
        preprocessors=[],
        profile=False,
        node_encoder={
            "encoder_name": "GCNNodeEncoder",
            "node_latent_dim": 10,
            "embedding_dist": "gaussian",
        },
        load_existing=False,
    )

    standardizer = LivePercentile()  # skip first 4 elements that are indices

    # pipeline =  model | evaluator
    # csv_reader >> pipeline
    # csv_reader.start()

    feature_extraction = feature_extractor | feature_buffer

    graph_rep = HomoGraphRepresentation(preprocessors=["to_float_tensor", "to_device"])
    
    cd_model=ConceptDriftWrapper(model, 1000, 50)

    protocol_splitter = MultilayerSplitter(
        pipelines=(standardizer | graph_rep | cd_model | evaluator)
    )
    
    pipeline = feature_extraction | protocol_splitter | collate_evaluator
    
    offline_reader >> pipeline
    offline_reader.start()
