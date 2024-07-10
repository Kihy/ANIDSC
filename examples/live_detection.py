import itertools
from ANIDSC import models
from ANIDSC.data_source import LiveSniffer, PacketReader, CSVReader
from ANIDSC.base_files import Processor, FeatureBuffer
from ANIDSC.feature_extractors import AfterImageGraph, AfterImage, FrequencyExtractor
from ANIDSC.models import AE, GCNNodeEncoder, HomoGraphRepresentation
from ANIDSC.base_files import (
    BaseEvaluator,
    MultilayerSplitter,
    CollateEvaluator,
    ConceptDriftWrapper,
)
from ANIDSC.models.gnnids import LinearNodeEncoder, NodeEncoderWrapper
from ANIDSC.normalizer import LivePercentile
import warnings


def get_pipeline(
    model_name,
    from_pcap=True,
    pipeline_type="vanilla",
    node_encoder_type="LinearNodeEncoder",
    dist_type="gaussian",
):
    if model_name == "KitNET":
        preprocessors = []
    else:
        preprocessors = ["to_float_tensor", "to_device"]

    standardizer = LivePercentile()
    evaluator = BaseEvaluator(
        [
            "detection_rate",
            "median_score",
            "median_threshold",
            "pos_count",
            "batch_size",
        ],
        log_to_tensorboard=(pipeline_type == "vanilla"),
        save_results=(pipeline_type == "vanilla"),
        draw_graph_rep_interval=0,
    )

    if pipeline_type == "vanilla":
        model = getattr(models, model_name)(preprocessors=preprocessors, profile=False)

        if from_pcap:
            feature_extractor = AfterImage()
            feature_buffer = FeatureBuffer(buffer_size=256)
            pipeline = (
                feature_extractor | feature_buffer | standardizer | model | evaluator
            )
        else:
            pipeline = standardizer | model | evaluator

    elif pipeline_type == "graph_cdd":
        # graph cdd model
        model = getattr(models, model_name)(preprocessors=[])
        node_encoder = getattr(models, node_encoder_type)(15, dist_type)
        encoder_model = NodeEncoderWrapper(node_encoder, model)
        cd_model = ConceptDriftWrapper(encoder_model, 1000, 50)

        standardizer = LivePercentile()

        graph_rep = HomoGraphRepresentation(preprocessors=preprocessors)

        collate_evaluator = CollateEvaluator(log_to_tensorboard=True, save_results=True)
        protocol_splitter = MultilayerSplitter(
            pipeline=(standardizer | graph_rep | cd_model | evaluator)
        )

        if from_pcap:
            feature_extractor = AfterImageGraph(["TCP", "UDP", "ARP", "ICMP"])
            feature_buffer = FeatureBuffer(buffer_size=256)
            pipeline = (
                feature_extractor
                | feature_buffer
                | protocol_splitter
                | collate_evaluator
            )
        else:
            pipeline = protocol_splitter | collate_evaluator

    return pipeline


def load_pipeline(
    dataset_name,
    fe_name,
    benign_file,
    model_name,
    from_pcap=True,
    pipeline_type="vanilla",
    node_encoder_type="LinearNodeEncoder",
    dist_type="gaussian",
):
    evaluator = BaseEvaluator(
        [
            "detection_rate",
            "median_score",
            "median_threshold",
            "pos_count",
            "batch_size",
        ],
        log_to_tensorboard=(pipeline_type == "vanilla"),
        save_results=(pipeline_type == "vanilla"),
        draw_graph_rep_interval=0,
    )

    if pipeline_type == "vanilla":
        standardizer = LivePercentile.load_pickle(
            "scalers", dataset_name, fe_name, benign_file, "LivePercentile"
        )
        model = getattr(models, model_name).load_pickle(
            "models", dataset_name, fe_name, benign_file, model_name
        )

        if from_pcap:
            feature_buffer = FeatureBuffer(buffer_size=256)

            feature_extractor = AfterImage.load_pickle(
                "feature_extractors", dataset_name, fe_name, benign_file, fe_name
            )
            pipeline = (
                feature_extractor | feature_buffer | standardizer | model | evaluator
            )
        else:
            pipeline = standardizer | model | evaluator

    elif pipeline_type == "graph_cdd":
        collate_evaluator = CollateEvaluator(log_to_tensorboard=True, save_results=True)
        protocol_splitter = MultilayerSplitter.load_pickle(
            "models",
            dataset_name,
            fe_name,
            benign_file,
            f"MultlayerSplitter(LivePercentile-HomoGraphRepresentation-ConceptDriftWrapper(NodeEncoderWrapper({node_encoder_type}({dist_type})-{model_name}))-BaseEvaluator)",
        )

        if from_pcap:
            feature_buffer = FeatureBuffer(buffer_size=256)
            feature_extractor = AfterImageGraph.load_pickle(
                "feature_extractors", dataset_name, fe_name, benign_file, fe_name
            )

            pipeline = (
                feature_extractor
                | feature_buffer
                | protocol_splitter
                | collate_evaluator
            )
        else:
            pipeline = protocol_splitter | collate_evaluator
    return pipeline


def feature_extraction(
    dataset_name, file_name, benign_file="", pipeline_type="vanilla"
):
    print(f"extracting {file_name}")
    offline_reader = PacketReader(dataset_name, file_name)
    feature_buffer = FeatureBuffer(buffer_size=256)

    if pipeline_type == "vanilla":
        if benign_file != "":
            feature_extractor = AfterImage.load_pickle(
                "feature_extractors",
                dataset_name,
                "AfterImage",
                benign_file,
                "AfterImage",
            )
        else:
            feature_extractor = AfterImage()
    elif pipeline_type == "graph_cdd":
        protocols = ["TCP", "UDP", "ARP", "ICMP"]
        if benign_file != "":
            feature_extractor = AfterImageGraph.load_pickle(
                "feature_extractors",
                dataset_name,
                f"AfterImageGraph({','.join(protocols+['Other'])})",
                benign_file,
                f"AfterImageGraph({','.join(protocols+['Other'])})",
            )
        else:
            feature_extractor = AfterImageGraph(protocols)

    pipeline = feature_extractor | feature_buffer

    offline_reader >> pipeline
    offline_reader.start()


def uq_feature_extraction(pipeline_type="vanilla"):
    # benign
    dataset_name = "../datasets/UQ_IoT_IDS21"
    benign_file = "benign/whole_week"
    feature_extraction(dataset_name, benign_file, pipeline_type=pipeline_type)

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
        feature_extraction(
            dataset_name, file_name, benign_file, pipeline_type=pipeline_type
        )


def uq_benign(
    model_name,
    from_pcap=True,
    pipeline_type="vanilla",
    node_encoder_type="LinearNodeEncoder",
    dist_type="gaussian",
):
    dataset_name = "../datasets/UQ_IoT_IDS21"
    file_name = "benign/whole_week"

    pipeline = get_pipeline(
        model_name,
        from_pcap,
        pipeline_type=pipeline_type,
        node_encoder_type=node_encoder_type,
        dist_type=dist_type,
    )

    if pipeline_type == "vanilla":
        fe_name = "AfterImage"
    elif pipeline_type == "graph_cdd":
        fe_name = "AfterImageGraph(TCP,UDP,ARP,ICMP,Other)"

    if from_pcap:
        # data sources
        offline_reader = PacketReader(dataset_name, file_name)  #
    else:
        if pipeline_type == "vanilla":
            offline_reader = CSVReader(dataset_name, "AfterImage", fe_name, file_name)
        elif pipeline_type == "graph_cdd":
            offline_reader = CSVReader(
                dataset_name, "AfterImageGraph", fe_name, file_name
            )

    offline_reader >> pipeline
    offline_reader.start()


def uq_malicious(
    model_name,
    from_pcap=True,
    pipeline_type="vanilla",
    node_encoder_type="LinearNodeEncoder",
    dist_type="gaussian",
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
    if pipeline_type == "vanilla":
        fe_name = "AfterImage"
    elif pipeline_type == "graph_cdd":
        fe_name = "AfterImageGraph(TCP,UDP,ARP,ICMP,Other)"

    for d, a in itertools.product(devices, attacks):
        file_name = f"malicious/{d}/{a}"

        if from_pcap:
            # data sources
            offline_reader = PacketReader(dataset_name, file_name)  #
        else:
            if pipeline_type == "vanilla":
                offline_reader = CSVReader(
                    dataset_name, "AfterImage", fe_name, file_name
                )
            else:
                offline_reader = CSVReader(
                    dataset_name, "AfterImageGraph", fe_name, file_name
                )

        pipeline = load_pipeline(
            dataset_name,
            fe_name,
            benign_file,
            model_name,
            from_pcap,
            pipeline_type=pipeline_type,
            node_encoder_type=node_encoder_type,
            dist_type=dist_type,
        )
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
    file_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    for file_name in file_names:

        # data sources
        offline_reader = PacketReader(dataset_name, f"{file_name}-WorkingHours")  #

        offline_reader >> pipeline
        offline_reader.start()


def cic_cdd():
    model = AE(preprocessors=["to_float_tensor", "to_device"], profile=False)
    cd_model = ConceptDriftWrapper(model, 1000, 50)
    standardizer = LivePercentile()
    evaluator = BaseEvaluator(
        ["detection_rate", "median_score", "median_threshold"],
        log_to_tensorboard=True,
        save_results=True,
        draw_graph_rep_interval=0,
    )
    pipeline = standardizer | cd_model | evaluator

    dataset_name = "../datasets/CIC_IDS_2017"
    file_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    for file_name in file_names:

        # data sources
        offline_reader = CSVReader(
            dataset_name, "AfterImage", "AfterImage", f"{file_name}-WorkingHours"
        )  #

        offline_reader >> pipeline
        offline_reader.start()


def cic_graph_cdd():
    dataset_name = "../datasets/CIC_IDS_2017"
    file_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

    protocols = ["TCP", "UDP", "HTTP", "HTTPS", "SSH", "FTP"]
    feature_buffer = FeatureBuffer(buffer_size=256)
    model = AE(preprocessors=[], profile=False)
    node_encoder = GCNNodeEncoder(10, "gaussian")
    encoder_model = NodeEncoderWrapper(node_encoder, model)
    standardizer = LivePercentile()
    feature_extractor = AfterImageGraph(protocols)
    graph_rep = HomoGraphRepresentation(preprocessors=["to_float_tensor", "to_device"])
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
        draw_graph_rep_interval=0,
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
        draw_graph_rep_interval=0,
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

    pipeline_type = "graph_cdd"
    node_encoder_type = "LinearNodeEncoder"
    dist_type = "gaussian"
    # # feature extraction
    uq_feature_extraction(pipeline_type=pipeline_type)

    # # evaluate models
    model_names = ["AE", "GOAD", "VAE", "ICL", "SLAD", "KitNET"]

    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=RuntimeWarning)
        for model_name in model_names:
            uq_benign(
                model_name,
                False,
                pipeline_type=pipeline_type,
                node_encoder_type=node_encoder_type,
                dist_type=dist_type,
            )
            uq_malicious(
                model_name,
                False,
                pipeline_type=pipeline_type,
                node_encoder_type=node_encoder_type,
                dist_type=dist_type,
            )
