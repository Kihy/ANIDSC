from ANIDSC import models
from ANIDSC.adversarial_attacks.liuer_mihou import LiuerMihouAttack, PcapSaver
from ANIDSC.base_files.evaluator import BaseEvaluator
from ANIDSC.base_files.feature_extractor import FeatureBuffer
from ANIDSC.data_source.offline_sources import PacketReader
from ANIDSC.feature_extractors.after_image import AfterImage
from ANIDSC.normalizer.t_digest import LivePercentile


METRICS=[
            "detection_rate",
            "median_score",
            "median_threshold",
            "pos_count",
            "batch_size",
        ]

if __name__=="__main__":
    model_name="AE"
    dataset_name = "../datasets/Test_Data"
    benign_file = "benign_lenovo_bulb"
    file_name=f"malicious_ACK_Flooding"
    
    fe_name="AfterImage"
    
    model = getattr(models, model_name).load_pickle(
            "models", dataset_name, fe_name, benign_file, model_name
        )
    
    feature_extractor = AfterImage.load_pickle(
                "feature_extractors", dataset_name, fe_name, benign_file, fe_name
            )
    
    standardizer = LivePercentile.load_pickle(
            "scalers", dataset_name, fe_name, benign_file, "LivePercentile"
    )
    
    feature_buffer = FeatureBuffer(buffer_size=256)
    
    pcap_saver=PcapSaver()
    
    evaluator = BaseEvaluator(
        METRICS,
        log_to_tensorboard=True,
        save_results=True,
        draw_graph_rep_interval=0,
    )
    
    lm_attack=LiuerMihouAttack(feature_extractor, model, standardizer)
    
    pipeline = lm_attack | pcap_saver | feature_extractor | feature_buffer | standardizer | model | evaluator
    
    pipeline.set_save(False)
    
    offline_reader = PacketReader(dataset_name, file_name)
    offline_reader >> pipeline
    
    offline_reader.start()
    