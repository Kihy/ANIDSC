from ANIDSC.evaluations.evaluator import BaseEvaluator
from ANIDSC.feature_buffers.tabular import TabularFeatureBuffer
from ANIDSC.data_sources import PacketReader, CSVReader
from ANIDSC.feature_extractors.after_image import AfterImage
from ANIDSC.models.base_model import BaseOnlineODModel

from ANIDSC.normalizer.t_digest import LivePercentile
from ANIDSC.templates import METRICS



if __name__ == "__main__":
    
    
    dataset_name = "./datasets/UQ-IoT-IDS"
    benign_file = "benign_samples/whole_week"
    
    dataset_name = "./datasets/UQ-IoT-IDS"
    benign_file = "benign_samples/whole_week"
    
    offline_reader = PacketReader(dataset_name, benign_file, max_records=10000)
    
    feature_extractor=AfterImage()
    feature_buffer=TabularFeatureBuffer()
    scaler=LivePercentile()
    detector=BaseOnlineODModel("ANIDSC.models.torch_models.autoencoder.AE")
    evaluator=BaseEvaluator(METRICS)
    
    pipeline = feature_extractor|feature_buffer|scaler|detector|evaluator
    
    
    offline_reader >> pipeline
    offline_reader.start()
    