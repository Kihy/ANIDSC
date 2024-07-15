from ANIDSC import models
from ANIDSC.adversarial_attacks.liuer_mihou import LiuerMihouAttack
from ANIDSC.base_files.feature_extractor import FeatureBuffer
from ANIDSC.data_source.offline_sources import PacketReader
from ANIDSC.feature_extractors.after_image import AfterImage


if __name__=="__main__":
    model_name="AE"
    dataset_name = "../datasets/Test_Data"
    benign_file = "benign_lenovo_bulb"
    file_name=f"malicious_Port_Scanning"
    
    fe_name="AfterImage"
    
    model = getattr(models, model_name).load_pickle(
            "models", dataset_name, fe_name, benign_file, model_name
        )
    feature_extractor = AfterImage.load_pickle(
                "feature_extractors", dataset_name, fe_name, benign_file, fe_name
            )
    
    feature_buffer = FeatureBuffer(buffer_size=256)
    
    lm_attack=LiuerMihouAttack(feature_extractor, model)
    
    pipeline = lm_attack | feature_extractor | feature_buffer
    
    offline_reader = PacketReader(dataset_name, file_name)
    offline_reader >> pipeline
    
    offline_reader.start()
    