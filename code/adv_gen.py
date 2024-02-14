from feature_extractors.after_image import AfterImage
from feature_extractors.pyflowmeter import PyFlowMeter
from datasets.custom_dataset import *
from torch.utils.data import DataLoader
from models.sklearn_models import *
from metrics.od_metrics import *
from adversarial_attacks.liuer_mihou import LiuerMihouAttack
from models.kitsune import KitNET
from models.base_model import *
import pyshark
from models.misc_models import *
from .pipelines.od_pipelines import OutlierDetectionEvaluator


fe="AfterImage"
for device in ["Lenovo_Bulb_1"]: #"Smart_TV","Cam_1","Raspberry_Pi_telnet","Smart_Clock_1","Google_Nest_Mini_1","Smartphone_1":
    dataset_name=f"uq/{device}" 
    for a in ["Port_Scanning","Service_Detection"]: #"ACK_Flooding","SYN_Flooding","UDP_Flooding",:            
        file_name=f"malicious/{a}"

        model = load_pkl_model("Lenovo_Bulb_1", "Kitsune")
        
        adversarial_attack=LiuerMihouAttack(fe=fe,
                                            dataset_name=dataset_name, 
                                            model=model,
                                            metrics=[count, detection_rate])
        {"file_name":file_name}>>adversarial_attack
        
        evaluator = OutlierDetectionEvaluator(
            batch_size=128,
            files={"dataset_name":dataset_name,
                   "fe_name":fe,
                "file_name":f"adverarial/{adversarial_attack.name}/{model.model_name}/{file_name}"},
            dataset_name=dataset_name,
            test_split=0.9,
            metrics=[count, detection_rate],
        )
        
        {"model": model} >> evaluator

        
        
   


