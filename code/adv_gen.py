from feature_extractors.after_image import AfterImage
from feature_extractors.pyflowmeter import PyFlowMeter
from datasets.custom_dataset import *
from torch.utils.data import DataLoader
from models.sklearn_models import *
from models.trainer import *
from metrics.od_metrics import *
from adversarial_attacks.liuer_mihou import LiuerMihouAttack
from models.kitsune import KitNET
from models.base_model import *
import pyshark
from models.misc_models import *



for device in ["Lenovo_Bulb_1"]: #"Smart_TV","Cam_1","Raspberry_Pi_telnet","Smart_Clock_1","Google_Nest_Mini_1","Smartphone_1":
    dataset_name=f"uq/{device}" 

    for a in ["Port_Scanning","Service_Detection"]: #"ACK_Flooding","SYN_Flooding","UDP_Flooding",:
        with open(f"../../datasets/{dataset_name}/after_image/state.pkl", "rb") as pf:
            state=pickle.load(pf)
            
        file_name=f"malicious/{a}"
        fe=AfterImage(state=state)
        
        model = load_pkl_model("Lenovo_Bulb_1", "Kitsune")
        
        adversarial_attack=LiuerMihouAttack(fe=fe, model=model)
        {"dataset_name":dataset_name, "file_name":file_name}>>adversarial_attack
        
        for adv in ["LM/Kitsune"]:
            with open(f"../../datasets/{dataset_name}/{fe.name}/state.pkl", "rb") as pf:
                state=pickle.load(pf)
            fe=AfterImage(state=state)
            
            adv_file_name=f"adversarial/{adv}/{file_name}"
            {"dataset_name":dataset_name, "file_name":adv_file_name}>>fe
            
   


