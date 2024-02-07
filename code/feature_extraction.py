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

def extract_clean_uq_data():
    f=AfterImage()
    devices=["Smartphone_1"] #"Lenovo_Bulb_1","Smart_TV","Cam_1","Raspberry_Pi_telnet","Smart_Clock_1","Google_Nest_Mini_1",
    attacks=["ACK_Flooding", "SYN_Flooding", "UDP_Flooding", "Port_Scanning", "Service_Detection"]

    for device in devices:
        dataset_name=f"uq/{device}" 
        {"dataset_name":dataset_name, "file_name":f"benign/{device}",
        "state":None}>>f
        
        for a in attacks:
            file_name=f"malicious/{a}"
            # parse attacks based on previous state
            with open(f"../../datasets/{dataset_name}/{f.name}/state.pkl", "rb") as pf:
                state=pickle.load(pf)
            {"dataset_name":dataset_name, "file_name":file_name, "state":state}>>f

if __name__=="__main__":
    extract_clean_uq_data()