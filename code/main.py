from feature_extractors.after_image import AfterImage
from datasets.custom_dataset import *
from torch.utils.data import DataLoader
from models.sklearn_models import *
from models.trainer import *
from metrics.od_metrics import *
from adversarial_attacks.liuer_mihou import LiuerMihouAttack
from models.kitsune import KitNET
from models.base_model import *

# feature extraction pipeline
# with open("../../datasets/uq/benign/netstat/Lenovo_Bulb_1.pkl", "rb") as pf:
#        nstat=pickle.load(pf)
# fe=AfterImage(nstat=nstat)

# pcap_files=["../../datasets/uq/malicious/Lenovo_Bulb_1/adversarial/Lenovo_Bulb_1_ACK_Flooding_lm.pcap"]

# pcap_files>>fe

# save scaler as sklearn model
files = {
    "benign": ["Lenovo_Bulb_1"],
    "malicious": [
        "Lenovo_Bulb_1_Port_Scanning",
        "Lenovo_Bulb_1_Service_Detection",
    ],
    "adversarial": [
        "Lenovo_Bulb_1_ACK_Flooding_lm",
    ],
}

# model=SklearnOutlierDetector("MinMaxScaler","sklearn.preprocessing")
# trainer=OutlierDetectionTrainer(files=files, batch_size=None,
#                                 steps=["train","save"]
#                                 )
# {"model":model}>>trainer


# # training NIDS pipeline
# scaler=load_sklearn_model("Lenovo_Bulb_1","MinMaxScaler")
# model=SklearnOutlierDetector("LocalOutlierFactor", "sklearn.neighbors", model_params={"novelty":True}, preprocessors=[scaler.transform])

# model=KitNET(FM_grace_period=2200)

# trainer=OutlierDetectionPipeline(metrics=[special_f1,plot_scores],
#                                 batch_size=1024)

# {"files":files, "model":model}>>trainer

model = load_pkl_model("Lenovo_Bulb_1", "Kitsune")
trainer = OutlierDetectionPipeline(
    metrics=[mean_dr, plot_scores], steps=["eval"], batch_size=1024
)
{"files": files, "model": model} >> trainer

# mal_pcap="../../datasets/uq/malicious/Lenovo_Bulb_1/pcap/Lenovo_Bulb_1_ACK_Flooding.pcap"
# with open("../../datasets/uq/benign/netstat/Lenovo_Bulb_1.pkl", "rb") as pf:
#        nstat=pickle.load(pf)
# fe=AfterImage(nstat=nstat)
# adversarial_attack=LiuerMihouAttack(fe=fe, model=model,
#                                     )
# mal_pcap>>adversarial_attack
