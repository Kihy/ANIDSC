from itertools import product

from ANIDSC.evaluations import BasicSummarizer 

def get_file_names(dataset, adv_files=False):
    if dataset.endswith("CIC_IDS_2017"):
        return [f"{f}-WorkingHours" for f in ["Monday","Tuesday","Wednesday","Thursday", "Friday"]]
    elif dataset.endswith("UQ_IoT_IDS21"):
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

        return ["benign/whole_week"]+[f"malicious/{d}/{a}" for d, a in product(devices, attacks)]
    elif dataset.endswith("Test_Data"):
        files=["benign_lenovo_bulb", "malicious_ACK_Flooding","malicious_Service_Detection"]
        if adv_files:
            files+=["LiuerMihouAttack/malicious_ACK_Flooding"]
        return files
    else:
        raise ValueError("unknown dataset")
    
    
if __name__ == "__main__":
    dataset="../datasets/UQ_IoT_IDS21"
    # dataset="../datasets/Test_Data"
    
    # summarizer=BasicSummarizer([dataset],"AfterImage", get_file_names(dataset), calc_f1=True)
    # summarizer.plots()
    # summarizer.gen_summary()

    
    # summarizer=BasicSummarizer([dataset],"FrequencyExtractor", get_file_names(dataset), calc_f1=True)
    # # summarizer.plots()
    # summarizer.gen_summary()
    
        
    summarizer=BasicSummarizer([dataset], "AfterImageGraph(TCP,UDP,ARP,ICMP,Other)", get_file_names(dataset), calc_f1=True, col="protocol")
    summarizer.plots()
    summarizer.gen_summary()

