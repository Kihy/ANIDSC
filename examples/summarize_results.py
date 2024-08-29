from itertools import product

from matplotlib import pyplot as plt
import networkx as nx
from ANIDSC.evaluations import BasicSummarizer
from ANIDSC.utils import draw_graph 

def dummy_uq_graph():
    G_b=nx.DiGraph()
    G_m=nx.DiGraph()


    G_m.add_nodes_from([
        ("Router", {"node_as": 90,"idx":"1", "updated":"False"}),
        ("Attacker", {"node_as": 120,"idx":"5", "updated":"False"}),
        ("Smart TV", {"node_as": 130,"idx":"6", "updated":"False"}),
    ])

    G_b.add_nodes_from([
       
        ("Smart Clock", {"node_as": 0.24, "idx":"2", "updated":"False"}),
        ("Google Nest", {"node_as": 0.23, "idx":"3", "updated":"False"}),
        ("Camera", {"node_as": 0.20, "idx":"0", "updated":"False"}),
        ("Router", {"node_as": 0.19, "idx":"1", "updated":"False"}),
        ("Smart Plug", {"node_as": 0.19, "idx":"4", "updated":"False"}),
    ])
    G_b.graph["threshold"]="0.1"
    G_m.graph["threshold"]="0.1"
    
    edges_b = [
        ("Smart Plug", "Router"),
        ("Camera", "Router"),
        ("Smart Clock", "Router"),
        ("Smart Clock", "Google Nest"),
        ("Google Nest", "Router"),
    ]

    edges_m = [
        ("Attacker", "Smart TV"),
        ("Smart TV", "Router"),
    ]

    for i, j in edges_b:
        G_b.add_edge(i, j)
        G_b.add_edge(j, i)
        
    for i, j in edges_m:
        G_m.add_edge(i, j)
        G_m.add_edge(j, i)
        
    fig, ax = plt.subplots(ncols=2, figsize=(5,2.5))

    draw_graph(G_b, fig, ax[0], False, "Benign", {}, {0:"Camera",1:"Router",2:"Smart Clock",3:"Google Nest", 4:"Smart Plug",
                                             5:"Attacker",6:"Smart TV"})
    draw_graph(G_m, fig, ax[1], False, "ACK Flooding", {}, {0:"Camera",1:"Router",2:"Smart Clock",3:"Google Nest", 4:"Smart Plug",
                                             5:"Attacker",6:"Smart TV"})
        

    fig.tight_layout()
    fig_name = f"tmp.png"
    fig.savefig(fig_name)
    print(fig_name)

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
    # # # summarizer.plots()
    # summarizer.gen_summary()
    
        
    summarizer=BasicSummarizer([dataset], "AfterImageGraph(TCP,UDP,ARP,ICMP,Other)", get_file_names(dataset), calc_f1=True, col="protocol")
    summarizer.plots()
    summarizer.gen_summary()

