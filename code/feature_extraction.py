import feature_extractors
import itertools 
import pickle


    

def extract_network_features(dataset_name, fe_name, fe_config):
    f=getattr(feature_extractors, fe_name)(**fe_config)
    {"dataset_name":dataset_name, "file_name":f"benign/whole_week", "state":None}>>f
    
    #targetted attacks
    devices=["Lenovo_Bulb_1","Smart_TV","Cam_1","Raspberry_Pi_telnet","Smart_Clock_1","Google_Nest_Mini_1","Smartphone_1","Smartphone_2"]
    attacks=["ACK_Flooding", "SYN_Flooding", "UDP_Flooding", "Port_Scanning", "Service_Detection"]
    
    for device, attack in itertools.product(devices, attacks):
        with open(f"../../datasets/{dataset_name}/{f.name}/state.pkl", "rb") as pf:
            state=pickle.load(pf)
        f=getattr(feature_extractors, fe_name)(state=state,**fe_config)
        {"dataset_name":dataset_name, "file_name":f"malicious/{device}/{attack}"}>>f
        
    
    with open(f"../../datasets/{dataset_name}/{f.name}/state.pkl", "rb") as pf:
        state=pickle.load(pf)
    f=getattr(feature_extractors, fe_name)(state=state,**fe_config)
    {"dataset_name":dataset_name, "file_name":"malicious/Host_Discovery_10"}>>f

if __name__=="__main__":
    extract_network_features("UQ_IoT_IDS21","AfterImageGraph",{"graph_type":"homo"})
    # get_node_map("AfterImageGraph_homo","UQ_IoT_IDS21")