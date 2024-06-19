
import argparse
from ANIDSC.templates import train_models


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate models')
    parser.add_argument('model_idx', metavar='N', type=int,
                        help='index of model')
    parser.add_argument('--graph_based',action=argparse.BooleanOptionalAction,
                        help='whether to use graph based features')
    
    parser.add_argument('--embedding_dist', action='store', type=str, help='The embedding distribution.', default="gaussian")
    parser.add_argument('--GNN', action='store', type=str, help='The GNN model.', default="GCNNodeEncoder")
    
    parser.add_argument('--dataset_name', action='store', type=str, help='The dataset used.', default="UQ_IoT_IDS21")
    parser.add_argument('--fe_name', action='store', type=str, help='The feature extractor used.', default="AfterImageGraph_multi_layer")
    parser.add_argument('--file_name', nargs='+', type=str, help='The benign file name.', default="benign/whole_week")
    parser.add_argument('--batch_size', metavar='N', type=int, default=256,
                        help='batch size for training')
   
    args = parser.parse_args()

    devices = [
        "Smart_TV",
        "Raspberry_Pi_telnet",
        "Smart_Clock_1",
        "Google_Nest_Mini_1",
        "Smartphone_1",
        "Lenovo_Bulb_1",
        "Cam_1",
    ]  # ,
    
    attacks = [
        "Port_Scanning",
        "Service_Detection",
        "ACK_Flooding",
        "SYN_Flooding",
        "UDP_Flooding",
    ]  #

    batch_size = args.batch_size
    
    
    base_configs = {"n": 10000,
                    "n_features":100,
                    "device":"cuda",
                    "preprocessors":["to_device"]}
    
    if not args.graph_based:
        base_configs["preprocessors"].insert(0,"standardize")
        
        
    model_list = [
        ("AE", "AE", base_configs, "pth"),
        ("VAE", "VAE", base_configs, "pth"),
        ("ICL", "ICL", base_configs, "pth"),
        ("SLAD","SLAD", base_configs, "pth"),
        ("GOAD", "GOAD", base_configs, "pth"),
        ("KitNET", "KitNET", base_configs, "pkl"),
        ("ARCUS", "ARCUS", dict(base_configs, **{"seed" :10,
        "_model_type" :"RAPP",
        "_inf_type" :"ADP",
        "_batch_size" :256,
        "_min_batch_size":32,
        "_init_epoch" :5,
        "_intm_epoch" :1,
        "hidden_dim":24,
        "layer_num":3,
        "learning_rate":1e-4,
        "_reliability_thred" :0.95,
        "_similarity_thred" :0.80}), "pth")
    ]

    gnn_kwargs = {
        "l_features": 15,
        "n_features": 15,
        "embedding_dist": args.embedding_dist,
        "model_name":f"{args.GNN}-{args.embedding_dist}",
        "node_latent_dim": 15,
        "save_type":"pth"
    }
    patience = 1000
    confidence = 50
    metrics = ["detection_rate"]

    
    name, cls, conf, save_type = model_list[args.model_idx]
    conf["model_name"] = name
    conf["save_type"] = save_type
    
    if args.graph_based:
        conf["n_features"] = gnn_kwargs["node_latent_dim"]
        if cls=="KitNET":
            conf["FM_grace_period"]=100
        model_dict = {
                "cls":"MultiLayerOCDModel",
                "conf":{
                    "model_name":f"MultiLayerOCDModel-{patience}-{confidence}-{gnn_kwargs['model_name']}-{name}",
                    "base_model_cls": "OnlineCDModel",
                    "protocols":["UDP", "TCP","ARP", "ICMP"],
                    "save_type":"ensemble",
                    "base_model_config": {
                        "model_name": f"OnlineCDModel",
                        "base_model_cls": "GNNOCDModel",
                        "save_type": "ensemble",
                        "base_model_config": {
                            "model_name": "GNNOCDModel",
                            "save_type": "mix",
                            "gnn_cls": args.GNN,
                            "gnn_kwargs": gnn_kwargs,
                            "od_cls": cls,
                            "od_kwargs": conf,
                        },
                        "patience": patience,
                        "confidence": confidence,
                    }
                }
                }
        
        
    else:
        model_dict = {
                "cls": cls,
                "conf": conf
            }
        
    

    train_models(args.fe_name, args.dataset_name, model_dict, args.file_name, metrics, batch_size)
    
  
    