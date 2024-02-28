import json 
from pathlib import Path
import pandas as pd
import numpy as np 

def get_results(dataset_name):
    results_path=Path(f"../../datasets/{dataset_name}/results.json")
            
    with open(str(results_path)) as f:
        results=json.load(f)
    
    records=[]
    for model, contents in results.items():
        benign_dr=[]
        mal_dr=[]
        for file, metrics in contents.items():
            if file.startswith("benign"):
                benign_dr.append(metrics["detection_rate"])
            elif file.startswith("malicious"):
                mal_dr.append(metrics["detection_rate"])
        records.append([model, np.mean(benign_dr), np.mean(mal_dr)])
    
    records=pd.DataFrame(records,columns=["Model","Ben DR", "Mal DR"])
    print(records)

if __name__=="__main__":
    get_results("UQ_IoT_IDS21")