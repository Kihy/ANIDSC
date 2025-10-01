import itertools
from ANIDSC.pipeline.pipeline import Pipeline
from ANIDSC.templates import get_pipeline
import yaml


def feature_extraction(fe_name):
    dataset_name="../test_data"
    
    benign_file="benign_lenovo_bulb"
    malicious_files=["malicious_ACK_Flooding", "malicious_Port_Scanning", "malicious_Service_Detection"]
    
    # benign data
    template=get_pipeline("feature_extraction", dataset_name=dataset_name, file_name=benign_file, fe_name=fe_name, save_buffer=True)
            
    pipeline=Pipeline.load(template)
    pipeline.setup()
    pipeline.start()
    
    for file in malicious_files:
        
        
        # malicious data, assume that the attack is occuring straight after benign traffic 
        saved_file=f"{dataset_name}/{fe_name}/saved_components/pipeline/{benign_file}/PacketReader->{fe_name}->TabularFeatureBuffer(256).yaml"    
        
        with open(saved_file) as f:
            manifest = yaml.safe_load(f)
            
        manifest["attrs"]["manifest"]["data_source"]["attrs"]["file_name"]=file
        
        pipeline=Pipeline.load(manifest)
        
        pipeline.on_load()
        pipeline.start()

def basic_pipeline():
    fe_name="AfterImage"
    
    dataset_name="../test_data"
    
    benign_file="benign_lenovo_bulb"
    malicious_files=["malicious_ACK_Flooding", "malicious_Port_Scanning", "malicious_Service_Detection"]
    
    models=["torch_model.AE","torch_model.ICL","torch_model.Kitsune","torch_model.GOAD","torch_model.SLAD","torch_model.VAE"]
    
    # get fe_attrs
    saved_file=f"{dataset_name}/{fe_name}/saved_components/pipeline/{benign_file}/PacketReader->{fe_name}->TabularFeatureBuffer(256).yaml"
    with open(saved_file) as f:
        manifest = yaml.safe_load(f)
    fe_attrs=manifest["attrs"]["manifest"]["feature_extractor"]["attrs"]
    
    for model in models:
        template=get_pipeline("detection", dataset_name=dataset_name, file_name=benign_file, model_name=model, fe_name=fe_name, fe_attrs=fe_attrs)
        
        pipeline=Pipeline.load(template)
        pipeline.setup()
        pipeline.start()
        
        for file in malicious_files:
            
            saved_file=f"{dataset_name}/{fe_name}/saved_components/pipeline/{benign_file}/CSVReader->LivePercentile->OnlineOD({model})->BaseEvaluator.yaml"
        
            with open(saved_file) as f:
                manifest = yaml.safe_load(f)
                
            manifest["attrs"]["manifest"]["data_source"]["attrs"]["file_name"]=file
            
            pipeline=Pipeline.load(manifest)
            pipeline.on_load()
            pipeline.start()

def graph_pipeline():
    fe_name="AfterImageGraph"
    
    dataset_name="../test_data"
    
    benign_file="benign_lenovo_bulb"
    malicious_files=["malicious_ACK_Flooding", "malicious_Port_Scanning", "malicious_Service_Detection"]
    
    models=["torch_model.AE","torch_model.ICL","torch_model.Kitsune","torch_model.GOAD","torch_model.SLAD","torch_model.VAE"]
    node_encoders=["GCNNodeEncoder","GATNodeEncoder","LinearNodeEncoder"]
    
    # get fe_attrs
    saved_file=f"{dataset_name}/{fe_name}/saved_components/pipeline/{benign_file}/PacketReader->{fe_name}->TabularFeatureBuffer(256).yaml"
    with open(saved_file) as f:
        manifest = yaml.safe_load(f)
    fe_attrs=manifest["attrs"]["manifest"]["feature_extractor"]["attrs"]
    
    for model, node_encoder in itertools.product(models, node_encoders):
        template=get_pipeline("graph_detection", dataset_name=dataset_name, file_name=benign_file, model_name=model, fe_name=fe_name, fe_attrs=fe_attrs, node_encoder=node_encoder)
        
        pipeline=Pipeline.load(template)
        pipeline.setup()
        pipeline.start()
        
        for file in malicious_files:
            
            saved_file=f"{dataset_name}/{fe_name}/saved_components/pipeline/{benign_file}/CSVReader->MultilayerSplitter({node_encoder}->{model}).yaml"
        
            with open(saved_file) as f:
                manifest = yaml.safe_load(f)
                
            manifest["attrs"]["manifest"]["data_source"]["attrs"]["file_name"]=file
            
            pipeline=Pipeline.load(manifest)
            pipeline.on_load()
            pipeline.start()
    


if __name__ =="__main__":
    feature_extraction("AfterImage")
    basic_pipeline()
    
    feature_extraction("AfterImageGraph")
    graph_pipeline()