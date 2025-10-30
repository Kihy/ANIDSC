from ..pipeline import Pipeline
import yaml

from ..templates import get_template 

def run_file(file_iterator, pipeline_name, pipeline_vars):
    pipelines = []
    for state, file, dataset in file_iterator():        
        pipeline_vars["file_name"] = file
        pipeline_vars["dataset_name"] = dataset
        
        print(f"Running {state} pipeline {pipeline_name}:")
        print("-"*50)
        for k,v in pipeline_vars.items():
            print(f"{k:<20} {v:<20}")
        print("-"*50)

        if state == "new":
            pipeline = Pipeline.load(get_template(pipeline_name, **pipeline_vars))
            pipeline.setup()

            benign_path = pipeline.save_path

        elif state == "loaded":

            with open(benign_path) as f:
                manifest = yaml.safe_load(f)

            # datasource is always 0
            manifest["attrs"]["manifest"][0]["attrs"]["file_name"] = pipeline_vars[
                "file_name"
            ]
            manifest["attrs"]["manifest"][0]["file"] = None

            pipeline = Pipeline.load(manifest)
            pipeline.setup()
        pipeline.start()

        print("Execution completed successfully!")
        pipelines.append(pipeline)

    return pipelines

