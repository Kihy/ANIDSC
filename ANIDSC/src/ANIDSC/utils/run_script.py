import gc
from ..pipeline import Pipeline
import yaml
from textwrap import indent
from ..templates import get_template 

def yaml_align(data, indent=0):
    lines = []

    if isinstance(data, dict):
        # Longest key at this level
        max_len = max(len(str(k)) for k in data.keys())

        for k, v in data.items():
            key = str(k).ljust(max_len)
            prefix = " " * indent

            if isinstance(v, dict):
                lines.append(f"{prefix}{key}:")
                lines.extend(yaml_align(v, max_len + indent + 2))
            else:
                lines.append(f"{prefix}{key}: {v}")

    else:
        lines.append(" " * indent + str(data))

    return lines


def pprint(data):
    """Return fully aligned YAML-style string for the entire structure."""
    print("\n".join(yaml_align(data, indent=0)))

def run_file(file_iterator, pipeline_vars, return_pipeline=False):
    pipelines = []
    for state, file, dataset in file_iterator():        
        pipeline_vars["file_name"] = file
        pipeline_vars["dataset_name"] = dataset
        pipeline_name=pipeline_vars["pipeline_name"]
        
        print(f"Running {state} pipeline {pipeline_name}:")
        print("-"*50)
        pprint(pipeline_vars)
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
            manifest["attrs"]["run_identifier"] = pipeline_vars["run_identifier"]

            pipeline = Pipeline.load(manifest)
            pipeline.setup()
        else:
            raise ValueError("Unknown State", state)
        pipeline.start()

        print("Execution completed successfully!")
        if return_pipeline:
            pipelines.append(pipeline)
        else:
            del pipeline 
            gc.collect()

    return pipelines

