# ANIDSC
 Adversarial NIDS Chain

# Setting up directory
It is recommended to add code to existing code structure


The dataset folder structure should be something like datasets/{dataset_name}/{filename}
If you have labels, you can use that as part of filename, such as attack/file1.pcap 

Any pcap file should work, the main dataset used for testing is the UQ-IoT-IDS dataset: https://espace.library.uq.edu.au/view/UQ:17b44bb

Some test data are also available in datasets/test_data folder

All files should be run under the first ANIDSC directory. You will get file not found error if you dont.

# Examples
Running examples are in experiments folder, where it contains slurm and python script to run experiments and hyperparameter tuning.

Features files are also available for testing and debugging. The conversion is essentially put Pipeline variable to json file in configs file, and Hyperparameter Tuning variable to hyper_specs json file.


# Setting Up environment
install docker. 
run from top level directory (same level as the top level ANIDSC file):
```
docker build --pull --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g)  -t kihy/anidsc_image -f docker_root/Dockerfile.exp .
```

The visualisation/plotting container:
```
docker build --pull --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g)  -t kihy/anidsc_vis_image -f docker_root/Dockerfile.vis .
```

Push Docker images:
```
docker push kihy/anidsc_image:latest
```

verify gpu is intalled correctly via:
```
docker run --rm --gpus all -it kihy/anidsc_image python -c "import torch; print(torch.cuda.device_count(), torch.cuda.get_device_name(0))"
```


## running docker for debugging
To run the experiment container:
```
docker run --rm --gpus all -it \
  -v "$(pwd)/ANIDSC":/workspace/intrusion_detection/ANIDSC \
  -v "$(pwd)/datasets":/workspace/intrusion_detection/datasets \
  -v "$(pwd)/runs":/workspace/intrusion_detection/runs \
  -v "$(pwd)/experiments":/workspace/intrusion_detection/experiments \
  -v "$(pwd)/features":/workspace/intrusion_detection/features \
  -v "$(pwd)/.vscode":/workspace/intrusion_detection/.vscode\
  -w /workspace/intrusion_detection/ \
  -u $(id -u):$(id -g)  \
  kihy/anidsc_image
```

Run with:
```
python3 run_experiment.py run test_data --config experiments/config/metadata-extraction-template/protocol-meta-extraction.yaml --run_identifier meta_extraction
``` 


## running visualisation container

To run the visualisation container:
```
docker run --rm --gpus all -it \
  -v "$(pwd)/runs":/workspace/intrusion_detection/runs \
  -v "$(pwd)/visualisations":/workspace/intrusion_detection/visualisations \
  -v "$(pwd)/.vscode":/workspace/intrusion_detection/.vscode\
  -u $(id -u):$(id -g)  \
  -p 5007:5007 \
  -p 8080:8080 \
  -w /workspace/intrusion_detection \
  kihy/anidsc_vis_image
```


run whatever app you have
```
panel serve main.py --address 0.0.0.0 --port 5007 --dev

optuna-dashboard --host 0.0.0.0 --port 8080 sqlite:///optuna.db
```

Note:
The permissions are set as follows:
- within the docker container, the user is hostuser and hostgroup
- within the host system, it is the user and group that runs the command



## running slurm jobs
```
sbatch experiments/jobs/run_experiment.slurm <python script> [anything passed to the script]
```

Use `--dependency=afterok:xxxx` straight after `sbatch` to ensure dependency is correct

# Predefined Pipelines
Predefined pipelines are written in `templates.py`. Here we list some

## Meta Extraction
Extracts metadata (e.g., payload, size) for each packet, can be in any form but we use Dictionary and save as CSV

DataReader(pcap file) --> MetaExtractor --> FeatureBuffer(CSV)

## Feature Extraction 
Extracts features from metadata, either to tabular (CSV) or graph (NDJSON)

DataReader(CSV) --> FeatureExtractor --> FeatureBuffer(CSV/NDJSON)

## Detection

Detection is most complex, as many different methods of detection exists. At the most basic level, we have the basic detection pipeline:

DataReader --> Scaler --> Model --> Evaluator

For graph-based detection, we have lager:

DataReader --> Splitter (Scaler --> Node Embedder --> Model --> Evaluator) 

For generic graph based detection, we have:

DataReader --> GraphRepresentation --> Node Embedder --> Scaler (optional) --> Model --> Evaluator

# Structure 

```
ANIDSC
├── ANIDSC # source code
├── datasets # stores different datasets
├── docker_root # docker build files
├── experiments # experiments
│   ├── configs # experiment configs in json
│   ├── hyper_specs # hyperparameter configs for tuning with optuna
│   ├── jobs # slurm jobs
│   ├── logs # slurm logs
│   └── scripts # python scripts to run experiments
├── features # feature files for behaviour testing
│   └── steps # step file in python
├── runs # contains results from different runs
└── visualisations # visualisation scripts 
```

Each run (pipeline_name + run_identifier) refers to a single model. This is because each model may have different hyperparameters to tune. 

Directories:
Each run consists of three parts: 
- the file, which can be further decomposed into:
  - dataset
  - label (possibly multiple, e.g., attack/device)
  - file
- the pipeline, by default the name comes from the template 
- a run indentifier 

Depending on the parts of the pipeline, there might be different outputs:
- features folder from NumpyFeatureBuffer and DictFeatureBuffer
- graphs folder from JsonFeatureBuffer
- results folder from Evaluator


The structure of `runs` folder would be:

```
runs
└── {dataset}
    └── {pipeline}
        └── {run_id}
            ├── logs.out               # logs from slurm
            ├── optuna.db              # optuna database for tuning over the pipeline 
            ├── summary.csv            # summary of average precision across all files
            └── {label}
                └── {file}
                    ├── features.csv   # from NumpyFeatureBuffer, DictFeatureBuffer
                    ├── graphs.ndjson  # from JsonFeatureBuffer
                    └── results.csv    # from Evaluator
                    
```
Changing the structure require modifications in OutputWriter and BaseSaveMixin

