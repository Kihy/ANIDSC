# ANIDSC
 Adversarial NIDS Chain

# Setting up directory
It is recommended to add code to existing code structure

- ANIDSC folder contains all source code
- docker_root folder contains docker setup files
- experiments contains code to run experiments in slurm, but can be easily modified to run with docker 
- features folder contains feature files with behave for basic testing 
- visualisations folder contains visualisation code to analyse data.

The dataset folder structure should be something like datasets/{dataset_name}/pcap/{filename}
Any pcap file should work, the main dataset used for testing is the UQ-IoT-IDS dataset: https://espace.library.uq.edu.au/view/UQ:17b44bb
Some test data are also available in datasets/test_data/pcap folder


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
  -v "$(pwd)/experiments/scripts":/workspace/intrusion_detection/experiments \
  -v "$(pwd)/features":/workspace/intrusion_detection/features \
  -v "$(pwd)/.vscode":/workspace/intrusion_detection/.vscode\
  -w /workspace/intrusion_detection/ \
  -u $(id -u):$(id -g)  \
  kihy/anidsc_image
```

Script inside docker 
```
cd experiments

python3 real_experiment.py homogeneous uq_dataset --config "{\"fe_name\": \"SingleLayerGraphExtractor\",\"graph_rep\": \"CDD\", \"reader_type\":\"JsonGraphReader\", \"model_name\": \"MedianDetector\", \"node_embed\": \"PassThroughEmbedder\", \"run_identifier\": \"demo\"}"
```

## running visualisation container

To run the visualisation container:
```
docker run --rm --gpus all -it \
  -v "$(pwd)/datasets":/workspace/intrusion_detection/datasets \
  -v "$(pwd)/visualisations":/workspace/intrusion_detection/visualisations \
  -w /workspace/intrusion_detection/visualisations \
  -u $(id -u):$(id -g)  \
  -p 5007:5007 \
  kihy/anidsc_vis_image
```


run whatever app you have
```
panel serve app.py --address 0.0.0.0 --port 5007 --dev
panel serve multi-layer.py --address 0.0.0.0 --port 5007 --dev
```

Note:
The permissions are set as follows:
- within the docker container, the user is hostuser and hostgroup
- within the host system, it is the user and group that runs the command

## running behave jobs 
```
docker run --rm --gpus all -it \
  -v "$(pwd)/ANIDSC":/workspace/intrusion_detection/ANIDSC \
  -v "$(pwd)/datasets":/workspace/intrusion_detection/datasets \
  -v "$(pwd)/features":/workspace/intrusion_detection/features \
  -w /workspace/intrusion_detection/ \
  -u $(id -u):$(id -g)  \
  kihy/anidsc_image \
  behave features --stop 
```

## running slurm jobs
```
sbatch experiments/jobs/run_experiment.slurm test_dataset "experiments/configs/multi_feature.json"
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

