# ANIDSC
 Adversarial NIDS Chain

# Setting up directory
It is recommended to have two folders, one for dataset and one for actual code.

The dataset folder structure should be something like ./datasets/{dataset_name}/pcap/{filename}

There is no restriction on code structure 

# Setting Up environment
install docker

run
`docker run --gpus all -it --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm -v "path/to/this/folder":/workspace/ANIDSC -v "path/to/datasets":/workspace/datasets kihy/nids_framework`

if gpu is not set up you can remove --gpus all 

go to code folder 
`cd /workspace/ANIDSC`
and add code there.

The ANIDSC package should already be installed, just need to import.

# Running the code
The examples folder include example scripts to run this package:
* adv_gen.py shows how to conduct adversarial attack, still under development
* live_detection.py shows how to create NIDS pipeline 
* summarize_results.py contains functions to plot the anomaly scores

change dataset paths accordingly

There is also feature folder for scenario testing that can be used as example

# extending the code
To add a new compenents, extend classes described in under src/ANIDSC/base_files 