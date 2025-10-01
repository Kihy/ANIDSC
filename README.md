# ANIDSC
 Adversarial NIDS Chain

# Setting up directory
It is recommended to have two folders, one for dataset and one for actual code.

The dataset folder structure should be something like ./datasets/{dataset_name}/pcap/{filename}

Any pcap file should work, the main dataset used for testing is the UQ-IoT-IDS dataset: https://espace.library.uq.edu.au/view/UQ:17b44bb
Some test data are also available in ./test_data/pcap folder



There is no restriction on code structure 

# Setting Up environment
install docker

run
`docker run --gpus all -it --rm -v "/path/to/datasets":/workspace/intrusion_detection/datasets -v /path/to/experiment/scripts:/workspace/intrusion_detection/experiment -v kihy/anidsc_image`

if gpu is not set up you can remove --gpus all 


go to code folder 
`cd /workspace/ANIDSC`
and add code there.

The ANIDSC package should already be installed, just need to import.

# Running the code
There is a feature folder for scenario testing that can be used as example


