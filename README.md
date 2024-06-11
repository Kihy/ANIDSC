# ANIDSC
 Adversarial NIDS Chain

# Setting Up environment
install docker

run
`docker run --gpus all -it --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm -v "path/to/this/folder":/workspace/ANIDSC -v "path/to/datasets":/workspace/datasets -v "path/to/models":/workspace/models kihy/nids_framework`

# Running the code
The examples folder include example scripts to run this package
feature_extraction.py shows how to extract features
evaluate_models.py shows how to evaluate NIDS models
generate_report.py contains functions to obtain summary

# extending the code
To add a new feature extractor, inherit BaseTrafficFeatureExtractor from base_feature_extractor.py
To add a new model, inherit BaseOnlineODModel and one of the save mixins from base_model.py