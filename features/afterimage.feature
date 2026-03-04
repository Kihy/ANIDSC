@baseline 
Feature: Detection with AfterImage Features
    Background:
        Given The test_data file iterator
        And Pipeline variable: run_identifier -> afterimage_test

    Scenario: Empty folders
        Given folders in test_data with run identifier are empty

    @meta-extraction
    Scenario: Meta extraction
        Given Pipeline variable: reader_type -> PacketReader
        And Pipeline variable: meta_extractor -> ProtocolMetaExtractor
        And Pipeline variable: template_name -> metadata-extraction-template
        And Pipeline variable: pipeline_name -> protocol-meta-extraction
        And the pipeline variables are saved to config
        When the pipeline starts
        Then the pipeline should not fail
        And the components are saved

    @feature-extraction
    Scenario: Feature extraction
        Given Pipeline variable: reader_type -> CSVReader
        And Pipeline variable: prev_pipeline -> afterimage_test/protocol-meta-extraction
        And Pipeline variable: feature_extractor -> AfterImage
        And Pipeline variable: template_name -> feature-extraction-template
        And Pipeline variable: pipeline_name -> afterimage
        And the pipeline variables are saved to config
        When the pipeline starts
        Then the pipeline should not fail
        And the components are saved


    @detection
    Scenario Outline: Model Detection
        Given Pipeline variable: prev_pipeline -> afterimage_test/afterimage
        And Pipeline variable: reader_type -> CSVReader
        And Pipeline variable: gen_summary -> true
        And Pipeline variable: model_name -> <model>
        And Pipeline variable: model_params -> <model_params>
        And Pipeline variable: template_name -> scaled-detection-template
        And Pipeline variable: pipeline_name -> <model_name>
        And the pipeline variables are saved to config
        When the pipeline starts
        Then the pipeline should not fail
        And the components are saved
        And the results are written
        Examples:
            | model               | model_name | model_params                          |
            | torch_model.AE      | AE         | experiments/model_config/AE.yaml      |
            | torch_model.GOAD    | GOAD       | experiments/model_config/GOAD.yaml    |
            | torch_model.ICL     | ICL        | experiments/model_config/ICL.yaml     |
            | torch_model.Kitsune | Kitsune    | experiments/model_config/Kitsune.yaml |
            | torch_model.SLAD    | SLAD       | experiments/model_config/SLAD.yaml    |
            | torch_model.VAE     | VAE        | experiments/model_config/VAE.yaml     |

    @tuning
    Scenario Outline: Test Model Tuning
        Given Pipeline variable: run_identifier -> afterimage_test_tuning
        And Pipeline variable: prev_pipeline -> afterimage_test/afterimage
        And Pipeline variable: reader_type -> CSVReader
        And Pipeline variable: gen_summary -> true
        And Pipeline variable: model_name -> <model>
        And Pipeline variable: model_params -> <model_params>
        And Pipeline variable: template_name -> scaled-detection-template
        And Pipeline variable: pipeline_name -> <model_name>
        When the pipeline starts with optuna tuning enabled with 20 trials
        Then the pipeline should not fail
        And the optuna database is created
        Examples:
            | model               | model_name | model_params                          |
            | torch_model.AE      | AE         | experiments/model_config/AE.yaml      |
            | torch_model.GOAD    | GOAD       | experiments/model_config/GOAD.yaml    |
            | torch_model.ICL     | ICL        | experiments/model_config/ICL.yaml     |
            | torch_model.Kitsune | Kitsune    | experiments/model_config/Kitsune.yaml |
            | torch_model.SLAD    | SLAD       | experiments/model_config/SLAD.yaml    |
            | torch_model.VAE     | VAE        | experiments/model_config/VAE.yaml     |