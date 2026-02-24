Feature: Detection with multilayer models
    Scenario: Empty folders
        Given folders in test_data with test_run are empty

    Scenario: Meta extraction
        Given The test_data file iterator
        And Pipeline variable: reader_type -> PacketReader
        And Pipeline variable: meta_extractor -> ProtocolMetaExtractor
        And a meta_extraction pipeline
        When the pipeline starts
        Then the pipeline should not fail
        And the components are saved

    Scenario: Feature extraction
        Given The test_data file iterator
        And Pipeline variable: reader_type -> CSVReader
        And Pipeline variable: prev_pipeline -> meta_extraction/test_run
        And a graph_feature_extraction pipeline
        When the pipeline starts
        Then the pipeline should not fail
        And the components are saved

    # Scenario Outline: Feature-based Detection
    #     Given The test_data file iterator
    #     And Pipeline variable: prev_pipeline -> graph_feature_extraction-test_run
    #     And Pipeline variable: reader_type -> JsonGraphReader
    #     And Pipeline variable: graph_rep -> Barebone
    #     And Pipeline variable: model_name -> <model>
    #     And a multilayer_graph_feature pipeline
    #     When the pipeline starts
    #     Then the pipeline should not fail
    #     And the components are saved
    #     And the results are written
    #     Examples:
    #         | model                  |
    #         | torch_model.AE         |
    #         | torch_model.GOAD       |
    #         | torch_model.ICL        |
    #         | torch_model.SLAD       |
    #         | torch_model.VAE        |
    #         | sklearn_model.SGDOCSVM |

    Scenario Outline: Reconstruction Detection
        Given The test_data file iterator
        And Pipeline variable: prev_pipeline -> graph_feature_extraction/test_run
        And Pipeline variable: reader_type -> JsonGraphReader
        And Pipeline variable: graph_rep -> Barebone
        And Pipeline variable: model_name -> <model>
        And Pipeline variable: model_params -> {"latent_dim":16, "hidden_dim":32, "num_layers":2}
        And a multilayer_graph_recon pipeline
        When the pipeline starts
        Then the pipeline should not fail
        And the components are saved
        And the results are written
        Examples:
            | model                           |
            | torch_graph_model.StructuralGAE |
            | torch_graph_model.GINGAE        |
    
    Scenario: Test Model Tuning
        Given The test_data file iterator
        And Pipeline variable: prev_pipeline -> graph_feature_extraction/test_run
        And Pipeline variable: reader_type -> JsonGraphReader
        And Pipeline variable: graph_rep -> Barebone
        And Pipeline variable: model_name -> torch_graph_model.StructuralGAE
        And Hyperparameter Tuning variable: latent_dim -> {"type": "int", "low": 1,"high": 100}
        And Hyperparameter Tuning variable: hidden_dim -> {"type": "int", "low": 1,"high": 100}
        And Hyperparameter Tuning variable: n_trials -> 20
        And a multilayer_graph_recon pipeline
        When the pipeline starts with optuna tuning enabled
        Then the pipeline should not fail
        And the optuna database is created
