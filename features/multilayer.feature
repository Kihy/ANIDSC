Feature: Detection with multilayer models
    Scenario: Empty folders
        Given folders that starts with test_data MultiLayerGraphExtractor are empty
        Given folders that starts with test_data ProtocolMetaExtractor are empty

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
        And Pipeline variable: fe_name -> ProtocolMetaExtractor-test_run
        And a graph_feature_extraction pipeline
        When the pipeline starts
        Then the pipeline should not fail
        And the components are saved

    Scenario Outline: Detection
        Given The test_data file iterator
        And Pipeline variable: fe_name -> MultiLayerGraphExtractor(1)-test_run
        And Pipeline variable: reader_type -> JsonGraphReader
        And Pipeline variable: graph_rep -> Barebone
        And Pipeline variable: model_name -> <model>
        And a multilayer-graph-feature pipeline
        When the pipeline starts
        Then the pipeline should not fail
        And the components are saved
        And the results are written
        Examples:
            | model                  |
            | torch_model.AE         |
            | torch_model.GOAD       |
            | torch_model.ICL        |
            | torch_model.SLAD       |
            | torch_model.VAE        |
            | sklearn_model.SGDOCSVM |

