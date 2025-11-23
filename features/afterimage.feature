Feature: AfterImage
    Scenario: Empty folders
        Given test_data ProtocolMetaExtractor folder is empty
        Given test_data AfterImage folder is empty

    Scenario: Meta extraction
        Given The test_data file iterator
        And test_data ProtocolMetaExtractor folder is empty
        And Data Source: PacketReader
        And Meta Extractor: ProtocolMetaExtractor
        And a meta_extraction pipeline
        When the pipeline starts
        Then the pipeline should not fail
        And the components are saved

    Scenario: Feature extraction
        Given The test_data file iterator
        And folders that starts with test_data AfterImage are empty
        And Data Source: CSVReader
        And FE Name: ProtocolMetaExtractor
        And Feature Extractor: AfterImage
        And a feature_extraction pipeline
        When the pipeline starts
        Then the pipeline should not fail
        And the components are saved

    Scenario Outline: Model Detection from CSV Features
        Given The test_data file iterator
        And FE Name: AfterImage
        And Data Source: CSVReader
        And Model: <model>
        And a basic_detection pipeline
        When the pipeline starts
        Then the pipeline should not fail
        And the components are saved
        And the results are written
        Examples:
            | model               |
            | torch_model.AE      |
            | torch_model.GOAD    |
            | torch_model.ICL     |
            | torch_model.Kitsune |
            | torch_model.SLAD    |
            | torch_model.VAE     |


