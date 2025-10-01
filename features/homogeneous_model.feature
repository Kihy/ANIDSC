Feature: Detection with homogeneous models
    # Scenario: Empty folders
    #     Given test_data NetworkAccessGraphExtractor folder is empty

    # Scenario Outline: Feature extraction
    #     Given Dataset: test_data and File: <file>
    #     And Meta Extractor: ProtocolMetaExtractor and Feature Extractor: NetworkAccessGraphExtractor
    #     And a <state> feature_extraction pipeline
    #     When the pipeline starts
    #     Then the pipeline should not fail
    #     And the components are saved
    #     Examples:
    #         | state  | file                        |
    #         | new    | benign_lenovo_bulb          |
    #         | loaded | malicious_ACK_Flooding      |
    #         | loaded | malicious_Service_Detection |
    #         | loaded | malicious_Port_Scanning     |


    Scenario Outline: Detection with homogeneous models, ICL and Kitsune does not work well with small dimension input
        Given Dataset: test_data and File: <file>
        And Feature Extracted by ProtocolMetaExtractor and NetworkAccessGraphExtractor
        And Model: <model>
        And a <state> homogeneous pipeline
        When the pipeline starts
        Then the pipeline should not fail
        And the components are saved
        Examples:
            | model               | state  | file                        |
            | torch_model.AE      | new    | benign_lenovo_bulb          |
            | torch_model.AE      | loaded | malicious_ACK_Flooding      |
            | torch_model.AE      | loaded | malicious_Port_Scanning     |
            | torch_model.AE      | loaded | malicious_Service_Detection |
            | torch_model.GOAD    | new    | benign_lenovo_bulb          |
            | torch_model.GOAD    | loaded | malicious_ACK_Flooding      |
            | torch_model.GOAD    | loaded | malicious_Port_Scanning     |
            | torch_model.GOAD    | loaded | malicious_Service_Detection |
            | torch_model.SLAD    | new    | benign_lenovo_bulb          |
            | torch_model.SLAD    | loaded | malicious_ACK_Flooding      |
            | torch_model.SLAD    | loaded | malicious_Port_Scanning     |
            | torch_model.SLAD    | loaded | malicious_Service_Detection |
            | torch_model.VAE     | new    | benign_lenovo_bulb          |
            | torch_model.VAE     | loaded | malicious_ACK_Flooding      |
            | torch_model.VAE     | loaded | malicious_Port_Scanning     |
            | torch_model.VAE     | loaded | malicious_Service_Detection |
            | MedianDetector      | new    | benign_lenovo_bulb          |
            | MedianDetector      | loaded | malicious_ACK_Flooding      |
            | MedianDetector      | loaded | malicious_Port_Scanning     |
            | MedianDetector      | loaded | malicious_Service_Detection |
           
    