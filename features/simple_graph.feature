Feature: Feature extraction with NetworkAccessExtractor
    Scenario: Empty folders
        Given test_data NetworkAccessGraphExtractor folder is empty

    Scenario Outline: Feature extraction
        Given Dataset: test_data and File: <file>
        And Meta Extractor: ProtocolMetaExtractor and Feature Extractor: NetworkAccessGraphExtractor
        And a <state> feature_extraction pipeline
        When the pipeline starts
        Then the pipeline should not fail
        And the components are saved
        Examples:
            | state  | file                        |
            | new    | benign_lenovo_bulb          |
            | loaded | malicious_ACK_Flooding      |
            | loaded | malicious_Service_Detection |
            | loaded | malicious_Port_Scanning     |


    Scenario Outline: Check concept drift from simple features
        Given Dataset: test_data and File: <file>
        And Feature Extracted by ProtocolMetaExtractor and NetworkAccessGraphExtractor
        And Node Encoder: <node_encoder>
        And Model: <model>
        And a <state> graph_feature_detection pipeline
        When the pipeline starts
        Then the pipeline should not fail
        And the components are saved
        Examples:
            | node_encoder   | model               | state  | file                        |
            | GCNNodeEncoder | torch_model.AE      | new    | benign_lenovo_bulb          |
            | GCNNodeEncoder | torch_model.AE      | loaded | malicious_ACK_Flooding      |
            | GCNNodeEncoder | torch_model.AE      | loaded | malicious_Port_Scanning     |
            | GCNNodeEncoder | torch_model.AE      | loaded | malicious_Service_Detection |
            | GCNNodeEncoder | torch_model.GOAD    | new    | benign_lenovo_bulb          |
            | GCNNodeEncoder | torch_model.GOAD    | loaded | malicious_ACK_Flooding      |
            | GCNNodeEncoder | torch_model.GOAD    | loaded | malicious_Port_Scanning     |
            | GCNNodeEncoder | torch_model.GOAD    | loaded | malicious_Service_Detection |
            | GCNNodeEncoder | torch_model.ICL     | new    | benign_lenovo_bulb          |
            | GCNNodeEncoder | torch_model.ICL     | loaded | malicious_ACK_Flooding      |
            | GCNNodeEncoder | torch_model.ICL     | loaded | malicious_Port_Scanning     |
            | GCNNodeEncoder | torch_model.ICL     | loaded | malicious_Service_Detection |
            | GCNNodeEncoder | torch_model.Kitsune | new    | benign_lenovo_bulb          |
            | GCNNodeEncoder | torch_model.Kitsune | loaded | malicious_ACK_Flooding      |
            | GCNNodeEncoder | torch_model.Kitsune | loaded | malicious_Port_Scanning     |
            | GCNNodeEncoder | torch_model.Kitsune | loaded | malicious_Service_Detection |
            | GCNNodeEncoder | torch_model.SLAD    | new    | benign_lenovo_bulb          |
            | GCNNodeEncoder | torch_model.SLAD    | loaded | malicious_ACK_Flooding      |
            | GCNNodeEncoder | torch_model.SLAD    | loaded | malicious_Port_Scanning     |
            | GCNNodeEncoder | torch_model.SLAD    | loaded | malicious_Service_Detection |
            | GCNNodeEncoder | torch_model.VAE     | new    | benign_lenovo_bulb          |
            | GCNNodeEncoder | torch_model.VAE     | loaded | malicious_ACK_Flooding      |
            | GCNNodeEncoder | torch_model.VAE     | loaded | malicious_Port_Scanning     |
            | GCNNodeEncoder | torch_model.VAE     | loaded | malicious_Service_Detection |


