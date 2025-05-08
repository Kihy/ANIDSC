Feature: Run pipeline with graph features
    Scenario Outline: Process packets from an offline csv reader with LAGER
        Given a <state> graph pipeline with input from csv file initialized with dataset test_data, file <file>, feature extractor <fe_name>, model <model>, node encoder <node_encoder>
        When the pipeline starts
        Then the pipeline should not fail
        And the components are saved
        Examples:
            | state  | node_encoder   | model               | file                        | fe_name         |
            | new    | GCNNodeEncoder | torch_model.AE      | benign_lenovo_bulb          | AfterImageGraph |
            | new    | GCNNodeEncoder | torch_model.ICL     | benign_lenovo_bulb          | AfterImageGraph |
            | new    | GCNNodeEncoder | torch_model.Kitsune | benign_lenovo_bulb          | AfterImageGraph |
            | new    | GCNNodeEncoder | torch_model.GOAD    | benign_lenovo_bulb          | AfterImageGraph |
            | new    | GCNNodeEncoder | torch_model.SLAD    | benign_lenovo_bulb          | AfterImageGraph |
            | new    | GCNNodeEncoder | torch_model.VAE     | benign_lenovo_bulb          | AfterImageGraph |
            | loaded | GCNNodeEncoder | torch_model.ICL     | malicious_Service_Detection | AfterImageGraph |
            | loaded | GCNNodeEncoder | torch_model.AE      | malicious_Service_Detection | AfterImageGraph |
            | loaded | GCNNodeEncoder | torch_model.GOAD    | malicious_Service_Detection | AfterImageGraph |
            | loaded | GCNNodeEncoder | torch_model.SLAD    | malicious_Service_Detection | AfterImageGraph |
            | loaded | GCNNodeEncoder | torch_model.VAE     | malicious_Service_Detection | AfterImageGraph |
            | loaded | GCNNodeEncoder | torch_model.Kitsune | malicious_Service_Detection | AfterImageGraph |
            | loaded | GCNNodeEncoder | torch_model.ICL     | malicious_Port_Scanning     | AfterImageGraph |
            | loaded | GCNNodeEncoder | torch_model.AE      | malicious_Port_Scanning     | AfterImageGraph |
            | loaded | GCNNodeEncoder | torch_model.GOAD    | malicious_Port_Scanning     | AfterImageGraph |
            | loaded | GCNNodeEncoder | torch_model.SLAD    | malicious_Port_Scanning     | AfterImageGraph |
            | loaded | GCNNodeEncoder | torch_model.VAE     | malicious_Port_Scanning     | AfterImageGraph |
            | loaded | GCNNodeEncoder | torch_model.Kitsune | malicious_Port_Scanning     | AfterImageGraph |
            | loaded | GCNNodeEncoder | torch_model.ICL     | malicious_ACK_Flooding      | AfterImageGraph |
            | loaded | GCNNodeEncoder | torch_model.AE      | malicious_ACK_Flooding      | AfterImageGraph |
            | loaded | GCNNodeEncoder | torch_model.GOAD    | malicious_ACK_Flooding      | AfterImageGraph |
            | loaded | GCNNodeEncoder | torch_model.SLAD    | malicious_ACK_Flooding      | AfterImageGraph |
            | loaded | GCNNodeEncoder | torch_model.VAE     | malicious_ACK_Flooding      | AfterImageGraph |
            | loaded | GCNNodeEncoder | torch_model.Kitsune | malicious_ACK_Flooding      | AfterImageGraph |




