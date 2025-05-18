Feature: Basic chaining from CSV files
    Scenario Outline: Process packets from an offline csv reader
        Given a <state> basic pipeline with input from csv file initialized with dataset test_data, file <file>, feature extractor <fe_name> and model <model>
        When the pipeline starts
        Then the pipeline should not fail
        And the components are saved
        Examples:
            | state  | model               | file                        | fe_name            |
            | new    | BoxPlot             | benign_lenovo_bulb          | FrequencyExtractor |
            | loaded | BoxPlot             | malicious_Service_Detection | FrequencyExtractor |
            | loaded | BoxPlot             | malicious_Port_Scanning     | FrequencyExtractor |
            | loaded | BoxPlot             | malicious_ACK_Flooding      | FrequencyExtractor |
            | new    | torch_model.AE      | benign_lenovo_bulb          | AfterImage         |
            | new    | torch_model.ICL     | benign_lenovo_bulb          | AfterImage         |
            | new    | torch_model.Kitsune | benign_lenovo_bulb          | AfterImage         |
            | new    | torch_model.GOAD    | benign_lenovo_bulb          | AfterImage         |
            | new    | torch_model.SLAD    | benign_lenovo_bulb          | AfterImage         |
            | new    | torch_model.VAE     | benign_lenovo_bulb          | AfterImage         |
            | loaded | torch_model.ICL     | malicious_Service_Detection | AfterImage         |
            | loaded | torch_model.AE      | malicious_Service_Detection | AfterImage         |
            | loaded | torch_model.GOAD    | malicious_Service_Detection | AfterImage         |
            | loaded | torch_model.SLAD    | malicious_Service_Detection | AfterImage         |
            | loaded | torch_model.VAE     | malicious_Service_Detection | AfterImage         |
            | loaded | torch_model.Kitsune | malicious_Service_Detection | AfterImage         |
            | loaded | torch_model.ICL     | malicious_Port_Scanning     | AfterImage         |
            | loaded | torch_model.AE      | malicious_Port_Scanning     | AfterImage         |
            | loaded | torch_model.GOAD    | malicious_Port_Scanning     | AfterImage         |
            | loaded | torch_model.SLAD    | malicious_Port_Scanning     | AfterImage         |
            | loaded | torch_model.VAE     | malicious_Port_Scanning     | AfterImage         |
            | loaded | torch_model.Kitsune | malicious_Port_Scanning     | AfterImage         |
            | loaded | torch_model.ICL     | malicious_ACK_Flooding      | AfterImage         |
            | loaded | torch_model.AE      | malicious_ACK_Flooding      | AfterImage         |
            | loaded | torch_model.GOAD    | malicious_ACK_Flooding      | AfterImage         |
            | loaded | torch_model.SLAD    | malicious_ACK_Flooding      | AfterImage         |
            | loaded | torch_model.VAE     | malicious_ACK_Flooding      | AfterImage         |
            | loaded | torch_model.Kitsune | malicious_ACK_Flooding      | AfterImage         |



