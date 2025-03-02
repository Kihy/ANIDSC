@core
Feature: Basic chaining from CSV files
    Scenario Outline: Process packets from an offline csv reader
        Given a csv file initialized with dataset "datasets/Test_Data", file "benign_lenovo_bulb", and feature extractor AfterImage
        And a new basic pipeline with <model>
        When the PacketReader starts
        Then the pipeline should not fail
        And the components are saved
        And the results are logged
        Examples:
            | model                              |
            | torch_models.autoencoder.AE        |
            | torch_models.icl.ICL               |
            | torch_models.kitsune_torch.Kitsune |
            | torch_models.goad.GOAD             |
            | torch_models.slad.SLAD             |
            | torch_models.autoencoder.VAE       |

    Scenario Outline: Process packets from an offline csv reader
        Given a csv file initialized with dataset "datasets/Test_Data", file "<file>", and feature extractor AfterImage
        And a loaded basic pipeline with <model>
        When the PacketReader starts
        Then the pipeline should not fail
        And the components are saved
        And the results are logged
        Examples:
            | model                              | file                        |
            | torch_models.icl.ICL               | malicious_Service_Detection |
            | torch_models.autoencoder.AE        | malicious_Service_Detection |
            | torch_models.goad.GOAD             | malicious_Service_Detection |
            | torch_models.slad.SLAD             | malicious_Service_Detection |
            | torch_models.autoencoder.VAE       | malicious_Service_Detection |
            | torch_models.kitsune_torch.Kitsune | malicious_Service_Detection |
            | torch_models.icl.ICL               | malicious_Port_Scanning     |
            | torch_models.autoencoder.AE        | malicious_Port_Scanning     |
            | torch_models.goad.GOAD             | malicious_Port_Scanning     |
            | torch_models.slad.SLAD             | malicious_Port_Scanning     |
            | torch_models.autoencoder.VAE       | malicious_Port_Scanning     |
            | torch_models.kitsune_torch.Kitsune | malicious_Port_Scanning     |
            | torch_models.icl.ICL               | malicious_ACK_Flooding      |
            | torch_models.autoencoder.AE        | malicious_ACK_Flooding      |
            | torch_models.goad.GOAD             | malicious_ACK_Flooding      |
            | torch_models.slad.SLAD             | malicious_ACK_Flooding      |
            | torch_models.autoencoder.VAE       | malicious_ACK_Flooding      |
            | torch_models.kitsune_torch.Kitsune | malicious_ACK_Flooding      |




