@core
Feature: NIDS Chain with Graph Representation
    Scenario Outline: Process packets from an offline csv reader
        Given a csv file initialized with dataset "test_data", file "benign_lenovo_bulb", and feature extractor AfterImageGraph
            And a new graph pipeline with <model>
        When the PacketReader starts
        Then the pipeline should not fail
            And the components are saved
            And the results are logged
    Examples:
        | model   |
        | ICL     |
        | AE      |
        | Kitsune |
        | GOAD    |
        | SLAD    |
        | VAE     |

    Scenario Outline: Process packets from an offline csv reader
        Given a csv file initialized with dataset "test_data", file "<file>", and feature extractor AfterImageGraph
            And a loaded graph pipeline with <model>
        When the PacketReader starts
        Then the pipeline should not fail
            And the components are saved
            And the results are logged
    Examples:
        | model   | file                        |
        | ICL     | malicious_Service_Detection |
        | AE      | malicious_Service_Detection |
        | Kitsune | malicious_Service_Detection |
        | GOAD    | malicious_Service_Detection |
        | SLAD    | malicious_Service_Detection |
        | VAE     | malicious_Service_Detection |
        | ICL     | malicious_ACK_Flooding      |
        | AE      | malicious_ACK_Flooding      |
        | Kitsune | malicious_ACK_Flooding      |
        | GOAD    | malicious_ACK_Flooding      |
        | SLAD    | malicious_ACK_Flooding      |
        | VAE     | malicious_ACK_Flooding      |
        | ICL     | malicious_Port_Scanning     |
        | AE      | malicious_Port_Scanning     |
        | Kitsune | malicious_Port_Scanning     |
        | GOAD    | malicious_Port_Scanning     |
        | SLAD    | malicious_Port_Scanning     |
        | VAE     | malicious_Port_Scanning     |

