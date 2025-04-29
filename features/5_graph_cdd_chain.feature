@core
Feature: NIDS Chain with Graph Representation and concept drift detection

    Scenario Outline: Process packets from an offline csv reader
        Given a csv file initialized with dataset "test_data", file "benign_lenovo_bulb", and feature extractor AfterImageGraph
            And a new cdd graph pipeline with <model> and <cdd_framework>
        When the PacketReader starts
        Then the pipeline should not fail
            And the components are saved
            And the results are logged
    Examples:
        | model   | cdd_framework |
        | ICL     | ARCUS         |
        | AE      | ARCUS         |
        | Kitsune | ARCUS         |
        | GOAD    | ARCUS         |
        | SLAD    | ARCUS         |
        | VAE     | ARCUS         |
        | ICL     | DriftSense    |
        | AE      | DriftSense    |
        | Kitsune | DriftSense    |
        | GOAD    | DriftSense    |
        | SLAD    | DriftSense    |
        | VAE     | DriftSense    |

    Scenario Outline: Process packets from an offline csv reader
        Given a csv file initialized with dataset "test_data", file "<file>", and feature extractor AfterImageGraph
            And a loaded cdd graph pipeline with <model> and <cdd_framework>
        When the PacketReader starts
        Then the pipeline should not fail
            And the components are saved
            And the results are logged
    Examples:
        | model   | file                        | cdd_framework |
        | ICL     | malicious_Service_Detection | ARCUS         |
        | AE      | malicious_Service_Detection | ARCUS         |
        | Kitsune | malicious_Service_Detection | ARCUS         |
        | GOAD    | malicious_Service_Detection | ARCUS         |
        | SLAD    | malicious_Service_Detection | ARCUS         |
        | VAE     | malicious_Service_Detection | ARCUS         |
        | ICL     | malicious_ACK_Flooding      | ARCUS         |
        | AE      | malicious_ACK_Flooding      | ARCUS         |
        | Kitsune | malicious_ACK_Flooding      | ARCUS         |
        | GOAD    | malicious_ACK_Flooding      | ARCUS         |
        | SLAD    | malicious_ACK_Flooding      | ARCUS         |
        | VAE     | malicious_ACK_Flooding      | ARCUS         |
        | ICL     | malicious_Port_Scanning     | ARCUS         |
        | AE      | malicious_Port_Scanning     | ARCUS         |
        | Kitsune | malicious_Port_Scanning     | ARCUS         |
        | GOAD    | malicious_Port_Scanning     | ARCUS         |
        | SLAD    | malicious_Port_Scanning     | ARCUS         |
        | VAE     | malicious_Port_Scanning     | ARCUS         |
        | ICL     | malicious_Service_Detection | DriftSense    |
        | AE      | malicious_Service_Detection | DriftSense    |
        | Kitsune | malicious_Service_Detection | DriftSense    |
        | GOAD    | malicious_Service_Detection | DriftSense    |
        | SLAD    | malicious_Service_Detection | DriftSense    |
        | VAE     | malicious_Service_Detection | DriftSense    |
        | ICL     | malicious_ACK_Flooding      | DriftSense    |
        | AE      | malicious_ACK_Flooding      | DriftSense    |
        | Kitsune | malicious_ACK_Flooding      | DriftSense    |
        | GOAD    | malicious_ACK_Flooding      | DriftSense    |
        | SLAD    | malicious_ACK_Flooding      | DriftSense    |
        | VAE     | malicious_ACK_Flooding      | DriftSense    |
        | ICL     | malicious_Port_Scanning     | DriftSense    |
        | AE      | malicious_Port_Scanning     | DriftSense    |
        | Kitsune | malicious_Port_Scanning     | DriftSense    |
        | GOAD    | malicious_Port_Scanning     | DriftSense    |
        | SLAD    | malicious_Port_Scanning     | DriftSense    |
        | VAE     | malicious_Port_Scanning     | DriftSense    |

