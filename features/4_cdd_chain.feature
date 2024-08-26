@core
Feature: Chaining of components with Concept Drift Detection
    Scenario Outline: Process packets from an offline csv reader with ARCUS
        Given a csv file initialized with dataset "../datasets/Test_Data", file "benign_lenovo_bulb", and feature extractor AfterImage
            And a new CDD pipeline with <cdd_framework> over <model>
        When the PacketReader starts
        Then the pipeline should not fail
            And the components are saved
            And the results are logged
    Examples:
        | model   | cdd_framework |
        | Kitsune | ARCUS         |
        | GOAD    | ARCUS         |
        | SLAD    | ARCUS         |
        | ICL     | ARCUS         |
        | VAE     | ARCUS         |
        | AE      | ARCUS         |
        | Kitsune | DriftSense    |
        | GOAD    | DriftSense    |
        | SLAD    | DriftSense    |
        | ICL     | DriftSense    |
        | VAE     | DriftSense    |
        | AE      | DriftSense    |

    Scenario Outline: Process packets from an offline csv reader with saved ARCUS
        Given a csv file initialized with dataset "../datasets/Test_Data", file "<file>", and feature extractor AfterImage
            And a loaded CDD pipeline with <cdd_framework> over <model>
        When the PacketReader starts
        Then the pipeline should not fail
            And the components are saved
            And the results are logged
    Examples:
        | model   | cdd_framework | file                        |
        | Kitsune | ARCUS         | malicious_Service_Detection |
        | GOAD    | ARCUS         | malicious_Service_Detection |
        | SLAD    | ARCUS         | malicious_Service_Detection |
        | ICL     | ARCUS         | malicious_Service_Detection |
        | VAE     | ARCUS         | malicious_Service_Detection |
        | AE      | ARCUS         | malicious_Service_Detection |
        | Kitsune | DriftSense    | malicious_Service_Detection |
        | GOAD    | DriftSense    | malicious_Service_Detection |
        | SLAD    | DriftSense    | malicious_Service_Detection |
        | ICL     | DriftSense    | malicious_Service_Detection |
        | VAE     | DriftSense    | malicious_Service_Detection |
        | AE      | DriftSense    | malicious_Service_Detection |
        | Kitsune | ARCUS         | malicious_Port_Scanning     |
        | GOAD    | ARCUS         | malicious_Port_Scanning     |
        | SLAD    | ARCUS         | malicious_Port_Scanning     |
        | ICL     | ARCUS         | malicious_Port_Scanning     |
        | VAE     | ARCUS         | malicious_Port_Scanning     |
        | AE      | ARCUS         | malicious_Port_Scanning     |
        | Kitsune | DriftSense    | malicious_Port_Scanning     |
        | GOAD    | DriftSense    | malicious_Port_Scanning     |
        | SLAD    | DriftSense    | malicious_Port_Scanning     |
        | ICL     | DriftSense    | malicious_Port_Scanning     |
        | VAE     | DriftSense    | malicious_Port_Scanning     |
        | AE      | DriftSense    | malicious_Port_Scanning     |
        | Kitsune | ARCUS         | malicious_ACK_Flooding      |
        | GOAD    | ARCUS         | malicious_ACK_Flooding      |
        | SLAD    | ARCUS         | malicious_ACK_Flooding      |
        | ICL     | ARCUS         | malicious_ACK_Flooding      |
        | VAE     | ARCUS         | malicious_ACK_Flooding      |
        | AE      | ARCUS         | malicious_ACK_Flooding      |
        | Kitsune | DriftSense    | malicious_ACK_Flooding      |
        | GOAD    | DriftSense    | malicious_ACK_Flooding      |
        | SLAD    | DriftSense    | malicious_ACK_Flooding      |
        | ICL     | DriftSense    | malicious_ACK_Flooding      |
        | VAE     | DriftSense    | malicious_ACK_Flooding      |
        | AE      | DriftSense    | malicious_ACK_Flooding      |

