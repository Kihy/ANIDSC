@core
Feature: Basic chaining from CSV files
    Scenario Outline: Process packets from an offline csv reader
        Given a csv file initialized with dataset "../datasets/Test_Data", file "benign_lenovo_bulb", and feature extractor AfterImage
            And a new basic pipeline with <model>
        When the PacketReader starts
        Then the pipeline should not fail
            And the components are saved
            And the results are logged
    Examples:
        | model   |
        | AE      |
        | ICL     |
        | Kitsune |
        | GOAD    |
        | SLAD    |
        | VAE     |

    Scenario Outline: Process packets from an offline csv reader
        Given a csv file initialized with dataset "../datasets/Test_Data", file "<file>", and feature extractor AfterImage
            And a loaded basic pipeline with <model>
        When the PacketReader starts
        Then the pipeline should not fail
            And the components are saved
            And the results are logged
    Examples:
        | model   | file                        |
        | ICL     | malicious_Service_Detection |
        | AE      | malicious_Service_Detection |
        | GOAD    | malicious_Service_Detection |
        | SLAD    | malicious_Service_Detection |
        | VAE     | malicious_Service_Detection |
        | Kitsune | malicious_Service_Detection |
        | ICL     | malicious_Port_Scanning     |
        | AE      | malicious_Port_Scanning     |
        | GOAD    | malicious_Port_Scanning     |
        | SLAD    | malicious_Port_Scanning     |
        | VAE     | malicious_Port_Scanning     |
        | Kitsune | malicious_Port_Scanning     |
        | ICL     | malicious_ACK_Flooding      |
        | AE      | malicious_ACK_Flooding      |
        | GOAD    | malicious_ACK_Flooding      |
        | SLAD    | malicious_ACK_Flooding      |
        | VAE     | malicious_ACK_Flooding      |
        | Kitsune | malicious_ACK_Flooding      |



    