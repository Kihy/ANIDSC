Feature: Basic chaining of components

    Scenario: Process packets with boxplot
        Given a PacketReader initialized with dataset "../datasets/Test_Data" and file "benign_lenovo_bulb"
            And a feature extraction pipeline with frequency analysis
            And the output folder is empty
            And a basic boxplot model
        When the PacketReader starts
        Then the pipeline should not fail
            And the components are saved
            And the results are logged

    Scenario Outline: Process packets from an offline pcap reader
        Given a PacketReader initialized with dataset "../datasets/Test_Data" and <file>
            And a feature extraction pipeline with AfterImage
        When the PacketReader starts
        Then the pipeline should not fail
            And the components are saved
    Examples:
        | file                        |
        | benign_lenovo_bulb          |
        | malicious_Port_Scanning     |
        | malicious_Service_Detection |

    Scenario: Process packets directly from an offline pcap reader
        Given a PacketReader initialized with dataset "../datasets/Test_Data" and file "benign_lenovo_bulb"
            And a feature extraction pipeline with AfterImage
            And a basic NIDS model with AE
        When the PacketReader starts
        Then the pipeline should not fail
            And the components are saved
            And the results are logged



    Scenario Outline: Process packets from an offline csv reader
        Given a csv file initialized with dataset "../datasets/Test_Data", file "benign_lenovo_bulb", and feature extractor "AfterImage"
            And a basic NIDS model with <model>
        When the PacketReader starts
        Then the pipeline should not fail
            And the components are saved
            And the results are logged
    Examples:
        | model  |
        | ICL    |
        | AE     |
        | KitNET |
        | GOAD   |
        | SLAD   |
        | VAE    |

    Scenario Outline: Process packets from an offline csv reader
        Given a csv file initialized with dataset "../datasets/Test_Data", file "malicious_Port_Scanning", and feature extractor "AfterImage"
            And a basic NIDS model with <model>
        When the PacketReader starts
        Then the pipeline should not fail
            And the components are saved
            And the results are logged
    Examples:
        | model  |
        | ICL    |
        | AE     |
        | KitNET |
        | GOAD   |
        | SLAD   |
        | VAE    |

    Scenario Outline: Process packets from an offline csv reader
        Given a csv file initialized with dataset "../datasets/Test_Data", file "malicious_Service_Detection", and feature extractor "AfterImage"
            And a basic NIDS model with <model>
        When the PacketReader starts
        Then the pipeline should not fail
            And the components are saved
            And the results are logged
    Examples:
        | model  |
        | ICL    |
        | AE     |
        | KitNET |
        | GOAD   |
        | SLAD   |
        | VAE    |
