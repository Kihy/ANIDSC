Feature: NIDS Chain with Graph Representation

    Scenario Outline: Process packets from an offline pcap reader
        Given a PacketReader initialized with dataset "../datasets/Test_Data" and <file>
            And a feature extraction pipeline with AfterImageGraph
        When the PacketReader starts
        Then the pipeline should not fail
            And the components are saved
    Examples:
        | file                        |
        | benign_lenovo_bulb          |
        | malicious_Port_Scanning     |
        | malicious_Service_Detection |

    Scenario Outline: Process packets from an offline csv reader
        Given a csv file initialized with dataset "../datasets/Test_Data", file "benign_lenovo_bulb", and feature extractor AfterImageGraph
            And a new graph pipeline with <model>
        When the PacketReader starts
        Then the pipeline should not fail
            And the components are saved
            And the results are logged
    Examples:
        | model  |
        | ICL    |
        | AE     |
        | Kitsune |
        | GOAD   |
        | SLAD   |
        | VAE    |

    Scenario Outline: Process packets from an offline csv reader
        Given a csv file initialized with dataset "../datasets/Test_Data", file "malicious_Port_Scanning", and feature extractor AfterImageGraph
            And a loaded graph pipeline with <model>
        When the PacketReader starts
        Then the pipeline should not fail
            And the components are saved
            And the results are logged
    Examples:
        | model  |
        | ICL    |
        | AE     |
        | Kitsune |
        | GOAD   |
        | SLAD   |
        | VAE    |

    Scenario Outline: Process packets from an offline csv reader
        Given a csv file initialized with dataset "../datasets/Test_Data", file "malicious_Service_Detection", and feature extractor AfterImageGraph
            And a loaded graph pipeline with <model>
        When the PacketReader starts
        Then the pipeline should not fail
            And the components are saved
            And the results are logged
    Examples:
        | model  |
        | ICL    |
        | AE     |
        | Kitsune |
        | GOAD   |
        | SLAD   |
        | VAE    |
