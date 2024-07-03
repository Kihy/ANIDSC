Feature: Chaining of components with Concept Drift Detection

    Scenario Outline: Process packets from an offline pcap reader
        Given a PacketReader initialized with dataset "../datasets/Test_Data" and file "sample_lenovo_bulb"
        And a feature extraction pipeline with AfterImage
        And the output folder is empty
        And a CDD NIDS model with <model>
        When the PacketReader starts
        Then the pipeline should not fail
        And the components are saved 
        And the results are logged
        Examples:
            | model |
            | ARCUS |
            | GOAD  |
            | SLAD  |
            | KitNET|
            | ICL   |
            | VAE   |
            | AE    |

    Scenario Outline: Process packets from an offline csv reader
        Given a csv file initialized with dataset "../datasets/Test_Data", file "sample_lenovo_bulb", and feature extractor "AfterImage"
        And a CDD NIDS model with <model>
        When the PacketReader starts
        Then the pipeline should not fail
        And the components are saved 
        And the results are logged
        Examples:
            | model |
            | ARCUS |
            | GOAD  |
            | SLAD  |
            | KitNET|
            | ICL   |
            | VAE   |
            | AE    |