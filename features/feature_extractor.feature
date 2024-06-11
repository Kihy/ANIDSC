Feature: Feature Extraction
    To generate features for classification,
            I want to extract features based on pcap files

    Scenario Outline: Pcap features
        Given we want to extract features from <filename> in <dataset> dataset
        When The feature extractor is AfterImageGraph with <state> information
        Then The process should not report error

        Examples: Pcap files
            | dataset   | filename           | state |
            | Test_Data | sample_lenovo_bulb | None  |