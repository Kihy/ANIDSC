Feature: Offline Detection Pipeline
    A pipeline that reads input from pcap file and outputs scores

    Scenario Outline: Pcap features
        Given we want to extract features from <filename> in <dataset> dataset
        When The feature extractor is AfterImage with <state> information
        Then The process should not report error

        Examples: Pcap files
            | dataset   | filename           | state |
            | Test_Data | sample_lenovo_bulb | None  |