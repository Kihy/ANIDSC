@core
Feature: Evaluate results of chains
    Scenario: Draw detection rate and anomaly scores of all models
        Given a Summarizer initialized with dataset "datasets/Test_Data" and feature extractor AfterImage and files:
            | file                        |
            | benign_lenovo_bulb          |
            | malicious_ACK_Flooding      |
            | malicious_Port_Scanning     |
            | malicious_Service_Detection |
        When we plot the results
        Then the pipeline should not fail

    # Scenario: Draw detection rate and anomaly scores of all models
    #     Given a Summarizer initialized with dataset "../datasets/Test_Data" and feature extractor AfterImageGraph and files:
    #         | file                        |
    #         | benign_lenovo_bulb          |
    #         | malicious_Port_Scanning     |
    #         | malicious_Service_Detection |
    #         | malicious_ACK_Flooding      |
    #     When we plot the results
    #     Then the pipeline should not fail