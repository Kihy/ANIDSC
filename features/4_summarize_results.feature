Feature: Summarize results from experiments

    Scenario: Plot summarized results
        Given a Summarizer initialized with dataset:
            | dataset   |
            | test_data |
        And initialized with feature extractor:
            | feature_extractor |
            | AfterImage        |
            | AfterImageGraph   |
        When we plot the results
        Then the plots are plotted
