Feature: Summarize results from experiments
    Scenario:
        Given a Summarizer initialized with dataset test_data and feature extractor AfterImage
        When we plot the results
        Then the plots are plotted
