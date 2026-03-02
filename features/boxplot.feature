Feature: Detection with Trivial Boxplot Features
    Background:
        Given The test_data file iterator
        And Pipeline variable: run_identifier -> box_plot_test

    Scenario: Empty folders
        Given folders in test_data with run identifier are empty

    Scenario: Meta extraction
        Given Pipeline variable: reader_type -> PacketReader
        And Pipeline variable: meta_extractor -> ProtocolMetaExtractor
        And Pipeline variable: template_name -> metadata-extraction-template
        And Pipeline variable: pipeline_name -> protocol-meta-extraction
        And the pipeline variables are saved to config
        When the pipeline starts
        Then the pipeline should not fail
        And the components are saved


    Scenario: Feature extraction
        Given Pipeline variable: reader_type -> CSVReader
        And Pipeline variable: prev_pipeline -> box_plot_test/protocol-meta-extraction
        And Pipeline variable: feature_extractor -> FrequencyExtractor
        And Pipeline variable: template_name -> feature-extraction-template
        And Pipeline variable: pipeline_name -> frequency
        And the pipeline variables are saved to config
        When the pipeline starts
        Then the pipeline should not fail
        And the components are saved


    Scenario: BoxPlot Detection
        Given Pipeline variable: prev_pipeline -> box_plot_test/frequency
        And Pipeline variable: reader_type -> CSVReader
        And Pipeline variable: gen_summary -> true
        And Pipeline variable: model_name -> trad_model.MedianDetector
        And Pipeline variable: model_params -> {}
        And Pipeline variable: template_name -> basic-detection-template
        And Pipeline variable: pipeline_name -> boxplot
        And the pipeline variables are saved to config
        When the pipeline starts
        Then the pipeline should not fail
        And the components are saved
        And the results are written

        