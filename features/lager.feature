@baseline
Feature: Detection with AfterImage Features
    Background:
        Given The test_data file iterator
        And Pipeline variable: run_identifier -> lager_test


    Scenario: Empty folders
        Given folders in test_data with run identifier are empty

    @meta-extraction
    Scenario: Meta extraction
        Given Pipeline variable: reader_type -> PacketReader
        And Pipeline variable: meta_extractor -> ProtocolMetaExtractor
        And Pipeline variable: template_name -> metadata-extraction-template
        And Pipeline variable: pipeline_name -> protocol-meta-extraction
        And the pipeline variables are saved to config
        When the pipeline starts
        Then the pipeline should not fail
        And the components are saved

    @feature-extraction
    Scenario: Feature extraction
        Given Pipeline variable: reader_type -> CSVReader
        And Pipeline variable: prev_pipeline -> lager_test/protocol-meta-extraction
        And Pipeline variable: feature_extractor -> AfterImageGraph
        And Pipeline variable: fe_attr -> {"protocol_map": {"TCP": 0, "UDP": 1, "ICMP": 2, "ARP": 3, "Other": 4}}
        And Pipeline variable: template_name -> feature-extraction-template
        And Pipeline variable: pipeline_name -> afterimage-graph
        And the pipeline variables are saved to config
        When the pipeline starts
        Then the pipeline should not fail
        And the components are saved


    @detection
    Scenario Outline: Model Detection
        Given Pipeline variable: prev_pipeline -> lager_test/afterimage-graph
        And Pipeline variable: reader_type -> CSVReader
        And Pipeline variable: gen_summary -> true
        And Pipeline variable: model_name -> <model>
        And Pipeline variable: model_params -> <model_params>
        And Pipeline variable: template_name -> lager-template
        And Pipeline variable: embedder_name -> <embedder>
        And Pipeline variable: embedder_params -> <embedder_params>
        And Pipeline variable: protocol_map -> {"TCP": 0, "UDP": 1, "ICMP": 2, "ARP": 3, "Other": 4}
        And Pipeline variable: graph_rep -> HomoGraphRepresentation
        And Pipeline variable: pipeline_name -> <model_name>
        And the pipeline variables are saved to config
        When the pipeline starts
        Then the pipeline should not fail
        And the components are saved
        And the results are written
        Examples:
            | model               | model_name | model_params                            | embedder    | embedder_params                        |
            | torch_model.AE      | AE         | experiments/configs/models/AE.yaml      | GCNEmbedder | experiments/configs/embedders/GNN.yaml |
            | torch_model.GOAD    | GOAD       | experiments/configs/models/GOAD.yaml    | GCNEmbedder | experiments/configs/embedders/GNN.yaml |
            | torch_model.ICL     | ICL        | experiments/configs/models/ICL.yaml     | GCNEmbedder | experiments/configs/embedders/GNN.yaml |
            | torch_model.Kitsune | Kitsune    | experiments/configs/models/Kitsune.yaml | GCNEmbedder | experiments/configs/embedders/GNN.yaml |
            | torch_model.SLAD    | SLAD       | experiments/configs/models/SLAD.yaml    | GCNEmbedder | experiments/configs/embedders/GNN.yaml |
            | torch_model.VAE     | VAE        | experiments/configs/models/VAE.yaml     | GCNEmbedder | experiments/configs/embedders/GNN.yaml |
            | torch_model.AE      | AE         | experiments/configs/models/AE.yaml      | GATEmbedder | experiments/configs/embedders/GNN.yaml |
            | torch_model.GOAD    | GOAD       | experiments/configs/models/GOAD.yaml    | GATEmbedder | experiments/configs/embedders/GNN.yaml |
            | torch_model.ICL     | ICL        | experiments/configs/models/ICL.yaml     | GATEmbedder | experiments/configs/embedders/GNN.yaml |
            | torch_model.Kitsune | Kitsune    | experiments/configs/models/Kitsune.yaml | GATEmbedder | experiments/configs/embedders/GNN.yaml |
            | torch_model.SLAD    | SLAD       | experiments/configs/models/SLAD.yaml    | GATEmbedder | experiments/configs/embedders/GNN.yaml |
            | torch_model.VAE     | VAE        | experiments/configs/models/VAE.yaml     | GATEmbedder | experiments/configs/embedders/GNN.yaml |

    @tuning
    Scenario Outline: Test Model Tuning. 
        Given Pipeline variable: prev_pipeline -> lager_test/afterimage-graph
        And Pipeline variable: run_identifier -> lager_test_tuning
        And Pipeline variable: reader_type -> CSVReader
        And Pipeline variable: gen_summary -> true
        And Pipeline variable: model_name -> <model>
        And Pipeline variable: model_params -> <model_params>
        And Pipeline variable: template_name -> lager-template
        And Pipeline variable: embedder_name -> <embedder>
        And Pipeline variable: embedder_params -> <embedder_params>
        And Pipeline variable: protocol_map -> {"TCP": 0, "UDP": 1, "ICMP": 2, "ARP": 3, "Other": 4}
        And Pipeline variable: graph_rep -> HomoGraphRepresentation
        And Pipeline variable: pipeline_name -> <model_name>
        When the pipeline starts with optuna tuning enabled with 20 trials
        Then the pipeline should not fail
        And the optuna database is created
        Examples:
            | model               | model_name | model_params                            | embedder    | embedder_params                        |
            | torch_model.AE      | AE         | experiments/configs/models/AE.yaml      | GCNEmbedder | experiments/configs/embedders/GNN.yaml |
            | torch_model.GOAD    | GOAD       | experiments/configs/models/GOAD.yaml    | GCNEmbedder | experiments/configs/embedders/GNN.yaml |
            | torch_model.ICL     | ICL        | experiments/configs/models/ICL.yaml     | GCNEmbedder | experiments/configs/embedders/GNN.yaml |
            | torch_model.Kitsune | Kitsune    | experiments/configs/models/Kitsune.yaml | GCNEmbedder | experiments/configs/embedders/GNN.yaml |
            | torch_model.SLAD    | SLAD       | experiments/configs/models/SLAD.yaml    | GCNEmbedder | experiments/configs/embedders/GNN.yaml |
            | torch_model.VAE     | VAE        | experiments/configs/models/VAE.yaml     | GCNEmbedder | experiments/configs/embedders/GNN.yaml |
            | torch_model.AE      | AE         | experiments/configs/models/AE.yaml      | GATEmbedder | experiments/configs/embedders/GNN.yaml |
            | torch_model.GOAD    | GOAD       | experiments/configs/models/GOAD.yaml    | GATEmbedder | experiments/configs/embedders/GNN.yaml |
            | torch_model.ICL     | ICL        | experiments/configs/models/ICL.yaml     | GATEmbedder | experiments/configs/embedders/GNN.yaml |
            | torch_model.Kitsune | Kitsune    | experiments/configs/models/Kitsune.yaml | GATEmbedder | experiments/configs/embedders/GNN.yaml |
            | torch_model.SLAD    | SLAD       | experiments/configs/models/SLAD.yaml    | GATEmbedder | experiments/configs/embedders/GNN.yaml |
            | torch_model.VAE     | VAE        | experiments/configs/models/VAE.yaml     | GATEmbedder | experiments/configs/embedders/GNN.yaml |