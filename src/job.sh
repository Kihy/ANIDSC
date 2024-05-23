#!/bin/bash
# python evaluate_models.py 0 --graph_based --embedding_dist gaussian --GNN GATNodeEncoder
# python evaluate_models.py 1 --graph_based --embedding_dist gaussian --GNN GATNodeEncoder
# python evaluate_models.py 2 --graph_based --embedding_dist gaussian --GNN GATNodeEncoder
# python evaluate_models.py 3 --graph_based --embedding_dist gaussian --GNN GATNodeEncoder
# python evaluate_models.py 5 --graph_based --embedding_dist gaussian --GNN GATNodeEncoder
# python evaluate_models.py 6 --graph_based
# python evaluate_models.py 4 --graph_based 


# synthetic data 
# python evaluate_models.py 0 --graph_based --embedding_dist gaussian --GNN LinearNodeEncoder --dataset_name FakeGraphData --fe_name SyntheticFeatureExtractor --file_name benign/feature_correlation_test
# python evaluate_models.py 1 --graph_based --embedding_dist gaussian --GNN LinearNodeEncoder --dataset_name FakeGraphData --fe_name SyntheticFeatureExtractor --file_name benign/feature_correlation_test
# python evaluate_models.py 2 --graph_based --embedding_dist gaussian --GNN LinearNodeEncoder --dataset_name FakeGraphData --fe_name SyntheticFeatureExtractor --file_name benign/feature_correlation_test
# python evaluate_models.py 3 --graph_based --embedding_dist gaussian --GNN LinearNodeEncoder --dataset_name FakeGraphData --fe_name SyntheticFeatureExtractor --file_name benign/feature_correlation_test
# python evaluate_models.py 5 --graph_based --embedding_dist gaussian --GNN LinearNodeEncoder --dataset_name FakeGraphData --fe_name SyntheticFeatureExtractor --file_name benign/feature_correlation_test

# CIC-IDS dataset
python evaluate_models.py 0 --graph_based --embedding_dist gaussian --GNN GATNodeEncoder --dataset_name CIC_IDS_2017 --fe_name AfterImageGraph --file_name Monday-WorkingHours Tuesday-WorkingHours Wednesday-WorkingHours Thursday-WorkingHours Friday-WorkingHours