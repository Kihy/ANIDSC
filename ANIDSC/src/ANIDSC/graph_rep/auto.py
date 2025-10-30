from collections import defaultdict, deque
import json
import random
import sys
from typing import Tuple


from ..graph_rep.base import GraphRepresentation
import numpy as np


import networkx as nx
from scipy.stats import ks_2samp




class Concept:
    """
    Implements the Reservoir Sampling algorithm to sample k items from a stream.
    """

    def __init__(self, idx, min_length, max_length):
        """
        Initializes the ReservoirSampler.

        Args:
            min_length (int): Minimum number of items before sampling is considered initialized
            max_length (int): The desired size of the sample (the reservoir).
        """
        if min_length <= 0 or max_length <= 0:
            raise ValueError("min_length and max_length must be positive integers")
        if min_length > max_length:
            raise ValueError("min_length cannot be greater than max_length")

        self.idx=idx
        self.min_length = min_length
        self.max_length = max_length
        self.reservoir = []
        self.count = 0  # Number of items processed so far

    def __repr__(self):
        return str(self.idx)


    def add_item(self, item):
        """
        Adds an item from the stream to the sampling process.

        Args:
            item: The item to be considered for inclusion in the reservoir.
        """
        self.count += 1

        if len(self.reservoir) < self.max_length:
            # Fill the reservoir with the first max_length items
            self.reservoir.append(item)
        else:
            # For subsequent items, decide whether to replace an existing item
            # The probability of replacement is max_length / self.count
            j = random.randrange(self.count)  # Random index from 0 to count-1
            if j < self.max_length:
                self.reservoir[j] = item

    def pop_first(self):
        """
        Removes and returns the first item from the reservoir.

        Returns:
            The first item in the reservoir, or None if empty.
        """
        if self.reservoir:
            return self.reservoir.pop(0)
        return None

    def add_batch(self, batch):
        """
        Adds a batch of items to the reservoir.

        Args:
            batch: Iterable containing items to add
        """
        # Handle different types of batch inputs
        if hasattr(batch, "shape") and len(batch.shape) > 1:
            # For numpy arrays with multiple dimensions, iterate over rows
            for item in batch:
                self.add_item(item)
        elif hasattr(batch, "__iter__"):
            # For any other iterable
            for item in batch:
                self.add_item(item)
        else:
            # Single item
            self.add_item(batch)

    def get_sample(self):
        """
        Returns the current sample (the reservoir).

        Returns:
            list: A list containing the sampled items.
        """
        return np.vstack(
            self.reservoir
        )  # Return a copy to prevent external modification

    def is_initialized(self):
        """
        Check if the reservoir has enough samples to be considered initialized.

        Returns:
            bool: True if count >= min_length
        """
        return self.count >= self.min_length

    def size(self):
        """
        Returns the current size of the reservoir.

        Returns:
            int: Number of items in the reservoir
        """
        return len(self.reservoir)

    def __eq__(self, other):
        if not isinstance(other, Concept):
            return NotImplemented
        return (
            self.min_length == other.min_length
            and self.max_length == other.max_length
            and self.count == other.count
            and (self.get_sample() == other.get_sample()).all()
        )


class FutureConcept:
    """
    Manages future concept data and tracks which concept each item should be assigned to.
    """

    def __init__(self, min_length):
        """
        Initialize the FutureConcept.

        Args:
            min_length (int): Minimum number of items before initialization
        """
        if min_length <= 0:
            raise ValueError("min_length must be positive")

        self.min_length = min_length
        self.store = deque()
        self.future_idx = deque()

    def add_item(self, item):
        """
        Add an item to the future concept store.

        Args:
            item: The item to add
        """
        self.store.append(item)

    def add_future_idx(self, idx, count):
        """
        Assign future indices for items in the store.

        Args:
            idx (int): The concept index to assign
            count (int): Number of items to assign to this concept
        """
        if count < 0:
            raise ValueError("count must be non-negative")
        self.future_idx.extend([idx for _ in range(count)])

    def has_assigned_future(self):
        """
        Check if any items have been assigned to concepts.

        Returns:
            bool: True if there are assigned items
        """
        return len(self.future_idx) > 0

    def is_initialized(self):
        """
        Check if the future concept has enough data to be initialized.

        Returns:
            bool: True if store has at least min_length items
        """
        return len(self.store) >= self.min_length

    def get_sample(self):
        """
        Get a copy of the current store.

        Returns:
            list: Copy of items in the store
        """
        if len(self.store) > 0:
            return np.vstack(self.store)
        else:
            return np.array([[]])

    def pop_item_with_idx(self):
        """
        Pop the oldest item and its assigned concept index.

        Returns:
            tuple: (item, concept_idx) or (None, None) if empty
        """
        if self.store and self.future_idx:
            return self.store.popleft(), self.future_idx.popleft()
        return None, None

    def clear_assignments(self):
        """Clear all future index assignments."""
        self.future_idx.clear()

    def __eq__(self, other):
        if not isinstance(other, FutureConcept):
            return NotImplemented
        return (
            self.min_length == other.min_length
            and (self.get_sample() == other.get_sample()).all()
            and self.future_idx == other.future_idx
        )


def ks_test_multivariate(X, Y, p=0.05):
    """
    Performs Kolmogorov-Smirnov test on multivariate data.

    Args:
        X, Y: Arrays to compare (should have same number of features)
        p: Significance threshold

    Returns:
        bool: True if any of the p-values is less than threshold (indicating drift)
    """
    if not hasattr(X, "shape") or not hasattr(Y, "shape"):
        # Convert to numpy arrays if they aren't already
        X = np.array(X)
        Y = np.array(Y)

    # Handle 1D case
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    if len(Y.shape) == 1:
        Y = Y.reshape(-1, 1)

    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have the same number of features")

    p_values = []
    for i in range(X.shape[1]):
        _, p_val = ks_2samp(X[:, i], Y[:, i])
        p_values.append(p_val)

    return np.any(np.array(p_values) < p)


def min_max_scaler(attr, concepts):
    """
    Scales attributes using min-max normalization based on concept percentiles.

    Args:
        attr: Attribute values to scale
        concepts: Reference concept data

    Returns:
        Scaled attribute values
    """
    concepts_array = np.array(concepts)
    if len(concepts_array.shape) == 1:
        concepts_array = concepts_array.reshape(-1, 1)

    attr_array = np.array(attr)
    reshaped = False
    if len(attr_array.shape) == 1:
        attr_array = attr_array.reshape(1, -1)
        reshaped = True

    percentiles = np.percentile(concepts_array, [1, 99], axis=0)

    # Avoid division by zero
    denominator = percentiles[1] - percentiles[0]
    denominator = np.where(denominator == 0, 1, denominator)

    scaled = (attr_array - percentiles[0]) / denominator

    if reshaped:
        scaled = scaled[0]
    return scaled


def standardizer(attr, concepts):
    """
    Standardizes attributes using robust statistics based on concept percentiles.

    Args:
        attr: Attribute values to standardize
        concepts: Reference concept data

    Returns:
        Standardized attribute values
    """
    concepts_array = np.array(concepts)
    if len(concepts_array.shape) == 1:
        concepts_array = concepts_array.reshape(-1, 1)

    attr_array = np.array(attr)
    reshaped = False
    if len(attr_array.shape) == 1:
        reshaped = True
        attr_array = attr_array.reshape(1, -1)

    percentiles = np.percentile(concepts_array, [16, 50, 84], axis=0)

    # Avoid division by zero
    denominator = percentiles[2] - percentiles[0]
    denominator = np.where(denominator == 0, 1, denominator)

    standardized = (attr_array - percentiles[1]) / denominator

    if reshaped:
        standardized = standardized[0]
    return standardized


class ConceptDetector:
    """
    Advanced concept drift detector using reservoir sampling with future concept assignment.
    """

    def __init__(self, name, min_length, max_length, drift_detector, scaler):
        """
        Initialize the concept detector.

        Args:
            min_length: Minimum samples before initialization
            max_length: Maximum reservoir size
            drift_detector: Function to detect drift between samples
            scaler: Function to scale/normalize data
        """
        self.min_length = min_length
        self.max_length = max_length
        self.name=name
        self.concept_count=0
        self.concepts = [Concept(self.concept_count, min_length, max_length)]
        self.current_idx = 0
        self.drift_detector = drift_detector
        self.scaler = scaler
        self.future_concept = FutureConcept(min_length)

    def __repr__(self):
        return self.name 
    
    def update(self, attr):
        """
        Update the concept detector with new attribute data.

        Args:
            attr: New attribute data

        Returns:
            Scaled attribute data (always returns a scaled version)
        """
        # Convert to numpy array for consistent handling

        attr_array = np.array(list(attr.values()))
        
        if np.all(attr_array==0):
            
            update_attr={f"original_{k}": v for k,v in attr.items()}
            update_attr.update({k: 0 for i, k in enumerate(attr)})
            attr.update(update_attr)
            
            attr["concept_idx"] = -1
            return attr 

        # Phase 1: Initialize current concept
        if not self.concepts[self.current_idx].is_initialized():
            self.concepts[self.current_idx].add_item(attr_array)

            
            update_attr={f"original_{k}": v for k,v in attr.items()}
            update_attr.update({k: None for i, k in enumerate(attr)})
            attr.update(update_attr)
            
            attr["concept_idx"] = None
            return attr

        # Phase 2: Initialize future concept
        if not self.future_concept.is_initialized():
            self.future_concept.add_item(attr_array)

            update_attr={f"original_{k}": v for k,v in attr.items()}
            update_attr.update({k: None for i, k in enumerate(attr)})
            attr.update(update_attr)
            
            attr["concept_idx"] = None
            return attr

        # Phase 3: Drift detection and assignment (if not already assigned)
        if not self.future_concept.has_assigned_future():
            current_array = self.concepts[self.current_idx].get_sample()
            future_array = self.future_concept.get_sample()

            if self.drift_detector(current_array, future_array):
                # Drift detected - check if it matches any existing concept
                candidates = []

                for idx, concept in enumerate(self.concepts):
                    candidate_sample = concept.get_sample()
                    candidate_array = np.array(candidate_sample)

                    if not self.drift_detector(future_array, candidate_array):
                        # Found matching existing concept
                        candidates.append(idx)

                if len(candidates) == 0:
                    # New concept discovered
                    self.concept_count+=1
                    self.concepts.append(Concept(self.concept_count, self.min_length, self.max_length))
                    print(f"{self.name}: adding concept {self.concept_count}", file=sys.stderr)
                    
                    new_concept_idx = len(self.concepts) - 1
                    self.concepts[new_concept_idx].add_batch(future_array)
                    self.future_concept.add_future_idx(
                        new_concept_idx, len(future_array)
                    )
                    self.current_idx = new_concept_idx
                elif len(candidates) == 1:
                    new_idx = candidates[0]
                    # Switch to existing matching concept
                    self.future_concept.add_future_idx(new_idx, len(future_array))
                    self.current_idx = new_idx
                else:
                    # switch to first matching concept
                    new_idx = candidates[0]
                    self.future_concept.add_future_idx(new_idx, len(future_array))
                    self.current_idx = new_idx

                    print(f"{self.name}: Merging {candidates} into {candidates[0]}", file=sys.stderr)
                    print(f"{self.name}: {self.concepts}",end=" ", file=sys.stderr)
                    # merge other ones
                    for i in candidates[1:]:
                        self.concepts[new_idx].add_batch(self.concepts[i].get_sample())

                    # delete merged concepts:
                    for i in sorted(candidates[1:], reverse=True):
                        self.concepts.pop(i)
                    print(f"-> {self.concepts}", file=sys.stderr)

            # Check for high confidence same distribution (with higher p-value threshold)
            elif not self.drift_detector(current_array, future_array, p=0.95):
                # Highly confident they are the same - assign all to current concept
                self.future_concept.add_future_idx(self.current_idx, len(future_array))
            else:
                # Uncertain - process one item at a time
                self.future_concept.add_future_idx(self.current_idx, 1)

        # Phase 4: Process assigned items
        return_attr, assigned_idx = self.future_concept.pop_item_with_idx()

        # Scale the item using its assigned concept
        assigned_concept_sample = self.concepts[assigned_idx].get_sample()

        scaled = self.scaler(return_attr, assigned_concept_sample)

        # Add the item to its assigned concept
        self.concepts[assigned_idx].add_item(return_attr)

        # Add new item to future concept for next iteration
        self.future_concept.add_item(attr_array)

        
        update_attr={f"original_{k}": v for k,v in attr.items()}
        update_attr.update({k: scaled[i] for i, k in enumerate(attr)})
        attr.update(update_attr)
        attr["concept_idx"] = self.current_idx
        return attr

    def get_current_concept_id(self):
        """Get the ID of the current active concept."""
        return self.current_idx

    def get_num_concepts(self):
        """Get the total number of discovered concepts."""
        return len(self.concepts)

    def __eq__(self, other):
        if not isinstance(other, ConceptDetector):
            return NotImplemented
        return (
            self.min_length == other.min_length
            and self.max_length == other.max_length
            and self.current_idx == other.current_idx
            and self.concepts == other.concepts
            and self.future_concept == other.future_concept
            and self.drift_detector is other.drift_detector
            and self.scaler is other.scaler
        )


class ConceptStore:
    """
    Manages concept detectors for multiple nodes/entities.
    """

    def __init__(self, min_length, max_length, drift_detector, scaler):
        """
        Initialize the concept store.

        Args:
            min_length: Minimum samples before initialization
            max_length: Maximum reservoir size
            drift_detector: Function to detect drift between samples
            scaler: Function to scale/normalize data
        """
        self.min_length = min_length
        self.max_length = max_length
        self.drift_detector = drift_detector
        self.scaler = scaler
        self.concepts = {}

    def update(self, graph):
        """
        Update concept detectors for all nodes in the graph.

        Args:
            graph: NetworkX-like graph object with nodes and attributes

        Returns:
            dict: Mapping of node_id -> scaled attributes
        """
        updated_graph = graph.copy()
        for node, attrs in updated_graph.nodes(data=True):
            if node not in self.concepts:
                self.concepts[node] = ConceptDetector( node, 
                    self.min_length, self.max_length, self.drift_detector, self.scaler
                )

            scaled_result = self.concepts[node].update(attrs)

            updated_graph.nodes[node].update(scaled_result)

        return updated_graph

    def get_concept_info(self, node=None):
        """
        Get information about concepts for a specific node or all nodes.

        Args:
            node: Specific node to get info for (if None, returns info for all nodes)

        Returns:
            dict: Concept information
        """
        if node is not None:
            if node in self.concepts:
                detector = self.concepts[node]
                return {
                    "current_concept_id": detector.get_current_concept_id(),
                    "num_concepts": detector.get_num_concepts(),
                }
            return None
        else:
            return {
                node_id: {
                    "current_concept_id": detector.get_current_concept_id(),
                    "num_concepts": detector.get_num_concepts(),
                }
                for node_id, detector in self.concepts.items()
            }

    def __eq__(self, other):
        if not isinstance(other, ConceptStore):
            return NotImplemented
        return (
            self.min_length == other.min_length
            and self.max_length == other.max_length
            and self.drift_detector is other.drift_detector
            and self.scaler is other.scaler
            and self.concepts == other.concepts
        )


class CDD(GraphRepresentation):
    def __init__(
        self,
    ):
        super().__init__()
        self.concept_store = ConceptStore(60, 300, ks_test_multivariate, standardizer)

    def transform(self, X):
        """updates data with x and output graph representation after update

        Args:
            x (_type_): input data features

        Returns:
            Tuple[torch.Tensor,torch.Tensor,torch.Tensor]: tuple of node features, edge indices, edge features
        """

        scaled_graph = self.concept_store.update(X)

        # remove unscaled nodes for model
        filtered_graph = scaled_graph.copy()

        to_remove = [
            n
            for n, d in filtered_graph.nodes(data=True)
            if d.get("concept_idx") is None
        ]
        # remove them
        filtered_graph.remove_nodes_from(to_remove)

        if nx.is_empty(filtered_graph):
            return None
        else:
            return filtered_graph
