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

    def __init__(self, idx, min_length, max_length, malicious):
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

        self.idx = idx
        self.min_length = min_length
        self.max_length = max_length
        self.reservoir = []
        self.malicious = malicious
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
    Uses concept IDs instead of indices for stable references.
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
        self.future_ids = deque()  # Changed from future_idx to future_ids

    def add_item(self, item):
        """
        Add an item to the future concept store.

        Args:
            item: The item to add
        """
        self.store.append(item)

    def add_future_id(self, concept_id, count):
        """
        Assign future concept IDs for items in the store.

        Args:
            concept_id: The concept ID to assign
            count (int): Number of items to assign to this concept
        """
        if count < 0:
            raise ValueError("count must be non-negative")
        self.future_ids.extend([concept_id for _ in range(count)])

    def add_future_idx(self, idx, count):
        """
        Legacy method for backwards compatibility.
        Delegates to add_future_id.
        """
        self.add_future_id(idx, count)

    def has_assigned_future(self):
        """
        Check if any items have been assigned to concepts.

        Returns:
            bool: True if there are assigned items
        """
        return len(self.future_ids) > 0

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
            numpy.ndarray: Stacked array of items in the store
        """
        if len(self.store) > 0:
            return np.vstack(self.store)
        else:
            return np.array([[]])

    def pop_item_with_id(self):
        """
        Pop the oldest item and its assigned concept ID.

        Returns:
            tuple: (item, concept_id) or (None, None) if empty
        """
        if self.store and self.future_ids:
            return self.store.popleft(), self.future_ids.popleft()
        return None, None

    def pop_item_with_idx(self):
        """
        Legacy method for backwards compatibility.
        Delegates to pop_item_with_id.
        """
        return self.pop_item_with_id()

    def remap_ids(self, id_mapping):
        """
        Remap concept IDs in the future assignments queue.

        This is used when concepts are merged - any references to merged
        concept IDs are updated to point to the primary (surviving) concept ID.

        Args:
            id_mapping (dict): Dictionary mapping old_id -> new_id
                              e.g., {3: 1, 5: 1} means concepts 3 and 5
                              were merged into concept 1
        """
        if not id_mapping:
            return

        # Create new deque with remapped IDs
        remapped_ids = deque()
        for concept_id in self.future_ids:
            # Use mapped ID if it exists, otherwise keep original
            new_id = id_mapping.get(concept_id, concept_id)
            remapped_ids.append(new_id)

        self.future_ids = remapped_ids

    def clear_assignments(self):
        """Clear all future ID assignments."""
        self.future_ids.clear()

    def __eq__(self, other):
        if not isinstance(other, FutureConcept):
            return NotImplemented
        return (
            self.min_length == other.min_length
            and (self.get_sample() == other.get_sample()).all()
            and self.future_ids == other.future_ids
        )

    def __repr__(self):
        return (
            f"FutureConcept(min_length={self.min_length}, "
            f"store_size={len(self.store)}, "
            f"assigned={len(self.future_ids)})"
        )


def ks_test_multivariate(X, Y):
    """
    Performs Kolmogorov-Smirnov test on multivariate data. Returns minimum p value across all dimensions
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

    return np.min(np.array(p_values))


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
        self.name = name
        self.concept_count = 0
        self.concepts = {0: Concept(self.concept_count, min_length, max_length, False)}
        self.current_id = 0
        self.drift_detector = drift_detector
        self.scaler = scaler
        self.p = 0.01
        self.future_concept = FutureConcept(min_length)

    def __repr__(self):
        return self.name

    def update(self, attr):
        """
        Update the concept detector with new attribute data.

        Simplified logic using concept IDs instead of indices:
        1. Handle special cases (trivial/initialization)
        2. Buffer incoming data in future_concept
        3. When buffer is ready, detect drift once and assign all items
        4. Return oldest buffered item with its assignment

        Args:
            attr: New attribute data

        Returns:
            Scaled attribute data with concept_id
        """
        attr_array = np.array(list(attr.values()))

        # Special case: trivial concept (all zeros)
        if np.all(attr_array == 0):
            update_attr = {f"original_{k}": v for k, v in attr.items()}
            update_attr.update({k: 0 for k in attr})
            attr.update(update_attr)
            attr["concept_id"] = -1
            attr["malicious_concept"] = False
            return attr

        # Phase 1 & 2: Initialize current and future concepts
        # During initialization, we don't scale - just collect data
        if not self.concepts[self.current_id].is_initialized():
            self.concepts[self.current_id].add_item(attr_array)
            return self._make_unscaled_attr(attr, None)

        if not self.future_concept.is_initialized():
            self.future_concept.add_item(attr_array)
            return self._make_unscaled_attr(attr, None)

        # Phase 3: Detect drift when future buffer is full and unassigned
        if not self.future_concept.has_assigned_future():
            self._detect_and_assign_drift()

        # Phase 4: Process next item from the buffer
        return self._process_buffered_item(attr, attr_array)

    def _make_unscaled_attr(self, attr, concept_id):
        """Helper: Create unscaled attribute dictionary"""
        update_attr = {f"original_{k}": v for k, v in attr.items()}
        update_attr.update({k: None for k in attr})
        update_attr.update({"malicious_concept": False})
        attr.update(update_attr)
        attr["concept_id"] = concept_id
        return attr

    def _detect_and_assign_drift(self):
        """Helper: Detect drift and assign future buffer to appropriate concept"""
        current_sample = self.concepts[self.current_id].get_sample()
        future_sample = self.future_concept.get_sample()

        p_value = self.drift_detector(current_sample, future_sample)
        

        # No drift detected with high confidence - assign all to current
        if p_value > 1 - self.p:
            self.future_concept.add_future_id(self.current_id, len(future_sample))
            return

        # Drift detected - find or create matching concept
        if p_value < self.p:
            target_id = self._find_or_create_matching_concept(future_sample)
            self.future_concept.add_future_id(target_id, len(future_sample))
            self.current_id = target_id
            return

        self.future_concept.add_future_id(self.current_id, 1)

    def _find_or_create_matching_concept(self, future_sample):
        """Helper: Find existing matching concept or create new one"""
        # Find all concepts that match the future sample


        matching_ids = []
        non_matching_p = []
        for concept_id, concept in self.concepts.items():
            candidate_sample = concept.get_sample()
            p_value = self.drift_detector(future_sample, np.array(candidate_sample))
           
            if p_value > self.p:
                matching_ids.append(concept_id)
            else:
                non_matching_p.append(p_value)

        # No matches - create new concept
        if not matching_ids:

            # check maximum p value, if its still very small, its malicious concept
            if np.max(non_matching_p) < 1e-4:
                malicious = True
            else:
                malicious = False

            self.concept_count += 1
            new_id = self.concept_count
            self.concepts[new_id] = Concept(
                new_id, self.min_length, self.max_length, malicious
            )
            self.concepts[new_id].add_batch(future_sample)

            print(
                f"{self.name}: adding concept {new_id}, malicious: {malicious}, maxp p {np.max(non_matching_p)}",
                file=sys.stderr,
            )
            return new_id

        # One match - use it
        if len(matching_ids) == 1:
            return matching_ids[0]

        # Multiple matches - merge them
        return self._merge_concepts(matching_ids)

    def _merge_concepts(self, matching_ids):
        """Helper: Merge multiple matching concepts into the first one"""
        primary_id = matching_ids[0]

        print(
            f"{self.name}: Merging concepts {matching_ids} into {primary_id}",
            file=sys.stderr,
        )

        # Merge all data into primary concept
        for concept_id in matching_ids[1:]:
            self.concepts[primary_id].add_batch(self.concepts[concept_id].get_sample())
            # if any is malicious, all is malicious
            if self.concepts[concept_id].malicious:
                self.concepts[primary_id].malicious = True

        # Update future_concept queue to remap merged IDs to primary
        id_mapping = {old_id: primary_id for old_id in matching_ids[1:]}
        if hasattr(self.future_concept, "remap_ids"):
            self.future_concept.remap_ids(id_mapping)

        # Remove merged concepts from dictionary
        for concept_id in matching_ids[1:]:
            del self.concepts[concept_id]

        print(f"{self.name}: Now have {len(self.concepts)} concepts", file=sys.stderr)
        return primary_id

    def _process_buffered_item(self, attr, new_attr_array):
        """Helper: Process the oldest item from buffer and add new item"""
        # Get oldest buffered item and its assigned concept ID
        item_data, assigned_id = self.future_concept.pop_item_with_id()

        # Safety check
        if assigned_id not in self.concepts:
            print(
                f"{self.name}: Warning - invalid ID {assigned_id}, using current {self.current_id}",
                file=sys.stderr,
            )
            assigned_id = self.current_id

        # Scale using assigned concept
        concept_sample = self.concepts[assigned_id].get_sample()
        scaled = self.scaler(item_data, concept_sample)

        # Add processed item to its concept
        self.concepts[assigned_id].add_item(item_data)

        # Add new incoming item to future buffer
        self.future_concept.add_item(new_attr_array)

        # Build return dictionary
        update_attr = {f"original_{k}": v for k, v in attr.items()}
        update_attr.update({k: scaled[i] for i, k in enumerate(attr)})
        update_attr.update({"malicious_concept": self.concepts[assigned_id].malicious})
        attr.update(update_attr)
        attr["concept_id"] = assigned_id

        return attr

    def get_current_concept_id(self):
        """Get the ID of the current active concept."""
        return self.current_id

    def get_num_concepts(self):
        """Get the total number of discovered concepts."""
        return len(self.concepts)

    def __eq__(self, other):
        if not isinstance(other, ConceptDetector):
            return NotImplemented
        return (
            self.min_length == other.min_length
            and self.max_length == other.max_length
            and self.current_id == other.current_id
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
                self.concepts[node] = ConceptDetector(
                    node,
                    self.min_length,
                    self.max_length,
                    self.drift_detector,
                    self.scaler,
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
        self.concept_store = ConceptStore(60, 3600, ks_test_multivariate, standardizer)

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
            n for n, d in filtered_graph.nodes(data=True) if d.get("concept_id") in [None, -1]
        ]

        # remove them
        filtered_graph.remove_nodes_from(to_remove)

        if nx.is_empty(filtered_graph):
            return None
        else:
            return filtered_graph
