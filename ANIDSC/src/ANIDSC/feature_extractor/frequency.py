from collections import deque

from ANIDSC.feature_buffer.tabular import NumpyFeatureBuffer

from ..utils.helper import compare_dicts

from ..save_mixin.pickle import PickleSaveMixin
import numpy as np
from ..component.feature_extractor import BaseFeatureExtractor

class FrequencyState:
    def __init__(self, time_window):
        self.time_window = time_window
        self.sliding_window = deque()
        self.last_timestamp = None
    
    
    def update(self, traffic_vector):
        self.sliding_window.extend(traffic_vector['timestamp'])
        if self.last_timestamp is None:
            self.last_timestamp = self.sliding_window[0]

        frequencies = []
        while self.sliding_window and (self.sliding_window[-1] - self.last_timestamp) > self.time_window:
            window_end = self.last_timestamp + self.time_window

            # evict entries that are older than the current window start
            while self.sliding_window and self.sliding_window[0] < self.last_timestamp:
                self.sliding_window.popleft()

            # count entries that fall within [last_timestamp, window_end)
            count = sum(1 for t in self.sliding_window if t < window_end)
            frequencies.append([self.last_timestamp, count])
            self.last_timestamp = window_end

        return np.array(frequencies) if frequencies else None
    
    def __eq__(self, other):
        # 1) Ensure same type
        if not isinstance(other, FrequencyState):
            return NotImplemented
        # 2) Compare attribute dictionaries
        return compare_dicts(self.__dict__, other.__dict__, self.__class__)

class FrequencyExtractor(PickleSaveMixin, BaseFeatureExtractor):
    def __init__(self, time_window=1, **kwargs):
        """simple frequency feature extractor based on time windows

        Args:
            time_window (int, optional): length of time window. Defaults to 1.
        """                
        super().__init__(**kwargs)
        self.time_window = time_window
        self.state = FrequencyState(self.time_window)
        
        
    
    def peek(self, traffic_vectors):
        pass
    
    @property
    def headers(self):
        return ["timestamp", "frequency"]
    
    def setup(self):
        pass 
    
    def teardown(self):
        pass
  
    
    def update(self, traffic_vector):
        return self.state.update(traffic_vector)