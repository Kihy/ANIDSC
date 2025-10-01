from collections import deque

from ANIDSC.feature_buffer.tabular import TabularFeatureBuffer

from ..utils.helper import compare_dicts

from ..save_mixin.pickle import PickleSaveMixin
import numpy as np
from ..component.feature_extractor import BaseFeatureExtractor

class FrequencyState:
    def __init__(self, time_window):
        self.time_window=time_window 
        self.sliding_window=deque()
        self.last_timestamp=None
    
    def update(self, traffic_vector):
        self.sliding_window.append(traffic_vector['timestamp'])
        self.last_timestamp=traffic_vector['timestamp']
        while (self.sliding_window[-1]-self.sliding_window[0])>self.time_window:
            self.sliding_window.popleft()
        return np.array([[len(self.sliding_window)]])
    
    def __eq__(self, other):
        # 1) Ensure same type
        if not isinstance(other, FrequencyState):
            return NotImplemented
        # 2) Compare attribute dictionaries
        return compare_dicts(self.__dict__, other.__dict__, self.__class__)

class FrequencyExtractor(PickleSaveMixin, BaseFeatureExtractor):
    def __init__(self, time_window=10, **kwargs):
        """simple frequency feature extractor based on time windows

        Args:
            time_window (int, optional): length of time window. Defaults to 10.
        """                
        super().__init__(**kwargs)
        self.time_window=time_window
        
        self.state=FrequencyState(self.time_window)     # collection of timestamps
        
        self.feature_buffer = TabularFeatureBuffer(buffer_size=256)
        self.feature_buffer.attach_to(self)
    
    def peek(self, traffic_vectors):
        pass
    
    def get_headers(self):
        return ["frequency"]

    
    def update(self, traffic_vector):
        return self.state.update(traffic_vector)