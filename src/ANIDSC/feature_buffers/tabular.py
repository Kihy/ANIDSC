

import numpy as np

from .base_buffer import BaseFeatureBuffer
from ..save_mixin.null import NullSaveMixin

from numpy.typing import NDArray


class TabularFeatureBuffer(NullSaveMixin, BaseFeatureBuffer):
    
            
    def save_buffer(self)->NDArray:
        """saves buffer

        Returns:
            NDArray: the buffered features as numpy array
        """        

        batch_data=np.vstack(self.feature_list)
        
        if self.save_features:
            np.savetxt(
                    self.feature_file,
                    batch_data,
                    delimiter=",",
                    fmt="%s",
                )
            
        if self.save_meta:
            np.savetxt(
                self.meta_file, np.vstack(self.meta_list), delimiter=",", fmt="%s"
            )
        
        self.feature_list = []
        self.meta_list = []
        
        return batch_data
    
    def __str__(self):
        return f"TabularFeatureBuffer({self.buffer_size})"
        