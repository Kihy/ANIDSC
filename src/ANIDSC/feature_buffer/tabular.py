import numpy as np

from ..component.feature_buffer import BaseFeatureBuffer
from ..save_mixin.null import NullSaveMixin

from numpy.typing import NDArray


class TabularFeatureBuffer(NullSaveMixin, BaseFeatureBuffer):
    @property 
    def file_type(self):
        return "csv"
    
    def save_buffer(self) -> NDArray:
        """saves buffer

        Returns:
            NDArray: the buffered features as numpy array
        """

        if len(self.data_list)==0:
            return 
        
        batch_data = np.vstack(self.data_list)
        
        np.savetxt(
            self.save_file,
            batch_data,
            delimiter=",",
            fmt="%s",
        )
        self.data_list = []


    def __str__(self):
        return f"TabularFeatureBuffer({self.buffer_size})"
