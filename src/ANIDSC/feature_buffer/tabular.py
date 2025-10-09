import os
from ..save_mixin.pickle import PickleSaveMixin
import numpy as np

from ..component.feature_buffer import BaseFeatureBuffer

from numpy.typing import NDArray


class NumpyFeatureBuffer(PickleSaveMixin, BaseFeatureBuffer):
    @property 
    def file_type(self):
        return "csv"
    
    def setup(self):
        super().setup()
        # write header when file is empty
        headers = self.request_attr("headers")
        
        if os.stat(self.feature_path).st_size == 0:
            self.save_file.write(",".join(headers) + "\n")
    
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
        
        return batch_data
    
    def __str__(self):
        return f"NumpyFeatureBuffer({self.folder_name})"




class DictFeatureBuffer(PickleSaveMixin, BaseFeatureBuffer):
    @property 
    def file_type(self):
        return "csv"
    
    def setup(self):
        super().setup()
        # write header when file is empty
        headers = self.request_attr("headers")
        
        if os.stat(self.feature_path).st_size == 0:
            self.save_file.write(",".join(headers) + "\n")
    
    def save_buffer(self) -> NDArray:
        """saves buffer

        Returns:
            NDArray: the buffered features as numpy array
        """

        if len(self.data_list)==0:
            return 
        
        batched=self.data_list.copy()
        
        for data in self.data_list:
            self.save_file.write(",".join(map(str, data.values()))+"\n")
        
        
        
        self.data_list = []
        
        return batched
    
    def __str__(self):
        return f"NumpyFeatureBuffer({self.folder_name})"
