from socket import getservbyport

from ..feature_buffer.tabular import TabularFeatureBuffer
from ..component.feature_extractor import BaseMetaExtractor
from ..save_mixin.pickle import PickleSaveMixin



class TimeMetaExtractor(PickleSaveMixin, BaseMetaExtractor):
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)
        
        
    def get_meta_vector(self, packet):
        return {"timestamp": float(packet[0].time)}

    

    def get_headers(self):
        """return the feature names of traffic vectors

        Returns:
            list: names of traffic vectors
        """
        return ["timestamp"]
