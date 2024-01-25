from abc import ABC, abstractmethod
import numpy as np
from tqdm import tqdm 



class BaseTrafficFeatureExtractor(ABC):
    
    @abstractmethod
    def setup(self, path, **kwargs):
        pass 

    
    @abstractmethod
    def get_meta_headers(self):
        pass
    @abstractmethod
    def get_headers(self):
        pass
    
    @abstractmethod
    def teardown(self, num_rows):
        pass
    
    def __rrshift__(self, other):
        if isinstance(other, str):
            self.extract_features(other)
        if isinstance(other, list):
            for o in other:
                self.extract_features(o)
                
    
    def extract_features(self, path):
        self.setup(path)
        
        self.feature_file.write(",".join(self.get_headers())+"\n")
        self.meta_file.write(",".join(self.get_meta_headers())+"\n")
        
        features_list=[]
        meta_list=[]
        count=0
        skipped=0
        for packet in tqdm(self.packets):
            meta = self.get_traffic_vector(packet)
            
            
            if meta is None:
                skipped+=1
                continue
            
            feature=self.update(meta)
            features_list.append(feature)
            meta_list.append(meta)
            count+=1 
            
            if count%1e4==0:
                np.savetxt(self.feature_file, np.vstack(features_list), delimiter=",",fmt="%.7f")
                np.savetxt(self.meta_file, np.vstack(meta_list), delimiter=",", fmt="%s")
                features_list=[]
                meta_list=[]
        
        # save remaining
        np.savetxt(self.feature_file, np.vstack(features_list), delimiter=",",fmt="%.7f")
        np.savetxt(self.meta_file, np.vstack(meta_list), delimiter=",", fmt="%s")
                

        self.teardown(count)
        
        print(f"skipped: {skipped} written: {count}")