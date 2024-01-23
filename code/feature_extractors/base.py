from abc import ABC, abstractmethod
import numpy as np
from tqdm import tqdm 

class BaseTrafficFeatureExtractor(ABC):
    
    @abstractmethod
    def setup_input_stream(self, path):
        pass 

    @abstractmethod
    def setup_output_file(self):
        pass
    
    @abstractmethod
    def get_meta_headers(self):
        pass
    @abstractmethod
    def get_headers(self):
        pass
    
    @abstractmethod
    def finish(self):
        pass
    
    def __rrshift__(self, other):
        self.extract_features(other)
    
    def extract_features(self, path):
        packets=self.setup_input_stream(path)
        feature_file, meta_file=self.setup_output_file()
        
        feature_file.write(",".join(self.get_headers())+"\n")
        meta_file.write(",".join(self.get_meta_headers())+"\n")
        
        features_list=[]
        meta_list=[]
        count=0
        skipped=0
        for packet in tqdm(packets):
            features, meta = self.get_feature(packet)
            if features is None:
                skipped+=1
                continue
            features_list.append(features)
            meta_list.append(meta)
            count+=1 
            
            if count%1e4==0:
                np.savetxt(feature_file, np.vstack(features_list), delimiter=",",fmt="%.7f")
                np.savetxt(meta_file, np.vstack(meta_list), delimiter=",", fmt="%s")
                features_list=[]
                meta_list=[]
        
        # save remaining
        np.savetxt(feature_file, np.vstack(features_list), delimiter=",",fmt="%.7f")
        np.savetxt(meta_file, np.vstack(meta_list), delimiter=",", fmt="%s")
                
        meta_file.close()
        feature_file.close()
        self.finish()
        
        print(f"skipped: {skipped} written: {count}")