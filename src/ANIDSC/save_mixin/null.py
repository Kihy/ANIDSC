from ..save_mixin.basemixin import BaseSaveMixin

class NullSaveMixin(BaseSaveMixin):
    def save(self):
        print(f"skipping save for {str(self)}")
        
    @classmethod
    def load(cls, path):        
        print("No load")
        
    @property
    def save_type(self):
        return "none"