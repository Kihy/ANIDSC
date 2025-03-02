class NullSaveMixin:
    def save(self):
        print("didn't save")
        
    @classmethod
    def load(cls, folder, dataset_name, fe_name, file_name, name, suffix=''):
        """Load an object from a file using pickle."""
        
        print("didn't load")