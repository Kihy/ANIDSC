class NullSaveMixin:
    def save(self):
        super().save()
        print(f"skipping save for {str(self)}")
        
    @classmethod
    def load(cls, path):
        """Load an object from a file using pickle."""
        
        print("didn't load")