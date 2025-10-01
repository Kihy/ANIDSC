from abc import abstractmethod


class BaseSaveMixin:
    
    @property
    def save_path(self):
        fe_name = self.perform_action("feature_extractor", "__str__")
        if fe_name is None:
            fe_name = self.get_attr("data_source", "fe_name")

        return f"{self.get_attr('data_source','dataset_name')}/{fe_name}/saved_components/{self.component_type}/{self.get_attr('data_source','file_name')}/{str(self)}.{self.save_type}"

    
    @property
    @abstractmethod
    def save_type(self):
        pass
    
    @abstractmethod
    def save(self):
        pass 
    
    @classmethod
    @abstractmethod
    def load(cls, path):
        pass 