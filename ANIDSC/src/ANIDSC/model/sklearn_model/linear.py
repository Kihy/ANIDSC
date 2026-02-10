from sklearn.linear_model import SGDOneClassSVM
from .base_sklearn_model import BaseSklearnModel
from ...utils.helper import compare
class SGDOCSVM(BaseSklearnModel):
    def init_model(self):
        return SGDOneClassSVM(nu=0.05)
    
    def __eq__(self, other):
        if not isinstance(other, SGDOCSVM):
            return False

        return compare(self.model.get_params(), other.model.get_params())
               