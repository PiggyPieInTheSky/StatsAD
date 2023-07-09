# unfinished

from abc import ABC, abstractmethod    

class TestBaseClass(ABC):
    @abstractmethod
    def fit(self, X1, X2=None):
        pass
    @abstractmethod
    def test(self, X1, X2=None, sig:int=0.05):
        pass
    @abstractmethod
    def fit_test(self, X1, X2=None, sig:int=0.05):
        pass

class TransformerBasClass(ABC):

    @abstractmethod
    def transform(self, X1, X2=None):
        pass
    @abstractmethod
    def fit_transform(self, X1, X2=None):
        pass

class PredictorBaseClass(ABC):

    @abstractmethod
    def predict(self, X1, X2=None):
        pass
    @abstractmethod
    def fit_predict(self, X1, X2=None):
        pass