# unfinished

from abc import ABC, abstractmethod    

class TestBaseClass(ABC):
    @abstractmethod
    def fit(self, X):
        pass
    @abstractmethod
    def test(self, X, sig:int=0.05):
        pass
    @abstractmethod
    def fit_test(self, X, sig:int=0.05):
        pass

class TransformerBasClass(ABC):

    @abstractmethod
    def transform(self, X):
        pass
    @abstractmethod
    def fit_transform(self, X):
        pass

class PredictorBaseClass(ABC):

    @abstractmethod
    def predict(self, X):
        pass
    @abstractmethod
    def fit_predict(self, X):
        pass