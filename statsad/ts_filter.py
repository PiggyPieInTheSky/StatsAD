from utils import TestBaseClass, PredictorBaseClass

class ARFilter(TestBaseClass, PredictorBaseClass):
    """Anomaly detection based on filtering from an auto regressive model"""
    
    def __init__(self) -> None:
        pass

    def fit(self, X):
        pass

    def fit_test(self, X, sig: int = 0.05):
        pass

    def test(self, X, sig:int=0.05):
        pass

    def fit_predict(self, X):
        pass

    def predict(self, X):
        pass