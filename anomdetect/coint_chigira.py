import numpy as np
from collections import deque
import pandas as pd
from sklearn.utils.validation import check_array

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from .utils._internal import prepare_parameters_VAR

class ChigiraCointTest(BaseEstimator):
    """
    Chigira, H. (2008). A test of cointegration rank based on principal component analysis. Applied Economics Letters, 15(9), 693-696.
    """
    def __init__(self, n_selected_components=None, spec='c', earlybreak=False, PCAModel=None, adf_spec='n', adf_max_lag=None) -> None:
        self.n_selected_components = n_selected_components
        self.spec = spec
        self.earlybreak = earlybreak
        self.PCAModel = PCAModel
        self.adf_spec = adf_spec
        self.adf_max_lag=adf_max_lag

        self._IsFitted = False

    def fit(self, X):
        X_NSty = check_array(X)

        # fit VAR and PCA
        self._fit_pca(X_NSty)
        # perform the rank tests
        self._calc_coint(self.pcaY_Chigira, sig=1)
        self._IsFitted = True

        return self

    def fit_test(self, X, sig=0.05):
        #TODO: validate sig
        X_NSty = check_array(X)

        # fit VAR and PCA
        self._fit_pca(X_NSty)
        # perform the rank tests
        cointRank = self._calc_coint(self.pcaY_Chigira, sig=sig)
        self._IsFitted = True

        return cointRank
    
    def test(self, sig=0.05):
        if self._IsFitted == False:
            raise NotFittedError('Please fit this instance of the ChigiraCointTest class first.')
        #TODO: validate sig

        ttRank = len(self.test_results_['p_value'])
        cointRank = ttRank
        for i in range(0, ttRank):
            if self.test_results_['p_value'][i] > sig:
                cointRank = i
                break
        return cointRank

    def _fit_pca(self, X_NSty):
        estVAR = VAR(endog=X_NSty)
        # estimate a VAR model with desired specification
        # this step is to remove trend and mean of each series
        rstVAR = estVAR.fit(maxlags=0, trend=self.spec)
        # extract the residual. 
        X_Chigira = rstVAR.resid
        rstVAR = prepare_parameters_VAR(rstVAR)
        rstVAR.__ct_ar_last_t__ = X_NSty.shape[0]

        if isinstance(self.PCAModel, PCA):
            estPCA = self.PCAModel
        else:
            estPCA = PCA(n_components=self.n_selected_components)
        pcaY_Chigira = estPCA.fit_transform(X_Chigira)
        
        self.VARModel_ = rstVAR
        self.PCAModel_ = estPCA
        self.n_selected_components_ = estPCA.n_components_
        self.x_chigira_ = X_Chigira
        self.pcaY_Chigira = pcaY_Chigira

        # self.VARModel_.__ct_max_lag__ = 0
        # self.VARModel_.__ct_spec__ = self.VARModel_.trend

        
        #return pcaY_Chigira

    def _calc_coint(self, pcaY_Chigira, sig=0.05):
        cointRank = self.PCAModel_.n_components_
        dictRst = {'rank_r': deque(), 'p_value': deque(), 'adf_lag': deque()}
        for i in range(0, self.PCAModel_.n_components_):
            rstADF = adfuller(pcaY_Chigira[:,self.PCAModel_.n_components_-i-1], regression=self.adf_spec, maxlag=self.adf_max_lag, autolag='AIC')
            dictRst['rank_r'].append("r <= {0}".format(i))
            dictRst['p_value'].append(rstADF[1])
            dictRst['adf_lag'].append(rstADF[2])
            # we won't find the rank if sig is meaningless. 
            # This does not stop the calculation for all hypotheses
            if sig >= 1:
                continue
            # if ADF p-value is greater than significance level, we cannot reject null of a unit root
            if rstADF[1] > sig and cointRank > i:
                cointRank = i
                if self.earlybreak:
                    break
        
        for crntKey in dictRst:
            if isinstance(dictRst[crntKey], deque):
                dictRst[crntKey] = list(dictRst[crntKey])

        self.test_results_ = dictRst
        return cointRank