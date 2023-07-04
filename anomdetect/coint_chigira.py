import numpy as np
from collections import deque
import pandas as pd
from sklearn.utils.validation import check_array

from arch.unitroot import PhillipsPerron
from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.ar_model import AutoReg
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from typing import Type

from .utils._internal import prepare_parameters_VAR

class ChigiraCointTest(BaseEstimator):
    """
    Chigira cointegration test -- A Principal Component Analysis based cointegration test for large variable sets

    This test decomposes I(1) series into PCA factors. The number of I(0) factors indicates the number of cointegration vectors.

    Parameters
    ----------
    n_selected_components: int or default to None, number of selected PCA components. If specified, only this number of top PCA components are used in the test. The rest are discarded as part of an implicit dimension reduction process.
    
    spec: str, indicates the deterministic terms for removing trend and constant as part of the preprocessing step. Accepted values should agree with the trend parameter of statsmodels.tsa.vector_ar.var_model.VAR.fit method. Default is 'c' for a single constant term.

    earlybreak: bool, indicates if test statistics should be calculated for all potential numbers of cointegration vector or stops when the first one is found. The first one is found when the ith smallest PCA factor is no longer stationary. 

    PCAModel: sklearn.decomposition.PCA model object or None, allows customization of the underlying PCA model. If not provided, a new model is created with default set up. When provided, n_selected_components is ignored.
    
    unit_root_test: str, specifies the unit root test to be used. The default value 'pp' is for Phillips Perron test. For shorter time series, use 'adf' for Augmented Dickey Fuller test.
    
    urt_spec: str, unit root test specification for the deterministic terms. Its value should agree with either the 'trend' parameter of arch.unitroot.PhillipsPerron or the 'regression' parameter of statsmodels.tsa.stattools.adfuller.

    urt_lags: int or default to None, number of the lags to use in Newey-West estimator of covariance for Phillips Perron test or the max lag for the Augmented Dickey Fuller test.

    phillips_perron_test_type: str, test type of the Phillips Perron test. The default value is 'tau'. 'rho' is another value that can be passed down to arch.unitroot.PhillipsPerron.

    adf_autolag: str or None, default is 'AIC', used to indicate the auto selection of ADF test's lag length. For more info, see statsmodels.tsa.stattools.adfuller.

    Reference
    ---------
    [*] Chigira, H. (2008). A test of cointegration rank based on principal component analysis. Applied Economics Letters, 15(9), 693-696.
    """
    def __init__(self
        , n_selected_components: None|int = None
        , spec:str='c'
        , earlybreak:bool=False
        , PCAModel:PCA|None=None
        , unit_root_test:str='pp', urt_spec:str='n', urt_lags:int|None=None
        , phillips_perron_test_type:str='tau'
        , adf_autolag:str|None='AIC', adf_no_ac:bool=False, adf_no_ac_lag_len:int=4
    ):
        self.n_selected_components = n_selected_components
        self.spec = spec
        self.earlybreak = earlybreak
        self.PCAModel = PCAModel
        if unit_root_test == 'adf':
            self.unit_root_test = 'adf'
        else:
            self.unit_root_test = 'pp'
        self.phillips_perron_test_type = phillips_perron_test_type
        self.urt_spec = urt_spec
        self.urt_lags=urt_lags
        self.adf_autolag = adf_autolag
        self.adf_no_ac = adf_no_ac
        self.adf_no_ac_lag_len = adf_no_ac_lag_len

        self._IsFitted = False

    def fit(self, X):
        '''Fits the Chigira test object. This method detrends the original data based on the specification of the spec class parameter and calculates the PCA factors.
        
        Parameters
        ----------
        X: an n by l array of data where n represents the number of time periods and l indicates the number of variables. All series (columns) in the data are assumed to be I(1). The test does not pre-test this condition.
        '''
        X_NSty = check_array(X)

        # fit VAR and PCA
        self._fit_pca(X_NSty)
        # perform the rank tests
        self._calc_coint(self.pcaY_Chigira, sig=1)
        self._IsFitted = True

        return self

    def fit_test(self, X, sig:int=0.05):
        """Fits the Chigira test object and tests the number of cointegration vectors. 
        This method detrends the original data based on the specification of the spec class parameter, calculates the PCA factors, checks the number of stationary PCA factors.
        
        Parameters
        ----------
        X: an n by l array of data where n represents the number of time periods and l indicates the number of variables. All series (columns) in the data are assumed to be I(1). The test does not pre-test this condition.

        sig: int, significance level of the test, between 0 and 1, default is 0.05.
        """
        
        if sig < 0  or sig > 1:
            raise ValueError('The sig input argument should be between 0 and 1.')

        X_NSty = check_array(X)

        # fit VAR and PCA
        self._fit_pca(X_NSty)
        # perform the rank tests
        cointRank = self._calc_coint(self.pcaY_Chigira, sig=sig)
        self._IsFitted = True

        return cointRank
    
    def test(self, sig:int=0.05):
        """Tests the number of cointegration vectors after the class is fitted to the model.
        
        Parameters
        ----------
        sig: int, significance level of the test, between 0 and 1, default is 0.05.
        """
        if self._IsFitted == False:
            raise NotFittedError('Please fit this instance of the ChigiraCointTest class first.')
        
        if sig < 0  or sig > 1:
            raise ValueError('The sig input argument should be between 0 and 1.')

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

    def _calc_coint(self, pcaY_Chigira, sig=0.05):
        cointRank = self.PCAModel_.n_components_
        dictRst = {'rank_r': deque(), 'p_value': deque(), 'lag': deque(), 'stat': deque()}
        for i in range(0, self.PCAModel_.n_components_):
            
            dictRst['rank_r'].append("r <= {0}".format(i))
            pvalue = 2
            if self.unit_root_test == 'adf':
                if self.adf_no_ac:
                    blag = self._ar_lag_sel(pcaY_Chigira[:,self.PCAModel_.n_components_-i-1], min(sig, 0.1))
                    rstADF = adfuller(pcaY_Chigira[:,self.PCAModel_.n_components_-i-1], regression=self.urt_spec, maxlag=blag, autolag=None)
                else:
                    rstADF = adfuller(pcaY_Chigira[:,self.PCAModel_.n_components_-i-1], regression=self.urt_spec, maxlag=self.urt_lags, autolag=self.adf_autolag)
                dictRst['lag'].append(rstADF[2])
                dictRst['stat'].append(rstADF[0])
                pvalue=rstADF[1]
            else:
                pp_rst = PhillipsPerron(pcaY_Chigira[:,self.PCAModel_.n_components_-i-1], lags=self.urt_lags, trend=self.urt_spec, test_type=self.phillips_perron_test_type)
                dictRst['lag'].append(pp_rst.lags)
                dictRst['stat'].append(pp_rst.stat)
                pvalue=pp_rst.pvalue
            dictRst['p_value'].append(pvalue)
            
            # we won't find the rank if sig is meaningless. 
            # This does not stop the calculation for all hypotheses
            if sig >= 1:
                continue
            # if ADF p-value is greater than significance level, we cannot reject null of a unit root
            if pvalue > sig and cointRank > i:
                cointRank = i
                if self.earlybreak:
                    break
        
        for crntKey in dictRst:
            if isinstance(dictRst[crntKey], deque):
                dictRst[crntKey] = list(dictRst[crntKey])

        self.test_results_ = dictRst
        return cointRank

    def _ar_lag_sel(self, y_t, sig=0.05):
        
        if self.urt_spec == 'ctt':
            urt_spec = 'ct'
        else:
            urt_spec = self.urt_spec
        
        if urt_spec == 'n':
            df_adj_det = 0
        else:
            df_adj_det = len(urt_spec)

        # find the optimal lag for Johansen by estimating a best fitted VAR based on AIC
        best_lagged_AIC = 9999999999
        best_lag = 0
        for lagL in range(1, self.urt_lags+1):
            # ensure all lagged models have the same starting period after lags are used
            ts_y = y_t[(self.urt_lags-lagL):]

            # fit an AR model
            lagAR = AutoReg(endog=ts_y, lags=lagL, trend=urt_spec).fit()
            # test residual autocorrelation with up to self.adf_no_ac_lag_len lags
            ljung_box_rst = lagAR.test_serial_correlation(lags=df_adj_det+lagL+self.adf_no_ac_lag_len)
            # if all self.adf_no_ac_lag_len lags of the residuals show no autocorrelation
            num_no_pass_lag = ljung_box_rst.dropna().assign(test_pass=lambda df_:df_['LB P-value']<0.01).query('test_pass==False').shape[0]
            
            # if no auto correlation in the residual and AIC is better
            if num_no_pass_lag == 0 and best_lagged_AIC > lagAR.aic:
                # pick the lag
                best_lagged_AIC = lagAR.aic
                best_lag = lagL
        return best_lag    