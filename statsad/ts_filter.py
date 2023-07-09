from typing import Optional, Union
import numpy as np
from .utils import TestBaseClass, PredictorBaseClass
from sklearn.utils.validation import check_array
from sklearn.exceptions import NotFittedError
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.ar_model import ar_select_order
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
from statsmodels.tsa.statespace.varmax import VARMAX, VARMAXResults
from statsmodels.tsa.vector_ar.var_model import VAR, VARResults
from statsmodels.tsa.ar_model import AutoReg, AutoRegResults

#TODO: deal with problems of Ljung-Box
# https://stats.stackexchange.com/questions/6455/how-many-lags-to-use-in-the-ljung-box-test-of-a-time-series/205262#205262
# https://stats.stackexchange.com/questions/148004/testing-for-autocorrelation-ljung-box-versus-breusch-godfrey
class ARFilter(TestBaseClass, PredictorBaseClass):
    """Anomaly detection based on filtering from an auto regressive model"""
    
    def __init__(self
        , trend:str='c'
        , autolag:Optional[str]='aic', autolag_full_search:bool=False
        , max_auto_lags:int=5, autolag_init_p:int=1, autolag_min_p:int=1
        , max_ma_lags:int=0, autolag_init_q:int=0, autolag_min_q:int=0
        , auto_corr_sig=0.05, auto_corr_max_lag=4
        , differencing:bool=False
        , ts_model:Optional[Union[VARResults,VARMAXResults,AutoRegResults,SARIMAXResults]]=None
        , ml_max_iter=100
    ) -> None:
        self.trend = trend
        self.autolag = autolag
        self.autolag_full_search = autolag_full_search
        self.max_auto_lags = max_auto_lags
        self.max_ma_lags = max_ma_lags
        self.autolag_init_p = autolag_init_p
        self.autolag_init_q = autolag_init_q
        self.autolag_min_p = autolag_min_p
        self.autolag_min_q = autolag_min_q
        self.auto_corr_sig = auto_corr_sig
        self.auto_corr_max_lag = auto_corr_max_lag
        self.differencing = differencing
        self.ts_model = ts_model
        self.ml_max_iter = ml_max_iter

        

    def fit(self, X1, X2=None):
        X1chk = check_array(X1)
        X2chk = check_array(X2) if X2 is not None else None
            
        #https://www.statsmodels.org/dev/examples/notebooks/generated/statespace_sarimax_faq.html
        n_obs, n_vars = X1chk.shape

        if self.autolag is not None and self.autolag not in ['aic', 'bic']:
            raise ValueError("autolag class parameter can only take values of 'aic', 'bic' or None.")

        if self.autolag_min_p<0 or self.max_auto_lags<0 or self.autolag_init_p<0 \
            or self.auto_corr_max_lag<0 \
            or self.max_ma_lags<0 or self.autolag_init_q<0 or self.autolag_min_q<0:
            raise ValueError(f'Neither of the following lag parameters should be negative: autolag_min_p (={self.autolag_min_p}), max_auto_lags (={self.max_auto_lags}), autolag_init_p (={self.autolag_init_p}), max_ma_lags (={sefl.max_ma_lags}), autolag_init_q (={self.max_ma_lags}), autolag_min_q  (={self.max_ma_lags}), auto_corr_max_lag (={self.auto_corr_max_lag}).')
        elif self.autolag_min_p > self.max_auto_lags:
            raise ValueError('autolag_min_p parameter cannot be higher than max_auto_lags.')
        elif self.autolag_init_p > self.max_auto_lags or self.autolag_init_p < self.autolag_min_p:
            raise ValueError(f'autolag_init_p (={self.autolag_init_p}) should be bounded between autolag_min_p (={self.autolag_min_p}) and max_auto_lags (={self.max_auto_lags}).')
        elif self.autolag_min_q > self.max_ma_lags:
            raise ValueError('autolag_min_q parameter cannot be higher than max_ma_lags.')
        elif self.autolag_init_q > self.max_ma_lags or self.autolag_init_q < self.autolag_min_q:
            raise ValueError(f'autolag_init_q (={self.autolag_init_q}) should be bounded between autolag_min_q (={self.autolag_min_q}) and max_ma_lags (={self.max_ma_lags}).')
        
        # self._estimate_model
        # make the first difference for VAR type models
        if self.differencing and n_vars != 1:
            endogX = X1chk[1:,:] - X1chk[0:-1,:]
            if X2chk is not None:
                exogX = X2chk[1:,:]
            else:
                exogX = None
        else:
            endogX, exogX = X1chk, X2chk

        if self.autolag is not None:
            ar_p, ma_q = self._auto_lag(endogX, exogX, self.max_auto_lags, self.max_ma_lags)
        else:
            ar_p, ma_q = self.max_auto_lags, self.max_ma_lags
        
        estRst = self._estimate_model(endogX, exogX, ar_p, ma_q)
        self.ts_modl_ = estRst
        self.ar_p_, self.ma_q_ = ar_p, ma_q         
        self.in_d_ = 1 if self.differencing else 0

        # calculate the variance-covariance matrix of the residuals. need it for T2 stat calculation
        if n_vars > 1:
            resid = estRst.resid
            self.covMatInv_ = np.linalg.inv(np.cov(resid, rowvar=False, ddof=1))
        else:
            resid = estRst.resid.reshape(-1,1)
            self.covMatInv_ = 1 / np.var(resid, ddof=1)

    def fit_test(self, X1, X2=None, sig: int = 0.05):
        pass

    def test(self, X1, X2=None, sig:int=0.05):
        pass

    def fit_predict(self, X1, X2=None):
        pass

    def predict(self, X1, X2=None):
        pass

    def _estimate_model(self, endogX, exogX, ar_p, ma_q):
        n_vars = endogX.shape[1]

        if n_vars == 1: # if single time series
            ar_d = 1 if self.differencing else 0
            if ma_q == 0: # auto regressive
                return AutoReg(endog=endogX, exog=exogX
                        , lags=ar_p
                        , trend=self.trend
                    ).fit()
            else: # auto regressive moving average
                return SARIMAX(endog=endogX, exog=exogX
                        , order=(ar_p, ar_d, ma_q)
                    ).fit(maxiter=self.ml_max_iter)

        else: # if multiple time series
            if self.max_ma_lags == 0: # auto regressive
                return VAR(endog=endogX
                        , exog=exogX
                    ).fit(maxlags=ar_p, ic=None, trend=self.trend)
            else: # auto regressive moving average
                return VARMAX(endog=endogX, exog=exogX
                        , order=(ar_p, ma_q)
                        , trend=self.trend
                    ).fit(maxiter=self.ml_max_iter)

    def _auto_lag(self, endogX, exogX, max_auto_lags, max_ma_lags=0):
        import warnings

        start_p = min(max(self.autolag_init_p, self.autolag_min_p), max_auto_lags)
        start_q = min(max(self.autolag_init_q, self.autolag_min_q), max_ma_lags)

        best_metric = default_metric = 9999999
        init_metric = default_metric + 1
        bestModel, best_p, best_q = None, -1, -1

        traversed = set()
        while(True):
            for d_p in [0,-1,1]:
                ar_p = start_p + d_p
                if ar_p < self.autolag_min_p or ar_p > max_auto_lags:
                    continue
                for d_q in ([0,-1,1] if max_ma_lags>0 else [0]):
                    ma_q = start_q + d_q
                    if ma_q < self.autolag_min_q or ma_q > max_ma_lags:
                        continue

                    crntOrder = (ar_p, ma_q)
                    if crntOrder in traversed:
                        continue 
                    traversed.add(crntOrder)

                    #SARIMAX and VARMAX cannot estimate 0-order models
                    if max_ma_lags !=0 and ar_p == 0 and ma_q == 0:
                        continue

                    # lag the exog properly
                    crnt_metric = default_metric
                    with warnings.catch_warnings(record=True) as warningStack:
                        #estimate model, remove leading rows so that AIC/BIC can compare the same time periods of data for all ar_p values
                        estRst = self._estimate_model(endogX[(max_auto_lags-ar_p):,:]
                            , None if exogX is None else exogX[(max_auto_lags-ar_p):,:]
                            , ar_p, ma_q)
                        
                        # test auto-correlation on the residuals
                        if isinstance(estRst.model, VAR):
                            rstWT = estRst.test_whiteness(self.auto_corr_max_lag+ar_p, self.auto_corr_sig)
                            IsAutoCorr = rstWT.pvalue <= rstWT.signif
                        elif isinstance(estRst.model, AutoReg):
                            df_adj_det = 0 if self.trend == 'n' else len(self.trend)
                            exog_df_adj = 0 if exogX is None else exogX.shape[1]
                            ad_adj = df_adj_det+ar_p+ma_q+exog_df_adj
                            rstJB = estRst.test_serial_correlation(lags=ad_adj+self.auto_corr_max_lag, model_df=ad_adj).dropna()
                            num_auto_lags = 1 if rstJB.shape[0]==0 else rstJB.assign(test_pass=lambda df_:df_['LB P-value']<self.auto_corr_sig).query('test_pass==False').shape[0]
                            IsAutoCorr = num_auto_lags != 0
                        else:
                            n_vars = endogX.shape[1]
                            det_df_adj = 0 if self.trend == 'n' else len(self.trend)
                            exog_df_adj = 0 if exogX is None else exogX.shape[1]
                            df_adj = (det_df_adj + (ar_p+ma_q)*n_vars + exog_df_adj)*n_vars + (1+n_vars)*n_vars/2
                            autotest_rst = estRst.test_serial_correlation(lags=df_adj+self.auto_corr_max_lag-1, method='ljungbox', df_adjust=True)
                            p_values = autotest_rst[:,1,:]
                            # if none of the lags + series rejected the null of no auto correlation
                            IsAutoCorr = np.sum(p_values[~np.isnan(p_values)] < self.auto_corr_sig) != 0
                        if IsAutoCorr:
                            # discard this p,q combo
                            crnt_metric = default_metric + 2
                        else:
                            crnt_metric = getattr(estRst, self.autolag) #aic or bic

                    if crnt_metric < best_metric:
                        # choose the current p,q combo
                        best_metric = crnt_metric
                        bestModel = estRst
                        best_p, best_q = ar_p, ma_q

                    if d_p==d_q==0:
                        init_metric = crnt_metric

            # if the initial metric is the best compared to the adjacent neighbors in (p,q) space
            if self.autolag_full_search == False and init_metric <= best_metric:
                break
            
            if ar_p >= max_auto_lags and ma_q >= max_ma_lags:
                break

            if best_metric < default_metric:
                init_metric = best_metric

            # never found a feasible model
            if self.autolag_full_search or bestModel is None:
                start_p, start_q = ar_p, ma_q
            else:
                # start next round with the current round's best p and q
                start_p, start_q = best_p, best_q
                    
        return best_p, best_q