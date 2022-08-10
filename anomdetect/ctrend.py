from msilib.schema import Error
import numpy as np
from collections import deque
import pandas as pd
from scipy.stats import f as fdist
from scipy.interpolate import interp1d
from sklearn.utils.validation import check_array
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.decomposition import PCA
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.nonparametric.kde import KDEUnivariate
from .coint_chigira import ChigiraCointTest
class CommonTrend(BaseEstimator):
    def __init__(self
        , pca_model=None
        , sig_chigira=0.05, n_selected_components=None, spec_chigira='c'
        , n_stationary_components=None
        , spec_var_sty_comp = 'n'
        , max_lag_sty_comp = None, lag_criterion_sty_comp='aic'
        , var_res_dist_sty = None #None, 'norm', 'kde'
        , ci_out_of_control_side_sty = 'greater' #'less', 'two-sided', 'greater'
        , kde_kernel_sty = 'gau', kde_bw_sty='silverman'
        , spec_var_nonsty_comp = 'n'
        , max_lag_nonsty_comp = None, lag_criterion_nonsty_comp='aic'
        , var_res_dist_nonsty = None #None, 'norm', 'kde'
        , ci_out_of_control_side_nonsty = 'greater' #'less', 'two-sided', 'greater'
        , kde_kernel_nonsty = 'gau', kde_bw_nonsty='silverman'
        , var_res_normality_sig = 0.05
    ) -> None:
        self.pca_model=pca_model
        # if using Chigira
        self.sig_chigira=sig_chigira
        self.n_selected_components=n_selected_components
        self.spec_chigira=spec_chigira
        # if using supplied PCA 
        self.n_stationary_components = n_stationary_components
        # VAR parameters for stationary factors
        self.spec_var_sty_comp=spec_var_sty_comp
        self.max_lag_sty_comp=max_lag_sty_comp
        self.lag_criterion_sty_comp=lag_criterion_sty_comp
        self.var_res_dist_sty=var_res_dist_sty
        self.ci_out_of_control_side_sty=ci_out_of_control_side_sty
        self.kde_kernel_sty=kde_kernel_sty
        self.kde_bw_sty=kde_bw_sty
        # VAR parameters for non-stationary factors
        self.spec_var_nonsty_comp=spec_var_nonsty_comp
        self.max_lag_nonsty_comp=max_lag_nonsty_comp
        self.lag_criterion_nonsty_comp=lag_criterion_nonsty_comp
        self.var_res_dist_nonsty=var_res_dist_nonsty
        self.ci_out_of_control_side_nonsty=ci_out_of_control_side_nonsty
        self.kde_kernel_nonsty=kde_kernel_nonsty
        self.kde_bw_nonsty=kde_bw_nonsty

        self.var_res_normality_sig=var_res_normality_sig

        self._IsFitted = False
    
    def fit(self, nonstyX, styX=None):
        IsChigira = self.pca_model is None or isinstance(self.pca_model, ChigiraCointTest)
        
        X_Sty = None if styX is None else check_array(styX)
        if nonstyX is not None:
            X_NSty = check_array(nonstyX)
            if X_NSty.shape[0] != X_Sty.shape[0]:
                raise ValueError('nonstyX and styX input arguments should have the same number of rows.')
        else: 
            X_NSty = None

        if IsChigira: #if using Chigira coint test
            NumStyComp, pcaY, chigira = self._chigira_coint(X_NSty)
            self.ChigiraModel_ = chigira
            self.PCAModel_ = chigira.PCAModel_
            # # remove the first row from the stationary input variables (styX) because Chigira already used the first period among the non-stationary vars for the coint test.
            # X_Sty = X_Sty[1:,:]
        else: #if use supplied PCA model
            pcaY = self._pca_decomposition(X_NSty)
            NumStyComp = self.n_stationary_components
            self.ChigiraModel_ = None
            self.PCAModel_ = self.pca_model

        if NumStyComp == 0:
            raise ValueError('The variables in nonstyX are not cointegrated. Common trend model is not appropriate.')

        self.n_components_kasa_sty_ = NumStyComp
        self.n_components_kasa_nonsty_ = self.PCAModel_.components_.shape[0] - NumStyComp
        self.n_components_sty_ = NumStyComp if X_Sty is None else NumStyComp + X_Sty.shape[1]
        self.n_components_nonsty_ = self.n_components_kasa_nonsty_
        self.n_selected_components_ = self.PCAModel_.n_components_
        self.n_components_ = self.n_components_nonsty_ + self.n_components_sty_

        # perform Kasa decomposition to get the non-stationary and stationary factors of the combined input matrix [nonstyX, X_Sty]
        beta_comp_NSty, beta_Sty, kasaNSty_comp, kasaSty_comp = self._kasa_decomposition(pcaY, X_Sty, self.PCAModel_.components_, NumStyComp)
        self.components_kasa_nonsty_ = beta_comp_NSty
        self.components_kasa_sty_ = beta_Sty

        self._kasaX_nonsty = kasaNSty_comp
        self._kasaX_sty = kasaSty_comp
        
        # find control limit for the non-stationary factors
        if self.n_components_nonsty_ > 0:
            mdlVAR_NSty, covMatInv_NSty, ci_fxX_NSty = self._var_control_limit_est(
                Xinput = kasaNSty_comp
                # VAR specification usually does not have a constant (spec_var_nonsty_comp='n') because principal components are mean-zero and not expected to have trend. 
                , var_spec = self.spec_var_nonsty_comp
                , var_max_lag = self.max_lag_nonsty_comp
                , var_lag_criterion = self.lag_criterion_nonsty_comp
                , dist_assump = self.var_res_dist_nonsty
                , out_of_control_side = self.ci_out_of_control_side_nonsty
                , kernel = self.kde_kernel_nonsty
                , kernel_bw = self.kde_bw_nonsty
            )

            self.VARModel_nonsty_ = mdlVAR_NSty
            self.covMatInv_nonsty_ = covMatInv_NSty
            self.total_lag_nonsty_ = mdlVAR_NSty.k_var
            self._ci_fxn_nonsty = ci_fxX_NSty
        else:
            self.VARModel_nonsty_ = None
            self.covMatInv_nonsty_ = 0
            self.total_lag_nonsty_ = 0
        
        # find control limit for the stationary factors
        mdlVAR_Sty, covMatInv_Sty, ci_fxn_Sty = self._var_control_limit_est(
            Xinput = kasaSty_comp
            # VAR specification should depend on whether the stationary variables should be de-trended.
            , var_spec = self.spec_var_sty_comp
            , var_max_lag = self.max_lag_sty_comp
            , var_lag_criterion = self.lag_criterion_sty_comp
            , dist_assump = self.var_res_dist_sty
            , out_of_control_side = self.ci_out_of_control_side_sty
            , kernel = self.kde_kernel_sty
            , kernel_bw = self.kde_bw_sty
        )

        self.VARModel_sty_ = mdlVAR_Sty
        self.covMatInv_sty_ = covMatInv_Sty
        self.total_lag_sty_ = mdlVAR_Sty.k_ar
        self._ci_fxn_sty = ci_fxn_Sty

        self._IsFitted = True

        return self

    def _chigira_coint(self, X_NSty):
        IsInputModelFitted = getattr(self.pca_model, 'n_selected_components_', None) is not None

        if isinstance(self.sig_chigira, float) == False:
            raise ValueError('The sig_chigira parameter should be given a float value.')
        
        # perform the Chigira cointegration test
        if isinstance(self.pca_model, ChigiraCointTest): # if the model is already fitted
            chigira = self.pca_model
        else: # if we need to fit the model first
            # if isinstance(self.n_selected_components, float) == False and isinstance(self.n_selected_components, int) == False:
            #     raise ValueError('The n_selected_components parameter must be a number of float or int type.')
            # if self.n_selected_components <= 0:
            #     raise ValueError('The n_selected_components parameter (={0}) cannot be a non-positive number.'.format(self.n_selected_components))

            chigira = ChigiraCointTest(n_selected_components=self.n_selected_components)
        
        if IsInputModelFitted:
            NumStyComp = chigira.test(self.sig_chigira)
        else:
            #X_NSty = check_array(X_NSty)
            NumStyComp = chigira.fit_test(X_NSty, self.sig_chigira)
        pcaY = chigira.pcaY_Chigira

        return (NumStyComp, pcaY, chigira)

    def _pca_decomposition(self, X_NSty):
        # X_NSty = check_array(X_NSty)

        if isinstance(self.n_stationary_components, int) == False:
            raise ValueError('n_stationary_components parameter must be an integer.')
        if self.n_stationary_components > X_NSty.shape[1]:
            raise ValueError('n_stationary_components parameter cannot be bigger than the number of features in nonstyX.')

        IsInputModelFitted = getattr(self.pca_model, 'n_components_', None) is not None

        if IsInputModelFitted:
            pcaY = self.pca_model.transform(X_NSty)
        else:
            pcaY = self.pca_model.fit_transform(X_NSty)
        
        if self.n_stationary_components > self.pca_model.n_components_:
            raise ValueError('n_stationary_components (={0}) parameter cannot be bigger than the total number of available principal components (={1}).'.format(self.n_stationary_components, self.pca_model.n_components_))

        return pcaY

    def _kasa_decomposition(self, pcaY, styX, pca_components, n_pca_sty_components):
        n_pca_components = pca_components.shape[0]
        n_pca_nonsty_components = n_pca_components - n_pca_sty_components

        # decompose the variables into stationary factors and non-stationary (common trend) factors
        if styX is not None: # if there are stationary variables
            # find the stationary factors among combined inputs of [pcaY, styX]
            kasaSty_comp = np.concatenate([pcaY[:,-n_pca_sty_components:], styX], axis=1)

            if n_pca_nonsty_components > 0: # if there are non-stationary factors among the cointegrated variables
                # get the coefficients (\Beta_{\perp}) for the non-stationary factors of the variables
                beta_comp_NSty = np.concatenate(
                    [
                        pca_components[0:n_pca_nonsty_components,:].transpose()
                        , np.zeros((styX.shape[1], n_pca_nonsty_components))
                    ]
                    , axis=0
                )
                # nonstationary factors of the combined [pcaY, styX] input matrix
                kasaNSty_comp = pcaY[:,0:n_pca_nonsty_components]
            else: # if there isn't any non-stationary factors among the cointegrated variabels, aka Johansen full rank
                beta_comp_NSty = 0
                # no non-stationary factors of the combined [pcaY, styX] input matrix
                kasaNSty_comp = 0
            
            # get the coefficients (\Beta) for the stationary factors of the variables
            # the (0,0) spot contains the coefficients to make the nonstationary vars stationary. 
            # the (0,1) spot is a zero matrix of size (num of PCA selected factors, num of stationary vars from input)
            # the (1,0) spot is a zero matrix of size (num of stationary vars from inputs, num of stationary factors for the non-stationary inputs)
            # the (1,1) spot is a square identity matrix to ensure the statioanry input vars are not changed. 
            beta_Sty = np.block([
                [pca_components[-n_pca_sty_components:,:].transpose(), np.zeros((n_pca_components, styX.shape[1]))]
                , [np.zeros((styX.shape[1], n_pca_sty_components)), np.identity(styX.shape[1])]
            ])
        else: # if there isn't any stationary input variables provides
            # find the stationary factors among combined inputs of [pcaY, styX]
            # this would be the last few principal components
            kasaSty_comp = pcaY[:,-n_pca_sty_components:]
            
            if n_pca_nonsty_components > 0: # if there are non-stationary factors among the cointegrated variables
                # get the coefficients (\Beta_{\perp}) for the non-stationary factors of the variables
                beta_comp_NSty = pca_components[0:n_pca_nonsty_components,:].transpose()
                # nonstationary factors of the combined [pcaY, styX] input matrix
                kasaNSty_comp = pcaY[:,0:n_pca_nonsty_components]
            else:  # if there isn't non-stationary factors among the cointegrated variabels, aka Johansen full rank
                beta_comp_NSty = 0
                # no non-stationary factors of the combined [pcaY, styX] input matrix
                kasaNSty_comp = 0

            # get the coeffients (\Beta) for the stationary factors of the variables
            beta_Sty = pca_components[-n_pca_sty_components:,:].transpose()

        return (beta_comp_NSty, beta_Sty, kasaNSty_comp, kasaSty_comp)

    def _var_control_limit_est(self, Xinput, var_max_lag=None, var_lag_criterion='aic', var_spec='n', dist_assump=None, out_of_control_side='greater', kernel='gau', kernel_bw='silverman'):
        # estimate the VAR model on the Kasa decomposed data
        mdlVAR = VAR(endog=Xinput).fit(maxlags=var_max_lag, ic=var_lag_criterion, trend=var_spec)
        # grab the residuals
        resid = mdlVAR.resid
        # calculate the variance-covariance matrix of the residuals. need it for T2 stat calculation
        covMatInv = np.linalg.inv(np.cov(resid, rowvar=False))

        if dist_assump is None:
            if isinstance(self.var_res_normality_sig, float) == False:
                raise ValueError('The var_res_normality_sig parameter must be assigned a float value.')
            # Jarque-Bera normality test. null: skewness and kurtosis are jointly 0.
            if mdlVAR.test_normality().pvalue < self.var_res_normality_sig:
                dist_assump='kde'
            else:
                dist_assump='norm'

        # generate the confidence interval function
        if dist_assump == 'norm': # if the residuals follow a normal distribution
            if out_of_control_side == 'less': #'less', 'two-sided'
                ci_fxn = lambda x: resid.shape[1] * (resid.shape[0]-1) * fdist.ppf(x,resid.shape[1],resid.shape[0]-resid.shape[1]) / (resid.shape[0]-resid.shape[1])
            elif out_of_control_side == 'two-sided':
                ci_fxn = lambda x: resid.shape[1] * (resid.shape[0]-1) * fdist.ppf([x/2, 1-x/2],resid.shape[1],resid.shape[0]-resid.shape[1]) / (resid.shape[0]-resid.shape[1])
            else: # 'greater', default
                ci_fxn = lambda x: resid.shape[1] * (resid.shape[0]-1) * fdist.ppf(1-x,resid.shape[1],resid.shape[0]-resid.shape[1]) / (resid.shape[0]-resid.shape[1])
        else: # if residuals are not normally distributed, use KDE
            # calculate the Hotelling's T2 statistic on the training data
            T2stats = np.sum(resid.dot(covMatInv) * resid, axis=1)
            # KDE on the train data T2 stats
            estKDE = KDEUnivariate(T2stats).fit(kernel=kernel, bw=kernel_bw)
            # linearly interpolate the inverse CDF function
            ppf_fxn = interp1d(x=estKDE.cdf, y=estKDE.support)
            # finalize the confidence interval function
            if out_of_control_side == 'less':
                ci_fxn = lambda x: ppf_fxn(x)
            elif out_of_control_side == 'two-sided':
                ci_fxn = lambda x: ppf_fxn([x/2, 1-x/2])
            else: # 'greater', default
                ci_fxn = lambda x: ppf_fxn(1-x)

        return (mdlVAR, covMatInv, ci_fxn)

    def fit_transform(self, nonstyX, styX=None):
        
        self._IsFitted = True

    def fit_test(self, nonstyX, testX, styX=None):
        
        self._IsFitted = True

    def transform(self, nonstyX, styX=None):
        if not self._IsFitted:
            raise NotFittedError('Please fit this instance of the CommonTrend class first.')

    def test(self, testX):
        if not self._IsFitted:
            raise NotFittedError('Please fit this instance of the CommonTrend class first.')

        



