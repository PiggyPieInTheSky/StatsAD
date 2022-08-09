import numpy as np
from collections import deque
import pandas as pd
from sklearn.utils.validation import check_array
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.decomposition import PCA
from statsmodels.tsa.vector_ar.var_model import VAR
from .coint_chigira import ChigiraCointTest

class CommonTrend(BaseEstimator):
    def __init__(self
        , pca_model=None
        , sig_chigira=0.05, n_selected_components=None, spec_chigira='c'
        , n_stationary_components=None
        , max_lag_sty_comp = None, lag_criterion_sty_comp='aic'
        , spec_var_sty_comp = 'n'
        , var_res_dist_sty = None #None, 'norm', 'kde'
        , max_lag_nonsty_comp = None, lag_criterion_nonsty_comp='aic'
        , spec_var_nonsty_comp = 'n'
        , var_res_dist_nonsty = None #None, 'norm', 'kde'
    ) -> None:
        self.pca_model=pca_model
        # if using Chigira
        self.sig_chigira=sig_chigira
        self.n_selected_components=n_selected_components
        self.spec_chigira=spec_chigira
        # if using supplied PCA 
        self.n_stationary_components = n_stationary_components
        # VAR parameters for stationary components
        self.max_lag_sty_comp=max_lag_sty_comp
        self.lag_criterion_sty_comp=lag_criterion_sty_comp
        self.spec_var_sty_comp=spec_var_sty_comp
        self.var_res_dist_sty=var_res_dist_sty
        # VAR parameters for non-stationary components
        self.max_lag_nonsty_comp=max_lag_nonsty_comp
        self.lag_criterion_nonsty_comp=lag_criterion_nonsty_comp
        self.spec_var_nonsty_comp=spec_var_nonsty_comp
        self.var_res_dist_nonsty=var_res_dist_nonsty

        self._IsFitted = False
    
    def fit(self, nonstyX, styX=None):
        IsChigira = self.pca_model is None or isinstance(self.pca_model, ChigiraCointTest)
        
        X_Sty = None if styX is None else check_array(styX)

        if IsChigira: #if using Chigira coint test
            NumStyComp, pcaY, chigira = self._chigira_coint(nonstyX)
            self.ChigiraModel_ = chigira
            self.PCAModel_ = chigira.PCAModel_
            # remove the first row from the stationary input variables (styX) because Chigira already used the first period among the non-stationary vars for the coint test.
            X_Sty = X_Sty[1:,:]
        else: #if use supplied PCA model
            pcaY = self._pca_decomposition(nonstyX)
            NumStyComp = self.n_stationary_components
            self.ChigiraModel_ = None
            self.PCAModel_ = self.pca_model

        if NumStyComp == 0:
            raise ValueError('The variables in nonstyX are not cointegrated. Common trend model is not appropriate.')

        # perform Kasa decomposition to get the non-stationary and stationary components of the combined input matrix [nonstyX, X_Sty]
        beta_comp_NSty, beta_Sty, kasaNSty_comp, kasaSty_comp = self._kasa_decomposition(pcaY, X_Sty, self.PCAModel_.components_, NumStyComp)

        n_pca_nonsty_components = self.PCAModel_.components_.shape[0] - NumStyComp
        # estimate variance covariance matrix after fitting VAR on the non-stationary components
        if n_pca_nonsty_components > 0:
            # VAR specification usually does not have a constant (spec_var_nonsty_comp='n') because principal components are mean-zero and not expected to have trend. 
            mdlVAR_NSty = VAR(endog=kasaNSty_comp).fit(maxlags=self.max_lag_nonsty_comp, ic=self.lag_criterion_nonsty_comp, trend=self.spec_var_nonsty_comp)
            covMatInv_NSty = np.linalg.inv(np.cov(mdlVAR_NSty.resid, rowvar=False))

            #TODO: calculate based on self.var_res_dist_sty
            #==============================================
            #==============================================
            #==============================================
            T2Stats_NSty = np.sum(mdlVAR_NSty.resid.dot(covMatInv_NSty) * mdlVAR_NSty.resid, axis=1)
            #==============================================
            #==============================================
            #==============================================

            self.Total_Lag_NSty_ = mdlVAR_NSty.k_var + IsChigira
        else:
            mdlVAR_NSty = None
            covMatInv_NSty = 0
            self.Total_Lag_NSty_ = 0

        self.VARModel_NSty_ = mdlVAR_NSty
        

        # estimate variance covariance matrix after fitting VAR on the stationary components
        # VAR specification should depend on whether the stationary variables should be de-trended.
        mdlVAR_Sty = VAR(endog=kasaSty_comp).fit(maxlags=self.max_lag_sty_comp, ic=self.lag_criterion_sty_comp, trend=self.spec_var_sty_comp)
        covMatInv_Sty = np.linalg.inv(np.cov(mdlVAR_Sty.resid, rowvar=False))


        T2Stats_Sty = np.sum(mdlVAR_Sty.resid.dot(covMatInv_Sty) * mdlVAR_Sty.resid, axis=1)



        self._IsFitted = True

    def _chigira_coint(self, X_NSty):
        IsInputModelFitted = getattr(self.pca_model, 'n_selected_components_', None) is not None

        if isinstance(self.sig_chigira, float) == False:
            raise ValueError('The sig_chigira parameter should be given a float value.')
        
        # perform the Chigira cointegration test
        if isinstance(self.pca_model, ChigiraCointTest): # if the model is already fitted
            chigira = self.pca_model()
        else: # if we need to fit the model first
            if isinstance(self.n_selected_components, float) == False and isinstance(self.n_selected_components, int) == False:
                raise ValueError('The n_selected_components parameter must be a number of float or int type.')
            if self.n_selected_components <= 0:
                raise ValueError('The n_selected_components parameter (={0}) cannot be a non-positive number.'.format(self.n_selected_components))
            
            chigira = ChigiraCointTest(n_selected_components=self.n_selected_components)
        
        if IsInputModelFitted:
            NumStyComp = chigira.test(self.sig_chigira)
        else:
            X_NSty = check_array(X_NSty)
            NumStyComp = chigira.fit_test(X_NSty, self.sig_chigira)
        pcaY = chigira.pcaY_Chigira

        return (NumStyComp, pcaY, chigira)

    def _pca_decomposition(self, X_NSty):
        X_NSty = check_array(X_NSty)

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

        # decompose the variables into stationary components and non-stationary (common trend) components
        if styX is not None: # if there are stationary variables
            # find the stationary components among combined inputs of [pcaY, styX]
            kasaSty_comp = np.concatenate([pcaY[:,-n_pca_sty_components:], styX], axis=1)

            if n_pca_nonsty_components > 0: # if there are non-stationary components among the cointegrated variables
                # get the coefficients (\Beta_{\perp}) for the non-stationary components of the variables
                beta_comp_NSty = np.concatenate(
                    [
                        pca_components[0:n_pca_nonsty_components,:].transpose()
                        , np.zeros((styX.shape[1], n_pca_nonsty_components))
                    ]
                    , axis=0
                )

                # get the coefficients (\Beta) for the stationary components of the variables
                # the (0,0) spot contains the coefficients to make the nonstationary vars stationary. 
                # the (0,1) spot is a zero matrix of size (num of PCA selected components, num of stationary vars from input)
                # the (1,0) spot is a zero matrix of size (num of stationary vars from inputs, num of stationary components for the non-stationary inputs)
                # the (1,1) spot is a square identity matrix to ensure the statioanry input vars are not changed. 
                beta_Sty = np.block([
                    [pca_components[-n_pca_sty_components:,:].transpose(), np.zeros((n_pca_components, styX.shape[1]))]
                    , [np.zeros((styX.shape[1], n_pca_sty_components)), np.identity(styX.shape[1])]
                ])

                # nonstationary components of the combined [pcaY, styX] input matrix
                kasaNSty_comp = pcaY[:,0:n_pca_nonsty_components]
            else: # if there isn't any non-stationary components among the cointegrated variabels, aka Johansen full rank
                beta_comp_NSty = 0
                # get the coeffients (\Beta) for the stationary components of the variables
                beta_Sty = pca_components[-n_pca_sty_components:,:].transpose()
                # no non-stationary components of the combined [pcaY, styX] input matrix
                kasaNSty_comp = 0
        else: # if there isn't any stationary input variables provides
            # find the stationary components among combined inputs of [pcaY, styX]
            # this would be the last few principal components
            kasaSty_comp = pcaY[:,-n_pca_sty_components:]
            
            if n_pca_nonsty_components > 0: # if there are non-stationary components among the cointegrated variables
                # get the coefficients (\Beta_{\perp}) for the non-stationary components of the variables
                beta_comp_NSty = pca_components[0:n_pca_nonsty_components,:].transpose()
                # nonstationary components of the combined [pcaY, styX] input matrix
                kasaNSty_comp = pcaY[:,0:n_pca_nonsty_components]
            else:  # if there isn't non-stationary components among the cointegrated variabels, aka Johansen full rank
                beta_comp_NSty = 0
                # no non-stationary components of the combined [pcaY, styX] input matrix
                kasaNSty_comp = 0

            # get the coeffients (\Beta) for the stationary components of the variables
            beta_Sty = pca_components[-n_pca_sty_components:,:].transpose()

        return (beta_comp_NSty, beta_Sty, kasaNSty_comp, kasaSty_comp)

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

        



