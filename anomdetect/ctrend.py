import numpy as np
from collections import deque
import pandas as pd
from scipy.stats import f as fdist
from scipy.interpolate import interp1d
from sklearn.utils.validation import check_array
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
#from sklearn.decomposition import PCA
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.ar_model import ar_select_order
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.nonparametric.kde import KDEUnivariate

from .utils._internal import prepare_parameters_VAR, forecast_var_one_step_ahead
from .utils._oos_predict import __var_oos_predict as var_oos_predict


#TODO: check PCA whiening effect on predicting original X with principal components
#TODO: check if other solvers of PCA produces orthogonal length 1 PCs
class CommonTrend(BaseEstimator):
    def __init__(self
        , pca_model=None
        , sig_chigira=0.05, n_selected_components=None, spec_chigira='c'
        , n_stationary_components=None
        , spec_var_sty_factor = 'n'
        , max_lag_sty_factor = None, lag_criterion_sty_factor='aic'
        , var_res_dist_sty = None #None, 'norm', 'kde'
        , ci_out_of_control_side_sty = 'greater' #'less', 'two-sided', 'greater'
        , kde_kernel_sty = 'gau', kde_bw_sty='silverman'
        , spec_var_nonsty_factor = 'n'
        , max_lag_nonsty_factor = None, lag_criterion_nonsty_factor='aic'
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
        # Parameters for stationary factors
        self.spec_var_sty_factor=spec_var_sty_factor
        self.max_lag_sty_factor=max_lag_sty_factor
        self.lag_criterion_sty_factor=lag_criterion_sty_factor
        self.var_res_dist_sty=var_res_dist_sty
        self.ci_out_of_control_side_sty=ci_out_of_control_side_sty
        self.kde_kernel_sty=kde_kernel_sty
        self.kde_bw_sty=kde_bw_sty
        # Parameters for non-stationary factors
        self.spec_var_nonsty_factor=spec_var_nonsty_factor
        self.max_lag_nonsty_factor=max_lag_nonsty_factor
        self.lag_criterion_nonsty_factor=lag_criterion_nonsty_factor
        self.var_res_dist_nonsty=var_res_dist_nonsty
        self.ci_out_of_control_side_nonsty=ci_out_of_control_side_nonsty
        self.kde_kernel_nonsty=kde_kernel_nonsty
        self.kde_bw_nonsty=kde_bw_nonsty

        self.var_res_normality_sig=var_res_normality_sig

        self._IsFitted = False
    
    def fit(self, nonstyX, styX=None):
        from .coint_chigira import ChigiraCointTest
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
        else: #if use supplied PCA model
            pcaY = self._pca_decomposition(X_NSty)
            NumStyComp = self.n_stationary_components
            self.ChigiraModel_ = None
            self.PCAModel_ = self.pca_model

        if NumStyComp == 0:
            raise ValueError('The variables in nonstyX are not cointegrated. Common trend model is not appropriate.')

        self.n_features_nonsty = self.PCAModel_.n_features_
        self.n_features_sty = 0 if X_Sty is None else X_Sty.shape[1]
        self.n_features_ = self.n_features_nonsty + self.n_features_sty

        self.n_components_nonsty_sty_ = NumStyComp
        self.n_components_nonsty_nonsty_ = self.PCAModel_.components_.shape[0] - NumStyComp
        self.n_components_sty_ = NumStyComp if X_Sty is None else NumStyComp + X_Sty.shape[1]
        self.n_components_nonsty_ = self.n_components_nonsty_nonsty_
        self.n_selected_components_ = self.PCAModel_.n_components_
        self.n_components_ = self.n_components_nonsty_ + self.n_components_sty_

        # perform Kasa decomposition to get the non-stationary and stationary factors of the combined input matrix [nonstyX, X_Sty]
        beta_comp_NSty, beta_Sty, kasaNSty_factor, kasaSty_factor = CommonTrend.kasa_decomposition(pcaY, self.PCAModel_.components_, NumStyComp, X_Sty)
        self.kasa_nonsty_loading_ = beta_comp_NSty # common trend loading matrix
        self.kasa_sty_loading_ = beta_Sty # stationary loading matrix

        self._kasaX_nonsty = kasaNSty_factor
        self._kasaX_sty = kasaSty_factor
        
        # find control limit for the non-stationary factors
        if self.n_components_nonsty_ > 0:
            mdlVAR_NSty, covMatInv_NSty, ci_fxX_NSty = self._var_control_limit_est(
                Xinput = kasaNSty_factor
                , IsI1 = True
                # VAR specification usually does not have a constant (spec_var_nonsty_factor='n') because principal components are mean-zero and not expected to have trend. 
                , var_spec = self.spec_var_nonsty_factor
                , var_max_lag = self.max_lag_nonsty_factor
                , var_lag_criterion = self.lag_criterion_nonsty_factor
                , dist_assump = self.var_res_dist_nonsty
                , out_of_control_side = self.ci_out_of_control_side_nonsty
                , kernel = self.kde_kernel_nonsty
                , kernel_bw = self.kde_bw_nonsty
            )

            self.VARModel_nonsty_ = mdlVAR_NSty
            self.covMatInv_nonsty_ = covMatInv_NSty
            # non-stationary vars are first-differenced before estimated with VAR
            # thus, the extra 1 lag
            self.total_lag_nonsty_ = mdlVAR_NSty.__ct_max_lag__ + 1
            self._ci_fxn_nonsty = ci_fxX_NSty
        else:
            self.VARModel_nonsty_ = None
            self.covMatInv_nonsty_ = 0
            self.total_lag_nonsty_ = 0
        
        # find control limit for the stationary factors
        mdlVAR_Sty, covMatInv_Sty, ci_fxn_Sty = self._var_control_limit_est(
            Xinput = kasaSty_factor
            , IsI1 = False
            # VAR specification should depend on whether the stationary variables should be de-trended.
            , var_spec = self.spec_var_sty_factor
            , var_max_lag = self.max_lag_sty_factor
            , var_lag_criterion = self.lag_criterion_sty_factor
            , dist_assump = self.var_res_dist_sty
            , out_of_control_side = self.ci_out_of_control_side_sty
            , kernel = self.kde_kernel_sty
            , kernel_bw = self.kde_bw_sty
        )

        self.VARModel_sty_ = mdlVAR_Sty
        self.covMatInv_sty_ = covMatInv_Sty
        self.total_lag_sty_ = mdlVAR_Sty.__ct_max_lag__
        self._ci_fxn_sty = ci_fxn_Sty

        self._IsFitted = True

        return self

    def _chigira_coint(self, X_NSty):
        from .coint_chigira import ChigiraCointTest
        IsInputModelFitted = getattr(self.pca_model, 'n_selected_components_', None) is not None

        if isinstance(self.sig_chigira, float) == False:
            raise ValueError('The sig_chigira parameter should be given a float value.')
        
        # perform the Chigira cointegration test
        if isinstance(self.pca_model, ChigiraCointTest): # PCA comes from Chigira cointegration test
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

    @staticmethod
    def kasa_decomposition(pcaY: np.ndarray, pca_eigenvec: np.ndarray, n_pca_sty_components: int, styX=None):
        """Performs augmented Kasa (1992) decomposition of the principal components of cointegrated I(1) features.
        
        Parameters
        ==========
            pcaY:                   an n-by-m 2d numpy array that represents the selected top PCA components. n is number of time periods / data observations. m is the number of selected PCA components. Note, this is the transformed X value if using sklearn.decomposiiton.PCA.

            pca_eigenvec:           an m-by-l 2d numpy array that denotes the PCA eigenvectors (factor loading matrix). m is the number of selected PCA components. l is the total number of original features. Note this is the PCA.component_ attribute if from sklearn.

            n_pca_sty_components:   an int >0. Number of stationary principal components. This is the r value from Johansen / Chigira type cointegration test.

            styX:                   an optional 2d numpy array that has extra exogenous stationary time series that can be appended to the stationary part of the results. The number of rows must match that of pcaY.

        Return
        ======
            A 4-element tuple with the following elements:
            beta_comp_NSty:     a 2d numpy array that contains the coefficients applied to the original features to produce the principal components components for the (non-stationary) common trend components

            beta_Sty:           a 2d numpy array that contains the coefficients applied to the original features to produce the principal components components for the stationary components. If extra stationary features are supplied to styX, the original features are aligned as [I(1) features, styX].
            
            kasaNSty_factor:    an n by (m-n_pca_sty_components) 2d numpy array that contains the (non-stationary) principal components for the common trend components. This contains the top m-r principal components where r (=n_pca_sty_components) is the number of cointegration vectors.
            
            kasaSty_factor:     an n by n_pca_sty_components 2d numpy array that contains the principal components for the stationary components. If styX is supplied, the column layout follows [the stationary PCs, styX].

            Note: suppose the *detrended* oringal features are X = [nStyX, styX] where nStyX is the I(1) features that generated pcaY. The following should hold: 1) X.dot(beta_comp_NSty) := kasaNSty_factor, and 2) X.dot(beta_Sty) := kasaSty_factor.

        Reference
        =========
        Kasa, K. (1992). Common stochastic trends in international stock markets. Journal of monetary Economics, 29(1), 95-124.

        Lin, Y., Kruger, U., Gu, F., Ball, A., & Chen, Q. (2019). Monitoring nonstationary and dynamic trends for practical process fault diagnosis. Control Engineering Practice, 84, 139-158.
        """
        if pcaY.shape[1] != pca_eigenvec.shape[0]:
            #TODO: add error message
            raise ValueError("")
        if n_pca_sty_components > pca_eigenvec.shape[0]:
            #TODO: add error message
            raise ValueError("")
        if styX is not None and styX.shape[0] != pcaY.shape[0]:
            raise ValueError("styX input argument has {0} rows and is inconsistent with pcaY that has {1} rows.".format(styX.shape[0], pcaY[0]))

        n_pca_eigenvec = pca_eigenvec.shape[0]
        n_pca_nonsty_components = n_pca_eigenvec - n_pca_sty_components

        # decompose the variables into stationary factors and non-stationary (common trend) factors
        if styX is not None: # if there are stationary variables
            # find the stationary factors among combined inputs of [pcaY, styX]
            kasaSty_factor = np.concatenate([pcaY[:,-n_pca_sty_components:], styX], axis=1)

            if n_pca_nonsty_components > 0: # if there are non-stationary factors among the cointegrated variables
                # get the coefficients (\Beta_{\perp}) for the non-stationary factors of the variables
                beta_comp_NSty = np.concatenate(
                    [
                        pca_eigenvec[0:n_pca_nonsty_components,:].transpose()
                        , np.zeros((styX.shape[1], n_pca_nonsty_components))
                    ]
                    , axis=0
                )
                # nonstationary factors of the combined [pcaY, styX] input matrix
                kasaNSty_factor = pcaY[:,0:n_pca_nonsty_components]
            else: # if there isn't any non-stationary factors among the cointegrated variabels, aka Johansen full rank
                beta_comp_NSty = 0
                # no non-stationary factors of the combined [pcaY, styX] input matrix
                kasaNSty_factor = 0
            
            # get the coefficients (\Beta) for the stationary factors of the variables
            # the (0,0) spot contains the coefficients to make the nonstationary vars stationary. 
            # the (0,1) spot is a zero matrix of size (num of PCA selected factors, num of stationary vars from input)
            # the (1,0) spot is a zero matrix of size (num of stationary vars from inputs, num of stationary factors for the non-stationary inputs)
            # the (1,1) spot is a square identity matrix to ensure the statioanry input vars are not changed. 
            beta_Sty = np.block([
                [pca_eigenvec[-n_pca_sty_components:,:].transpose(), np.zeros((n_pca_eigenvec, styX.shape[1]))]
                , [np.zeros((styX.shape[1], n_pca_sty_components)), np.identity(styX.shape[1])]
            ])
        else: # if there isn't any stationary input variables provides
            # find the stationary factors among combined inputs of [pcaY, styX]
            # this would be the last few principal components
            kasaSty_factor = pcaY[:,-n_pca_sty_components:]
            
            if n_pca_nonsty_components > 0: # if there are non-stationary factors among the cointegrated variables
                # get the coefficients (\Beta_{\perp}) for the non-stationary factors of the variables
                beta_comp_NSty = pca_eigenvec[0:n_pca_nonsty_components,:].transpose()
                # nonstationary factors of the combined [pcaY, styX] input matrix
                kasaNSty_factor = pcaY[:,0:n_pca_nonsty_components]
            else:  # if there isn't non-stationary factors among the cointegrated variabels, aka Johansen full rank
                beta_comp_NSty = 0
                # no non-stationary factors of the combined [pcaY, styX] input matrix
                kasaNSty_factor = 0

            # get the coeffients (\Beta) for the stationary factors of the variables
            beta_Sty = pca_eigenvec[-n_pca_sty_components:,:].transpose()

        return (beta_comp_NSty, beta_Sty, kasaNSty_factor, kasaSty_factor)

    def _var_control_limit_est(self, Xinput, IsI1=False, var_max_lag=None, var_lag_criterion='aic', var_spec='n', dist_assump=None, out_of_control_side='greater', kernel='gau', kernel_bw='silverman'):
        if IsI1: # if I(1)
            # make the first difference
            Xinput = Xinput[1:,:] - Xinput[0:-1,:]
        
        if Xinput.shape[1] > 1: # if more than one variable in Xinput
            # estimate the VAR model on the Kasa decomposed data
            mdlVAR = VAR(endog=Xinput).fit(maxlags=var_max_lag, ic=var_lag_criterion, trend=var_spec)
            resid = mdlVAR.resid
        else: # if only one variable in Xinput
            if var_max_lag is None:
                var_max_lag = round(12 * (Xinput.shape[0]/100.)**(1./4))
            ar_lag_select_rst = ar_select_order(endog=Xinput, maxlag=var_max_lag, ic=var_lag_criterion, trend='n')
            ar_max_lag = 0 if ar_lag_select_rst.ar_lags is None else ar_lag_select_rst.ar_lags[-1]
            #ar_max_lag = 1
            mdlVAR = AutoReg(endog=Xinput, lags=ar_max_lag, trend=var_spec).fit()
            resid = mdlVAR.resid.reshape(-1,1)

        mdlVAR.__ct_ar_last_t__ = Xinput.shape[0]
        mdlVAR = prepare_parameters_VAR(mdlVAR)
        
        # calculate the variance-covariance matrix of the residuals. need it for T2 stat calculation
        if Xinput.shape[1] > 1:
            covMatInv = np.linalg.inv(np.cov(resid, rowvar=False, ddof=1))
        else:
            covMatInv = 1 / np.var(resid, ddof=1)

        if dist_assump is None:
            if isinstance(self.var_res_normality_sig, float) == False:
                raise ValueError('The var_res_normality_sig parameter must be assigned a float value.')
            # Jarque-Bera style normality test. null: skewness and kurtosis are jointly 0.
            normality_pvalue = mdlVAR.test_normality().pvalue if Xinput.shape[1] > 1 else jarque_bera(resid)[1]
            if normality_pvalue < self.var_res_normality_sig:
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
            mdlVAR.__ct_kde__ = estKDE

        return (mdlVAR, covMatInv, ci_fxn)

    def fit_transform(self, nonstyX, styX=None):
        """Fit the model, then returns the principal components based on Kasa decomposition. 

        Parameters
        ==========
        nonstyX:    an n by l 2d array that contains the data for I(1) features where n is the number of rows / time periods; l is the total number of I(1) features

        styX:       an optional n by k 2d array that contains the extra / exogenous stationary variables.
        
        Return
        ======
        an n by (m+k) 2d numpy array that contains principal components where m is the number of selected PCA components from the I(1) input data and equals the n_selected_components_ attribute. The column layout is as the following: n_components_nonsty_nonsty_ number of principal components for the stochastic common trend, n_components_nonsty_sty_ number of principal components for the stationary components, and n_features_sty number of extra stationary features (if provided).
        """
        self.fit(nonstyX=nonstyX, styX=styX)
        return np.concatenate([self._kasaX_nonsty, self._kasaX_sty], axis=1)

    def fit_predict(self, nonstyX: np.ndarray, styX=None):
        """Computes essential components for decomposition and testing, then decomposes I(1) data into stochastic common trend components and stationary components based on Kasa (1992).
        
        Parameters
        ==========
        nonstyX:    an n by l 2d array that contains the data for I(1) features where n is the number of rows / time periods; l is the total number of I(1) features

        styX:       an optional n by k 2d array that contains the extra / exogenous stationary variables.
        
        Return
        ======
        Znonsty:    an n by (l+k) 2d numpy array of non-stationary components of the augmented input data matrix [nonstyX, styX]. The first l columns represent the stochastic common trend components of the origal I(1) data nonstyX. The rest of the columns should all be zero.
        
        Zsty:       an n by (l+k) 2d numpy array that represents the stationary components of the augmented input data. The first l columns represents the stationary components of the original I(1) data nonstyX. The rest of the columns, if k != 0, are the verbatim copies of styX.
        
        Zdet:       an n by (l+k) 2d numpy array of the deterministic terms. It always includes averages of the original features. When Chigira method is used during fit, the time trend, if specified, is also included in this variable.

        Note: Znonsty + Zsty + Zdet should equal to the augmented input data matrix [nonstyX, styX].

        Reference
        =========
        Kasa, K. (1992). Common stochastic trends in international stock markets. Journal of monetary Economics, 29(1), 95-124.

        Lin, Y., Kruger, U., Gu, F., Ball, A., & Chen, Q. (2019). Monitoring nonstationary and dynamic trends for practical process fault diagnosis. Control Engineering Practice, 84, 139-158.
        """
        # should return the converted X explained by PCs: X*beta*beta'
        self.fit(nonstyX=nonstyX, styX=styX)

        nonsty_means = self.PCAModel_.mean_.reshape(1,-1)
        if getattr(self, 'ChigiraModel_', None) is not None:
            # Chigira de-means and de-trends the features with a VAR model of lag 0. 
            # We need to find the trended values
            nonsty_means = self.ChigiraModel_.VARModel_.fittedvalues + nonsty_means
        Zdet_nonsty = np.concatenate([nonsty_means, np.zeros((nonsty_means.shape[0], self.n_features_ - nonsty_means.shape[1]))], axis=1)

        # weight given to the tread / mean values for the common trends based on explained variance ratio
        # Chigira detrends the non-stationary features while PCA demeans them. 
        # There isn't a unique way of distributing the mean to the common trend and stationary parts of the cointegrated data
        # here, we simply use the variance ratio. 
        # Note that the PCA model may not report all components if PCA.n_components is not equal to PCA.n_features.
        # Thus we need to recalculate the explained variance ratio.
        # var_ratio = np.sum(self.PCAModel_.explained_variance_[0:self.n_components_nonsty_]) / np.sum(self.PCAModel_.explained_variance_)

        # common trend component ======================
        if self.n_components_nonsty_!= 0: # if there is common trend
            # use the non-stationary principal components to predict the original I(1) features
            Xct_detrended = self._kasaX_nonsty.dot(self.PCAModel_.components_[0:self.n_components_nonsty_nonsty_,:])
            # distribute the mean values for the non-stationary (common trend) principal components
            # Xct_trend = nonsty_means * var_ratio
            if self.n_features_nonsty > 0: # if extra stationary features were given
                # pad extra zeros in the columns for the extra stationary features
                Znonsty = np.concatenate([
                    Xct_detrended #+ Xct_trend
                    , np.zeros((self._kasaX_nonsty.shape[0], self.n_features_nonsty))
                    ], axis=1
                )
            else:
                Znonsty = Xct_detrended #+ Xct_trend
        else:
            Znonsty = 0

        # stationary component ======================
        # use stationary PCs to predict the original I(1) features
        Xsty_detrended = self._kasaX_sty[:,0:self.n_components_nonsty_sty_].dot(self.PCAModel_.components_[self.n_components_nonsty_nonsty_:,:])
        # distribute the mean values for the stationary principal components
        # Xsty_trend = nonsty_means * (1-var_ratio)
        if self.n_features_nonsty > 0: # if extra stationary features were given
            # add the extra stationary features
            Zsty = np.concatenate([
                    Xsty_detrended #+ Xsty_trend
                    , self._kasaX_sty[:,self.n_components_nonsty_sty_:]
                    ], axis=1
                )
        else:
            Zsty = Xsty_detrended #+ Xsty_trend

        # note that the sum of the below should recover the original data: [nonstyX, styX]
        return Znonsty, Zsty, Zdet_nonsty

    def fit_test(self, nonstyX, styX=None, sig=0.05):
        self.fit(nonstyX=nonstyX, styX=styX)

        if self.VARModel_nonsty_.resid.ndim == 1:
            resid = self.VARModel_nonsty_.resid.reshape(-1,1)
        else:
            resid = self.VARModel_nonsty_.resid
        # get the test results for the nonstationary factors
        T2_nonsty, violation_nonsty = self._violation_check(
            resid, self.covMatInv_nonsty_
            , sig
            , self.ci_out_of_control_side_nonsty, self._ci_fxn_nonsty
        )

        if self.VARModel_sty_.resid.ndim == 1:
            resid = self.VARModel_sty_.resid.reshape(-1,1)
        else:
            resid = self.VARModel_sty_.resid
        # get the test results for the stationary factors
        T2_sty, violation_sty = self._violation_check(
            resid, self.covMatInv_sty_
            , sig
            , self.ci_out_of_control_side_sty, self._ci_fxn_sty
        )

        return {
            'T2_nonsty': T2_nonsty
            , 'T2_sty': T2_sty
            , 'control_limit_nonsty': self._ci_fxn_nonsty(sig)
            , 'control_limit_sty': self._ci_fxn_sty(sig)
            , 'violation_nonsty': violation_nonsty
            , 'violation_sty': violation_sty
        }

    def transform(self, nonstyX, styX=None, start_step=1):
        """Transforms data into principal components after the model is fitted / Kasa decomposition.

        Parameters
        ==========
        nonstyX:    an n by l 2d array that contains the data for I(1) features where n is the number of rows / time periods; l is the total number of I(1) features

        styX:       an optional n by k 2d array that contains the extra / exogenous stationary variables.
        
        Return
        ======
        an n by (m+k) 2d numpy array that contains principal components where m is the number of selected PCA components from the I(1) input data and equals the n_selected_components_ attribute. The column layout is as the following: n_components_nonsty_nonsty_ number of principal components for the stochastic common trend, n_components_nonsty_sty_ number of principal components for the stationary components, and n_features_sty number of extra stationary features (if provided).
        """
        if not self._IsFitted:
            raise NotFittedError('Please fit this instance of the CommonTrend class first.')
        
        if nonstyX.shape[1] != self.n_features_nonsty:
            raise ValueError("nonstyX has {0} columns but should have {1} instead to match with the I(1) training data".format(nonstyX.shape[0], self.n_features_nonsty))
        if self.n_features_sty != 0: 
            if styX is None:
                raise ValueError("styX is required since it is given during training.")
            elif styX.shape[1] != self.n_features_sty:
                raise ValueError("styX has {0} columns but should have {1} instead to match with the I(1) training data".format(styX.shape[0], self.n_features_sty))
            elif styX.shape[0] != nonstyX.shape[0]:
                raise ValueError(f"nonstyX has {nonstyX.shape[0]} rows and does not match the number of rows (={styX.shape[0]}) in styX.")
            else:
                styX = check_array(styX)

        nonstyX = check_array(nonstyX)
        
        # find the means / trend
        nonsty_means = self.PCAModel_.mean_.reshape(1,-1)
        if getattr(self, 'ChigiraModel_', None) is not None:
            # Chigira de-means and de-trends the features with a VAR model of lag 0. 
            # We need to find the trended values
            nonsty_means = var_oos_predict(self.ChigiraModel_.VARModel_, start_step=start_step, fcst_steps=nonstyX.shape[0]) + nonsty_means
            nonstyX_detrended = nonstyX - nonsty_means
        else:
            nonstyX_detrended = nonstyX

        nonstyX_pcaY = self.PCAModel_.transform(nonstyX_detrended)

        if styX is None:
            return nonstyX_pcaY
        
        return np.concatenate([nonstyX_pcaY, styX], axis=1)


    def test(self, X_test, sig=0.05, lagged_X_test=None, start_period=1):
        if not self._IsFitted:
            raise NotFittedError('Please fit this instance of the CommonTrend class first.')
        
        if X_test is None:
            return None
        else:
            X_test = check_array(X_test)

        data_lag_len = np.max([self.total_lag_nonsty_, self.total_lag_sty_])
        if lagged_X_test is None:
            if X_test.shape[0] <= data_lag_len:
                raise ValueError('X_test must have more than {0} rows due to the use of lags.'.format(data_lag_len))
        else:
            if lagged_X_test.shape[0] < data_lag_len:
                raise ValueError('lagged_X_test must have more than {0} rows due to the use of lags.'.format(data_lag_len))
            elif lagged_X_test.shape[0] > data_lag_len:
                # ignore the extra rows for the early periods
                lagged_X_test = lagged_X_test[-data_lag_len:,:]
            
            # add the lagged periods to the top of the test data
            X_test_combined = np.concatenate([lagged_X_test, X_test], axis=0)

        # detrend nonstationary vars based on the estimated VAR model
        Xnonsty_trend = forecast_var_one_step_ahead(X_test_combined[:,0:self.n_features_nonsty], self.ChigiraModel_.VARModel_, start_period-data_lag_len)
        Xnonsty_detrend = X_test_combined[:,0:self.n_features_nonsty] - Xnonsty_trend
        X_test_combined = np.concatenate([Xnonsty_detrend, X_test_combined[:,self.n_features_nonsty:]], axis=1)
        
        Xnonsty_detrend.dot(self.ChigiraModel_.PCAModel_.components_.transpose())

        # decompose to the non-stationary factors
        nonstyFactor = X_test_combined.dot(self.kasa_nonsty_loading_)[(data_lag_len-self.total_lag_nonsty_):,:]
        # first-difference the non-stationary factors
        d_nonstyFactor = nonstyFactor[1:,:] - nonstyFactor[0:-1,:]
        # get the test results for the non-stationary factors
        T2_nonsty, violation_nonsty = self._factor_2_teststat(
            d_nonstyFactor, self.VARModel_nonsty_, start_period
            , sig, self.covMatInv_nonsty_
            , self.ci_out_of_control_side_nonsty, self._ci_fxn_nonsty
        )

        # decompose to stationary factors
        styFactor = X_test_combined.dot(self.kasa_sty_loading_)[(data_lag_len-self.total_lag_sty_):,:]
        # get the test results for the stationary factors
        T2_sty, violation_sty = self._factor_2_teststat(
            styFactor, self.VARModel_sty_, start_period
            , sig, self.covMatInv_sty_
            , self.ci_out_of_control_side_sty, self._ci_fxn_sty
        )

        return {
            'T2_nonsty': T2_nonsty
            , 'T2_sty': T2_sty
            , 'control_limit_nonsty': self._ci_fxn_nonsty(sig)
            , 'control_limit_sty': self._ci_fxn_sty(sig)
            , 'violation_nonsty': violation_nonsty
            , 'violation_sty': violation_sty
        }

    def predict(self, nonstyX, styX=None, start_step:int=1):
        """After the model is trained, decomposes new data from the same data generating processes of the training data into stochastic common trend components and stationary components based on Kasa (1992).
        
        Parameters
        ==========
        nonstyX:    an n by l 2d array that contains the data for I(1) features where n is the number of rows / time periods; and l is the total number of I(1) features before PCA.

        styX:       an optional n by k 2d array that contains the extra / exogenous stationary variables.

        start_step: an optional int that indicates the time period for the first row in the data. This is ignored if Chigira is not used as the PCA model or the time trend is not specified in Chigira. 1 means the first period after the end of the estimation data.
        
        Return
        ======
        Znonsty:    an n by (l+k) 2d numpy array of non-stationary components of the augmented input data matrix [nonstyX, styX]. The first l columns represent the stochastic common trend components of the origal I(1) data nonstyX. The rest of the columns should all be zero.
        
        Zsty:       an n by (l+k) 2d numpy array that represents the stationary components of the augmented input data. The first l columns represents the stationary components of the original I(1) data nonstyX. The rest of the columns, if k != 0, are the verbatim copies of styX.
        
        Zdet:       an n by (l+k) 2d numpy array of the deterministic terms. It always includes averages of the original features. When Chigira method is used during fit, the time trend, if specified, is also included in this variable.

        Note: Znonsty + Zsty + Zdet should equal to the augmented input data matrix [nonstyX, styX].

        Reference
        =========
        Kasa, K. (1992). Common stochastic trends in international stock markets. Journal of monetary Economics, 29(1), 95-124.

        Lin, Y., Kruger, U., Gu, F., Ball, A., & Chen, Q. (2019). Monitoring nonstationary and dynamic trends for practical process fault diagnosis. Control Engineering Practice, 84, 139-158.
        """
        
        if not self._IsFitted:
            raise NotFittedError('Please fit this instance of the CommonTrend class first.')

        if nonstyX.shape[1] != self.n_features_nonsty:
            raise ValueError("nonstyX has {0} columns but should have {1} instead to match with the I(1) training data".format(nonstyX.shape[0], self.n_features_nonsty))
        if self.n_features_sty != 0: 
            if styX is None:
                raise ValueError("styX is required since it is given during training.")
            elif styX.shape[1] != self.n_features_sty:
                raise ValueError("styX has {0} columns but should have {1} instead to match with the I(1) training data".format(styX.shape[0], self.n_features_sty))
            elif styX.shape[0] != nonstyX.shape[0]:
                raise ValueError(f"nonstyX has {nonstyX.shape[0]} rows and does not match the number of rows (={styX.shape[0]}) in styX.")
            else:
                styX = check_array(styX)

        nonstyX = check_array(nonstyX)

        # find the means / trend
        nonsty_means = self.PCAModel_.mean_.reshape(1,-1)
        if getattr(self, 'ChigiraModel_', None) is not None:
            # Chigira de-means and de-trends the features with a VAR model of lag 0. 
            # We need to find the trended values
            nonsty_means = var_oos_predict(self.ChigiraModel_.VARModel_, start_step=start_step, fcst_steps=nonstyX.shape[0]) + nonsty_means
        Zdet_nonsty = np.concatenate([nonsty_means, np.zeros((nonsty_means.shape[0], self.n_features_ - nonsty_means.shape[1]))], axis=1)

        nonstyX_detr = nonstyX - nonsty_means

        # weight given to the tread / mean values for the common trends based on explained variance ratio
        # Chigira detrends the non-stationary features while PCA demeans them. 
        # There isn't a unique way of distributing the mean to the common trend and stationary parts of the cointegrated data
        # here, we simply use the variance ratio. 
        # Note that the PCA model may not report all components if PCA.n_components is not equal to PCA.n_features.
        # Thus we need to recalculate the explained variance ratio.
        # var_ratio = np.sum(self.PCAModel_.explained_variance_[0:self.n_components_nonsty_]) / np.sum(self.PCAModel_.explained_variance_)

        augX_detr = np.concatenate([nonstyX_detr, styX], axis=1)

        # common trend component ======================
        if self.n_components_nonsty_!= 0: # if there is common trend
            # first dot product calculates the principal components. 2nd dot product uses the principal components to predict the original I(1) features.
            Xct_detrended = augX_detr.dot(self.kasa_nonsty_loading_).dot(self.PCAModel_.components_[0:self.n_components_nonsty_nonsty_,:])
            # distribute the mean values for the non-stationary (common trend) principal components
            # Xct_trend = nonsty_means * var_ratio
            if self.n_features_nonsty > 0: # if extra stationary features were given
                # pad extra zeros in the columns for the extra stationary features
                Znonsty = np.concatenate([
                    Xct_detrended #+ Xct_trend
                    , np.zeros((nonstyX.shape[0], self.n_features_nonsty))
                    ], axis=1
                )
            else:
                Znonsty = Xct_detrended #+ Xct_trend
        else:
            Znonsty = 0

        # stationary component ======================
        # get the stationary PCs
        Xsty_components = augX_detr.dot(self.kasa_sty_loading_)
        # use stationary PCs to predict the original I(1) features
        Xsty_detrended = Xsty_components[:,:self.n_components_nonsty_sty_].dot(self.PCAModel_.components_[-self.n_components_nonsty_sty_:,:])
        # distribute the mean values for the stationary principal components
        # Xsty_trend = nonsty_means * (1-var_ratio)
        if self.n_features_nonsty > 0: # if extra stationary features were given
            # add the extra stationary features
            Zsty = np.concatenate([
                    Xsty_detrended #+ Xsty_trend
                    , styX
                    ], axis=1
                )
        else:
            Zsty = Xsty_detrended #+ Xsty_trend

        # note that the sum of the below should recover the original data: [nonstyX, styX]
        return Znonsty, Zsty, Zdet_nonsty

    def _perform_test(self):
        pass

    def _factor_2_teststat(self, factor, VARModel, start_period, covMatInv, sig, ci_out_of_control_side, ci_fxn):
        # get the residuals
        yhat = forecast_var_one_step_ahead(factor, VARModel, start_period)
        resid = factor[VARModel.__ct_max_lag__:,:] - yhat
        
        return self._violation_check(resid, covMatInv, sig, ci_out_of_control_side, ci_fxn)

    def _violation_check(self, resid, covMatInv, sig, ci_out_of_control_side, ci_fxn):
        T2Stats = np.sum(resid.dot(covMatInv) * resid, axis=1)
        control_limit = ci_fxn(sig)
        if ci_out_of_control_side == 'less':
            violation = T2Stats < control_limit
        elif ci_out_of_control_side == 'two-sided':
            violation = np.logical_or(T2Stats < control_limit[0], T2Stats > control_limit[1])
        else: # 'greater', default
            violation = T2Stats > control_limit
        return (T2Stats, violation)
