from .interface import TestBaseClass
from typing import Optional, Union
import numpy as np
from sklearn.utils.validation import check_array
from scipy.stats import f as fdist
from scipy.stats import norm as normdist
from scipy.interpolate import interp1d
from statsmodels.nonparametric.kde import KDEUnivariate
from sklearn.exceptions import NotFittedError

class ProcesssControlADTester(TestBaseClass):
    """A tester of outliers in process control setting

    Parameters
    ----------
    kde_kernel: str, bool, or default to None, to indicate the kernel of the Kernel Density Estimation. If None, asymptotic distribution is used for the given test statistic. For more info, see documentation on statsmodels.nonparametric.kde.KDEUnivariate.fit method.
    
    kde_bw: str or float to indicate the bandwidth of KDE. Ignored if kde_kernel is False or None. Default is 'silverman'. For more info, see documentation on statsmodels.nonparametric.kde.KDEUnivariate.fit method.
    
    kde_gridsize: int or default to None for the gridsize input argument for statsmodels.nonparametric.kde.KDEUnivariate.fit method.
    
    kde_model: an estimated kde class or default to None. If this class is provided, no KDE is estimated and all other kde_* parameters are ignored. The class must implement cdf and support attributes to provide an array of empirical cdf values and their corresponding support values.

    asymp_ci_fxn: function with one scalar input, to calculate the control limit based on the asymptotic distribution. The scalar input is for the significance level. This argument must be provided if KDE is not used: eg kde_kernel = False and kde_model is None.
    """
    def __init__(self
        , kde_kernel:Optional[Union[str,bool]]=False
        , kde_bw='silverman'
        , kde_gridsize:Optional[int]=None
        , kde_model=None
        , asymp_ci_fxn=None
    ) -> None:
        if kde_kernel is None:
            kde_kernel = False
        elif kde_kernel == True:
            kde_kernel = 'gau'
        
        if kde_kernel == False and kde_model is None and asymp_ci_fxn is None:
            raise ValueError('When kde_kernel is False or None or not provided, asymp_ci_fxn cannot be None.')

        # if test_statistic not in ['t2', 'q']:
        #     raise ValueError(f'test_statistic can only be "t2" or "q", but "{test_statistic}" is given.')

        # self.test_statistic = test_statistic
        self.kde_kernel = kde_kernel
        self.kde_bw = kde_bw
        self.kde_gridsize = kde_gridsize
        self.kde_model = kde_model
        self.asymp_ci_fxn = asymp_ci_fxn

    def fit(self, X1, X2=None):
        '''Estimate the control limit function of the given test statistic.

        Parameters
        ==========
        X1: a one-dimensional array or a column / row vector of test statistics.
        X2: ignored
        '''
        # resid = X1
        # if X2 is None:
        #     raise ValueError('X2 canont be None.')

        # if self.test_statistic == 't2': # Hotelling's T2
        #     covMatInv = X2
        #     ndim = np.ndim(covMatInv)
        #     if ndim not in [0,2]:
        #         raise ValueError('For Hotelling\'s T2 statistic, X2 must be a scalar or a matrix.')
        #     elif ndim == 2:
        #         dim = np.shape(covMatInv)
        #         if dim[0] != dim[1]:
        #             raise ValueError('For Hotelling\'s T2 statistic, X2 is not a square matrix.')
        #         if np.all(covMatInv - covMatInv.T < 0.00001):
        #             covMatInv = (covMatInv + covMatInv.T) / 2 # ensure symmetry
        #         else:
        #             raise ValueError('For Hotelling\'s T2 statistic, X2 is not symmetric.')

        #     # calculate the Hotelling's T2 statistic based on residuals: Mahalanobis distance of 
        #     test_stats = np.sum(resid.dot(covMatInv) * resid, axis=1)
        # else: # Q-statistic
        #     lambdas = X1

        test_stats = self._train_dt_check(X1)

        if self.kde_kernel == False: # use asymptotic distribution
            self.ci_fxn_ = self.asymp_ci_fxn
            # if self.test_statistic == 't2':
            #     # F distribution for Hotelling's T2 statistic
            #     self.ci_fxn_ = lambda x: test_stats.shape[1] * (test_stats.shape[0]-1) * fdist.ppf(1-x,test_stats.shape[1],test_stats.shape[0]-test_stats.shape[1]) / (test_stats.shape[0]-test_stats.shape[1])
            # else: # Q-statistic
            #     thetas = np.zeros(3)
            #     for i in [1,2,3]:
            #         thetas[i-1] = np.sum(lambdas ** i)
            #     h0 = 1 - 2 * thetas[0] * thetas[2] / 3 / (thetas[1] ** 2)
            #     ci_v1 = np.sqrt(2 * thetas[1] * h0 * h0) / thetas[0]
            #     ci_v2 = 1 + thetas[1] * h0 * (h0-1) / (thetas[0] ** 2)
            #     # normal distribution for Q-statistic
            #     self.ci_fxn_ = lambda x: thetas[0] * (normdist.ppf(1-x) * ci_v1 + ci_v2) ** (1/h0)
        else:
            if self.kde_model is None:
                # if self.test_statistic == 'q':
                #     # Q-statistic is the L2 norm of the residuals
                #     test_stats = np.sum(resid **2, axis=1)
                # train the kde on the test statistics
                self.kde_model_ = KDEUnivariate(test_stats).fit(kernel=self.kde_kernel, bw=self.kde_bw, gridsize=self.kde_gridsize)
            else:
                self.kde_model_ = self.kde_model
            
            # linearly interpolate the inverse CDF function
            ppf_fxn = interp1d(x=self.kde_model_.cdf, y=self.kde_model_.support)
            # finalize the confidence interval function
            self.ci_fxn_ = lambda x: ppf_fxn(1-x)
            

    def test(self, X1, X2=None, sig: float = 0.05):
        '''Checks if an outlier is present

        Parameters
        ==========
        X1:  a one-dimensional array, a column / row vector or a scalar of test statistic(s).
        sig: float, significance level, between 0 and 1.
        X2:  ignored.

        Return
        ======
        An array of boolean values to indicate the corresponding elements the input is an outlier or not.
        '''
        ci_fxn = getattr(self, 'ci_fxn_', None)
        if ci_fxn is None:
            NotFittedError()
        test_stats = self._train_dt_check(X1)
        return test_stats>ci_fxn(sig)
        

    def fit_test(self, X1, X2=None, sig: float = 0.05):
        self.fit(X1)
        return self.test(X1, sig=sig)

    def _train_dt_check(X1):
        ndim = np.ndim(X1)
        if ndim == 0:
            return X1
        elif ndim == 1:
            return X1.reshape((-1,1))
        elif ndim > 2:
            raise ValueError('Number of dimensions is too large.')
        
        dim = np.shape(X1)
        if dim[1] > 1:
            if dim[0] == 0:
                return np.reshape(X1, (-1,1))
            raise ValueError('X1 should be a one dimensional array.')
        return X1

