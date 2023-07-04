import numpy as np
from collections import deque
from statsmodels.tsa.ar_model import AutoRegResults, AutoRegResultsWrapper
from statsmodels.tsa.vector_ar.var_model import VARResults, VARResultsWrapper

def prepare_parameters_VAR(mdlVAR):
    if isinstance(mdlVAR, VARResults) or isinstance(mdlVAR, VARResultsWrapper):
        mdlVAR.__ct_max_lag__ = mdlVAR.k_ar
        mdlVAR.__ct_spec__ = mdlVAR.trend
        paramMat = mdlVAR.params
        IsAR = False
    elif isinstance(mdlVAR, AutoRegResults) or isinstance(mdlVAR, AutoRegResultsWrapper):
        mdlVAR.__ct_max_lag__ = 0 if mdlVAR.ar_lags is None else mdlVAR.ar_lags[-1]
        mdlVAR.__ct_spec__ = mdlVAR.model.trend
        paramMat = mdlVAR.params.reshape(-1,1)
        IsAR = True

    # prepare for parameters that will be used to forecast
    if mdlVAR.__ct_spec__ == 'n':
        mdlVAR.__ct_const_params__ = 0
        mdlVAR.__ct_trend_params__ = 0
        AR_params = paramMat if mdlVAR.__ct_max_lag__ != 0 else 0
    elif mdlVAR.__ct_spec__ == 'c':
        mdlVAR.__ct_const_params__ = paramMat[[0],:]
        mdlVAR.__ct_trend_params__ = 0
        AR_params = paramMat[1:paramMat.shape[0],:] if mdlVAR.__ct_max_lag__ != 0 else 0
    elif mdlVAR.__ct_spec__ == 't':
        mdlVAR.__ct_const_params__ = 0
        mdlVAR.__ct_trend_params__ = paramMat[[0],:]
        AR_params = paramMat[1:paramMat.shape[0],:] if mdlVAR.__ct_max_lag__ != 0 else 0
    else: # 'ct'
        mdlVAR.__ct_const_params__ = paramMat[[0],:]
        mdlVAR.__ct_trend_params__ = paramMat[[1],:]
        AR_params = paramMat[2:paramMat.shape[0],:] if mdlVAR.__ct_max_lag__ != 0 else 0
    
    # the AR matrix is reversed so that the early observations' coefficients have lower row index
    if mdlVAR.__ct_max_lag__ != 0:
        if IsAR: # if one variable
            mdlVAR.__ct_ar_params__ = np.flip(AR_params, axis=0)
        else:
            # block reverse the parameter matrix so that the earliest period shows up on top of the parameter matrix
            # mdlVAR.params dimension: num_vars*num_lags, num_vars. The row is divided into blocks of peirods. Within each period for a single column, var is flattend. For example, ((i-1)*n_var:i*n_var , 1) is for the 2nd var (col index=1) and the ith lag i=1,2,3...
            mdlVAR.__ct_ar_params__ = np.concatenate(np.array_split(AR_params, mdlVAR.__ct_max_lag__, axis=0)[::-1])
    
    return mdlVAR

def forecast_var_one_step_ahead(Xinput, mdlVAR, start_period=1):
    '''
    start_period=1 is the first period after the train data.
    '''
    yhat = deque()
    time_shift = start_period + mdlVAR.__ct_ar_last_t__ #mdlVAR.__ct_max_lag__ + start_period
    for i in range(0, Xinput.shape[0]-mdlVAR.__ct_max_lag__):
        yhat_crnt = mdlVAR.__ct_const_params__ + (i + time_shift) * mdlVAR.__ct_trend_params__
        if mdlVAR.__ct_max_lag__ > 0:
            yhat_crnt += Xinput[i:(mdlVAR.__ct_max_lag__+i),:].flatten().reshape(1,-1).dot(mdlVAR.__ct_ar_params__)
        yhat.append(yhat_crnt)
    yhat = np.concatenate(yhat, axis=0)
    return yhat
    