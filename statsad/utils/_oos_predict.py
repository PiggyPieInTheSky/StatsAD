import numpy as np
# from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.vector_ar.vecm import VECM
from statsmodels.tsa.vector_ar.var_model import VAR

def __vecm_oos_predict(vecm_mdl, lag_data=None, start_step=None, fcst_steps=1, pred_itvl_sig=None, itvl_shift=0):
    """performs out of sample forecast from a fitted VECM model with the ability to specify the lagged data and arbitrary starting time trend

    Parameters
    ==========
        vecm_mdl:   dict

        lag_data:   2d numpy array. # of cols = # of endog vars in VAR model. If not enough rows are provided, the function automatically adds rows form the bottom of the estimation data

        start_Step: int, >=1. starting point of the first forecast in the output. 1 means the 1st forecast period after the end of estimaiton data.

        fcst_steps: int, >=1. number of forecast periods

        pred_itvl_sig:  sig lvl of the prediciton interval. If None or 0, no preidction interval is returned

        itvl_shift:     int, >=0. The default margin of the error for the prediciton interval (when itvl_shift==0) starts at the period right after estimaiton data (period 1) and ends at the end of forecast (period = fcst_steps). This assumes any periods before start_step are realized and have no uncertainty. When itvl_shift > 0, the maring of error comes from those for periods from itvl_shift+1 to itvl_shift+fcst_steps.

    Return
    ======
        If pred_itvl_sig is None or 0, only point estimate forecast will be returned as a 2d numpy array with fcst_steps rows.
        If pred_itvl_sig is specified, a 3-entry tuple is returned with point estimate, lower bound and upper bound in each entry. Each entry is a 2d numpy array with fcst_steps rows. 
    """
    if pred_itvl_sig is None:
        pred_itvl_sig = 0

    if pred_itvl_sig < 0:
        raise ValueError('itvl_shift cannot be negative.')

    if lag_data is None: #lagged data is not provided
        # use the last few observations form the estimation data
        # we assume the forecast starts at one period after the esitmation data
        lag_data = vecm_mdl.model.endog[-vecm_mdl.k_ar:]
        if start_step is not None and start_step < 1:
            raise ValueError(f"start_step cannot be less than 1. {start_step} is given.")
    elif lag_data.shape[1] != vecm_mdl.model.endog.shape[1]:
        raise ValueError("lag_data has less variables in column (={0}) than those used in estimaiton (={1}). Please do not include deterministic terms (constant, trend).".format(lag_data.shape[1], vecm_mdl.endog.shape[1]))
    elif lag_data.shape[0] != vecm_mdl.k_var:
        # not enough lagged rows are given, add from the bottom of estiamtion data
        lag_data = np.concatenate([vecm_mdl.model.endog[-(vecm_mdl.k_ar-lag_data.shape[0]):,:], lag_data], axis=0)

    IsCointConst = 'ci' in vecm_mdl.deterministic # constant term incointegration relation
    IsCointTrend = 'li' in vecm_mdl.deterministic # time trend in cointegration relation
    IsUnresConst = 'co' in vecm_mdl.deterministic # constant term outside of cointegration relation
    IsUnresTrend = 'lo' in vecm_mdl.deterministic # time trend outside of cointegration relaiton

    # initialize the forecast resturn structure
    ecm_fcst = np.zeros((fcst_steps, vecm_mdl.model.endog.shape[1]))

    # initialize the time_trend_t value based on start_step input argument
    if IsCointTrend or IsUnresTrend:
        if start_step is None:
            #speficy the time trend starting point at the last period of the estimation data
            time_trend_t = vecm_mdl.model.endog.shape[0] # since it is in the cointegration relation (as matched with y_{t-1} term on RHS), the first forecast period's time trend is the last esitmaiton period
        else:
            time_trend_t = vecm_mdl.model.endog.shape[0] + start_step - 1

    # loop through each forecast period
    for iPeriod in range(fcst_steps):
        # initialize the structure for the current period's forecast
        ecm_1stp_fcst = np.zeros((1, vecm_mdl.model.endog.shape[1]))

        # compute the AR terms. since this is ECM, there is at least one lag from VAR's perspective
        # the var_rep property:
        #   1. does not contain deterministic terms
        #   2. for lag 1: \alpha \times \beta\prime + indentiy (to move the y_{t-1} in \Delta y_t on the LHS to the RHS) + the coefficient for \Delta y{t-1} in \Gamma.
        #   3. for lag = i \in [2, k_ar): coefficient for the lag in \Gamma (for y_{t-i} \Delta y_{t-i}) - coefficient for the previous lag (for y{t-i} from \Delta y_{t-i+1})
        #   4. for lag = k_ar: coeffcient of the last lag in \Gamma

        ecm_det_term = None
        if IsCointConst or IsCointTrend: # if deterministic term(s) within cointegration relation
            mx_coint_det = vecm_mdl.det_coef_coint.dot(vecm_mdl.alpha.transpose())
            # compute the deterministic terms
            if IsCointConst and IsCointTrend: # const + time trend within coint relation
                ecm_det_term = np.array([[1, time_trend_t]]).dot(mx_coint_det)
            elif IsCointConst == False and IsCointTrend: # only time trend within coint relation
                ecm_det_term = mx_coint_det * time_trend_t
            elif IsCointConst and IsCointTrend == False: # only const within coint relation
                ecm_det_term= mx_coint_det
        else:
            if vecm_mdl.deterministic != 'n':
                raise Exception("ECM forecast is not implmented for the deterministic terms '{0}'.".format(vecm_mdl.deterministic))

        if ecm_det_term is not None:
            ecm_1stp_fcst += ecm_det_term
        
        # roll forward the lag_data structure with the current period's forecast
        if vecm_mdl.k_ar == 1: # only one lag
            lag_data = ecm_1stp_fcst
        else:
            # add the current period's forecst to the bottom of the lag data and remove the first peirod in the data
            lag_data = np.concatenate([lag_data[1:], ecm_1stp_fcst], axis=0)

        ecm_fcst[iPeriod,] =  ecm_1stp_fcst

        if IsCointTrend or IsUnresTrend:
            time_trend_t += 1

    if pred_itvl_sig == 0:
        return ecm_fcst

    # look for forecast_interval at the top level, not the one within a class
    itvlRaw = vecm_mdl.predict(steps=fcst_steps+itvl_shift, alpha=pred_itvl_sig)

    # recover the "maring of error" fron the bounds and add th enew point estimates
    fcst_lower = itvlRaw[1][itvl_shift:,] - itvlRaw[0][itvl_shift:,] + ecm_fcst
    fcst_upper = itvlRaw[2][itvl_shift:,] - itvlRaw[0][itvl_shift:,] + ecm_fcst

    return ecm_fcst, fcst_lower, fcst_upper

def __var_oos_predict(var_mdl, lag_data=None, start_step=1, fcst_steps=1, pred_itvl_sig=None, itvl_shift=0):
    """performs out of sample forecast from a fitted VAR model with the ability to specify the lagged data and arbitary starting time trend
    
    Parameters
    ==========
        var_mdl:   statsmodels.tsa.vector_ar.var_model.VARResultsWrapper

        lag_data:   2d numpy array. # of cols = # of endog vars in VAR model. If not enough rows are provided, the function automatically adds rows form the bottom of the estimation data

        start_step: int. starting point of the first forecast in the output. 1 means the 1st forecast period after the end of estimaiton data.

        fcst_steps: int, >=1. number of forecast periods

        pred_itvl_sig:  sig lvl of the prediciton interval. If None or 0, no preidction interval is returned

        itvl_shift:     int, >=0. The default margin of the error for the prediciton interval (when itvl_shift==0) starts at the period right after estimaiton data (period 1) and ends at the end of forecast (period = fcst_steps). This assumes any periods before start_step are realized and have no uncertainty. When itvl_shift > 0, the maring of error comes from those for periods from itvl_shift+1 to itvl_shift+fcst_steps.

    Return
    ======
        If pred_itvl_sig is None or 0, only point estimate forecast will be returned as a 2d numpy array with fcst_steps rows.
        If pred_itvl_sig is specified, a 3-entry tuple is returned with point estimate, lower bound and upper bound in each entry. Each entry is a 2d numpy array with fcst_steps rows. 

    """

    if pred_itvl_sig is None:
        pred_itvl_sig = 0

    if pred_itvl_sig < 0:
        raise ValueError('itvl_shift cannot be negative.')

    if start_step is None:
        start_step = 1

    IsConst = 'c' in var_mdl.trend
    IsTrend = 't' in var_mdl.trend

    # edge case: model with deterministic terms only
    if var_mdl.k_ar == 0:
        # move the trend value to reflect user's adjustment
        start_trend_value = var_mdl.endog.shape[0] + start_step
        if IsConst and IsTrend: # if consttant and trend terms are in the model
            # build a matrix of 1s in first column and progress trend vlaues in the 2nd
            x_var = np.concatenate([np.ones((fcst_steps,1)), np.arange(start_trend_value, start_trend_value+fcst_steps, step=1).reshape((-1,1))], axis=1)
            # build the forecast X*beta
            fcst_ptest = x_var.dot(var_mdl.params)
        elif IsConst: # if only the constant term is in the model
            # repeat the constant term as forecast
            fcst_ptest = np.repeat(var_mdl.params.reshape((-1,1)), repeats=fcst_steps, axis=0)
        else: # if no trend and no constant (not sure if this is possible). note: there is no trend only option in statsmodels
            # return all zeros
            fcst_ptest = np.zeros((fcst_steps, var_mdl.endog.shape[1]))
        #return results
        if pred_itvl_sig == 0: #if the user does not request prediction intervals
            return fcst_ptest
        else: # if the user requests prediciton intervals
            # deterministic terms don't have bounds
            return fcst_ptest, fcst_steps, fcst_steps

    if lag_data is None: # lagged data is not provided
        lag_data = var_mdl.endog[-var_mdl.k_ar:]
        if lag_data.shape[1] != var_mdl.endog.shape[1]:
            raise ValueError("lag_data has less variables in column (={0}) than those used in estimaiton (={1}). Please do not include deterministic terms (constant, trend).".format(lag_Data.shape[1], vecm_mdl.endog.shape[1]))
        elif lag_data.shape[0] != var_mdl.k_ar:
            # not enough lagged rows are given, add from the bottom of the estimation data
            lag_data = np.concatenate([var_mdl.endog[-(var_mdl.k_ar-lag_data.shape[0]):,:], lag_data], axis=0)

    time_diff = 0 # used to adjust the time trend when the user specifies an arbitrary starting point
    if IsTrend:
        time_diff = var_mdl.params.loc['trend'].reshape((-1,1)) * (start_step - 1)

    # return point estimate if prediciton interval is not requested
    if pred_itvl_sig == 0:
        return var_mdl.forecast(lag_data, steps=fcst_steps) + time_diff

    # get the raw VAR forecast
    itvlRaw = var_mdl.forecast_interval(lag_data, steps=fcst_steps+itvl_shift, alpha=pred_itvl_sig)
    # add the time shift if start_step is provided
    fcst_ptest = itvlRaw[0][0:fcst_steps, :] + time_diff
    # align the margin of error for the lower and upper bounds based on whether or not margin of error should be shifted
    fcst_lb = itvlRaw[1][itvl_shift:,] - itvlRaw[0][itvl_shift:,] + fcst_ptest
    fcst_ub = itvlRaw[2][itvl_shift:,] - itvlRaw[0][itvl_shift:,] + fcst_ptest

    return fcst_ptest, fcst_lb, fcst_ub
