# from sklearn.ensemble import VotingRegressor
# import lightgbm as lgb
# import xgboost as xgb
# from sklearn.pipeline import Pipeline
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.impute import KNNImputer
import itertools
import statsmodels.api as sm

import catboost as cat
import numpy as np

def get_model():
    # we should decrease the num_iterations of catboost
    cat_model = cat.CatBoostRegressor(
        iterations=800,
        loss_function="MAPE",
        verbose=0,
        grow_policy='SymmetricTree',
        learning_rate=0.035,
        colsample_bylevel=0.8,
        max_depth=5,
        l2_leaf_reg=0.2,
        # max_leaves = 17,
        subsample=0.70,
        max_bin=4096,
    )
    return cat_model

def get_linear_pred(g):
    gg = g.copy().reset_index(0, drop=True)
    g = g.dropna()
    linear_model = np.poly1d(np.polyfit(g.month_number, g.y, 1))
    yhat = linear_model(gg.month_number)
    gg['yhat'] = yhat
    return gg

def arimax(y, order, seasonal_order, exog=None):
    model = sm.tsa.statespace.SARIMAX(y, exog=exog, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(maxiter=300, disp=False)
    return results

def get_best_params(y, exog=None, period=0, steps=2):
    r1 = r2 = r3 = range(steps)
    pdq = list(itertools.product(r1, r2, r3))
    if period:
        seasonal_pdq = [(x[0], x[1], x[2], period) for x in list(itertools.product(r1, r2, r3))]
    else:
        seasonal_pdq = [(0, 0, 0, 0)]
    aic_min = np.inf
    for order in pdq:
        for seasonal_order in seasonal_pdq:
            results = arimax(y, order, seasonal_order, exog=exog)
            if results.aic < aic_min:
                aic_min = results.aic
                best_params = [order, seasonal_order, results.aic]
    return best_params

def predict_arima(g, exog=False, target_month_number=46):
    gg = g.copy().reset_index(0, drop=True)
    y = g['y']
    if exog:
        # x = g[['edu_post_hs_ratio', 'gender_male_ratio', 'race_white_ratio', 'race_black_ratio']]
        best_params = get_best_params(y, exog=x)
    else:
        best_params = get_best_params(y)
    best_results = arimax(y, best_params[0], best_params[1])
    pred = best_results.get_prediction(start=target_month_number - 20, end=target_month_number, dynamic=False)
    gg = gg.join(pred.predicted_mean, on=gg.month_number, how='outer')
    gg['yhat'] = gg['predicted_mean']
    gg = gg.drop(['predicted_mean'], axis=1)
    return gg

