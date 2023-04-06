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

