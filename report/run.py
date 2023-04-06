import numpy as np
import pandas as pd

import feature_engineering
import models
import metric

def iterative_train(d, features, month_first=35, month_last=40):
    for train_cutoff in range(month_first, month_last+1):
        print(train_cutoff)
        model = models.get_model()
        train_idxs = (0 <= d.month_number) & (d.month_number < train_cutoff)
        valid_idxs = (d.month_number == train_cutoff)
        model.fit(d.loc[train_idxs, features], d.loc[train_idxs, 'y'])

        d.loc[valid_idxs, 'yhat'] = model.predict(d.loc[valid_idxs, features])
        d.loc[valid_idxs, 'y'] = d.loc[valid_idxs, 'y'].fillna(d.loc[valid_idxs, 'yhat'])
        d, _ = feature_engineering.add_lag(d, target='y')
        d, _ = feature_engineering.add_rolling(d, target='y')


    test_idxs = (month_first <= d.month_number) & (d.month_number <= month_last)
    print('SMAPE:', metric.get_smape(d.loc[test_idxs, 'ytrue'], d.loc[test_idxs, 'yhat']))
    return d, model

d, test, features = feature_engineering.get_train_test()
d, features_temp = feature_engineering.add_all_features(d)
features += features_temp
d, model = iterative_train(d, features)
d.to_csv('trained.csv', index=False)
