import numpy as np
import pandas as pd

import feature_engineering
import models
import metric

# def iterative_train(d, features, month_first=35, month_last=40):
#     for train_cutoff in range(month_first, month_last+1):
#         print(train_cutoff)
#         model = models.get_model()
#         train_idxs = (0 <= d.month_number) & (d.month_number < train_cutoff)
#         valid_idxs = (d.month_number == train_cutoff)
#         model.fit(d.loc[train_idxs, features], d.loc[train_idxs, 'y'])

#         d.loc[valid_idxs, 'yhat'] = model.predict(d.loc[valid_idxs, features])
#         d.loc[valid_idxs, 'y'] = d.loc[valid_idxs, 'y'].fillna(d.loc[valid_idxs, 'yhat'])
#         d, _ = feature_engineering.add_lag(d, target='y')
#         d, _ = feature_engineering.add_rolling(d, target='y')


#     test_idxs = (month_first <= d.month_number) & (d.month_number <= month_last)
#     print('SMAPE:', metric.get_smape(d.loc[test_idxs, 'ytrue'], d.loc[test_idxs, 'yhat']))
#     return d, model

def iterative_train(d, features, month_first=34, month_last=40):
    for train_cutoff in range(month_first, month_last+1):
        print(train_cutoff)
        model = models.get_model()
        train_idxs = (0 <= d.month_number) & (d.month_number < train_cutoff) & (d.lastactive > 140)
        valid_idxs = (d.month_number == train_cutoff)
        # print(d.loc[train_idxs & d.target.isna(), ['month_number', 'target', 'y', 'ytrue', 'active']])
        model.fit(d.loc[train_idxs, features], d.loc[train_idxs, 'target'].clip(-0.0043, 0.0045))

        d.loc[valid_idxs, 'target_hat'] = model.predict(d.loc[valid_idxs, features])
        print(d.loc[:, ['cfips', 'month_number', 'target', 'target_hat', 'y', 'ytrue', 'active']].head(50))
        d.loc[valid_idxs, 'target'] = d.loc[valid_idxs, 'target'].fillna(d.loc[valid_idxs, 'target_hat'])
        d.loc[valid_idxs, 'yhat'] = d.loc[valid_idxs, 'y']*d.loc[valid_idxs, 'target_hat'] + 1
        print(d.loc[:, ['cfips', 'month_number', 'target', 'target_hat', 'y', 'ytrue', 'yhat', 'active']].head(50))
        # d, _ = feature_engineering.add_lag(d, target='target')
        # d, _ = feature_engineering.add_rolling(d, target='target')



    test_idxs = (month_first <= d.month_number) & (d.month_number <= month_last)
    print('SMAPE:', metric.get_smape(d.loc[test_idxs, 'ytrue'], d.loc[test_idxs, 'yhat']))
    return d, model

d, test, features = feature_engineering.get_train_test(month_number_split=35)
d, features_temp = feature_engineering.add_all_features(d, target='target')
features += features_temp
print(features)
d, model = iterative_train(d, features)

# prefix = 'trained_ratio'
# model.save_model(f'{prefix}_model')
# d.to_csv(f'{prefix}.csv', index=False)
