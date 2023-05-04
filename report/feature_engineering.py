import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# based on https://www.kaggle.com/code/vadimkamaev/score-1-382/notebook
data_dir = '../data/'

def get_train_test(month_number_split=35):
    # combine the original training set with the revealed data
    d = pd.concat([
        pd.read_csv(f'{data_dir}/train.csv'),
        pd.read_csv(f'{data_dir}/revealed_test.csv')
    ]).sort_values(by=['cfips', 'first_day_of_month']).reset_index(0, drop=True)
    d = d.sort_values(['cfips', 'row_id']).reset_index(0, drop=True)
    d['first_day_of_month'] = pd.to_datetime(d['first_day_of_month'])
    # d['county'] = d.groupby('cfips')['county'].ffill()
    # d['state'] = d.groupby('cfips')['state'].ffill()
    d['county_i'] = (d['county'] + d['state']).factorize()[0]
    d['state_i'] = d['state'].factorize()[0]
    d['month_number'] = d.groupby(['cfips'])['row_id'].cumcount()
    d = d.rename({'microbusiness_density': 'y'}, axis=1)

    test = d.loc[d.month_number >= month_number_split, :]
    train = d.copy()
    train.loc[train.month_number >= month_number_split, 'y'] = np.nan
    train['ytrue'] = d['y']

    features = ['county_i', 'state_i']
    return train, test, features

# def get_raw(train, test):
#     d = pd.concat([train, test]).sort_values(['cfips', 'row_id']).reset_index(0, drop=True)
#     d['first_day_of_month'] = pd.to_datetime(d['first_day_of_month'])
#     d['county'] = d.groupby('cfips')['county'].ffill()
#     d['state'] = d.groupby('cfips')['state'].ffill()
#     d['county_i'] = (d['county'] + d['state']).factorize()[0]
#     d['state_i'] = d['state'].factorize()[0]
#     d['month_number'] = d.groupby(['cfips'])['row_id'].cumcount()
#     d['y'] = d['microbusiness_density']
#     d = d.drop('microbusiness_density', axis=1)
#     features = ['state_i']
#     return d, features

def add_lag(d, target='y', max_lag=8):
    features = []
    for lag in range(1, max_lag):
        d[f'{target}_lag_{lag}'] = d.groupby('cfips')[target].shift(lag).bfill()
        d[f'dif_{lag}'] = (d[target] / d[f'{target}_lag_{lag}']).fillna(1).clip(0, None) - 1
        d.loc[(d[f'{target}_lag_{lag}']==0),f'dif_{lag}'] = 0
        d.loc[(d[target]>0) & (d[f'{target}_lag_{lag}']==0), f'dif_{lag}'] = 1
        d[f'dif_{lag}'] = d[f'dif_{lag}'].abs()
        # d[f'{target}_lag_{lag}'] = d.groupby('cfips')[target].shift(lag)
        features.append(f'{target}_lag_{lag}')
        # d[f'active_lag_{lag}'] = d.groupby('cfips')['active'].diff(lag)
        # feats.append(f'active_lag_{lag}')
    return d, features

def add_rolling(d, target='y', lags=[1]):

    def get_rolling(s, window):
        return s.rolling(window, min_periods=1).sum()

    features = []
    for lag in lags:
        for window in [2, 4, 6, 8, 10]:
            group = d.groupby('cfips')[f'{target}_lag_{lag}']
            d[f'{target}_roll_{window}_{lag}'] = group.transform(get_rolling, window=window)
            features.append(f'{target}_roll_{window}_{lag}')
    return d, features

def add_internal_census(d):
    census = pd.read_csv(f'{data_dir}/census_starter.csv')
    features = list(census.columns)
    features.remove('cfips')
    d = d.merge(census, on='cfips', how='left')
    return d, features

def add_external_census(d):
    '''
    data: https://www2.census.gov/programs-surveys/popest/datasets/2020-2021/counties/totals/co-est2021-alldata.csv
    schema: https://www2.census.gov/programs-surveys/popest/technical-documentation/file-layouts/2020-2021/CO-EST2021-ALLDATA.pdf
    '''
    census = pd.read_csv(f'{data_dir}/co-est2021-alldata.csv', encoding='latin-1')
    census['cfips'] = census.STATE*1000 + census.COUNTY
    features = [
        'SUMLEV', 'REGION', 'DIVISION', 'ESTIMATESBASE2020',
        'POPESTIMATE2020', 'POPESTIMATE2021',
        'NPOPCHG2020', 'NPOPCHG2021',
        'BIRTHS2020', 'BIRTHS2021',
        'DEATHS2020', 'DEATHS2021',
        'NATURALCHG2020', 'NATURALCHG2021',
        'INTERNATIONALMIG2020', 'INTERNATIONALMIG2021',
        'DOMESTICMIG2020', 'DOMESTICMIG2021',
        'NETMIG2020', 'NETMIG2021',
        'RESIDUAL2020', 'RESIDUAL2021',
        'GQESTIMATESBASE2020', 'GQESTIMATES2020', 'GQESTIMATES2021',
        'RBIRTH2021', 'RDEATH2021', 'RNATURALCHG2021', 'RINTERNATIONALMIG2021', 'RDOMESTICMIG2021', 'RNETMIG2021'
    ]
    d = d.merge(census, on='cfips', how='left')
    return d, features


# def add_coords(d, features):
#     '''
#     https://www.kaggle.com/datasets/alejopaullier/usa-counties-coordinates
#     '''
#     coords = pd.read_csv(f'{data_dir}/cfips_location.csv').drop('name', axis=1)
#     d = d.merge(coords, on='cfips')
#     features += ['lng', 'lat']
#     return d, features

def add_target(d):
    # lag = 1
    # d[f'mbd_lag_{lag}'] = d.groupby('cfips')['y'].shift(lag).bfill()
    # d['dif'] = (d['y'] / d[f'mbd_lag_{lag}']).fillna(1).clip(0, None) - 1
    # d.loc[(d[f'mbd_lag_{lag}']==0), 'dif'] = 0
    # d.loc[(d[f'y']>0) & (d[f'mbd_lag_{lag}']==0), 'dif'] = 1
    # d['dif'] = d['dif'].abs()

    d['target'] = d.groupby('cfips')['y'].shift(-1)
    d['target'] = d['target']/d['y'] - 1
    features = ['target']
    return d, features

def add_all_features(d, target='y'):
    features = []
    d['lastactive'] = d.groupby('cfips')['active'].transform('last')
    d, f0 = add_target(d)
    d, f1 = add_lag(d, target=target)
    d, f2 = add_rolling(d, target=target)
    d, f3 = add_internal_census(d)
    d, f4 = add_external_census(d)
    # print(d.loc[:, ['cfips', 'month_number', 'y', 'ytrue', 'target']].head(100))
    features += f0 + f1 # + f2 + f3 + f4
    return d, features
