import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import metric

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def plot(d, nrows=2, ncols=6):
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*3, nrows*3))
    for i, cfips in enumerate(d.cfips.unique()):
        if i >= nrows*ncols:
            break
        idx_row = int(i / ncols)
        idx_col = i % ncols
        if nrows == 1:
            ax = axs[idx_col]
        else:
            ax = axs[idx_row, idx_col]


        g = d.loc[d.cfips == cfips, :].reset_index(0, drop=True)
        yrange = g.ytrue.max() - g.ytrue.min()
        ax.plot(g['month_number'], g['ytrue'], 'o-', label='raw')
        last = pd.Series([g['ytrue'].tolist()[-1]])
        yhat = pd.concat([g['yhat'], last])
        yhat = yhat.tolist()[1:]
        ax.plot(g['month_number'], yhat + np.random.normal(loc=0, scale=0.05*yrange, size=len(yhat)), 'o-', label='predicted')
        cfips = g.cfips.unique()[0]
        ax.set_title(f'cfips={cfips}')
        # ax.set_ylim()

        if idx_row == nrows - 1:
            ax.set_xlabel('month_number')
        ax.set_xlim(0, 40)
        if idx_row == 0 and idx_col == ncols-1:
            ax.legend(fancybox=False)
        if idx_col == 0:
            ax.set_ylabel('y')

    fig.tight_layout()
    plt.savefig('gb.pdf', bbox_inches='tight')

d = pd.read_csv('test.csv')
d['month_number'] = d['dcount']
d['ytrue'] = d['microbusiness_density']
d['yhat'] = d['ypred']
cfips_list = [10001, 10003, 10005, 1001, 1003, 1005, 1007, 1009, 1011, 1013, 1015, 1017]
d['cfips_sort'] = d['cfips'].map({cfips: i for i, cfips in enumerate(cfips_list)})
d = d.loc[d.cfips.isin(cfips_list), :]
d = d.sort_values(by=['cfips_sort', 'month_number'])
d = d.loc[d.month_number <= 40, :]
print(list(d.columns))
month_first = 35
month_last = 40
test_idxs = (month_first <= d.month_number) & (d.month_number <= month_last)
smape = metric.get_smape(d.loc[test_idxs, 'ytrue'], d.loc[test_idxs, 'yhat'])
print(smape)
# print(d.loc[:, ['microbusiness_density', 'ypred', 'ypred_last']].head(100))
plot(d)
