import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import features
import models

import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)


data, d, test = features.get_train_test()
model = models.get_model()

# p = d.groupby('cfips', group_keys=True).apply(models.predict_arima).reset_index(0, drop=True)

def plot(d, nfigs=18, ncols=6):
    nrows = int(np.ceil(nfigs/ncols))
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*3, nrows*3), sharey=False, sharex=True)

    for i, cfips in enumerate(d.cfips.unique()):
        if i >= nfigs:
            break
        idx_row = int(i / ncols)
        idx_col = i % ncols
        if nrows == 1:
            ax = axs[idx_col]
        else:
            ax = axs[idx_row, idx_col]

        g = d.loc[d.cfips == cfips, :].reset_index(0, drop=True)
        ax.plot(g['month_number'], g['y'], 'o-', label='raw')
        ax.plot(g['month_number'], g['yhat'], 'o-', label='predicted')
        cfips = g.cfips.unique()[0]
        ax.set_title(f'cfips={cfips}')
        ax.set_xlabel('month_number')
        if idx_row == 0 and idx_col == ncols-1:
            ax.legend(fancybox=False)
        if idx_col == 0:
            ax.set_ylabel('y')

    fig.tight_layout()
    plt.show()

plot(p)
