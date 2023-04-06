import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot(d, nrows=10, ncols=10):
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*3, nrows*3), sharey=True, sharex=True)
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
        ax.plot(g['month_number'], g['ytrue'], 'o-', label='raw')
        ax.plot(g['month_number'], g['yhat'], 'o-', label='predicted')
        cfips = g.cfips.unique()[0]
        ax.set_title(f'cfips={cfips}')

        if idx_row == nrows - 1:
            ax.set_xlabel('month_number')
        ax.set_xlim(0, 40)
        if idx_row == 0 and idx_col == ncols-1:
            ax.legend(fancybox=False)
        if idx_col == 0:
            ax.set_ylabel('y')

    fig.tight_layout()
    plt.show()

d = pd.read_csv('trained.csv')
print(d.loc[:, ['y', 'yhat']])
plot(d)
