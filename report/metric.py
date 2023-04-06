import numpy as np

def get_smape(y_true, y_pred):
    '''
    https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
    '''
    smap = np.zeros(len(y_true))

    num = np.abs(y_true - y_pred)
    dem = (np.abs(y_true) + np.abs(y_pred)) / 2

    pos_ind = (y_true!=0)|(y_pred!=0)
    smap[pos_ind] = num[pos_ind] / dem[pos_ind]

    return 100*np.mean(smap)
