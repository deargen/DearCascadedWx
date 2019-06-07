import os
import numpy as np
from sklearn.utils import shuffle
from keras.utils import to_categorical

__author__ = 'Sungsoo Park'

def get_risk_group(x_trn, c_trn, s_trn, high_risk_th, low_risk_th):
    hg = []
    lg = []
    for n,os in enumerate(s_trn):
        if os <= high_risk_th and c_trn[n] == 0:
            hg.append(x_trn[n])
        if os > low_risk_th:
            lg.append(x_trn[n])

    return np.asarray(hg), np.asarray(lg)

def get_train_val(hg, lg, is_categori_y, seed):
    x_all = np.concatenate([hg, lg])
    hg_y = np.ones(len(hg))
    lg_y = np.zeros(len(lg))
    y_all = np.concatenate([hg_y, lg_y])
    if is_categori_y:            
        y_all = to_categorical(y_all, num_classes=2)
    x_all, y_all = shuffle(x_all, y_all, random_state=seed)
    n = len(x_all)
    dev_index = n * 4 // 5

    return x_all[:dev_index], y_all[:dev_index], x_all[dev_index:], y_all[dev_index:]

def get_train_val_dfs(x_all, c_all, s_all, seed):
    e_all = 1 - c_all
    x_all, e_all, s_all = shuffle(x_all, e_all, s_all, random_state=seed)
    n = len(x_all)
    dev_index = n * 4 // 5

    return x_all[:dev_index], e_all[:dev_index], s_all[:dev_index], x_all[dev_index:], e_all[dev_index:], s_all[dev_index:]

def get_train(hg, lg, is_categori_y, seed):
    x_all = np.concatenate([hg, lg])
    hg_y = np.ones(len(hg))
    lg_y = np.zeros(len(lg))
    y_all = np.concatenate([hg_y, lg_y])
    if is_categori_y:
        y_all = to_categorical(y_all, num_classes=2)
    x_all, y_all = shuffle(x_all, y_all, random_state=seed)
    return x_all, y_all
