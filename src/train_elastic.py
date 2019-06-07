import numpy as np
import argparse
import scipy
from tqdm import tqdm
import _pickle as cPickle
from sklearn.preprocessing import MinMaxScaler
from lifelines.utils import concordance_index
from glmnet_py import *
import operator
import glmnet
import glmnetCoef

__author__ = 'Bonggun Shin', 'Sungsoo Park'

def elastic_coxph(x, surv, cen, x_names, alp=False, lam=False):
    y = np.stack([surv, 1 - cen], axis=1)

    lam = [0.1,0.5,1]
    alp = [0.1,0.5,1]

    result = []
    for a in tqdm(alp):
        fit = glmnet.glmnet(x=x.copy(), y=y.copy(), family='cox', alpha=a, nlambda=100)

        for l in lam:
            beta = glmnetCoef.glmnetCoef(fit, s=scipy.float64([l]), exact=False)
            beta = beta.flatten()
            try:
                features = x_names[np.array([int(i) for i, e in enumerate(beta) if e != 0])]
            except:
                features = []
            result.append((a, l, features, beta))
    return result


def run(cancer, target_dir):
    target_split = cPickle.load(open(target_dir + cancer + '/mrna_split.cPickle', 'rb'))
    x_trn = np.log2(target_split[0] + 1)
    x_dev = np.log2(target_split[4] + 1)
    x_tst = np.log2(target_split[8] + 1)

    scaler = MinMaxScaler()
    print(scaler.fit(x_trn))
    x_trn = scaler.transform(x_trn)
    x_dev = scaler.transform(x_dev)
    x_tst = scaler.transform(x_tst)

    target_split = (x_trn, target_split[1], target_split[2], target_split[3],
                    x_dev, target_split[5], target_split[6], target_split[7],
                    x_tst, target_split[9], target_split[10], target_split[11], target_split[12])

    x_trn, c_trn, s_trn, i_trn, x_dev, c_dev, s_dev, i_dev, x_tst, c_tst, s_tst, i_tst, x_names = target_split
    s_trn = s_trn + 1e-10
    s_dev = s_dev + 1e-10
    s_tst = s_tst + 1e-10

    result = elastic_coxph(x_trn, s_trn, c_trn, x_names)

    cindex_dev = {}
    cindex_tst = {}

    for i in range(len(result)):
        alp, lam, features, beta = result[i]
        s_pred_dev = np.sum(-x_dev * beta, 1)
        s_pred_tst = np.sum(-x_tst * beta, 1)
        c_dev_r = np.array([int(bool(k) == False) for k in c_dev])
        c_tst_r = np.array([int(bool(k) == False) for k in c_tst])
        c_index_dev = concordance_index(s_dev, s_pred_dev, c_dev_r)
        c_index_tst = concordance_index(s_tst, s_pred_tst, c_tst_r)

        cindex_dev[(alp, lam)] = c_index_dev
        cindex_tst[(alp, lam)] = c_index_tst

    k, item = max(cindex_dev.items(), key=operator.itemgetter(1))
    cindex_dev_max = item
    cindex_tst_max = cindex_tst[k]

    print(cancer, ' C-index of test set : ',cindex_tst_max)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', default="LUAD", type=str)  # cancer_type
    parser.add_argument('-target', default='../data/', type=str)
    args = parser.parse_args()

    run(args.d, args.target)