import argparse
import _pickle as cPickle
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import os
from matplotlib import pyplot as plt
import matplotlib as mpl
import pandas as pd

__author__ = "Bonggun Shin"

RANDOM_SEED = 0

def split_trn_dev_tst(filename, x_trndevtst, c_trndevtst, s_trndevtst, x_names):

    cv_trn = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=RANDOM_SEED)
    i_trn, i_devtst = cv_trn.split(x_trndevtst, c_trndevtst).__next__()

    x_trn = x_trndevtst[i_trn]
    x_devtst = x_trndevtst[i_devtst]

    c_trn = c_trndevtst[i_trn]
    c_devtst = c_trndevtst[i_devtst]

    s_trn = s_trndevtst[i_trn]
    s_devtst = s_trndevtst[i_devtst]


    cv_dev = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=RANDOM_SEED)
    i_dev, i_tst = cv_dev.split(x_devtst, c_devtst).__next__()

    x_dev = x_devtst[i_dev]
    x_tst = x_devtst[i_tst]

    c_dev = c_devtst[i_dev]
    c_tst = c_devtst[i_tst]

    s_dev = s_devtst[i_dev]
    s_tst = s_devtst[i_tst]


    with open(filename, 'wb') as handle:
        cPickle.dump((x_trn, c_trn, s_trn, i_trn, x_dev, c_dev, s_dev, i_dev, x_tst, c_tst, s_tst, i_tst, x_names), handle)

    return (x_trn, c_trn, s_trn, i_trn, x_dev, c_dev, s_dev, i_dev, x_tst, c_tst, s_tst, i_tst)


def split_trn_dev_tst_for_aggregated_data(filename, x_trndevtst, c_trndevtst, s_trndevtst, x_names, filename2, x_trndevtst2, c_trndevtst2, s_trndevtst2, x_names2):
    assert (x_trndevtst.shape[0] != x_trndevtst2.shape[0], "two input should have the same # of samples")

    cv_trn = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=RANDOM_SEED)
    i_trn, i_devtst = cv_trn.split(x_trndevtst, c_trndevtst).__next__()

    x_trn = x_trndevtst[i_trn]
    x_devtst = x_trndevtst[i_devtst]

    c_trn = c_trndevtst[i_trn]
    c_devtst = c_trndevtst[i_devtst]

    s_trn = s_trndevtst[i_trn]
    s_devtst = s_trndevtst[i_devtst]


    x_trn2 = x_trndevtst2[i_trn]
    x_devtst2 = x_trndevtst2[i_devtst]

    c_trn2 = c_trndevtst2[i_trn]
    c_devtst2 = c_trndevtst2[i_devtst]

    s_trn2 = s_trndevtst2[i_trn]
    s_devtst2 = s_trndevtst2[i_devtst]


    cv_dev = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=RANDOM_SEED)
    i_dev, i_tst = cv_dev.split(x_devtst, c_devtst).__next__()

    x_dev = x_devtst[i_dev]
    x_tst = x_devtst[i_tst]

    c_dev = c_devtst[i_dev]
    c_tst = c_devtst[i_tst]

    s_dev = s_devtst[i_dev]
    s_tst = s_devtst[i_tst]


    x_dev2 = x_devtst2[i_dev]
    x_tst2 = x_devtst2[i_tst]

    c_dev2 = c_devtst2[i_dev]
    c_tst2 = c_devtst2[i_tst]

    s_dev2 = s_devtst2[i_dev]
    s_tst2 = s_devtst2[i_tst]


    with open(filename, 'wb') as handle:
        cPickle.dump((x_trn, c_trn, s_trn, i_trn, x_dev, c_dev, s_dev, i_dev, x_tst, c_tst, s_tst, i_tst, x_names), handle)

    with open(filename2, 'wb') as handle:
        cPickle.dump(
            (x_trn2, c_trn2, s_trn2, i_trn, x_dev2, c_dev2, s_dev2, i_dev, x_tst2, c_tst2, s_tst2, i_tst, x_names2),
            handle)

        return (x_trn, c_trn, s_trn, i_trn, x_dev, c_dev, s_dev, i_dev, x_tst, c_tst, s_tst, i_tst), (
            x_trn2, c_trn2, s_trn2, i_trn, x_dev2, c_dev2, s_dev2, i_dev, x_tst2, c_tst2, s_tst2, i_tst)

def split(cancer_list, target_dir):
    print(
        "| | mRNA.split  | mRNA.c_ratio | miRNA.split | miRNA.c_ratio | methyl.split | methyl.c_ratio | mut.split | mut.c_ratio | all.split | all.c_ratio |")
    print("|------|-------|-------|--------|----------|-------|--------|-------|--------|-------|-------|")

    df_list = []
    for idx, cancer in enumerate(cancer_list):
        load_path_template = target_dir + cancer + '/%s.cPickle'

        with open(load_path_template % 'mrna', 'rb') as handle:
            mrna_dataset = cPickle.load(handle)

            x_trndevtst = mrna_dataset.drop('survival', axis=1).drop('censored', axis=1).values
            x_names = mrna_dataset.drop('survival', axis=1).drop('censored', axis=1).columns.values # gene names
            s_trndevtst = mrna_dataset['survival'].values
            c_trndevtst = mrna_dataset['censored'].values

            filename = (target_dir + cancer + '/%s_split.cPickle') % 'mrna'
            mrna_dataset_split = split_trn_dev_tst(filename, x_trndevtst, c_trndevtst, s_trndevtst, x_names)
            mrna_split = "(%d/%d/%d, %d)" % (
                mrna_dataset_split[0].shape[0],
                mrna_dataset_split[4].shape[0],
                mrna_dataset_split[8].shape[0],
                mrna_dataset_split[0].shape[1])

            mrna_cratio = "(%.2f/%.2f/%.2f)" % (
                np.average(mrna_dataset_split[1]),
                np.average(mrna_dataset_split[5]),
                np.average(mrna_dataset_split[9]))

        print('| %s | %s | %s ' % (cancer, mrna_split, mrna_cratio))

if __name__=="__main__":
    cancer_list = ['LUAD','LUSC','READ','BRCA','BLCA']

    parser = argparse.ArgumentParser()
    parser.add_argument('-cancer', default='ALL', choices=cancer_list + ['ALL'], type=str)
    parser.add_argument('-target', default='../data/', type=str)
    args = parser.parse_args()

    if args.cancer!='ALL':
        cancer_list = [args.cancer]

    split(cancer_list, args.target)
