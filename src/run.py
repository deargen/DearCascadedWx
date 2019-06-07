import os
import _pickle as cPickle
import time
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import importlib
import pandas as pd
import argparse
from sklearn.utils import shuffle

__author__ = 'Bonggun Shin','Sungsoo Park'

class Timer(object):
    def __init__(self, name, callback):
        self.name = name
        self.callback = callback

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name)
        print('Elapsed: %s' % (time.time() - self.tstart))
        self.callback(self.name, time.time() - self.tstart)

class Experiment:
    def __init__(self, cancer, method_name="", fold=5, n_feature=50):
        self.cancer = cancer
        self.method_name = method_name
        self.fold = fold
        self.data_rel_path = '../data'
        self.fold_start = 0
        self.resumed = False
        self.n_feature = n_feature
        self.omics_type = 'mrna'
        self.method_type = 'cascaded'#only works in [CWX, RF, FISHER, TRACERATIO]

        if self.method_type == 'cascaded':
            self.out_folder = '../result_cas_'+self.omics_type
        if self.method_type == 'no cascaded':
            self.out_folder = '../result_nocas_'+self.omics_type

        if not os.path.exists(self.out_folder):
            os.makedirs(self.out_folder)
        if not os.path.exists(self.out_folder+'/'+method_name):
            os.makedirs(self.out_folder+'/'+method_name)

        self.filename = self.out_folder+'/%s.%s.%d.cpkl' % (self.cancer, self.method_name, self.n_feature)

        self.key_template = 'cindex.f%d.t%d'

        if os.path.isfile(self.filename) :
            self.fold_start, self.trn_index_list, self.tst_index_list, self.score_dev_list, self.score_list = cPickle.load(
                open(self.filename, 'rb'))

            self.resumed = True

            # print('resuming the exp')
            self.result = {}
        else:
            self.result = {}

    def record_time(self, name, t):
        self.result['time.'+name] = t

    def load_model(self):
        module = importlib.import_module('models.%s' % self.method_name.lower())
        class_ = getattr(module, 'Survival' + self.method_name)
        self.model_instance = class_(self.method_name, self.cancer, self.omics_type, self.out_folder)

    def load_data(self):
        # load data
        with Timer('loading', self.record_time):
            if self.omics_type == 'mrna':
                self.data = cPickle.load(open(self.data_rel_path+'/%s/' % self.cancer + 'mrna_split.cPickle', 'rb'))

            x_trn = np.log2(self.data[0] + 1)
            x_dev = np.log2(self.data[4] + 1)
            x_tst = np.log2(self.data[8] + 1)

            scaler = MinMaxScaler()
            scaler.fit(x_trn)
            x_trn = scaler.transform(x_trn)
            x_dev = scaler.transform(x_dev)
            x_tst = scaler.transform(x_tst)

            target_split = (x_trn, self.data[1], self.data[2], self.data[3],
                            x_dev, self.data[5], self.data[6], self.data[7],
                            x_tst, self.data[9], self.data[10], self.data[11], self.data[12])

        x_trn, c_trn, s_trn, i_trn, x_dev, c_dev, s_dev, i_dev, x_tst, c_tst, s_tst, i_tst, x_names = target_split

        # Elasticnet (glmnet) complains about 0 survival, so add epsilon to survival values
        s_trn = s_trn + 1e-10
        s_dev = s_dev + 1e-10
        s_tst = s_tst + 1e-10

        self.x_all = np.concatenate((x_trn, x_dev, x_tst), axis=0)
        self.c_all = np.concatenate((c_trn, c_dev, c_tst), axis=0)
        self.s_all = np.concatenate((s_trn, s_dev, s_tst), axis=0)
        self.i_all = np.concatenate((i_trn, i_dev, i_tst), axis=0)


        self.x_names = x_names
        self.org_x_names = x_names
        self.n = self.x_all.shape[0]

        if self.resumed is False:
            self.kf = KFold(n_splits=self.fold)
            self.kf.get_n_splits(self.x_all)

            self.trn_index_list = []
            self.tst_index_list = []
            for trn_index, tst_index in self.kf.split(self.x_all):
                self.trn_index_list.append(trn_index)
                self.tst_index_list.append(tst_index)


    def get_data(self, cvi):
        trn_index = self.trn_index_list[cvi]
        tst_index = self.tst_index_list[cvi]

        return self.x_all[trn_index], self.c_all[trn_index], self.s_all[trn_index], self.i_all[trn_index],\
               self.x_all[tst_index], self.c_all[tst_index], self.s_all[tst_index], self.i_all[tst_index]

    def run_experiment(self):
        if self.resumed is False:
            self.score_dev_list = []
            self.score_list = []

        for i in range(self.fold_start, self.fold):
            x_trn, c_trn, s_trn, i_trn, x_tst, c_tst, s_tst, i_tst = self.get_data(i)

            #pre filtering
            if self.method_name == 'FFNNCWX' or self.method_name == 'FFNNFSCORE' or self.method_name == 'FFNNRFS':

                if self.method_name == 'FFNNRFS':
                    variance_th = 0.18
                else:
                    variance_th = 0.0

                if self.method_name == 'FFNNCWX':
                    variance_th = 0.11
                    if self.method_type == 'no cascaded':
                        variance_th = 0.00

                xdf = pd.DataFrame(x_trn,columns=self.org_x_names)
                xdf_test = pd.DataFrame(x_tst,columns=self.org_x_names)
                sel_idx_trn = xdf.std() > variance_th#true or false
                sel_idx_test = xdf_test.std() > variance_th                
                sel_idx = sel_idx_trn & sel_idx_test
                xdf = xdf.loc[:, sel_idx]
                
                self.x_names = self.org_x_names[sel_idx]
                x_trn = xdf.values
                x_tst = x_tst[:,sel_idx]
                print(x_trn.shape)
                print(x_tst.shape)

            #tran & evaluation
            score_dev = self.model_instance.train(x_trn, c_trn, s_trn, self.x_names, i, self.n_feature)
            score = self.model_instance.evaluate(x_tst, c_tst, s_tst, i, self.n_feature, do_gse_eval = False)
            self.score_dev_list.append(score_dev)
            self.score_list.append(score)
            self.fold_start = i+1

            print('[Fold %d] cindex_dev=%f, cindex_tst=%f' % (i, score_dev, score))
            self.save_result()

    def save_result(self):
        cPickle.dump((self.fold_start, self.trn_index_list, self.tst_index_list, self.score_dev_list, self.score_list), open(self.filename, 'wb'))

    def get_aligned_result(self):
        mean = np.mean(self.score_list)
        std = np.std(self.score_list)

        result = [mean, std]
        result = result + self.score_list

        result = [str(self.n_feature)] + ['%.4f' % item for item in result]

        return result

    def print_result(self, sep='\t'):
        print("===================================== [%s] %s =====================================" % (
            self.method_name, self.cancer))
        n_fold = len(self.score_list)
        if self.fold == n_fold:
            print('Avg'+sep+sep.join(map(str, range(self.fold))))
            one_line = [str(np.mean(self.score_dev_list))]
            for i in range(self.fold):
                one_line.append(str(self.score_dev_list[i]))

            print(sep.join(one_line))

            one_line = [str(np.mean(self.score_list))]
            for i in range(self.fold):
                one_line.append(str(self.score_list[i]))

            print(sep.join(one_line))

        else:
            print('Avg' + sep + sep.join(map(str, range(n_fold))))
            one_line = [str(np.mean(self.score_dev_list))]
            for i in range(n_fold):
                one_line.append(str(self.score_dev_list[i]))

            print(sep.join(one_line))

            one_line = [str(np.mean(self.score_list))]
            for i in range(n_fold):
                one_line.append(str(self.score_list[i]))

            print(sep.join(one_line))



if __name__ == "__main__":
    
    cancer_list = ['LUAD', 'READ', 'BLCA', 'BRCA', 'LUSC']
    model_list = ['FFNNFISHER', 'FFNNFSCORE', 'FFNNLLL21', 'FFNNRELIEFF', 'FFNNRF', 'FFNNCONW',
                'FFNNTRACERATIO', 'FFNNXGB', 'FFNNCONW', 'FFNNSVM', 'FFNNCOX', 'FFNNRFS', 'FFNNDESEQ','FFNNCWX']

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', default=['LUAD'],
                        nargs='+', choices=cancer_list + ['ALL'], type=str)
    parser.add_argument('-m', default=['FFNNCWX'], nargs='+', choices=model_list + ['ALL'], type=str)
    parser.add_argument('-p', default='../data', type=str)
    parser.add_argument('-gpu', default='0', type=str)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if 'ALL' not in args.m:
        model_list = args.m

    if 'ALL' not in args.d:
        cancer_list = args.d

    n_feature_start = 1
    n_feature_end = 100
    manual_f_num = [1,25,50,75,100]
    for cancer in cancer_list:
        with open('./'+cancer+'_result.csv','at') as wFile:                
            for method_name in model_list:

                print('train')
                for i in range(n_feature_start, n_feature_end+1):
                # for i in manual_f_num:
                    exp = Experiment(cancer, method_name, n_feature=i)
                    exp.load_model()
                    exp.load_data()
                    exp.run_experiment()

                title = cancer+','+method_name+',dev,f1,f2,f3,f4,f5\n'
                wFile.writelines(title)
                print(method_name + ' show result')
                for i in range(n_feature_start, n_feature_end+1):
                # for i in manual_f_num:
                    exp = Experiment(cancer, method_name, n_feature=i)
                    out = ','.join(exp.get_aligned_result())
                    print(out)
                    wFile.writelines(out+'\n')
                wFile.writelines('\n')
