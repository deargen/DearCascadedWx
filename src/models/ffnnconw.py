from models.neuralnet import SurvivalNeuralNet
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.regularizers import L1L2
import numpy as np
import pandas as pd
import _pickle as cPickle
from keras.utils import to_categorical
from wx_hyperparam import WxHyperParameter
from wx_core import DoFeatureSelectionConnectionWeight
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score

import os
import models.utils as helper

class SurvivalFFNNCONW(SurvivalNeuralNet):
    def __init__(self, model_name, cancer, omics_type, out_folder, epochs=1000, vecdim=10):
        super(SurvivalFFNNCONW, self).__init__(model_name, cancer, omics_type, out_folder, epochs)
        self.vecdim = vecdim
        self.selected_idx = None
        self.random_seed = 1
        self.cancer_type = cancer
        self.omics_type = omics_type
        self.out_folder = out_folder

    def feature_selection(self, x, c, s, xnames, fold, sel_f_num, dev_index):  
        idx = np.where(c == 0)
        x = x[idx]
        s = s[idx]

        def get_sel_idx(feature_list, sel_feature_num):
            x_all, s_all = shuffle(x, s, random_state=self.random_seed)
            s_all = np.log10(s_all)

            n = len(x_all)
            dev_index = n * 4 // 5

            trn_x = x_all[:dev_index]
            trn_y = s_all[:dev_index]
            val_x = x_all[dev_index:]
            val_y = s_all[dev_index:]            

            feature_num = trn_x.shape[1]
            hp = WxHyperParameter(epochs=50, learning_ratio=0.001, batch_size = 16, verbose=True)
            sel_gene_num = sel_feature_num
            sel_idx, sel_genes, sel_weight, test_auc = DoFeatureSelectionConnectionWeight(trn_x, trn_y, val_x, val_y, val_x, val_y, feature_list, hp, n_sel=sel_gene_num)

            return sel_idx

        save_feature_file = self.out_folder+'/FFNNCONW/selected_features_'+self.cancer_type+'_'+self.omics_type+'_'+str(fold)+'.csv'
 
        if os.path.isfile(save_feature_file):
            df = pd.read_csv(save_feature_file)
            sort_index = df['index'].values
            final_sel_idx = sort_index[:sel_f_num]
        else:
            sel_f_num_write = len(xnames)
            final_sel_idx = get_sel_idx(xnames, sel_f_num_write)

            with open(save_feature_file,'wt') as wFile:
                wFile.writelines("gene,coef,index\n")
                for n,name in enumerate(xnames[final_sel_idx]):
                    wFile.writelines(str(name.split('|')[0])+','+str(sel_f_num_write - n)+','+str(final_sel_idx[n])+'\n')
                    
            final_sel_idx = final_sel_idx[:sel_f_num]

        return final_sel_idx

    def get_model(self, input_size, dropout):
        input_dim = input_size
        # reg = L1L2(l1=1.0, l2=0.5)
        reg = None
        inputs = Input((input_dim,))
        if dropout == 0.0:
            z = inputs#without dropout
        else:
            z = Dropout(dropout)(inputs)
        outputs = Dense(1, kernel_initializer='zeros', bias_initializer='zeros',
                        kernel_regularizer=reg,
                        activity_regularizer=reg,
                        bias_regularizer=reg)(z)
        model = Model(inputs=inputs, outputs=outputs)
        # model.summary()
        return model

    def preprocess_eval(self, x):
        x_new = x[:,self.sel_idx]
        return x_new

    def preprocess(self, x, c, s, xnames, fold, n_sel, dev_index):
        sel_idx = self.feature_selection(x, c, s, xnames, fold, n_sel, dev_index)
        self.sel_idx = sel_idx
        x_new = x[:,sel_idx]
        return x_new