from models.neuralnet import SurvivalNeuralNet
# from models.feedforwardnet import SurvivalFeedForwardNet
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.regularizers import L1L2
import numpy as np
import pandas as pd
import _pickle as cPickle
from keras.utils import to_categorical
from wx_hyperparam import WxHyperParameter
from wx_core import DoFeatureSelectionWX
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score
from sklearn import svm
import os
import models.utils as helper

class SurvivalFFNNDESEQ(SurvivalNeuralNet):
    def __init__(self, model_name, cancer, omics_type, out_folder, epochs=1000, vecdim=10):
        super(SurvivalFFNNDESEQ, self).__init__(model_name, cancer, omics_type, out_folder, epochs)
        self.vecdim = vecdim
        self.selected_idx = None
        self.random_seed = 1
        self.cancer_type = cancer
        self.omics_type = omics_type 
        self.out_folder = out_folder

    def feature_selection(self, x, c, s, xnames, fold, sel_f_num, dev_index):  
        def get_sel_idx_from_file(feature_list, deseq_file):
            df = pd.read_csv(deseq_file, sep='\t')
            df = df.sort_values(by=['padj'])

            f_names = df.index.values
            ret_sort_idx = []
            for n, name_ in enumerate(f_names):
                idx = np.where(xnames == name_)
                assert len(idx) == 1, "error multi index"
                ret_sort_idx.append(idx[0][0])
            return ret_sort_idx

        save_feature_file = self.out_folder+'/FFNNDESEQ/selected_features_'+self.cancer_type+'_'+self.omics_type+'_'+str(fold)+'.csv'    

        if os.path.isfile(save_feature_file):
            df = pd.read_csv(save_feature_file)
            sort_index = df['index'].values
            final_sel_idx = sort_index[:sel_f_num]
        else:
            sel_f_num_write = len(xnames)            

            deseq_result_file = './deseq_out/02out_'+self.cancer_type+'_fold'+str(fold)+'_DESeq2out.txt'
            final_sel_idx = get_sel_idx_from_file(xnames, deseq_result_file)

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