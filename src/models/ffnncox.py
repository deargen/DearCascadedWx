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
from lifelines import CoxPHFitter, KaplanMeierFitter
from sklearn.metrics import roc_auc_score
# from sklearn.feature_selection import VarianceThreshold
from tqdm import tqdm
import os
import models.utils as helper

class SurvivalFFNNCOX(SurvivalNeuralNet):
    def __init__(self, model_name, cancer, omics_type, out_folder, epochs=1000, vecdim=10):
        super(SurvivalFFNNCOX, self).__init__(model_name, cancer, omics_type, out_folder, epochs)
        self.vecdim = vecdim
        self.selected_idx = None
        self.random_seed = 1
        self.cancer_type = cancer
        self.omics_type = omics_type
        self.out_folder = out_folder

    def DoFeatureSelectionCPH(self, x, c, s, xnames, fold, sel_f_num, dev_index):
        variance_th = 0.15
        xdf = pd.DataFrame(x,columns=xnames)
        sel_idx = xdf.std() > variance_th#true or false
        xdf = xdf.loc[:, sel_idx]
        xnames = xnames[sel_idx]
        x = xdf.values

        gene_p_value = []
        for i in tqdm(range(0, x.shape[1])):
            subset_num = i
            cph_h_trn_stack = np.column_stack((x[:,subset_num:subset_num+1], c, s))
            cph_cols = xnames.copy().tolist()[subset_num:subset_num+1]
            cph_cols.append('E')
            cph_cols.append('S')
            cph_train_df= pd.DataFrame(cph_h_trn_stack,columns=cph_cols)
            cph = CoxPHFitter()
            cph.fit(cph_train_df,duration_col='S',event_col='E', step_size= 0.1, show_progress=False)
            f_scores = pd.DataFrame(cph.summary)['p'].values
            gene_p_value.append(f_scores[0])

        gene_p_value = np.asarray(gene_p_value)
        sort_idx = np.argsort(gene_p_value)
        f_name_sort = np.asarray(xnames)[sort_idx]
        f_score_sort = gene_p_value[sort_idx]

        return sort_idx, f_name_sort, f_score_sort#, auc 

    def feature_selection(self, x, c, s, xnames, fold, sel_f_num, dev_index):  
        save_feature_file = self.out_folder+'/FFNNCOX/selected_features_'+self.cancer_type+'_'+self.omics_type+'_'+str(fold)+'.csv'

        if os.path.isfile(save_feature_file):
            df = pd.read_csv(save_feature_file)
            sort_index = df['index'].values
            final_sel_idx = sort_index[:sel_f_num]
        else:
            sort_idx, f_name_sort, f_score_sort = self.DoFeatureSelectionCPH(x, c, s, xnames, fold, sel_f_num, dev_index)
            with open(save_feature_file,'wt') as wFile:
                wFile.writelines("gene,pvalue,index\n")
                for n,idx in enumerate(sort_idx):
                    wFile.writelines(str(f_name_sort[n].split('|')[0])+','+str(f_score_sort[n])+','+str(idx)+'\n')
                    
            final_sel_idx = sort_idx[:sel_f_num]

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