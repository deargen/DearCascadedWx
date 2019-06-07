import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense
#from keras.utils import to_categorical
from keras import backend as K
from keras import metrics, optimizers, applications, callbacks
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
import numpy as np
from wx_hyperparam import WxHyperParameter
import xgboost as xgb

__author__ = 'Sungsoo Park'

#set default global hyper paramerters
wx_hyperparam = WxHyperParameter(learning_ratio=0.001)

def cw_ann_reg_model(x_train, y_train, x_val, y_val, hyper_param=wx_hyperparam, hidden_layer_size=128):
    input_dim = len(x_train[0])
    inputs = Input((input_dim,))
    hidden = Dense(hidden_layer_size)(inputs)
    fc_out = Dense(1)(hidden)
    model = Model(input=inputs, output=fc_out)

    #build a optimizer
    sgd = optimizers.SGD(lr=hyper_param.learning_ratio, decay=hyper_param.weight_decay, momentum=hyper_param.momentum, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=[metrics.mse])        

    #call backs
    def step_decay(epoch):
        exp_num = int(epoch/10)+1       
        return float(hyper_param.learning_ratio/(10 ** exp_num))

    best_model_path="./slp_cw_ann_weights_best"+".hdf5"
    save_best_model = ModelCheckpoint(best_model_path, monitor="val_loss", verbose=hyper_param.verbose, save_best_only=True, mode='min')
    change_lr = LearningRateScheduler(step_decay)                                

    #run train
    history = model.fit(x_train, y_train, validation_data=(x_val,y_val), 
                epochs=hyper_param.epochs, batch_size=hyper_param.batch_size, shuffle=True, callbacks=[save_best_model, change_lr], verbose=hyper_param.verbose)

    #load best model
    model.load_weights(best_model_path)

    return model

def connection_weight(x_train, y_train, x_val, y_val, n_selection=100, hidden_layer_size=128, hyper_param=wx_hyperparam, num_cls=2):
    input_dim = len(x_train[0])

    # make model and do train
    model = cw_ann_reg_model(x_train, y_train, x_val, y_val, hyper_param=hyper_param, hidden_layer_size=hidden_layer_size)

    #load weights
    weights = model.get_weights()

    #get feature importance using connection weight algo (Olden 2004)
    wt_ih = weights[0]#.transpose() #input-hidden weights
    wt_ho = weights[1]#.transpose() #hidden-out weights
    dot_wt = wt_ih * wt_ho
    sum_wt = np.sum(dot_wt,axis=1)

    selected_idx = np.argsort(sum_wt)[::-1][0:n_selection]
    selected_weights = sum_wt[selected_idx]

    #get evaluation acc from best model
    loss, val_acc = model.evaluate(x_val, y_val)

    K.clear_session()

    return selected_idx, selected_weights, val_acc

def DoFeatureSelectionConnectionWeight(train_x, train_y, val_x, val_y, test_x, test_y, f_list, hp, n_sel = 14):
    ITERATION = 5
    feature_num = len(f_list)

    all_weight = np.zeros(feature_num)    
    all_count = np.ones(feature_num)

    accs = []
    for i in range(0, ITERATION):    
        sel_idx, sel_weight, test_acc = connection_weight(train_x, train_y, val_x, val_y, n_selection=min(n_sel*100, feature_num), hyper_param=hp)
        accs.append(test_acc)
        for j in range(0,min(n_sel*100, feature_num)):
            all_weight[sel_idx[j]] += sel_weight[j]
            all_count[sel_idx[j]] += 1        

    all_weight = all_weight / all_count
    sort_index = np.argsort(all_weight)[::-1]
    sel_index = sort_index[:n_sel]#top n_sel

    sel_index = np.asarray(sel_index)
    sel_weight =  all_weight[sel_index]
    gene_names = np.asarray(f_list)
    sel_genes = gene_names[sel_index]

    return sel_index, sel_genes, sel_weight, np.mean(accs,axis=0)

def DoFeatureSelectionWX(train_x, train_y, val_x, val_y, test_x, test_y, f_list, hp, n_sel = 14, sel_option='top'):
    ITERATION = 10
    feature_num = len(f_list)

    all_weight = np.zeros(feature_num)    
    all_count = np.ones(feature_num)

    accs = []
    for i in range(0, ITERATION):    
        sel_idx, sel_weight, test_acc = WxSlp(train_x, train_y, val_x, val_y, test_x, test_y, n_selection=min(n_sel*100, feature_num), hyper_param=hp)
        accs.append(test_acc)
        for j in range(0,min(n_sel*100, feature_num)):
            all_weight[sel_idx[j]] += sel_weight[j]
            all_count[sel_idx[j]] += 1        

    all_weight = all_weight / all_count
    sort_index = np.argsort(all_weight)[::-1]
    if sel_option == 'top':
        sel_index = sort_index[:n_sel]

    sel_index = np.asarray(sel_index)
    sel_weight =  all_weight[sel_index]
    gene_names = np.asarray(f_list)
    sel_genes = gene_names[sel_index]

    return sel_index, sel_genes, sel_weight, np.mean(accs,axis=0)

# from sklearn.metrics import roc_auc_score
def NaiveSLPmodel(x_train, y_train, x_val, y_val, hyper_param=wx_hyperparam):
    input_dim = len(x_train[0])
    inputs = Input((input_dim,))
    fc_out = Dense(2,  kernel_initializer='zeros', bias_initializer='zeros', activation='softmax')(inputs)
    model = Model(input=inputs, output=fc_out)

    #build a optimizer
    sgd = optimizers.SGD(lr=hyper_param.learning_ratio, decay=hyper_param.weight_decay, momentum=hyper_param.momentum, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    #call backs
    def step_decay(epoch):
        exp_num = int(epoch/10)+1
        return float(hyper_param.learning_ratio/(10 ** exp_num))

    best_model_path="./slp_wx_weights_best"+".hdf5"
    save_best_model = ModelCheckpoint(best_model_path, monitor="val_loss", verbose=hyper_param.verbose, save_best_only=True, mode='min')
    change_lr = LearningRateScheduler(step_decay)                                

    #run
    history = model.fit(x_train, y_train, validation_data=(x_val,y_val), 
                epochs=hyper_param.epochs, batch_size=hyper_param.batch_size, shuffle=True, callbacks=[save_best_model, change_lr])

    #load best model
    model.load_weights(best_model_path)

    return model

def WxSlp(x_train, y_train, x_val, y_val, test_x, test_y, n_selection=100, hyper_param=wx_hyperparam, num_cls=2):#suppot 2 class classification only now.
    sess = tf.Session()
    K.set_session(sess)

    input_dim = len(x_train[0])

    # make model and do train
    model = NaiveSLPmodel(x_train, y_train, x_val, y_val, hyper_param=hyper_param)

    #load weights
    weights = model.get_weights()

    #cacul WX scores
    num_data = {}
    running_avg={}
    tot_avg={}
    Wt = weights[0].transpose() #all weights of model
    Wb = weights[1].transpose() #all bias of model
    for i in range(num_cls):
        tot_avg[i] = np.zeros(input_dim) # avg of input data for each output class
        num_data[i] = 0.
    for i in range(len(x_train)):
        c = y_train[i].argmax()
        x = x_train[i]
        tot_avg[c] = tot_avg[c] + x
        num_data[c] = num_data[c] + 1
    for i in range(num_cls):
        tot_avg[i] = tot_avg[i] / num_data[i]

    #data input for first class
    wx_00 = tot_avg[0] * Wt[0]# + Wb[0]# first class input avg * first class weight + first class bias
    wx_01 = tot_avg[0] * Wt[1]# + Wb[1]# first class input avg * second class weight + second class bias

    #data input for second class
    wx_10 = tot_avg[1] * Wt[0]# + Wb[0]# second class input avg * first class weight + first class bias
    wx_11 = tot_avg[1] * Wt[1]# + Wb[1]# second class input avg * second class weight + second class bias

    wx_abs = np.zeros(len(wx_00))
    for idx, _ in enumerate(wx_00):
        wx_abs[idx] = np.abs(wx_00[idx] - wx_01[idx]) + np.abs(wx_11[idx] - wx_10[idx])

    selected_idx = np.argsort(wx_abs)[::-1][0:n_selection]
    selected_weights = wx_abs[selected_idx]

    #get evaluation acc from best model
    loss, test_acc = model.evaluate(test_x, test_y)

    K.clear_session()

    return selected_idx, selected_weights, test_acc