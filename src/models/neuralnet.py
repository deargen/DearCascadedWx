from models.base import BaseModel
import numpy as np
import operator
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Dense, concatenate, Dropout, Activation
from keras import optimizers, applications, callbacks
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from lifelines.utils import concordance_index
from keras.callbacks import LearningRateScheduler
from abc import ABCMeta, abstractmethod
from sklearn.metrics import roc_auc_score
import pandas as pd

__author__ = 'Bonggun Shin','Sungsoo Park'

class MyCallback(ModelCheckpoint):
    def __init__(self, filepath, data, real_save=True, patience=20):
        super(MyCallback, self).__init__(filepath, save_weights_only=True)
        self.patience = patience

        self.x_trn, self.c_trn, self.s_trn, self.x_dev, self.c_dev, self.s_dev = data

        self.cindex_dev = 0
        self.cindex_best_epoch = 0
        self.real_save = real_save
        self.filepath_template = self.filepath+'-%s'
        self.max_epoch = 100

    def print_status(self):
        print('\n=========================== [Best cindex (epoch = %d)] cindex=%f =================================='
              % (self.cindex_best_epoch, self.cindex_dev))


    def on_train_end(self, logs=None):
        print('[Best:on_train_end]')
        self.print_status()

    def on_epoch_end(self, epoch, logs=None):
        pred_dev = -np.exp(self.model.predict(self.x_dev, batch_size=1, verbose=0))
        cindex_dev = concordance_index(self.s_dev, pred_dev, self.c_dev)

        if self.cindex_dev < cindex_dev:
            self.cindex_dev = cindex_dev
            self.cindex_best_epoch = epoch
            if self.real_save is True:
                if self.save_weights_only:
                    self.model.save_weights(self.filepath, overwrite=True)
                else:
                    self.model.save(self.filepath, overwrite=True)

        else:
            if epoch - self.cindex_best_epoch > self.patience:
                self.model.stop_training = True
                print("Early stopping at %d" % epoch)

        if epoch > self.max_epoch:
                self.model.stop_training = True
                print("Stopping at max epoch %d" % epoch)

class SurvivalNeuralNet(BaseModel):
    def __init__(self, model_name, cancer, omics_type, out_folder, epochs=100):
        super(SurvivalNeuralNet, self).__init__(model_name, cancer, omics_type, out_folder)
        self.epochs = epochs
        self.model_name = model_name
        self.cancer = cancer
        self.omics_type = omics_type
        self.out_folder = out_folder

    @abstractmethod
    def get_model(self, input_size, dropout):
        pass

    def reset_weights(self):
        session = K.get_session()
        for layer in self.model.layers:
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel.initializer.run(session=session)

    @abstractmethod
    def preprocess(self, x, n_feature):
        pass

    def feature_selection(self, x, c, s, names, fold, sel_num):
        pass

    def train(self, x, c, s, names, fold, n_feature=50):
        #learning_ratio = 1e-3
        n = x.shape[0]
        dev_index = n * 3 // 4

        x = self.preprocess(x, c, s, names, fold, n_feature, dev_index)
        x_trn, x_dev = x[:dev_index], x[dev_index:]
        c_trn, c_dev = 1 - c[:dev_index], 1 - c[dev_index:]
        s_trn, s_dev = s[:dev_index], s[dev_index:]

        sort_idx = np.argsort(s_trn)[::-1]
        x_trn = x_trn[sort_idx]
        s_trn = s_trn[sort_idx]
        c_trn = c_trn[sort_idx]

        def nll(E, NUM_E):
            def loss(y_true, y_pred):
                hazard_ratio = K.exp(y_pred)
                log_risk = K.log(K.cumsum(hazard_ratio))
                uncensored_likelihood = K.transpose(y_pred) - log_risk
                censored_likelihood = uncensored_likelihood * E
                neg_likelihood = -K.sum(censored_likelihood) / NUM_E
                return neg_likelihood

            return loss

        input_size = len(x[0])

        cindex_dev = {}
        # for dropout in [0.0, 0.5]:
        for dropout in [0.0]:
            self.model = self.get_model(input_size, dropout)
            for lr in [0.1, 0.01, 0.001, 0.0001]:
                print('############## Run at ', fold, dropout, lr)                
                adam = optimizers.Adam(lr=lr)
                self.model.compile(loss=[nll(c_trn, np.sum(c_trn))], optimizer=adam)

                data = (x_trn, c_trn, s_trn, x_dev, c_dev, s_dev)
                modelpath = self.out_folder+'/%s/%s_(%d)_%0.1f_%0.5f.hdf5' % (self.model_name, self.cancer, fold, dropout, lr)

                checkpoint = MyCallback(modelpath, data)

                self.model.fit(x_trn, s_trn, epochs=self.epochs, batch_size=len(x_trn),
                            verbose=0, shuffle=False, callbacks=[checkpoint])
                self.model.load_weights(modelpath)
                pred_raw = self.model.predict(x_dev, batch_size=1, verbose=1)
                pred_dev = -np.exp(pred_raw)
                cindex_dev_max = concordance_index(s_dev, pred_dev, c_dev)

                cindex_dev[modelpath] = cindex_dev_max

                self.reset_weights()

        self.bestmodelpath, self.cindex_dev_max = max(cindex_dev.items(), key=operator.itemgetter(1))

        return self.cindex_dev_max

    def evaluate(self, x, c, s, fold, n_feature=50, do_gse_eval=False):
        x = self.preprocess_eval(x)
        self.model.load_weights(self.bestmodelpath)
        pred_raw = self.model.predict(x, batch_size=1, verbose=1)
        pred_tst = -np.exp(pred_raw)
        self.cindex_tst_max = concordance_index(s, pred_tst, 1-c)

        K.clear_session()

        return self.cindex_tst_max