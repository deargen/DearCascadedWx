from abc import ABCMeta, abstractmethod

class BaseModel(metaclass=ABCMeta):
    def __init__(self, model_name, cancer, omics_type, out_folder):
        # self.x_trn, self.c_trn, self.s_trn, self.x_tst, self.c_tst, self.s_tst = dataset
        # self.x_trn, self.c_trn, self.s_trn = dataset
        self.model_name = model_name
        self.cancer = cancer
        self.omics_type = omics_type
        self.data_rel_path = '../data'
        self.out_folder = out_folder

    @abstractmethod
    def train(self, x_trn, c_trn, s_trn, names, fold):
        pass

    @abstractmethod
    def evaluate(self, x, c, s, fold):
        pass
