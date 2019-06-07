import argparse
import time
import wget
import os
import tarfile
import _pickle as cPickle
import pandas as pd
import numpy as np

__author__ = "Bonggun Shin", "Eunji Heo", "Sungsoo Park"

class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name)
        print('Elapsed: %s' % (time.time() - self.tstart))


class Preprocessor(object):
    mrna_dir = 'gdac.broadinstitute.org_%s.Merge_rnaseqv2__illuminahiseq_rnaseqv2__unc_edu__Level_3__RSEM_genes_normalized__data.Level_3.2016012800.0.0/'
    clinical_dir = 'gdac.broadinstitute.org_%s.Clinical_Pick_Tier1.Level_4.2016012800.0.0/'
    filename_pattern = {
                         'mrna': mrna_dir+"%s.rnaseqv2__illuminahiseq_rnaseqv2__unc_edu__Level_3__RSEM_genes_normalized__data.data.txt",
                         'clinical': clinical_dir+"%s.clin.merged.picked.txt",
    }

    def __init__(self, cancer_type, base_path):
        self.cancer_type = cancer_type
        self.base_path = base_path
        self.have_tartget = False

    def _imputation(self):
        print("imputation")

        print("Before imputation", self.base_df.isnull().values.any())
        pids = self.base_df['PID']

        self.base_df = self.base_df.drop('PID', axis=1)
        self.base_df = self.base_df.apply(pd.to_numeric)
        self.base_df = self.base_df.fillna(self.base_df.mean())

        print("After imputation", self.base_df.isnull().values.any())
        print(self.base_df.isnull().any(1).nonzero()[0])

        self.base_df['PID'] = pids


    def targets(self):

        filename = self.base_path+self.filename_pattern['clinical'] % (self.cancer_type, self.cancer_type)
        df_all = pd.read_csv(filename, sep='\t')
        df_all.index = df_all.iloc[:, 0]
        df_all.drop(['Hybridization REF'], axis=1, inplace=True)
        df_all = df_all.loc[['days_to_death', 'days_to_last_followup']]
        df_all = df_all.T
        new_data = []
        for i, row in df_all.iterrows():
            one_row = []
            death = float(row['days_to_death'])
            follow = float(row['days_to_last_followup'])
            if not (np.isnan(death) and np.isnan(follow)):
                one_row.append(i.upper())
                if np.isnan(death):
                    # not dead => censored = 1
                    one_row.append(1)
                    one_row.append(follow)
                else:
                    # dead => consored=0
                    one_row.append(0)
                    one_row.append(death)
            if death < 0 or follow < 0:
                print('[%s] neg day?! follow(%f), death(%f)' % (i, follow, death))
                continue

            if len(one_row)==0:
                print ('[%s] both days_to_death and days_to_last_followup are NAN!!'%i)
                continue
            new_data.append(one_row)

        new_df = pd.DataFrame(data=new_data)
        new_df.columns = ['PID', 'censored', 'survival']
        new_df.index = new_df.PID
        # new_df.drop('PID',axis=1, inplace=True)
        self.target_total_df = new_df
        print("target_total_df", self.target_total_df.shape)
        self.have_tartget = True

    # == Feature types
    def pp_rna(self):

        pkl_path = self.base_path + "mrna.cPickle"
        if os.path.exists(pkl_path):
            self.mrna = cPickle.load(open(pkl_path, 'rb'))
            return self.mrna

        if not self.have_tartget:
            self.targets()

        luad_mrna = pd.read_csv(self.base_path + self.filename_pattern['mrna'] % (self.cancer_type, self.cancer_type), sep='\t', low_memory=False)
        luad_mrna.drop(0, axis=0, inplace=True)
        luad_mrna.index = luad_mrna['Hybridization REF']
        luad_mrna.drop(['Hybridization REF'], axis=1, inplace=True)
        luad_mrna.columns = ["-".join(x.split("-")[:3]) for x in luad_mrna.columns]
        luad_mrna = luad_mrna.T
        luad_mrna = luad_mrna[luad_mrna.index.isin(self.target_total_df.index.tolist())]
        luad_mrna['PID'] = luad_mrna.index.tolist()
        luad_mrna.drop_duplicates('PID', keep='first', inplace=True)
        self.base_df = luad_mrna

        self._imputation()
        # self.mrna = self.base_df.copy()
        self.mrna = pd.merge(self.base_df, self.target_total_df, left_on='PID', right_on='PID')
        self.mrna.index = self.mrna['PID']
        self.mrna.drop(['PID'], axis=1, inplace=True)

        cPickle.dump(self.mrna, open(pkl_path, 'wb'))
        return self.mrna


def download(download_list, target_dir):
    target_dir = target_dir+"%s/"

    base_path = 'http://gdac.broadinstitute.org/runs/stddata__2016_01_28/data/%s/20160128/'
    clinical_path = 'gdac.broadinstitute.org_%s.Clinical_Pick_Tier1.Level_4.2016012800.0.0.tar.gz'
    mrna_path = 'gdac.broadinstitute.org_%s.Merge_rnaseqv2__illuminahiseq_rnaseqv2__unc_edu__Level_3__RSEM_genes_normalized__data.Level_3.2016012800.0.0.tar.gz'

    def down_if_required(target_path, base_address, cancer, job_path, job_name):
        file_name = job_path % cancer
        if os.path.isfile(target_path + file_name):
            print('[File already exists]: ' + target_path + file_name)

        else:
            with Timer(job_name):
                url = base_address + file_name
                print(url)
                filename = wget.download(url, target_path)
                print(filename)

        def extract_nonexisting(directory, archive):
            for name in archive.getnames():
                if os.path.exists(os.path.join(directory, name)):
                    print(name, "already exists!")

                else:
                    archive.extract(name, path=directory)
                    print(name, "extracted.")

        with Timer("Extracting..."):
            with tarfile.open(target_path + file_name, "r:gz") as archive:
                extract_nonexisting(target_path, archive)


    def preprocess_and_save(target_path, cancer):
        pp = Preprocessor(cancer, target_path)

        with Timer("Preprocessing mrna..."):
            mrna = pp.pp_rna()
            print(mrna.values.shape)

    for cancer in download_list:
        print ("================================ Cancer: %s ================================" % cancer)
        base_address = base_path%cancer
        if not os.path.exists(target_dir%cancer):
            os.makedirs(target_dir%cancer)

        down_if_required(target_dir%cancer, base_address, cancer, clinical_path, "Clinical")
        down_if_required(target_dir%cancer, base_address, cancer, mrna_path, "mRNASeq")

        preprocess_and_save(target_dir%cancer, cancer)



if __name__=="__main__":

    cancer_list = ['LUAD','LUSC','READ','BLCA', 'BRCA']

    parser = argparse.ArgumentParser()
    parser.add_argument('-cancer', default='ALL', choices=cancer_list + ['ALL'], type=str)
    parser.add_argument('-target', default='../data/', type=str)
    args = parser.parse_args()

    if not os.path.exists(args.target):
        os.makedirs(args.target)

    if args.cancer=='ALL':
        download_list = cancer_list

    else:
        download_list = [args.cancer]

    download(download_list, args.target)