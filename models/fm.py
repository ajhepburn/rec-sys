from sklearn.datasets import load_svmlight_file
from sklearn.metrics import mean_squared_error, precision_score, recall_score, roc_auc_score
from os.path import join

from datetime import datetime

import pandas as pd
import numpy as np

import logging
import os
import sys


from os import environ

from fastFM import als
from tffm import TFFMRegressor

import xlearn as xl
import tensorflow as tf

import scipy

class FactorisationMachines:
    def __init__(self):
        self.dpath = '../data/csv/dataparser/'
        # self.dpath = '../data/libsvm/old/'
        self.logpath = 'logs/fm/'
        self.X_train, self.y_train = load_svmlight_file(join(self.dpath, 'train.libfm'))
        self.X_test, self.y_test = load_svmlight_file(join(self.dpath, 'test.libfm'))

    def logger(self, model_name):
        """Sets the logger configuration to report to both std.out and to log to ./log/models/<MODEL_NAME>/
        
        Also sets the formatting instructions for the log file, prints Time, Current Thread, Logging Type, Message.

        """
        directory = os.path.join(self.logpath, model_name)

        if not os.path.exists(directory):
            os.makedirs(directory)

        logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
                handlers=[
                    logging.FileHandler("{0}/{1}.log".format(directory, str(datetime.now())[:-7])),
                    logging.StreamHandler(sys.stdout)
                ])

    def libfm(self, loss='mcmc'):
        # ./libFM -train /media/ntfs/Workspace/Project/rec-sys/data/csv/dataparser/train.libfm -validation /media/ntfs/Workspace/Project/rec-sys/data/csv/dataparser/validation.libfm -test /media/ntfs/Workspace/Project/rec-sys/data/csv/dataparser/test.libfm -task r -dim '1,1,4' -iter 1000 -method mcmc -out /media/ntfs/Workspace/Project/rec-sys/data/csv/dataparser/out.libfm
        # predictions = pd.read_csv(join(self.dpath, 'out.libfm'), header=None).values.flatten()
        predictions = pd.read_csv(join('/home/alex/libfm_variant_test/out/', 'out-'+loss+'.libfm'), header=None).values.flatten()
        testY = self.y_test

        # prec = precision_score(testY, predictions.round(), average='weighted')
        # rec = recall_score(testY, predictions.round(), average='weighted') 
        # fmeasure = 2*((prec*rec)/(prec+rec))
        auc = roc_auc_score(testY, predictions, average='weighted')
        rmse = np.sqrt(mean_squared_error(testY, predictions))
        # print("LibFM RMSE: {}".format(rmse))
        return (auc, rmse)

    def fastfm(self):
        fm = als.FMRegression(n_iter=100, init_stdev=0.1, rank=4, l2_reg_w=0.1, l2_reg_V=0.5)
        fm.fit(self.X_train, self.y_train)
        y_pred = fm.predict(self.X_test)

        prec = precision_score(self.y_test, y_pred.round(), average='weighted')
        rec = recall_score(self.y_test, y_pred.round(), average='weighted') 
        fmeasure = 2*((prec*rec)/(prec+rec))
        auc = roc_auc_score(self.y_test, y_pred, average='macro')
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        return (auc, rmse)

    def tffm(self):
        # show_progress = True if not self.onlyResults else False
        X_train, y_train, X_test, y_test = self.X_train.todense(), np.transpose(self.y_train).flatten(), self.X_test.todense(), np.transpose(self.y_test).flatten()
        # if self.onlyResults: environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

        model = TFFMRegressor(
            order=2,
            rank=4,
            optimizer=tf.train.AdamOptimizer(learning_rate=0.1),
            n_epochs=100,
            batch_size=-1,
            init_std=0.001,
            input_type='dense'
        )

        model.fit(X_train, y_train, show_progress=True)
        predictions = model.predict(X_test)

        prec = precision_score(y_test, predictions.round(), average='weighted')
        rec = recall_score(y_test, predictions.round(), average='weighted') 
        fmeasure = 2*((prec*rec)/(prec+rec))
        auc = roc_auc_score(y_test, predictions, average='weighted')
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        model.destroy()
        print("Completed tffm evaluation.")
        return (auc, rmse)

    def xlearn(self):
        # Training task
        fm_model = xl.create_fm()
        fm_model.setTrain(join(self.dpath, 'train.libfm')) 
        fm_model.setValidate(join(self.dpath, 'validation.libfm'))
        
        fm_model.setQuiet()
        param = {'task':'weighted', 'lr':0.01, 'lambda':0.002, 'metric':'f1', 'opt':'ftrl'}

        fm_model.fit(param, join(self.dpath, 'model.out'))

        # Prediction task
        fm_model.setTest(join(self.dpath, 'test.libfm'))
        fm_model.setSigmoid()
        fm_model.predict(join(self.dpath, 'model.out'), join(self.dpath, 'output.txt'))
        predictions = pd.read_csv(join(self.dpath, 'output.txt'), header=None).values.flatten()
        print("Completed xLearn evaluation.")
        return np.sqrt(mean_squared_error(self.y_test, predictions))

    def run(self):
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        for prog in ['libfm']:
            print(prog, getattr(self, prog)())


fm = FactorisationMachines()
fm.run()