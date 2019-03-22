from os.path import join
from os import environ

from sklearn.datasets import load_svmlight_file
from sklearn.metrics import roc_auc_score, mean_squared_error

from fastFM import als
from tffm import TFFMRegressor

import xlearn as xl
import numpy as np
import pandas as pd
import tensorflow as tf

import scipy
import sys

class FactorisationMachines:
    def __init__(self, onlyResults=False, metric='rmse'):
        self._data_path = './data/libsvm/'
        self._libfm_path = './models/fm/libfm/'
        self._xlearn_path = './models/fm/xlearn/'
        self.X_train, self.y_train = load_svmlight_file(join(self._data_path,'train.libsvm'))
        self.X_test, self.y_test = load_svmlight_file(join(self._data_path,'test.libsvm'))

        self.onlyResults = onlyResults
        self.metric = metric

    def libfm(self):
        # ./libFM -train /media/ntfs/Workspace/Project/rec-sys/data/libsvm/train.libsvm -test /media/ntfs/Workspace/Project/rec-sys/data/libsvm/test.libsvm -task r -dim '1,1,4' -iter 1000 -method mcmc
        predictions = pd.read_csv(join(self._libfm_path, 'output.libfm'), header=None).values.flatten()
        testY = self.y_test
        rmse = np.sqrt(mean_squared_error(testY, predictions))
        if self.onlyResults: print("Completed LibFM evaluation.")
        else: print("LibFM RMSE: {}".format(rmse))
        return rmse

    def fastfm(self):
        fm = als.FMRegression(n_iter=100, init_stdev=0.1, rank=4, l2_reg_w=0.1, l2_reg_V=0.5)
        fm.fit(self.X_train, self.y_train)
        y_pred = fm.predict(self.X_test)
        return np.sqrt(mean_squared_error(self.y_test, y_pred))
        # fm = als.FMRegression(n_iter=0, rank=4)
        # fm.fit_predict(self.X_train, self.y_train, self.X_test)
        # last_pred = None
        # for i in range(100):
        #     last_pred = y_pred = fm.fit_predict(self.X_train, self.y_train, self.X_test, n_more_iter=1)
        #     if not self.onlyResults: print("Iteration {}, RMSE: {:.6f}".format(i, np.sqrt(mean_squared_error(y_pred, self.y_test))))
        # if self.onlyResults: print("Completed fastFM evaluation.")
        # return np.sqrt(mean_squared_error(last_pred, self.y_test))
        

    def tffm(self):
        show_progress = True if not self.onlyResults else False
        X_train, y_train, X_test, y_test = self.X_train.todense(), np.transpose(self.y_train).flatten(), self.X_test.todense(), np.transpose(self.y_test).flatten()
        if self.onlyResults: environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

        model = TFFMRegressor(
            order=2,
            rank=4,
            optimizer=tf.train.FtrlOptimizer(learning_rate=0.1),
            n_epochs=100,
            batch_size=-1,
            init_std=0.001,
            input_type='dense'
        )

        model.fit(X_train, y_train, show_progress=show_progress)
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        if not self.onlyResults: print('RMSE: {:.6f}'.format(rmse))
        model.destroy()
        if self.onlyResults: print("Completed tffm evaluation.")
        return rmse

    def xlearn(self):
        # Training task
        fm_model = xl.create_fm()
        fm_model.setTrain(join(self._data_path, 'train.libsvm')) 
        fm_model.setValidate(join(self._data_path, 'test.libsvm'))
        
        if self.onlyResults: fm_model.setQuiet()
        param = {'task':'reg', 'lr':0.1, 'lambda':0.002, 'metric':'rmse', 'opt':'ftrl'}

        fm_model.fit(param, join(self._xlearn_path, 'model.out'))

        # Prediction task
        fm_model.setTest(join(self._data_path, 'test.libsvm'))
        fm_model.setSigmoid()
        fm_model.predict(join(self._xlearn_path, 'model.out'), join(self._xlearn_path, 'output.txt'))
        predictions = pd.read_csv(join(self._xlearn_path, 'output.txt'), header=None).values.flatten()
        if self.onlyResults: print("Completed xLearn evaluation.")
        return np.sqrt(mean_squared_error(self.y_test, predictions))

    def run(self):
        print(self.libfm())
        print(self.fastfm())
        print(self.tffm())
        print(self.xlearn())

if __name__ == "__main__":
    fm = FactorisationMachines(onlyResults=False)
    fm.run()