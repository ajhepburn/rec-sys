import os
import shutil
import itertools
import sys

import pandas as pd
import numpy as np
import sklearn.preprocessing
import papermill as pm

import tensorflow as tf
from tensorflow.python.client import device_lib

import reco_utils.recommender.wide_deep.wide_deep_utils as wide_deep
from reco_utils.common import tf_utils
from reco_utils.dataset import movielens
from reco_utils.dataset.pandas_df_utils import user_item_pairs
from reco_utils.dataset.python_splitters import python_random_split, python_stratified_split
import reco_utils.evaluation.python_evaluation

# Recommend top k items
TOP_K = 10
RANKING_METRICS = ['map_at_k', 'ndcg_at_k', 'precision_at_k', 'recall_at_k']
RATING_METRICS = ['rmse', 'mae', 'rsquared', 'exp_var']
EVALUATE_WHILE_TRAINING = True

USER_COL = 'user_id'
ITEM_COL = 'item_id'
RATING_COL = 'target'
ITEM_FEAT_COL = 'item_features'

"""Hyperparameters"""
MODEL_TYPE = 'wide_deep'
EPOCHS = 50
BATCH_SIZE = 64
# Wide (linear) model hyperparameters
LINEAR_OPTIMIZER = 'Ftrl'
LINEAR_OPTIMIZER_LR = 0.0029  # Learning rate
LINEAR_L1_REG = 0.0           # L1 Regularization rate for FtrlOptimizer
LINEAR_MOMENTUM = 0.9         # Momentum for MomentumOptimizer
# DNN model hyperparameters
DNN_OPTIMIZER = 'Adagrad'
DNN_OPTIMIZER_LR = 0.1
DNN_L1_REG = 0.0           # L1 Regularization rate for FtrlOptimizer
DNN_MOMENTUM = 0.9         # Momentum for MomentumOptimizer or RMSPropOptimizer
DNN_HIDDEN_LAYER_1 = 0     # Set 0 to not use this layer
DNN_HIDDEN_LAYER_2 = 128   # Set 0 to not use this layer
DNN_HIDDEN_LAYER_3 = 256   # Set 0 to not use this layer
DNN_HIDDEN_LAYER_4 = 32    # DNN hidden units will be = [512, 256, 128, 128]
DNN_USER_DIM = 4
DNN_ITEM_DIM = 4
DNN_DROPOUT = 0.4
DNN_BATCH_NORM = 1        # 1 to use batch normalization, 0 if not.

NUM_CPUS = os.cpu_count()


class WideDeepModel:
    def __init__(self):
        self._logpath = './log/models/widedeep/'
        self._rpath = './data/csv/cashtags_clean.csv'
        self._modeldir = './models/widedeep/'

    def csv_to_df(self, months=3):
        df = pd.read_csv(self._rpath, sep='\t')
        
        df['count'] = df.groupby(
            ['user_id', 'item_tag_ids']
        ).user_id.transform('size')
        df = df[df['count'] < months*100]

        df_weights = df[['user_id', 'item_tag_ids', 'count']].drop_duplicates(
            subset=['user_id', 'item_tag_ids']
        )
        
        df_weights = df_weights.assign(target=df_weights['count'].div(
            df_weights.groupby('user_id')['count'].transform('sum')
        ))
        df_weights = df_weights.drop('count', axis=1)
        
        df_sector_industry = df[['user_id', 'item_tag_ids', 'item_sectors', 'item_industries']]
        df_sector_industry = df_sector_industry[['user_id', 'item_tag_ids', 'item_sectors', 'item_industries']].drop_duplicates(
            subset=['user_id', 'item_tag_ids']
        )

        df2 = pd.merge(df_weights, df_sector_industry, on=["user_id", "item_tag_ids"], how="left")
        df2 = df2.rename(columns={'item_tag_ids':'item_id'}) 

        df2['item_features_str'] = df2[['item_sectors', 'item_industries']].apply(lambda x: '|'.join(x), axis=1)
        df = df2[['user_id', 'item_id', 'target','item_features_str']]

        features_encoder = sklearn.preprocessing.MultiLabelBinarizer()
        df[ITEM_FEAT_COL] = features_encoder.fit_transform(
            df['item_features_str'].apply(lambda s: s.split("|"))
        ).tolist()
        df = df.drop('item_features_str', axis=1)
        return df

    def fit_wide_deep_model(self, data_splits: tuple, user_item_info: tuple):
        train, _ = data_splits
        users, items, item_feat_shape = user_item_info

        train_steps = EPOCHS * len(train) // BATCH_SIZE
        shutil.rmtree(self._modeldir, ignore_errors=True)

        DNN_HIDDEN_UNITS = []
        if DNN_HIDDEN_LAYER_1 > 0:
            DNN_HIDDEN_UNITS.append(DNN_HIDDEN_LAYER_1)
        if DNN_HIDDEN_LAYER_2 > 0:
            DNN_HIDDEN_UNITS.append(DNN_HIDDEN_LAYER_2)
        if DNN_HIDDEN_LAYER_3 > 0:
            DNN_HIDDEN_UNITS.append(DNN_HIDDEN_LAYER_3)
        if DNN_HIDDEN_LAYER_4 > 0:
            DNN_HIDDEN_UNITS.append(DNN_HIDDEN_LAYER_4)
        
        if MODEL_TYPE is 'deep' or MODEL_TYPE is 'wide_deep':
            print("DNN hidden units =", DNN_HIDDEN_UNITS)
            print("Embedding {} users to {}-dim vector".format(len(users), DNN_USER_DIM))
            print("Embedding {} items to {}-dim vector\n".format(len(items), DNN_ITEM_DIM))

        save_checkpoints_steps = max(1, train_steps // 5)
    
        # Model type is tf.estimator.DNNLinearCombinedRegressor, known as 'wide-and-deep'
        wide_columns, deep_columns = wide_deep.build_feature_columns(
            users=users[USER_COL].values,
            items=items[ITEM_COL].values,
            user_col=USER_COL,
            item_col=ITEM_COL,
            item_feat_col=ITEM_FEAT_COL,
            user_dim=DNN_USER_DIM,
            item_dim=DNN_ITEM_DIM,
            item_feat_shape=item_feat_shape,
            model_type=MODEL_TYPE,
        )

        # Optimizer specific parameters
        linear_params = {}
        if LINEAR_OPTIMIZER == 'Ftrl':
            linear_params['l1_regularization_strength'] = LINEAR_L1_REG
        elif LINEAR_OPTIMIZER == 'Momentum' or LINEAR_OPTIMIZER == 'RMSProp':
            linear_params['momentum'] = LINEAR_MOMENTUM

        dnn_params = {}
        if DNN_OPTIMIZER == 'Ftrl':
            dnn_params['l1_regularization_strength'] = DNN_L1_REG
        elif DNN_OPTIMIZER == 'Momentum' or DNN_OPTIMIZER == 'RMSProp':
            dnn_params['momentum'] = DNN_MOMENTUM

        print(linear_params, dnn_params)

        model = wide_deep.build_model(
            model_dir=self._modeldir,
            wide_columns=wide_columns,
            deep_columns=deep_columns,
            linear_optimizer=tf_utils.build_optimizer(LINEAR_OPTIMIZER, LINEAR_OPTIMIZER_LR, **linear_params),
            dnn_optimizer=tf_utils.build_optimizer(DNN_OPTIMIZER, DNN_OPTIMIZER_LR, **dnn_params),
            dnn_hidden_units=DNN_HIDDEN_UNITS,
            dnn_dropout=DNN_DROPOUT,
            dnn_batch_norm=(DNN_BATCH_NORM==1),
            log_every_n_iter=max(1, train_steps//20),  # log 20 times
            save_checkpoints_steps=save_checkpoints_steps
        )

        # Wide columns are the features for wide model, and deep columns are for DNN
        print("\nFeature specs:")
        for c in wide_columns + deep_columns:
            print(str(c)[:100], "...")

        steps = (train_steps, save_checkpoints_steps)
        return (
            model,
            steps
        )

    def train_and_eval(self, model, data_splits: tuple, user_item_info: tuple, steps: tuple):
        train, test = data_splits
        users, items, item_feat_shape = user_item_info
        train_steps, save_checkpoints_steps = steps

        cols = {
            'col_user': USER_COL,
            'col_item': ITEM_COL,
            'col_rating': RATING_COL,
            'col_prediction': 'prediction'
        }

        # Prepare ranking evaluation set, i.e. get the cross join of all user-item pairs
        ranking_pool = user_item_pairs(
            user_df=users,
            item_df=items,
            user_col=USER_COL,
            item_col=ITEM_COL,
            user_item_filter_df=train,  # Remove seen items
            shuffle=True
        )

        """ Training hooks to track training performance (evaluate on 'train' data) 
        """

        hooks = []
        evaluation_logger = None
        if EVALUATE_WHILE_TRAINING:
            class EvaluationLogger(tf_utils.Logger):
                def __init__(self):
                    self.eval_log = {}
                def log(self, metric, value):
                    if metric not in self.eval_log:
                        self.eval_log[metric] = []
                    self.eval_log[metric].append(value)
                    print("eval_{} = {}".format(metric, value))

                def get_log(self):
                    return self.eval_log

            evaluation_logger = EvaluationLogger()

            if len(RANKING_METRICS) > 0:
                hooks.append(
                    tf_utils.evaluation_log_hook(
                        model,
                        logger=evaluation_logger,
                        true_df=test,
                        y_col=RATING_COL,
                        eval_df=ranking_pool,
                        every_n_iter=save_checkpoints_steps,
                        model_dir=self._modeldir,
                        eval_fns=[getattr(reco_utils.evaluation.python_evaluation, m) for m in RANKING_METRICS],
                        **{**cols, 'k': TOP_K}
                    )
                )
            if len(RATING_METRICS) > 0:
                hooks.append(
                    tf_utils.evaluation_log_hook(
                        model,
                        logger=evaluation_logger,
                        true_df=test,
                        y_col=RATING_COL,
                        eval_df=test.drop(RATING_COL, axis=1),
                        every_n_iter=save_checkpoints_steps,
                        model_dir=self._modeldir,
                        eval_fns=[getattr(reco_utils.evaluation.python_evaluation, m) for m in RATING_METRICS],
                        **cols
                    )
                )

        print("Training steps = {}, Batch size = {} (num epochs = {})".format(train_steps, BATCH_SIZE, EPOCHS))

        train_fn = tf_utils.pandas_input_fn(
            df=train,
            y_col=RATING_COL,
            batch_size=BATCH_SIZE,
            num_epochs=None,  # None == run forever. We use steps=TRAIN_STEPS instead.
            shuffle=True,
            num_threads=NUM_CPUS-1
        )

        tf.logging.set_verbosity(tf.logging.INFO)

        try:
            model.train(
                input_fn=train_fn,
                hooks=hooks,
                steps=train_steps
            )
        except tf.train.NanLossDuringTrainingError:
            raise ValueError(
                "Training stopped with NanLossDuringTrainingError. Try other optimizers, smaller batch size and smaller learning rate."
            )
            
        if EVALUATE_WHILE_TRAINING:
            for m, v in evaluation_logger.get_log().items():
                pm.record("eval_{}".format(m), v)

        return model, ranking_pool, cols
    
    def test_and_export(self, model, test, eval_metrics: tuple, ranking_pool, cols):
        def test_preds():
            print("TESTING MENTION PREDICTION")
            if len(RATING_METRICS) > 0:
                predictions = list(model.predict(input_fn=tf_utils.pandas_input_fn(df=test)))
                prediction_df = test.drop(RATING_COL, axis=1)
                prediction_df['prediction'] = [p['predictions'][0] for p in predictions]
                prediction_df['prediction'].describe()
        
                for m in RATING_METRICS:
                    fn = getattr(reco_utils.evaluation.python_evaluation, m)
                    result = fn(test, prediction_df, **cols)
                    pm.record(m, result)
                    print(m, "=", result)

        def test_k():
            print("TESTING TOP N REC")
            if len(RANKING_METRICS) > 0:
                predictions = list(model.predict(input_fn=tf_utils.pandas_input_fn(df=ranking_pool)))
                prediction_df = ranking_pool.copy()
                prediction_df['prediction'] = [p['predictions'][0] for p in predictions]

                for m in RANKING_METRICS:
                    fn = getattr(reco_utils.evaluation.python_evaluation, m)
                    result = fn(test, prediction_df, **{**cols, 'k': TOP_K})
                    pm.record(m, result)
                    print(m, "=", result)
        
        for metric in eval_metrics:
            locals()['test_'+metric]()

    def run(self):
        data = self.csv_to_df()
        data_splits = (train, test) = python_stratified_split(
            data, filter_by="user", ratio=0.75,
            col_user=USER_COL, col_item=ITEM_COL, 
            seed=42
        )
        print("Train = {}, test = {}".format(len(train), len(test)))
        user_item_info = (_, items) = (
            data.drop_duplicates(USER_COL)[[USER_COL]].reset_index(drop=True), 
            data.drop_duplicates(ITEM_COL)[[ITEM_COL, ITEM_FEAT_COL]].reset_index(drop=True),
        )
        user_item_info = user_item_info + (len(items[ITEM_FEAT_COL][0]),)
        model, steps = self.fit_wide_deep_model(data_splits, user_item_info)
        model, ranking_pool, cols = self.train_and_eval(model, data_splits, user_item_info, steps)
        self.test_and_export(
            model,
            test,
            ('preds', 'k'),
            ranking_pool,
            cols
        )


if __name__ == "__main__":
    wdm = WideDeepModel()
    wdm.run()
