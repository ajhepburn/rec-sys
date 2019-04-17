from spotlight.interactions import Interactions
from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.sequence.implicit import ImplicitSequenceModel
from spotlight.sequence.representations import CNNNet
from spotlight.cross_validation import random_train_test_split
from spotlight.evaluation import precision_recall_score, mrr_score, sequence_precision_recall_score, sequence_mrr_score

from sklearn.model_selection import ParameterSampler

from datetime import datetime

import pandas as pd
import numpy as np

import hashlib
import logging
import sys
import json

NUM_SAMPLES = 100
DEFAULT_PARAMS = {
    'learning_rate': 0.01,
    'loss': 'pointwise',
    'batch_size': 256,
    'embedding_dim': 32,
    'n_iter': 10,
    'l2': 0.0
}

LEARNING_RATES = [1e-3, 1e-2, 5 * 1e-2, 1e-1]
LOSSES = ['bpr', 'hinge', 'pointwise']
BATCH_SIZE = [8, 16, 32, 256]
EMBEDDING_DIM = [8, 16, 32, 64, 128, 256]
N_ITER = list(range(5, 20))
L2 = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.0]

#df.groupby(['A','C'], group_keys=False).apply(lambda x: x.ix[x.B.idxmax()]).reset_index(drop=True)