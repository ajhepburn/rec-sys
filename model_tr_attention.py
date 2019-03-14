from tensorrec import TensorRec
from tensorrec.eval import fit_and_eval
from tensorrec.representation_graphs import (
    LinearRepresentationGraph, NormalizedLinearRepresentationGraph
)
from tensorrec.loss_graphs import BalancedWMRBLossGraph

from ds import get_movielens_100k

class TensorRecAttentionModel:
    def __init__(self):
        pass

    def run(self):
        train_interactions, test_interactions, user_features, item_features, _ = get_movielens_100k(negative_value=0)
        print(train_interactions)
        
if __name__ == "__main__":
    tra = TensorRecAttentionModel()
    tra.run()