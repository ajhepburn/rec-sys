import os, csv, sys, random, tensorrec
from collections import defaultdict
from scipy import sparse
from sklearn.preprocessing import MultiLabelBinarizer

class MetadataTensorRec:
    def __init__(self):
        self.rpath = './data/csv/metadata.csv'

    def load_csv(self, rpath):
        with open(self.rpath, 'r') as metadata_file:
            metadata_file_reader = csv.reader(metadata_file, delimiter="\t")
            raw_metadata = list(metadata_file_reader)
            raw_metadata_header = raw_metadata.pop(0)
        return (raw_metadata, raw_metadata_header)

    def map_to_internal_ids(self, raw_metadata):
        internal_user_ids = defaultdict(lambda: len(internal_user_ids))
        internal_item_ids = defaultdict(lambda: len(internal_item_ids))

        for row in raw_metadata:
            row[0] = internal_user_ids[int(row[0])]
            row[2] = internal_item_ids[int(row[2])]
        
        return (internal_user_ids, internal_item_ids)
    
    def train_test_split(self, raw_metadata):
        random.shuffle(raw_metadata)
        cutoff = int(.8 * len(raw_metadata))
        train = raw_metadata[:cutoff]
        test = raw_metadata[cutoff:]

        return (train, test)

    def interactions_to_sparse_matrix(self, interactions, n_users_items):
        n_users, n_items = n_users_items
        col_users, _ , col_items, _, _, _ = zip(*interactions)
        return sparse.coo_matrix(((1,)*len(col_items), (col_users, col_items)),
                                shape=(n_users, n_items))

    def cf_construct_indicator_features(self, n_users_items, sparse_train):
        n_users, n_items = n_users_items
        user_indicator_features = sparse.identity(n_users)
        item_indicator_features = sparse.identity(n_items)

        cf_model = tensorrec.TensorRec(n_components=5)

        print("Training collaborative filter")
        cf_model.fit(interactions=sparse_train,
                    user_features=user_indicator_features,
                    item_features=item_indicator_features,
                    user_batch_size=64)
        return (cf_model, (user_indicator_features, item_indicator_features))

    def evaluate_model(self, ranks, sparse_matrices):
        sparse_train, sparse_test = sparse_matrices
        train_recall_at_10 = tensorrec.eval.recall_at_k(
            test_interactions=sparse_train,
            predicted_ranks=ranks,
            k=10
        ).mean()
        test_recall_at_10 = tensorrec.eval.recall_at_k(
            test_interactions=sparse_test,
            predicted_ranks=ranks,
            k=10
        ).mean()
        print("Recall at 10: Train: {:.4f} Test: {:.4f}".format(train_recall_at_10,
                                                                test_recall_at_10))

    def add_item_metadata(self, metadata, internal_item_ids):
        raw_metadata, raw_metadata_header, internal_item_ids, n_items = metadata[0], metadata[1], internal_item_ids[0], internal_item_ids[1]
        internal_locations, internal_industries, internal_sectors, internal_cashtags = {}, {}, {}, {}


        for row in raw_metadata:
            row[2] = internal_item_ids[int(row[2])] 
            internal_industries[row[2]] = row[3]
            internal_sectors[row[2]] = row[4]
            internal_cashtags[row[2]] = row[5]

        print("Raw metadata example:\n{}\n{}".format(raw_metadata_header, 
                                                    raw_metadata[0]))

        industries = [internal_industries[internal_id]
                        for internal_id in list(internal_industries.keys())]
        # sectors = [internal_sectors[internal_id]
        #                 for internal_id in list(internal_sectors.keys())]
        # cashtags = [internal_cashtags[internal_id]
        #                 for internal_id in list(internal_cashtags.keys())]

        industry_features = MultiLabelBinarizer().fit_transform(industries)
        print("Size of MLB is {:.3f}MB".format(industry_features.data.nbytes/(1024**2)))
        # sector_features = MultiLabelBinarizer().fit_transform(sectors)
        # cashtag_features = MultiLabelBinarizer().fit_transform(cashtags)

        industry_features = sparse.coo_matrix(industry_features)
        # sector_features = sparse.coo_matrix(sector_features)
        # cashtag_features = sparse.coo_matrix(cashtag_features)
        print("Size of COO Matrix is {:.3f}MB".format(industry_features.data.nbytes/(1024**2)))
        return (industry_features)


    def cb_fit_model(self, industry_features, user_indicator_features, n_users_items, sparse_matrices):
        sparse_train, _ = sparse_matrices
        n_industries, n_items = n_users_items
        # Fit a content-based model using the genres as item features
        print("Training content-based recommender")
        content_model = tensorrec.TensorRec(
            n_components=n_industries, # movie_genre_features.shape[1]
            item_repr_graph=tensorrec.representation_graphs.FeaturePassThroughRepresentationGraph(),
            loss_graph=tensorrec.loss_graphs.WMRBLossGraph()
        )
        content_model.fit(interactions=sparse_train,
                        user_features=user_indicator_features,
                        item_features=industry_features,
                        n_sampled_items=int(n_items * .01),
                        user_batch_size=64)

        # Check the results of the content-based model
        print("Content-based recommender:")
        predicted_ranks = content_model.predict_rank(user_features=user_indicator_features,
                                                    item_features=industry_features)
        self.evaluate_model(predicted_ranks, sparse_matrices)

    def run(self, rpath):
        raw_metadata, raw_metadata_header = self.load_csv(rpath)

        internal_user_ids, internal_item_ids = self.map_to_internal_ids(raw_metadata)
        n_users_items = (len(internal_user_ids), len(internal_item_ids))

        train, test = self.train_test_split(raw_metadata)
        sparse_matrices = self.interactions_to_sparse_matrix(train, n_users_items), self.interactions_to_sparse_matrix(test, n_users_items)
        
        cf_model, (user_indicator_features, item_indicator_features) = self.cf_construct_indicator_features(n_users_items, sparse_matrices[0])
        predicted_ranks = cf_model.predict_rank(user_features=user_indicator_features,
                                        item_features=item_indicator_features)
        self.evaluate_model(predicted_ranks, sparse_matrices)
        industry_features = self.add_item_metadata((raw_metadata, raw_metadata_header), (internal_item_ids, len(internal_item_ids)))
        #self.cb_fit_model(industry_features, user_indicator_features, (industry_features.shape[1],len(internal_item_ids)), sparse_matrices)

if __name__ == "__main__":
    mtr = MetadataTensorRec()
    mtr.run('./data/csv/metadata.csv')
        


