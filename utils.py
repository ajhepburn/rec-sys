import itertools, copy
import numpy as np
from sklearn.metrics import mean_squared_error

class Utils:
    def calculate_mse(self, model, ratings, user_index=None):
        preds = model.predict_for_customers()
        if user_index:
            return mean_squared_error(ratings[user_index, :].toarray().ravel(),
                                    preds[user_index, :].ravel())
        
        return mean_squared_error(ratings.toarray().ravel(),
                                preds.ravel())

    def precision_at_k(self, model, ratings, k=5, user_index=None):
        if not user_index:
            user_index = range(ratings.shape[0])
        ratings = ratings.tocsr()
        precisions = []
        # Note: line below may become infeasible for large datasets.
        predictions = model.predict_for_customers()
        for user in user_index:
            # In case of large dataset, compute predictions row-by-row like below
            # predictions = np.array([model.predict(row, i) for i in xrange(ratings.shape[1])])
            top_k = np.argsort(-predictions[user, :])[:k]
            labels = ratings.getrow(user).indices
            precision = float(len(set(top_k) & set(labels))) / float(k)
            precisions.append(precision)
        return np.mean(precisions)   

    def print_log(self, row, header=False, spacing=12):
        top = ''
        middle = ''
        bottom = ''
        for r in row:
            top += '+{}'.format('-'*spacing)
            if isinstance(r, str):
                middle += '| {0:^{1}} '.format(r, spacing-2)
            elif isinstance(r, int):
                middle += '| {0:^{1}} '.format(r, spacing-2)
            elif isinstance(r, float):
                middle += '| {0:^{1}.5f} '.format(r, spacing-2)
            bottom += '+{}'.format('='*spacing)
        top += '+'
        middle += '|'
        bottom += '+'
        if header:
            print(top)
            print(middle)
            print(bottom)
        else:
            print(middle)
            print(top)

    def learning_curve(self, model, train, test, epochs, k=2, user_index=None):
        if not user_index:
            user_index = range(train.shape[0])
        prev_epoch = 0
        train_precision = []
        train_mse = []
        test_precision = []
        test_mse = []
        
        headers = ['epochs', 'p@k train', 'p@k test',
                'mse train', 'mse test']
        self.print_log(headers, header=True)
        
        for epoch in epochs:
            model.iterations = epoch - prev_epoch
            if not hasattr(model, 'user_vectors'):
                model.fit(train)
            else:
                model.fit_partial(train)
            train_mse.append(self.calculate_mse(model, train, user_index))
            train_precision.append(self.precision_at_k(model, train, k, user_index))
            test_mse.append(self.calculate_mse(model, test, user_index))
            test_precision.append(self.precision_at_k(model, test, k, user_index))
            row = [epoch, train_precision[-1], test_precision[-1],
                train_mse[-1], test_mse[-1]]
            self.print_log(row)
            prev_epoch = epoch
        return model, train_precision, train_mse, test_precision, test_mse  

    def grid_search_learning_curve(self, base_model, train, test, param_grid,
                               user_index=None, patk=3, epochs=range(2, 40, 2)):
        """
        "Inspired" (stolen) from sklearn gridsearch
        https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/model_selection/_search.py
        """
        curves = []
        keys, values = zip(*param_grid.items())
        for v in itertools.product(*values):
            params = dict(zip(keys, v))
            this_model = copy.deepcopy(base_model)
            print_line = []
            for k, v in params.items():
                setattr(this_model, k, v)
                print_line.append((k, v))

            print(' | '.join('{}: {}'.format(k, v) for (k, v) in print_line))
            _, train_patk, train_mse, test_patk, test_mse = self.learning_curve(this_model, train, test,
                                                                    epochs, k=patk, user_index=user_index)
            curves.append({'params': params,
                        'patk': {'train': train_patk, 'test': test_patk},
                        'mse': {'train': train_mse, 'test': test_mse}})
        return curves