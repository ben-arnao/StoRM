import xgboost as xgb
from tuner import Tuner


def get_score(dtrain,
              dtest,
              max_depth,
              eta,
              gamma,
              min_child_weight,
              max_delta_step,
              subsample,
              lambda_param,
              alpha,
              refresh_leaf,
              grow_policy,
              num_round
              ):

    param = {'max_depth': max_depth,
             'eta': eta,
             'gamma': gamma,
             'min_child_weight': min_child_weight,
             'max_delta_step': max_delta_step,
             'subsample': subsample,
             'lambda': lambda_param,
             'alpha': alpha,
             'refresh_leaf': refresh_leaf,
             'grow_policy': grow_policy,
             'objective': 'binary:logistic',
             'verbosity': 0}
    
    # uses 'https://github.com/dmlc/xgboost/blob/master/demo/guide-python/basic_walkthrough.py'
    # to build a basic example from

    bst = xgb.train(param, dtrain, num_round)
    preds = bst.predict(dtest)
    labels = dtest.get_label()
    
    error = (sum(1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i]) / float(len(preds)))
    return error


def build_fn(hp):
    hp.Param('num_round', [1, 2, 3, 4, 5], ordered=True)
    hp.Param('eta', [0.1, 0.2, 0.3, 0.4, 0.5], ordered=True)
    hp.Param('gamma', [1, 0.1, 0.01, 0.001, 0.0001, 0], ordered=True)
    hp.Param('max_depth', [2, 4, 6, 8, 10], ordered=True)
    hp.Param('min_child_weight', [0.5, 1, 2], ordered=True)
    hp.Param('max_delta_step', [1, 0.1, 0.001, 0.0001, 0], ordered=True)
    hp.Param('subsample', [0, 0.25, 0.5, 0.75, 1], ordered=True)
    hp.Param('lambda_param', [0.5, 1, 2, 3], ordered=True)
    hp.Param('alpha', [0.1, 0.001, 0.0001, 0], ordered=True)
    hp.Param('refresh_leaf', [0, 1], ordered=True)
    hp.Param('grow_policy', ['depthwise', 'lossguide'], ordered=True)


class MyTuner(Tuner):

    def run_trial(self, trial, *args):
        hp = trial.hyperparameters
        build_fn(hp)

        train = args[0]
        test = args[1]

        score = get_score(train,
                          test,
                          **hp.values)

        self.score_trial(trial, score)


tuner = MyTuner(build_fn=build_fn,
                objective_direction='min',
                init_random=5,
                randomize_axis_factor=0.5,
                max_iters=30,
                overwrite=True)

# download the datasets from 'https://github.com/dmlc/xgboost/tree/master/demo/data'
dtrain = xgb.DMatrix('C:/users/ben/documents/agaricus.txt.train')
dtest = xgb.DMatrix('C:/users/ben/documents/agaricus.txt.test')

tuner.search(dtrain, dtest)
