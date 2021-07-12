import copy
import os
import pickle
import hashlib
import random
import numpy as np
from pathlib import Path


class Tuner:

    def __init__(self,
                 project_dir=None,
                 build_fn=None,
                 randomize_axis_factor=0.7,
                 init_random=5,
                 objective_direction='max',
                 overwrite=True,
                 max_iters=100,
                 seed=None):

        if seed is not None:
            random.seed(seed)

        if project_dir:
            self.project_dir = project_dir
        else:
            self.project_dir = os.path.join('storm-tuner-default-project')

        Path(self.project_dir).mkdir(parents=True, exist_ok=True)

        self.max_iters = max_iters

        self.objective_direction = objective_direction
        self.build_fn = build_fn

        if os.path.exists(self._trials_path) and not overwrite:
            self.load()
            self._summarize_loaded_tuner()
        else:
            self.trials = []
            print('new tuner initialized')

        self.randomize_axis_factor = randomize_axis_factor
        self.init_random = init_random

        if len(self.trials) == 0:
            self.hyperparameters = HyperParameters()
            self.build_fn(self.hyperparameters)
        else:
            self.hyperparameters = self.trials[-1].hyperparameters.copy()

    def _summarize_loaded_tuner(self):
        if len(self.score_history) == 0:
            print('no valid trial scores recorded yet')
            return

        print(len(self.trials), 'trials loaded! (', len(self.score_history), 'valid trials )')
        if self.objective_direction == 'max':
            score_to_beat = max(self.score_history)
        else:
            score_to_beat = min(self.score_history)
        print('score to beat:', score_to_beat)
        print('best config:')
        print(self.get_best_trial().hyperparameters)

    @property
    def score_history(self):
        return [trial.score for trial in self.trials if not np.isnan(trial.score)]

    @property
    def _tried_so_far(self):
        return [trial.trial_id for trial in self.trials]

    def search(self, *args):
        print('\n-------- OPTIMIZING HYPERPARAMETERS --------\n')
        while True:
            if len(self.trials) >= self.max_iters:
                print('tuner finished {0} trials!'.format(len(self.trials)))
                print('final best config:')
                print(self.get_best_config())
                break
            trial = self._create_trial()
            trial.hyperparameters.initialized = True
            self.run_trial(trial, *args)
            self._on_trial_end(trial)

    def save(self):
        pickle.dump(self.trials, open(self._trials_path, "wb"))

    def load(self):  # all that needs to be loaded is the trials pickle file to fully reload the tuner state
        # probably want to eventually make this a json
        self.trials = pickle.load(open(self._trials_path, "rb"))

    @property
    def _trials_path(self):
        return os.path.join(self.project_dir, 'trials.p')

    def _on_trial_end(self, trial):
        self.save()
        self.report_trial(trial)

    def get_best_config(self):
        bt = self.get_best_trial()
        if bt is None:
            return None
        return bt.hyperparameters

    def _trial_is_new_best(self, trial):
        if np.isnan(trial.score):
            return False
        if len(self.score_history) == 1:
            return True
        if self.objective_direction == 'max':
            return trial.score > max(self.score_history[:-1])
        else:
            return trial.score < min(self.score_history[:-1])

    # just a method called after each trial to report the status of the tuner
    def report_trial(self, trial):

        # handle finding new best config
        if self._trial_is_new_best(trial):
            print('<><><> NEW BEST! <><><>')
            print(self.get_best_trial().hyperparameters)

        # print/log whatever you want to here
        print(len(self.trials), '| score:', round(trial.score, 8) if not np.isnan(trial.score) else None)

    def run_trial(self, trial, *args):
        raise NotImplementedError

    # completely randomize parameters not in use. This is useful to ensure when we activate an unused param,
    # it is not biased towards any particular value
    def _randomize_inactive(self):
        for param in self.hyperparameters.values.keys():
            if param not in self.hyperparameters.space:
                continue
            if param not in self.hyperparameters.active_params:
                self.hyperparameters.values[param] = random.choice(self.hyperparameters.space[param].values)

    # used when initializing the tuner. full random search.
    def _complete_random(self):
        for param in self.hyperparameters.values.keys():
            if param not in self.hyperparameters.space:
                continue
            self.hyperparameters.values[param] = random.choice(self.hyperparameters.space[param].values)

    # Here we build a model to see which params are actually used.
    # Then we create a hash only based off of active params.
    def _hash_active_params(self):
        self.hyperparameters.active_params = set()
        self.build_fn(self.hyperparameters)
        param_hash = self.hyperparameters.compute_values_hash()
        return param_hash

    def _draw_config(self):
        # do random configs for X user defined iters
        if len(self.score_history) < self.init_random:
            self._complete_random()
            param_hash = self._hash_active_params()
            while param_hash in self._tried_so_far:
                self._complete_random()
                param_hash = self._hash_active_params()
            return param_hash

        # copy the best set of hps
        self.hyperparameters = self.get_best_config().copy()

        self._randomize_inactive()

        while True:  # mutate params
            # get a random parameter to be mutated
            mod_param = random.choice(list(self.hyperparameters.values.keys()))

            if mod_param not in self.hyperparameters.space:
                raise Exception('HP not yet in space...', mod_param)

            # the parameter is set as 'ordered', only select an adjacent option.
            if self.hyperparameters.space[mod_param].ordered:

                # get the index of the current best value
                curr_ind = self.hyperparameters.space[mod_param].values.index(self.hyperparameters.values[mod_param])

                # deal with edge cases (index = 0 or len -1)
                if curr_ind == 0:
                    new_ind = curr_ind + 1
                elif curr_ind == len(self.hyperparameters.space[mod_param].values) - 1:
                    new_ind = curr_ind - 1

                else:  # randomly select value up or down
                    if bool(random.getrandbits(1)):
                        new_ind = curr_ind - 1
                    else:
                        new_ind = curr_ind + 1
                self.hyperparameters.values[mod_param] = self.hyperparameters.space[mod_param].values[new_ind]

            else:  # for unordered params, just choose a random value
                self.hyperparameters.values[mod_param] = random.choice(self.hyperparameters.space[mod_param].values)

            param_hash = self._hash_active_params()

            if random.uniform(0, 1) < self.randomize_axis_factor:  # another mutation
                continue
            elif param_hash in self._tried_so_far:  # conf already tested, mutate again
                self.hyperparameters = self.get_best_config().copy()
                self._randomize_inactive()
                continue
            else:
                return param_hash

    def _create_trial(self):
        param_hash = self._draw_config()
        return Trial(self.hyperparameters, param_hash)

    def score_trial(self, trial, trial_value):
        if trial_value is None:
            trial_value = np.nan
        trial.score = trial_value
        self._add_trial(trial)

    def _add_trial(self, trial):
        if trial.trial_id not in [trial.trial_id for trial in self.trials]:
            self.trials.append(trial)
        else:
            raise Exception('trying to add config trial that already exists')

    def get_best_trial(self):
        sorted_trials = sorted(
            [trial for trial in self.trials if not np.isnan(trial.score)],
            key=lambda trial: trial.score,
            reverse=self.objective_direction == 'max'
        )
        if len(sorted_trials) == 0:
            print('no valid trials recorded yet')
            return None
        else:
            return sorted_trials[0]


class Trial:
    def __init__(self, hyperparameters, param_hash):
        self.hyperparameters = hyperparameters.copy()
        self.trial_id = param_hash
        self.score = np.nan
        self.metrics = {}

    def _recalculate_hash(self):
        self.trial_id = self.hyperparameters.compute_values_hash()


class Param:
    def __init__(self, values, ordered):
        # should make values a set but need to preserve order if ordered
        self.values = values
        self.ordered = ordered


class HyperParameters:

    def __init__(self):
        self.space = {}
        self.values = {}
        self.active_params = set()
        self.initialized = False

    def __str__(self):
        res = ''
        for idx, param in enumerate(sorted(list(self.active_params))):
            if idx == 0:
                res += '\t' + str(param) + ' : ' + str(self.values[param])
            else:
                res += '\r\n\t' + str(param) + ' : ' + str(self.values[param])
        return res

    def copy(self):
        hps_copy = HyperParameters()
        hps_copy.space = copy.deepcopy(self.space)
        hps_copy.values = copy.deepcopy(self.values)
        hps_copy.active_params = copy.deepcopy(self.active_params)
        return hps_copy

    def _is_active(self, name):
        return name in self.active_params

    def _get_active_params(self):
        return {name: val for name, val in self.values.items() if self._is_active(name)}

    def compute_values_hash(self):
        values = self._get_active_params()
        keys = sorted(values.keys())
        s = ''.join(str(k) + '=' + str(values[k]) for k in keys)
        return hashlib.sha256(s.encode('utf-8')).hexdigest()[:32]

    def Param(self, name, values, ordered=False):
        self.active_params.add(name)  # mark param as active
        if name not in self.values or name not in self.space:  # register and randomize freshly encountered param
            if self.initialized:
                print('Attempting to define param \"{0}\" outside of the builder function. It is recommended to define '
                      'all params within the builder function so they are able to be flagged as active/inactive. This '
                      'is used to ensure virtually identical configurations are not tested twice.'.format(name))
            if len(values) > 10 and not ordered:
                print('Ordered param \"{0}\" has more than 10 values ({1}). It is recommended to keep the number of '
                      'potential values for an ordered parameter under 10'.format(name, len(values)))
            self.space[name] = Param(values, ordered)
            self.values[name] = random.choice(values)
        return self.values[name]  # retrieve param
