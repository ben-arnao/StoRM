import copy
import os
import pickle
import hashlib
import random
import time

import numpy as np


def display_hps(hyperparameters):
    params = list(hyperparameters.active_params)
    params.sort()
    for param in params:
        print('\t' + param, ':', hyperparameters.values[param])


class Tuner:

    def __init__(self,
                 project_dir=None,
                 build_fn=None,
                 randomize_axis_factor=0.5,
                 init_random=25,
                 objective_direction='max',
                 overwrite=False):

        self.project_dir = project_dir

        self.objective_direction = objective_direction
        self.hypermodel = HyperModel(build_fn)

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
            self.hypermodel.build(self.hyperparameters)
        else:
            self.hyperparameters = self.trials[-1].hyperparameters.copy()

    def _summarize_loaded_tuner(self):
        if len(self.score_history) == 0:
            print('no valid trial scores recorded yet')
            return

        print(len(self.trials), 'trials loaded!')
        if self.objective_direction == 'max':
            score_to_beat = max(self.score_history)
        else:
            score_to_beat = min(self.score_history)
        print('score to beat:', score_to_beat)
        print('best config:')
        display_hps(self.get_best_trial().hyperparameters)

    @property
    def score_history(self):
        return [trial.score for trial in self.trials if not np.isnan(trial.score)]

    @property
    def _tried_so_far(self):
        return [trial.trial_id for trial in self.trials]

    def search(self, *args):
        print('\n-------- OPTIMIZING HYPERPARAMETERS --------\n')
        while True:
            trial = self._create_trial()
            display_hps(trial.hyperparameters)
            self.run_trial(trial, *args)
            self._on_trial_end(trial)

    def save(self):
        pickle.dump(self.trials, open(self._trials_path, "wb"))

    def load(self):  # all that needs to be loaded is the trials to fully reload the tuner state
        self.trials = pickle.load(open(self._trials_path, "rb"))

    @property
    def _trials_path(self):
        return self.project_dir + '/trials.p'

    def _on_trial_end(self, trial):
        self.save()
        self.report_trial(trial)

    def retrieve_trial_metric(self, metric, only_last=True):
        arr = []
        has_value = False
        for trial in self.trials:
            if metric in trial.metrics.keys():
                arr.append(trial.metrics[metric])
                has_value = True
            else:
                arr.append(None)
        if has_value:
            if only_last:
                if arr[-1] is not None:
                    return arr[-1]
                else:
                    raise KeyError(metric + str(' does not exist in current trial metrics'))
            else:
                return arr
        else:
            raise KeyError(metric + str(' does not exist in any trial metrics'))

    def _handle_win(self, trial):
        print('<><><> NEW BEST! <><><>')
        display_hps(trial.hyperparameters)

        # dump the best config
        pickle.dump(trial.hyperparameters, open(self.project_dir + '/best_hps.p', "wb"))

    def _trial_is_new_best(self, trial):
        if self.objective_direction == 'max':
            return max(self.score_history) == trial.score
        else:
            return min(self.score_history) == trial.score

    def avg_minutes_per_trial(self, trailing_window_len):
        return round(np.mean(self.retrieve_trial_metric('elapsed', only_last=False)
                             [-trailing_window_len:]) / 60.0, 2)

    # just a method called after each trial to report the status of the tuner
    def report_trial(self, trial):
        # handle finding new best config
        if not np.isnan(trial.score) and (len(self.score_history) == 0 or self._trial_is_new_best(trial)):
            self._handle_win(trial)

        # print/log whatever you want to here
        print(len(self.score_history),
              '| score:', round(trial.score, 8) if not np.isnan(trial.score) else None,
              '| changed count:', self._params_changed_from_best(trial))

    def run_trial(self, trial, *args):
        raise NotImplementedError

    # useful for seeing how many axis the configuration has changed from the best config
    def _params_changed_from_best(self, trial):
        if len(self.trials) == 1:
            return 0
        best_trial = self.get_best_trial()

        # if current trial is the best trial (new best found), return the last best trial to compare to
        if trial.score == best_trial.score:
            best_trial = self.get_best_trial(next_best=True)

        # if this is first trial, there is no best yet
        if best_trial is None:
            return 0

        best_params = {param: best_trial.hyperparameters.values[param]
                       for param in best_trial.hyperparameters.active_params}
        curr_params = {param: trial.hyperparameters.values[param]
                       for param in trial.hyperparameters.active_params}

        changed = 0
        for param in best_params.keys():
            if param not in curr_params.keys() or best_params[param] != curr_params[param]:
                changed += 1
        return changed

    # completely randomize parameters not in use. This is useful to ensure when we activate an unused param,
    # it is not baised towards any particular value
    def _randomize_inactive(self):
        for param in self.hyperparameters.values.keys():
            if param not in self.hyperparameters.space:
                continue
            if param not in self.hyperparameters.active_params:
                self.hyperparameters.values[param] = random.choice(self.hyperparameters.space[param].values)

    # used when intializing the tuner. Full random search.
    def _complete_random(self):
        for param in self.hyperparameters.values.keys():
            if param not in self.hyperparameters.space:
                continue
            self.hyperparameters.values[param] = random.choice(self.hyperparameters.space[param].values)

    # Here we build a model to see which params are actually used.
    # Then we create a hash only based off of active params.
    def _hash_active_params(self):
        self.hyperparameters.active_params = set()
        self.hypermodel.build(self.hyperparameters)
        param_hash = self.hyperparameters.compute_values_hash()
        return param_hash

    def _draw_config(self):
        # do random configs for X user defines iters
        if len(self.trials) < self.init_random:
            self._complete_random()
            param_hash = self._hash_active_params()
            return param_hash

        # copy the best set of hps
        self.hyperparameters = self.get_best_config().copy()
        self._randomize_inactive()

        # mutate params
        while True:
            # get a random parameter to be mutated
            mod_param = random.choice(list(self.hyperparameters.values.keys()))

            # if chosen param does not exist in space skip (shouldn't need this, not sure why i have to have this
            # yet tuner seems to work fine anyway)
            if mod_param not in self.hyperparameters.space:
                continue

            # the parameter is set as 'ordered', only select an adjacent option.
            if self.hyperparameters.space[mod_param].ordered:

                # get the index of the current best value
                curr_ind = self.hyperparameters.space[mod_param].values.index(self.hyperparameters.values[mod_param])

                # deal with edge cases (index = 0 or len -1)
                if curr_ind == 0:
                    new_ind = curr_ind + 1
                elif curr_ind == len(self.hyperparameters.space[mod_param].values) - 1:
                    new_ind = curr_ind - 1
                else:
                    # randomly select value up or down
                    if bool(random.getrandbits(1)):
                        new_ind = curr_ind - 1
                    else:
                        new_ind = curr_ind + 1

                self.hyperparameters.values[mod_param] = self.hyperparameters.space[mod_param].values[new_ind]
            else:  # for unordered params, just choose a random value
                self.hyperparameters.values[mod_param] = random.choice(self.hyperparameters.space[mod_param].values)

            param_hash = self._hash_active_params()

            if random.uniform(0, 1) < self.randomize_axis_factor:
                continue
            elif param_hash in self._tried_so_far:
                # if the configuration has been tested already, start entire process over
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
            trial = np.nan
        trial.score = trial_value
        self.trials.append(trial)

    def get_best_trial(self, next_best=False):
        sorted_trials = sorted(
            [trial for trial in self.trials if not np.isnan(trial.score)],
            key=lambda trial: trial.score,
            reverse=self.objective_direction == 'max'
        )
        if len(sorted_trials) == 0:
            return None
        else:
            if next_best:
                return sorted_trials[1]
            return sorted_trials[0]

    def get_best_config(self):
        bt = self.get_best_trial()
        if bt is None:
            return None
        return bt.hyperparameters

    # probes a specific configuration given a HyperParameters object
    def probe(self, hps, *args):
        self.hyperparameters.values = hps.values
        self.hyperparameters.active_params = set()
        self.hypermodel.build(self.hyperparameters)
        display_hps(self.hyperparameters)
        param_hash = self.hyperparameters.compute_values_hash()
        if param_hash in self._tried_so_far:
            print('tried this config already...')
            return
        trial = Trial(self.hyperparameters, param_hash)
        start = time.time()
        self.run_trial(trial, *args)
        trial.metrics['elapsed'] = time.time() - start
        self._on_trial_end(trial)


class Trial:
    def __init__(self, hyperparameters, param_hash):
        self.hyperparameters = hyperparameters.copy()
        self.trial_id = param_hash
        self.score = None
        self.metrics = {}

    def _recalculate_hash(self):
        self.trial_id = self.hyperparameters.compute_values_hash()


class HyperModel:
    def __init__(self, build_model):
        self.build = build_model

    def build(self, hp):
        raise NotImplementedError


class Param:
    def __init__(self,
                 values,
                 ordered,
                 default):
        self.values = values
        self.ordered = ordered
        self.default = default


class HyperParameters:

    def __init__(self):
        self.space = {}
        self.values = {}
        self.active_params = set()

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

    def _retrieve(self,
                  name,
                  values,
                  ordered,
                  default):
        self.active_params.add(name)
        return self._retrieve_helper(name,
                                     values,
                                     ordered,
                                     default)

    def _retrieve_helper(self,
                         name,
                         values,
                         ordered,
                         default):
        if name in self.values:
            return self.values[name]
        else:
            return self._register(name,
                                  values,
                                  ordered,
                                  default)

    def _register(self,
                  name,
                  values,
                  ordered,
                  default):
        self.space[name] = Param(values,
                                 ordered,
                                 default)
        if default is not None:
            self.values[name] = default
        else:
            self.values[name] = random.choice(values)
        return self.values[name]

    def Param(self,
              name,
              values,
              ordered=False,
              default=None):
        return self._retrieve(name,
                              values,
                              ordered,
                              default)
