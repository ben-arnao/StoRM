import tensorflow as tf
import optuna
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Dropout, BatchNormalization, GaussianNoise
from storm_tuner import Tuner
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import random
import time

BATCHSIZE = 256
CLASSES = 10
EPOCHS = 30
TRIALS_PER_TUNE = 60
NUM_COMPARISONS = 50

# download dataset from https://archive.ics.uci.edu/ml/datasets/wine+quality
df = pd.read_csv('C:/Users/Ben/Downloads/winequality.csv')
target = 'quality'

x = df.loc[:, df.columns != target]
y = df[target]

# offset of target breaks sparse
y -= 3

num_classes = df[target].nunique()

storm_scores = []
optuna_scores = []


class BalancedSparseCategoricalAccuracy(keras.metrics.SparseCategoricalAccuracy):
    def __init__(self, name='balanced_sparse_categorical_accuracy', dtype=None):
        super().__init__(name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_flat = y_true
        if y_true.shape.ndims == y_pred.shape.ndims:
            y_flat = tf.squeeze(y_flat, axis=[-1])
        y_true_int = tf.cast(y_flat, tf.int32)

        cls_counts = tf.math.bincount(y_true_int)
        cls_counts = tf.math.reciprocal_no_nan(tf.cast(cls_counts, self.dtype))
        weight = tf.gather(cls_counts, y_true_int)
        return super().update_state(y_true, y_pred, sample_weight=weight)


for n in range(NUM_COMPARISONS):
    random.seed(time.time())
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random.randint(0, 100))
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    def build_model_storm(hp):
        n_layers = hp.Param('n_layers', [1, 2, 3, 4, 5], ordered=True)
        weight_decay = hp.Param('weight_decay', [0, 1e-5, 1e-4, 1e-3], ordered=True)
        num_hidden = hp.Param('num_hidden', [100, 200, 300, 400, 500], ordered=True)
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Flatten())
        for i in range(n_layers):
            model.add(
                tf.keras.layers.Dense(
                    num_hidden,
                    activation=hp.Param('activation', ['relu', 'tanh', 'elu', 'selu']),
                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                )
            )

            if hp.Param('batch_norm', [True, False]):
                model.add(BatchNormalization())

            if hp.Param('add_noise', [True, False]):
                model.add(GaussianNoise(hp.Param('noise_std', [1e-5, 1e-4, 1e-3, 1e-2, 1e-1], ordered=True)))

            if hp.Param('add_dropout', [True, False]):
                model.add(Dropout(hp.Param('dropout_rate', [0.1, 0.2, 0.3, 0.4, 0.5], ordered=True)))

        model.add(
            tf.keras.layers.Dense(CLASSES, kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
        )

        opt = hp.Param('optimizer', ['adam', 'sgd'])
        lr = hp.Param('learning_rate', [1e-2, 1e-3, 1e-4], ordered=True)
        if opt == 'adam':
            optimizer = Adam(lr=lr, epsilon=hp.Param('epsilon', [1e-12, 1e-10, 1e-8, 1e-6], ordered=True))
        else:
            optimizer = SGD(lr=lr, momentum=hp.Param('momentum', [0.85, 0.9, 0.95], ordered=True))

        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(loss=loss_fn, optimizer=optimizer, metrics=['accuracy'])

        return model


    class MyTuner(Tuner):

        def run_trial(self, trial, *args):
            hp = trial.hyperparameters
            model = build_model_storm(hp)
            history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=EPOCHS, batch_size=BATCHSIZE, verbose=0)
            score = np.mean(history.history['val_accuracy'][-5:])

            self.score_trial(trial, score)


    if __name__ == "__main__":
        tuner = MyTuner(build_fn=build_model_storm,
                        objective_direction='max',
                        init_random=10,
                        randomize_axis_factor=0.5,
                        max_iters=TRIALS_PER_TUNE,
                        overwrite=True)
        tuner.search()

        storm_scores.append(max(tuner.score_history))


    def build_model_optuna(trial):
        n_units = trial.suggest_int("n_units", 100, 500)
        n_layers = trial.suggest_int("n_layers", 1, 5)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Flatten())

        for i in range(n_layers):
            model.add(
                tf.keras.layers.Dense(
                    n_units,
                    activation=trial.suggest_categorical("activation", ['relu', 'tanh', 'elu', 'selu']),
                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                )
            )

            if trial.suggest_categorical("batch_norm", [True, False]):
                model.add(BatchNormalization())

            if trial.suggest_categorical("add_noise", [True, False]):
                model.add(GaussianNoise(trial.suggest_float("noise_value", 1e-5, 1e-1, log=True)))

            if trial.suggest_categorical("add_dropout", [True, False]):
                model.add(Dropout(trial.suggest_float("dropout", 0.05, 0.5)))

        model.add(
            tf.keras.layers.Dense(CLASSES, kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
        )

        kwargs = {}
        optimizer_options = ["Adam", "SGD"]
        optimizer_selected = trial.suggest_categorical("optimizer", optimizer_options)
        if optimizer_selected == "Adam":
            kwargs["learning_rate"] = trial.suggest_float("adam_learning_rate", 1e-4, 1e-2, log=True)
            kwargs["epsilon"] = trial.suggest_float(
                "adam_epsilon", 1e-12, 1e-6, log=True
            )
        elif optimizer_selected == "SGD":
            kwargs["learning_rate"] = trial.suggest_float(
                "sgd_opt_learning_rate", 1e-4, 1e-2, log=True
            )
            kwargs["momentum"] = trial.suggest_float("sgd_opt_momentum", 0.85, 0.95)

        optimizer = getattr(tf.optimizers, optimizer_selected)(**kwargs)

        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(loss=loss_fn, optimizer=optimizer, metrics=['accuracy'])

        return model


    def objective(trial):
        model = build_model_optuna(trial)
        history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=EPOCHS, batch_size=BATCHSIZE, verbose=0)
        score = np.mean(history.history['val_accuracy'][-5:])
        return score


    if __name__ == "__main__":
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=TRIALS_PER_TUNE)

        optuna_scores.append(study.best_trial.value)

    print('trial num:', len(storm_scores))
    print('storm average score:', np.mean(storm_scores))
    print('optuna average score:', np.mean(optuna_scores))
