from tensorflow import keras
import numpy as np
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam
from storm.tuner import Tuner

epochs = 50
trials_per_mode = 10
iters_per_trial = 100


def build_model(hp):
    model = Sequential()

    activation_fn = hp.Param('activation', ['relu', 'tanh', 'elu'])

    for x in range(hp.Param('num_layers', [1, 2, 3, 4], ordered=True)):
        model.add(Dense(hp.Param('kernel_size_' + str(x), [32, 64, 128], ordered=True)))

        model.add(Activation(activation_fn))

    hp.Param('batch_size', [256, 512, 1024], ordered=True)

    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    opt = hp.Param('optimizer', ['adam', 'sgd'])
    lr = hp.Param('learning_rate', [1e-3, 1e-4])
    if opt == 'adam':
        optimizer = Adam(lr=lr, epsilon=hp.Param('epsilon', [1e-10, 1e-8, 1e-6]))
    else:
        optimizer = SGD(lr=lr, momentum=hp.Param('momentum', [0.5, 0.9, 0.98]))

    model.compile(loss=loss_fn, optimizer=optimizer)
    return model


class MyTuner(Tuner):

    def run_trial(self, trial, *args):
        x_train = args[0]
        y_train = args[1]

        hp = trial.hyperparameters
        model = build_model(hp)

        batch_size = hp.values['batch_size']

        history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        score = np.mean(history.history['loss'][-5:])
        self.score_trial(trial, score)


(x_train, y_train), _ = keras.datasets.mnist.load_data()
x_train = np.reshape(x_train, (-1, 784))

random_scores = []
tuned_scores = []

for x in range(trials_per_mode):
    for mode in ['random', 'tune']:

        tuner = MyTuner(build_fn=build_model,
                        objective_direction='min',
                        init_random=5 if mode == 'tune' else iters_per_trial,
                        randomize_axis_factor=0.5,
                        max_iters=iters_per_trial,
                        overwrite=True)

        tuner.search(x_train, y_train)

        if mode == 'tune':
            tuned_scores.append(min(tuner.score_history))
        else:
            random_scores.append(min(tuner.score_history))

print('tuned scores mean:', np.mean(tuned_scores), '| stdev:', np.std(tuned_scores))
print('random scores mean:', np.mean(random_scores), '| stdev:', np.std(random_scores))
