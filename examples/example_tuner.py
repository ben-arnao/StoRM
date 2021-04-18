from tensorflow import keras
import numpy as np
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.models import Sequential
from tuner import Tuner


def build_model(hp):
    model = Sequential()

    # example of model-wide unordered categorical parameter
    activation_fn = hp.Param('activation', ['relu', 'tanh', 'elu'])

    # example of inline ordered parameter
    for x in range(hp.Param('num_layers', [1, 2, 3, 4, 5], ordered=True)):

        # example of per-block parameter
        model.add(Dense(hp.Param('kernel_size_' + str(x), [64, 128, 256], ordered=True)))

        model.add(Activation(activation_fn))

        # example of boolean param
        if hp.Param('use_batch_norm', [True, False]):
            model.add(BatchNormalization())

        if hp.Param('use_dropout', [True, False]):

            # example of nested param
            #
            # this param will not affect the configuration hash, if this block of code isn't executed
            # this is to ensure we do not test configurations that are functionally the same
            # but have different values for unused parameters
            model.add(Dropout(hp.Param('dropout_value', [0.1, 0.2, 0.3, 0.4, 0.5], ordered=True)))

    # example of supplementary paramteter that will be accessed elsewhere
    hp.Param('batch_size', [128, 256, 512], ordered=True)

    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(loss=loss_fn, optimizer='sgd')
    return model


def custom_score_function(x_train, y_train, model, batch_size):
    history = model.fit(x_train, y_train, epochs=25, batch_size=batch_size, verbose=2)
    # here we can defined custom logic to assign a score to a configuration
    return np.mean(history.history['loss'][-5:])


class MyTuner(Tuner):

    def run_trial(self, trial, *args):
        x_train = args[0]
        y_train = args[1]

        hp = trial.hyperparameters
        model = build_model(hp)

        # here we can access params generated in the builder function
        batch_size = hp.values['batch_size']

        score = custom_score_function(x_train,
                                      y_train,
                                      model=model,
                                      batch_size=batch_size)
        self.score_trial(trial, score)


tuner = MyTuner(project_dir='C:/TestProject',
                build_fn=build_model,
                objective_direction='min',
                init_random=5,
                randomize_axis_factor=0.5,
                overwrite=False)

(x_train, y_train), _ = keras.datasets.mnist.load_data()
x_train = np.reshape(x_train, (-1, 784))

# parameters passed through 'search' go directly to the 'run_trial' method
tuner.search(x_train, y_train)
