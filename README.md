# stochasticmutatortuner
A hyperparameter tuner for high-dimensional, categorical-parametered, intractable optimization problems (Neural Network)

# Motivations of this tuner

Neural network hyper parameter optimization is an especially challenging task due to 3 main reasons:

1) Parameters are highly codependent.

2) The search space can be highly non-convex and intractable.

3) For high-end performance where we want to make sure we don't miss out on the absolute best model, or for domains where there has not been extensive research and a general understanding on what types of choices work better than others, the dimensionality of the search space can get very large such that Bayesian Optimization-related methods are not very effective.

Recent research has discussed there is not a lot of reproducible evidence that show any of today's state of the art techniques significantly beat a plain old random search with some form of early stopping- https://arxiv.org/pdf/1902.07638.pdf

# How does this tuner attempt to solve these issues?

All of the points mentioned above make it very difficult if not impossible to do any sort of intelligently guided search for NN architecture/training hyperparameters. That is why i scrap the idea of building some sort surrogate function or gradient-based method to model high probability areas and opt to go for something simpler and hopefully more robust to the problems we're facing.

This tuner can be thought of as a combination of a grid search combined with random search, where the "distance" between the next evaluation candidate, and the overall best candidate, is probability based. The idea behind this tuner is to randomly mutate the current best configuration along different axes (and sometimes even multiple times along the same axis). The number of mutations made for the next evaluation candidate, is based on a user-defined probability. This approach aims to combine the benefits of tweaking a configuration in cases where we are near good minima and in cases where the feature set may have some level of independance, but also allowing the tuner to have the ability to mutate the model multiple times in one step, so that it can get out of local minima.

The default value for ```randomize_axis_factor``` is 0.5 which means that there is a 50% chance just one mutation will be made. There is a 25% chance two mutations will be made. A 12.5% chance that three mutations will be made, and so on.

My belief is that this tuner provides a good balance in addressing the issues stated above. Allowing enough freedom so that we do respect the non-convexness of the search space and co-dependency of variables, while also probalistically restricting how different the next evaluation candidate is from the current best, to provide some level of guidance and locality to the search.

# Usage

Here we define our hyper parameter space through providing our own model building method. All we need to do is define our HP space, and return an untrained model. Parameters used at train time can also be defined here. All parameters take the form: ```hp.Param('parameter_name', [value1, value2...], ordered=False)```. Setting a parameter to ordered=True, will ensure the tuner is only able to select adjacent values per a single mutation step.

```python
def build_model(hp):
    model = Sequential()
    
    # we can define train-time params in the build model function
    hp.Param('batch_size', [32, 64, 128, 256], ordered=True)
    
    model.add(Dense(10))
    
    activation_choices = ['tanh', 'softsign', 'selu', 'relu', 'elu', 'softplus']
    model.add(Activation(hp.Param('activation', activation_choices)))
    
    # example of nested parameter
    if hp.Param('dropout', [True, False]):
        model.add(Dropout(hp.Param('dropout_val', [0.1, 0.2, 0.3, 0.4, 0.5], ordered=True)))

    model.add(Dense(1))

    model.compile(loss='mse', optimizer=SGD(momentum=0.9))
    return model
```

We override the ```run_trial()``` method for our own Tuner, this encapsulates all the work of a single trial. All the run_trial method needs to do is assign a score to the trial ```self.score_trial(trial, score)```. How you use your model to make the score for the trial, is up to you (ex. K-fold cross validation, trailing average of epoch loss to mitigate variance, etc.). The ```self.hypermodel.build(hp)``` function called in ```run_trial``` is what will supply us with a blank model.

As we can see, any arguments you provide in the ```search()``` entry method, can be accessed in your ```run_trial()``` method.

```python
def get_model_score(model, params, X_train, y_train, X_test, y_test):
    history = model.fit(X_train,
                        y_train,
                        epochs=25,
                        validation_data=(X_test, y_test),
                        batch_size=params['batch_size'])
    return history.history['val_loss'][-1]

class MyTuner(tuner.engine.tuner.Tuner):
    def run_trial(self, trial, *args):
        hp = trial.hyperparameters
        X_train, y_train, X_test, y_test = args[0], args[1], args[2], args[3]
        model = self.hypermodel.build(hp)
        score = get_model_score(model, hp.values, X_train, y_train, X_test, y_test)
        self.score_trial(trial, score)
```

We initialize our Tuner and provide our training data

```python
tuner = MyTuner(project_dir='C:/myProject', objective_direction='min', hypermodel=build_model)
tuner.search(X_train, y_train, X_test, y_test)
```

# Customizing the tuner

One of the main goals of this library is to save the user from having to tune the tuner itself. An excess of tunable variables can confuse the user, make intuitive documentation more difficult, and even have a substantial negative effect on the tuner's performance if they are not set correctly.

With this tuner we have 2 main adjustable parameters to customize your search preferences...

```python

    def __init__(self,
                 project_dir='C:/myProject/',
                 build_fn=build_model,
                 objective_direction='min',
                 init_random=25,
                 randomize_axis_factor=0.5)
```

```init_random```: How many initial iterations to perform random search for. This is helpful for getting the search to an average/decent configuration, so that we don't waste too much time descending from a suboptimal starting point.

```randomize_axis_factor```: The main exploitative/explorative tradeoff parameter. A value closer to 1 means that steps will generally have more mutations. A value closer to 0 will mean steps are more likely to only do a single mutation.

# Other notes/features

-The tuner keep tracks of which parameters are in use by building a dummy model prior to hashing the configuration. When building the model, parameters the model building function actually draws from are flagged as active. For example, if we have a parameter to determine number of layers to use, if the number of layers is set to 1, parameters only applicable to layer 2+ will not be included in the hash. This allows us to ensure we never test configurations that are virtually identical. Since testing a configuration is expensive, i believe the upfront cost of just building a blank model is far less than by chance testing the same configuration twice.
-Tests coming shortly to provide comparisons between this tuner and random search
