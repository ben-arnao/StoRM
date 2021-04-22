# StoRM (Stochastic Random Mutator)
A hyperparameter tuner for high-dimensional, categorically-parametered, intractable optimization problems (Ex. Neural Network)

# Motivations of this tuner

Neural network hyper parameter optimization is an especially challenging task due to a few reasons:

1) Parameters are highly codependent. Adjusting a single parameter may not be enough to get over a saddle point, you will likely have to adjust many parameters simultaneously to escape local minima.

2) The search space can be highly non-convex and intractable, with many categorical, discrete-valued, conditional, and nested parameters. This sort of parameter makes it very difficult to generate any sort of quantitative probability model.

3) One might encounter a scenario where tuning a parameter will have very poor or very good results depending on what parameters are tuned with it, so attempting to model which parameters are more likely to be better will require a lot of trials to overcome this level of variance/noise. Even then, the best parameter value on average will not always be the best parameter value overall.

4) For high-end performance where local minima is not good enough and we want both a granular and board search, or for domains where there has not been extensive research and a general understanding on what types of choices work better than others, the dimensionality of the search space can get very large such that Bayesian Optimization-related methods are not very effective.

Recent research has discussed there is not a lot of reproducible evidence that show any of today's state of the art techniques significantly beat a plain old random search with some form of early stopping- https://arxiv.org/pdf/1902.07638.pdf

# How does this tuner attempt to solve these issues?

All of the issues mentioned above make it very difficult if not impossible to do any sort of intelligently guided search for NN architecture/training hyperparameters. That is why this tuner opts against attempting to build some sort surrogate function or gradient-based method to model the probability of the search space, and instead aims for something simpler and hopefully more robust to the problems we're facing. The user shouldn't expect a magic algorithm that takes the least amount of steps possible to reach global minima, but they should be able to expect something that beats brute force/random search for almost all use cases by a considerable amount, which is really what NN tuning needs at this stage.

The StoRM tuner can be thought of inuitively as a combination of a grid search combined with random search, where the "distance" between the current best configuration and the next evaluation candidate, is probability based. We randomly mutate the current best configuration along different axes (and sometimes even multiple times along the same axis). The number of mutations made for the next evaluation candidate, is based on a user-defined probability.

The default value for ```randomize_axis_factor``` is 0.5 which means that there is a 50% chance just one mutation will be made. There is a 25% chance two mutations will be made. A 12.5% chance that three mutations will be made, and so on.

This approach aims to address the issues stated above by allowing enough freedom so that we do respect the non-convexness of the search space and co-dependency of variables, while also probalistically restricting how different the next evaluation candidate is from the current best, to provide some level of guidance and locality to the search.

# Installation

```pip install storm-tuner```

# Usage

Here we define our hyperparameter space by providing our own configuration building method. All we need to do is define our HP space, and usually it will make sense for our function to return an untrained model. Parameters used at train time (ex. batch size) can also be defined here. All parameters take the form: ```hp.Param('parameter_name', [value1, value2...], ordered=False)```. Setting a parameter to ```ordered=True```, will ensure the tuner is only able to select adjacent values per a single mutation step. This is an important feature for parameters where there is ordinality.

*Keep in mind that your configuration building method does not necessarily need to return a model. There are two ways to use a parameter. You can define and use parameters inline like shown below, however you may also define parameters in the builder function but access parameters after running build_model() as well. There is no functional difference but in most cases we will want to define and use parameters in building our model, and only access train-time parameter elsewhere.

```python
def build_model(hp):
    model = Sequential()
    
    # we can define train-time params in the build model function
    hp.Param('batch_size', [32, 64, 128, 256], ordered=True)
    
    model.add(Dense(10))
    
    # here is a categorical parameter that most optimizers do not do well with
    activation_choices = ['tanh', 'softsign', 'selu', 'relu', 'elu', 'softplus']
    model.add(Activation(hp.Param('activation', activation_choices)))
    
    # example of nested parameter
    if hp.Param('dropout', [True, False]):
        model.add(Dropout(hp.Param('dropout_val', [0.1, 0.2, 0.3, 0.4, 0.5], ordered=True)))

    model.add(Dense(1))

    model.compile(loss='mse', optimizer=SGD(momentum=0.9))
    return model
```

We are required to override the ```run_trial()``` method for our own Tuner implementation, this encapsulates all the execution of a single trial. All the ```run_trial``` method needs to do is assign a score to the trial ```self.score_trial(trial, score)``` using a given parameter configuration ```trial.hyperparameters```. How the user generates a score for the configuration is entirely at their discretion.

The ```self.build_fn(hp)``` function called in ```run_trial``` is what will supply us with a blank model (as defined above).

As we can see, any arguments you provide in the ```search()``` entry method, can be accessed in your ```run_trial()``` method.

```python
from storm.tuner import Tuner

class MyTuner(Tuner):

    def run_trial(self, trial, *args):
        # retrieve hyperparameters
        hp = trial.hyperparameters
        
        # retrieve any parameters supplied via main search method
        X_train, y_train, X_test, y_test = args[0], args[1], args[2], args[3]
        
        # build our configuration
        model = self.build_fn(hp)
        
        # train model
        history = model.fit(X_train,
                            y_train,
                            epochs=25,
                            validation_data=(X_test, y_test),
                            batch_size=params['batch_size'])
                            
        # calculate score
        score = history.history['val_loss'][-1]
        
        # assign score to the trial
        self.score_trial(trial, score)
```

We initialize our Tuner and provide our training data

```python
tuner = MyTuner(objective_direction='min', build_fn=build_model)
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

```randomize_axis_factor```: The main exploitative/explorative tradeoff parameter. A value closer to 1 means that steps will generally have more mutations. A value closer to 0 will mean steps are more likely to only do a single mutation. A value of 0.5 seems reasonable in most cases, although for problems where you expect a large degree of parameter independance you may move the value closer to 0 and likewise for problems where you expect a great degree of parameter co-dependence you may set the value closer to 1.

# Design goals

The StoRM tuner is designed to be as simple as possible. The tuner supplies a parameter configuration and the user assigns this configuration a score. We leave it up to the user to implement any number of methodologies that might fit their goals and use cases. These can include:

- Techniques to reduce variance (k-fold cross validation, trailing average of epoch loss, average of multiple trains)
- Bandit search techniques, where we might abandon training of the current model if there is a high enough certainty that this model will not beat the best score at the end of the training. Because the tuner only cares if we beat the best score, not necessarily how much a trial lost, this means we can safely discard the configuration by just returning from our trial at this point. This will cause the trial's score to be defaulted to None so it is not tested again.
- This freedom allows the user to optimize parameters used at various stages of the experiment as well, ex. data pre-processing, model architecture, and training
- The tuner will also work with various branches of ML that utilize NNs for the model. For example, some reinforcement learning algorithms have another set of parameters to optimize.

# Other notes/features

The tuner keep tracks of which parameters are in use by building a dummy model prior to hashing the configuration. When building the model, parameters the model building function actually draws from are flagged as active. For example, if we have a parameter to determine number of layers to use, if the number of layers is set to 1, parameters only applicable to layer 2+ will not be included in the hash. This allows us to ensure we never test configurations that are virtually identical. Since testing a configuration is expensive, i believe the upfront cost of just building a blank model is far less than by chance testing the same configuration twice.

# Performance

Run ```compare_to_random_search.py``` from ```examples``` directory to compare performance to random search for yourself.

Here we can see that over ten trials each, StoRM has a clear advantage.

```tuned scores mean: 0.0013418583347811364 | stdev: 0.001806810901973602```

```random scores mean: 0.010490878883283586 | stdev: 0.006145158964894091```


