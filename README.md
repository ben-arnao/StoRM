# StoRM (Stochastic Random Mutator)
A robust hyperparameter tuner for high-dimensional, categorically and/or conditionally-parameterized, intractable optimization problems (Ex. Neural Network)

# Motivations of this tuner

Neural network hyper parameter optimization is an especially challenging task due to a few different reasons:

Parameters can be highly codependent. Adjusting a single parameter may not be enough to get over a saddle point, you will likely have to adjust many parameters simultaneously to escape local minima. 

You may have scenarios where adjusting a parameter can completely alter the performance of other parameters as well, making it very difficult to sample historically better values more often and run the risk of sampling values based on the modeling of a suboptimal parameter space. 

Attempting to model which parameters are more likely to be better will also require a lot of trials to overcome this level of variance/noise. Even then, as alluded to above, the best parameter value on average will not always be the best parameter value overall. 

The search space can be highly non-convex, with many categorical, discrete-valued, conditional, and nested parameters. This sort of parameter space makes it very difficult to generate any sort of quantitative probability model.

For high-end performance where local minima is not good enough and we want the best possible performance, or for domains where there has not been extensive research and there is not a general understanding on what types of choices work better than others, we might want to tune many parameters at once and the dimensionality of the search space can get very large, such that Bayesian Optimization-related methods are not very effective.

Recent research has discussed there is not a lot of reproducible evidence that show any of today's state of the art techniques significantly beat a plain old random search with some form of early stopping- https://arxiv.org/pdf/1902.07638.pdf

# How does this tuner attempt to solve these issues?

All of the issues mentioned above make it very difficult if not impossible to do any sort of intelligently guided search for NN architecture/training hyperparameters. That is why this tuner opts against attempting to build some sort of surrogate function or gradient-based method to model the probability of the search space, and instead aims for something simpler and hopefully more robust to the problems we're facing. The user shouldn't expect a magic algorithm that takes the least amount of steps possible to reach global minima, but they should be able to expect something that beats brute force/random search for almost all use cases by a respectable margin, which is really what NN tuning needs at this stage.

The StoRM tuner can be thought of intuitively as a combination of a grid search combined with random search, where the "distance" between the current best configuration and the next evaluation candidate, is probability based. We randomly mutate the current best configuration along different axes (and sometimes even multiple times along the same axis). The number of mutations made for the next evaluation candidate, is based on a user-defined probability.

The default value for ```randomize_axis_factor``` is 0.5 which means that there is a 50% chance just one mutation will be made. There is a 25% chance two mutations will be made. A 12.5% chance that three mutations will be made, and so on.

This approach aims to address the issues stated above by allowing enough freedom so that we do respect the non-convexities of the search space and co-dependency of variables, while also probabilistically restricting how different the next evaluation candidate is from the current best, to provide some level of guidance and locality to the search.

# Installation

```pip install storm-tuner```

# Usage

Here we define our hyperparameter space by providing our own configuration building method.

NOTE: The configuration building method is an important component of StoRM's functionality. Even though parameters can be accessed elsewhere, for example when the model is trained or during data preprocessing, all parameters must be defined in this method. This is because StoRM will execute this function in the background prior to the user defined execution of a trial. The reason for this is that StoRM will flag parameters that are actually drawn from and then create a hash of this particular configuration. This is a vital component as it ensures we never waste resources testing virtually identical configurations.

After we define our HP space, it will usually make the most sense for our function to return an untrained model at this point. However, one may opt to return more than a model in some circumstances (for example an optimizer) or they may even opt to not return anything at all and build the model later. This is entirely up to the user.

All parameters take the form: ```hp.Param('parameter_name', [value1, value2...], ordered=False)```. Setting a parameter to ```ordered=True```, will ensure the tuner is only able to select adjacent values per a single mutation step. This is an important feature for parameters where there is ordinality.

```python
def build_model(hp, *args):
    model = Sequential()
    
    # we can define train-time param in the build model function to be used later on in run_trial
    hp.Param('batch_size', [32, 64, 128, 256], ordered=True)
    
    # storm works easily with loops as well
    for x in range(hp.Param('num_layers', [1, 2, 3, 4], ordered=True)):
        model.add(Dense(hp.Param('kernel_size_' + str(x), [50, 100, 200], ordered=True)))
    
    # here is a categorical parameter that most tuners do not do well with
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

The ```self.build_fn(hp)``` function called in ```run_trial``` is what will supply us with a blank model (as mentioned above).

As we can see, any arguments you provide in the ```search()``` entry method, can be accessed in your ```run_trial()``` method. This is also true for ```build_model``` as well, if any parameters need to be passed in at this scope.

```python
from storm_tuner import Tuner

class MyTuner(Tuner):

    def run_trial(self, trial, *args):
        # retrieve hyperparameters
        hp = trial.hyperparameters
        
        # retrieve any parameters supplied via main search method
        X_train, y_train, X_test, y_test = args[0], args[1], args[2], args[3]
        
        # create our model/configuration
        model = self.build_fn(hp)
        
        # train model
        history = model.fit(X_train,
                            y_train,
                            epochs=25,
                            validation_data=(X_test, y_test),
                            
                            # here we access a parameter at train time
                            batch_size=hp.values['batch_size'])
                            
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
                 build_fn=build_model,
                 objective_direction='min',
                 init_random=10,
                 max_trials=100,
                 randomize_axis_factor=0.5)
```

```init_random```: How many initial iterations to perform random search for (3 -> 10+). This is helpful for getting the search to an average/decent configuration, so that we don't waste too much time descending from a suboptimal starting point.

```randomize_axis_factor```: The main exploitative/explorative tradeoff parameter (0 -> 1). A value closer to 1 means that steps will generally have more mutations. A value closer to 0 will mean steps are more likely to only do a single mutation. A value of 0.5 seems reasonable in most cases and will almost always be good enough.

For both of these parameters, the higher degree of parameter codependency that is expected and/or the more parameters that you are optimizing, it may be beneficial to set these values higher... ex. 10 initial random iterations, and 0.66 or 0.75 randomize acis factor.

# StoRM's design goals

The StoRM tuner is designed to be as simple as possible. The tuner supplies a parameter configuration and the user assigns this configuration a score. We leave it up to the user to implement any number of methodologies that might fit their goals and use cases. These can include:

- Techniques to reduce variance (k-fold cross validation, trailing average of epoch loss, average of multiple trains)
- Techniques where we might abandon training of the current model if there is a high enough certainty that this model will not beat the best score at the end of the training. *Because the tuner only cares if we beat the best score, not necessarily how much a trial lost, this means we can safely discard the configuration by just returning from our trial at this point. This will cause the trial's score to be defaulted to None so it is not tested again. Note: if we decide to run metrics on variables across all trials after tuning is complete, this may skew the results.*

Storm should be designed to be as generic as possible AND there is actually nothing specific to neural networks or a particular NN library coded in this project. This type of freedom also allows the user to optimize parameters used at various stages of the experiment as well, ex. data pre-processing, model architecture, and training.

Because of the tuner's experiment-agnostic approach, StoRM can be even more advantageous when used with various branches of ML that utilize NNs for the model yet have another set of hyper parameters to optimize that can make the search space even trickier and harder for traditional approaches to handle. For example, reinforcement learning.

# How to get the most out of StoRM

Of course, most of the success of StoRM revolves around the user's ability to parameterize the search space appropriately. StoRM will only be as good as the parameter space it operates on. A few things to keep in mind when parameterizing your search space...

- For a parameter with ordinal values that can also be turned completely of, such as dropout, one might consider adding an extra boolean parameter to unlock the dropout rate ordinal parameter. If optimization initializes to a suboptimal higher dropout value and dropout is not good for this particular problem, it will probably take more iterations to traverse the dropout value space than it would to turn dropout off for a configuration to escape this minimum.
- For ordinal parameters where it is not an option to use an additional "gateway" parameter, it is suggested to keep the amount of values under 10 and ideally around 5 for reasons explained above.
- For parameters that are coupled with one another (for example learning rate and weight decay). One might decide to parameterize weight decay as a factor of LR, instead of optimizing both separately. This way, we only search for the best step size to weight decay ratio, instead of forcing the model to try and find LR and WD values that meet at the right scale.
- Most NN hyper parameters are not very sensitive and it is far more important to find a good general area/scale for a parameter than it is for example to know that a learning rate of 1e-3 performs slightly better than 2e-3. We want to ensure there is a good distribution of values such that we capture the various points a parameter is commonly experimented with, yet do not have an overabundance of ordinal values so that our tuner has to stochastically traverse this space if initialized to a poor value. StoRM leaves it up to the user to provide the appropriate binning/sampling of values (log, exp, linear, etc.) which is very parameter-dependent.

In most cases the selection of values should be fairly intuitive...

batch size: [32, 64, 128, 256]

momentum: [0.8, 0.9, 0.98]

kernel size: [50, 100, 200, 400]

lr: [1e-2, 1e-3, 1e-4]

At the end of the day there is then nothing stopping the user from re-parameterizing their search space after narrowing in on promising areas from running storm tuner at a broader scope.

# StoRM is library-agnostic.

Although the examples here use Tensorflow/Keras, StoRM works with any library or algorithm (sklearn, pytorch, etc.). One simply defines any parameters we are optimizing in  ```build_fn```. The user can decide to return a model right here and utilize StoRM's inline parameterization, or they can opt to use parameters in ```run_trial```.

# What types of problems can StoRM be used for?

StoRM should be used for optimization problems where the parameter space can be high dimensional and has many categorical/conditional variables. StoRM should also be used when parameters do not need to be optimized at very fine levels, but rather we need to find good general choices. In short, StoRM will be most effective when there are many codependent decisions to be made.

StoRM will probably not be the best tuner to use if you are optimizing many real valued parameters that always have an affect on the target function, with low codependencies, and which can be sensitive to small changes such that we should offer the real valued spectrum of values, and not just a few bins to chose from. For these types of problems, Bayesian Optimization will still be more effective.

# Other notes/features

A StoRM ```Trial``` like the one used in the run trial method above, has a ```metrics dictionary``` to easily allows us to store any pertinent information to this trial for review later on.

# Performance

Run ```compare_to_random_search.py``` from ```examples``` directory to compare performance to random search for yourself.

Here we can see that over ten trials each, StoRM has a clear advantage.

```tuned scores mean: 0.0013418583347811364 | stdev: 0.001806810901973602```

```random scores mean: 0.010490878883283586 | stdev: 0.006145158964894091```


Run ```compare_to_optuna.py``` from ```examples``` directory to compare performance to a state of the art optimizer like Optuna.

Here we can see that with 100 tuning iterations and 25 trials per each tuner, StoRM has a slight edge and is definitely competitive.

```storm scores mean: 0.6252435888846715 std: 0.012562026804848855```

```optuna scores mean: 0.6011923088630041 std: 0.011612774221843973```
