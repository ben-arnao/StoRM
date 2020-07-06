# stochasticmutatortuner
A neural network hyperparameter tuner

# Motivations of this tuner

Neural network hyper parameter optimization is an especially challenging task due to 3 main reasons.

1) Parameters are highly co dependent, arguably on all axes.

2) The search space can be highly convex and intractable.

3) For high-end performance where we need to squeeze as much performance out of our model as possible, the search space can get very large.

Recent research has shown there is not much reproducible evidence that any of today's state of the art techniques significantly beat random search. https://arxiv.org/pdf/1902.07638.pdf

# How does this tuner attempt to solve these issues?

The tuner can be thought of as a combination of a restricted grid search combined with random search. The idea behind this tuner is to mutate the network along different axes and let the user choose how explorative the tuner should be. This approach allows for the tuner to combine the benefits of fine tuning a configuration for a slow and steady descent, but also allowing the tuner to have the freedom to mutate the network multiple times in one step, so that it can get out of local minima.

The default value ```randomize_axis_factor``` is 0.5 which means that there is a 50% chance, just one mutation will be made. There is a 25% chance 2 mutations will be made. A 12.5% chance that 3 mutations will be made, and so on.

My belief is that this tuner provides a good balance in addressing the issues above. Allowing enough freedom so that we do respect the convexness of the search space and co-dependency of variables while also restricting the the probability of hops, so that there is at least some guidance.

# Usage

Here we define our hyper parameter space through providing our own model building method. All we need to do is define our HP space, and return an untrained model. Parameters used at train time can also be defined here. All parameters take the form: ```hp.Choice('parameter_name', [value1, value2...], ordered=False)```. Setting a parameter to ordered=True, will ensure the tuner is only able to select adjacent values per a single mutation step.

```python
def build_model(hp):
    model = Sequential()
    
    # we can define train-time params in the build model function
    hp.Param('batch_size', [32, 64, 128, 256], ordered=True)
    
    for x in

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

We override the ```run_trial()``` method for our own Tuner, this encapsulates all the work of a single trial. All the run_trial method needs to do is assign a score to the trial ```self.score_trial(trial, score)```. How you use your model to make the score for the trial, is up to you (ie. K-fold cross validation). The ```self.hypermodel.build(hp)``` function called in ```run_trial``` is what will supply us with a blank model.

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

```init_random```: How many iterations to perform random search for. This is helpful for getting the search to an average configuration, so that we don't waste too much time descending from a sub

```randomize_axis_factor```: The main exploitative/explorative tradeoff parameter. A value closer to 1 means that steps will generally have more mutations. A value closer to 0 will mean steps are more likely to only do a single mutation.
