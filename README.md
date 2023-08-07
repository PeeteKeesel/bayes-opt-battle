<div align="center">

# Bayesian Optimization Comparison

![Python](https://img.shields.io/badge/python-3.11.4-green)
![optuna](https://img.shields.io/badge/optuna-3.2.0-blue)
![bayesian-optimization](https://img.shields.io/badge/bayesian--optimization-1.4.3-blue)
![scikit-optimize](https://img.shields.io/badge/scikit--optimize-0.9.0-blue)
![hyperopt](https://img.shields.io/badge/hyperopt-0.2.7-blue)

</div>

Ever wondered which bayesian optimization framework to use for your project? We try to help you with that :)  

> This repository provides a general comparison of different [bayesian optimization](https://en.wikipedia.org/wiki/Bayesian_optimization) frameworks. 

## :books: Table of Contents
- [Bayesian Optimization Comparison](#bayesian-optimization-comparison)
  - [:books: Table of Contents](#books-table-of-contents)
  - [:dart: Summary](#dart-summary)
  - [:bulb: Library Descriptions](#bulb-library-descriptions)
    - [:one: Optuna](#one-optuna)
    - [:two: BayesianOptimization](#two-bayesianoptimization)
    - [:three: BayesSearchCV](#three-bayessearchcv)
    - [:four: hyperopt](#four-hyperopt)
    - [:five: gp\_minimize](#five-gp_minimize)
  - [:pencil: My Notes](#pencil-my-notes)
  - [:calendar: ToDo's](#calendar-todos)

## :dart: Summary

| Library :robot: | Tune Time :hourglass:  | Notebook :closed_book: | 
| ---- | ---- | ---- |
| Optuna | | [optuna.ipynb](notebooks/optuna.ipynb) | 

## :bulb: Library Descriptions
### :one: [Optuna](https://optuna.org/)

```bash
# Pip
% pip install optuna
# Conda
% conda install -c conda-forge optuna
```

In Optuna you need to define an objective, create a study via [create_study]() and optimize your objective. An example implementation would be as follows:

```python
def objective(trial):
    # Defining the hyperparameter search space.
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    # ...
    
    # Build the pipeline
    clf = RandomForestClassifier(n_estimators=n_estimators,
                                 # ...  
                                 )
    
    pipeline.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    
    return acc

# Creating a study and running Optuna optimization.
study = optuna.create_study(study_name='my_optuna_study',
                            direction='maximize')
study.optimize(objective, 
               n_trials=100)

# Obtain the best found parameters.
best_params = study.best_params
```


While tuning Optuna provides logs in the following format 

```
[I 2023-08-07 11:32:52,097] A new study created in memory with name: my_optuna_study
[I 2023-08-07 11:32:52,296] Trial 0 finished with value: 0.15254237288135594 and parameters: {'n_estimators': 117, 'max_depth': 15, 'min_samples_split': 0.9872932015318743, 'min_samples_leaf': 0.12850314179489697}. Best is trial 0 with value: 0.15254237288135594.
[I 2023-08-07 11:32:52,590] Trial 1 finished with value: 0.0847457627118644 and parameters: {'n_estimators': 189, 'max_depth': 15, 'min_samples_split': 0.4880445134056033, 'min_samples_leaf': 0.30119199350593756}. Best is trial 0 with value: 0.15254237288135594.
[I 2023-08-07 11:32:52,774] Trial 2 finished with value: 0.0847457627118644 and parameters: {'n_estimators': 113, 'max_depth': 6, 'min_samples_split': 0.16337780557406967, 'min_samples_leaf': 0.3044302613012829}. Best is trial 0 with value: 0.15254237288135594.
...
```

### :two: [BayesianOptimization](https://github.com/bayesian-optimization/BayesianOptimization)

```bash
# Pip
$ pip install bayesian-optimization
# Conda
$ conda install -c conda-forge bayesian-optimization
```

### :three: [BayesSearchCV](https://scikit-optimize.github.io/stable/modules/generated/skopt.BayesSearchCV.html)

```bash
# Pip
$ pip install scikit-optimize
# Conda
% conda install -c conda-forge scikit-optimize
```

### :four: [hyperopt](http://hyperopt.github.io/hyperopt/)

```bash
# Pip
% pip install hyperopt
# Conda
% conda install -c conda-forge hyperopt
```

### :five: [gp_minimize](https://scikit-optimize.github.io/stable/modules/generated/skopt.gp_minimize.html)

```bash
# Pip
$ pip install scikit-optimize
# Conda
% conda install -c conda-forge scikit-optimize
```

## :pencil: My Notes 



## :calendar: ToDo's

