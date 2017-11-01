# scikit-ext : various scikit-learn extensions

### About
The `scikit_ext` package contains various scikit-learn extensions, built entirely on top of `sklearn` base classes.  The package is separated into two modules: [estimators](http://scikit-ext.s3-website-us-east-1.amazonaws.com/scikit_ext.html#module-scikit_ext.estimators) and [scorers](http://scikit-ext.s3-website-us-east-1.amazonaws.com/scikit_ext.html#module-scikit_ext.scorers). Full documentation can be found [here](http://scikit-ext.s3-website-us-east-1.amazonaws.com/index.html).

### Installation
[Package Index on PyPI](https://pypi.python.org/pypi/scikit-ext) <br> To install:
```
pip install scikit-ext
```

### Estimators
- `MultiGridSearchCV`: Extension to native sklearn `GridSearchCV` for multiple estimators and param_grids. Accepts a list
    of estimators and param_grids, iterating through each fitting 
    a `GridSearchCV` model for each estimator/param_grid. Chooses
    the best fitted `GridSearchCV` model. Inherits sklearn's `BaseSearchCV`
    class, so attributes and methods are all similar to `GridSearchCV`.
- `PrunedPipeline`: Extension to native sklearn `Pipeline` intended for text learning pipelines
    with a vectorization step and a feature selection step. Instead of remembering all
    vectorizer vocabulary elements and selecting appropriate features at prediction time,
    the extension prunes the vocabulary after fitting to only include elements who will
    ultimately survive the feature selection filter applied later in the pipeline. This reduces
    memory and improves prediction latency. Predictions will be identical to those made
    with a trained `Pipeline` model. Inherits sklearn's `Pipeline`
    class, so attributes and methods are all similar to `Pipeline`.
- `ZoomGridSearchCV`: Extension to native sklearn `GridSearchCV`. Fits multiple `GridSearchCV` models, updating
    the `param_grid` after each iteration. The update
    looks at successful parameter values for each 
    grid key. A new list of values is created which 
    expands the resolution of the search values centered
    around the best performing value of the previous fit.
    This allows the standard grid search process to start 
    with a small number of distant values for each parameter,
    and zoom in as the better performing corner of the 
    hyperparameter search space becomes clear.
- `IterRandomEstimator`: Meta-Estimator intended primarily for unsupervised 
    estimators whose fitted model can be heavily dependent
    on an arbitrary random initialization state.  It is   
    best used for problems where a `fit_predict` method
    is intended, so the only data used for prediction will be
    the same data on which the model was fitted.
- `OptimizedEnsemble`: An optimized ensemble class. Will find the optimal `n_estimators`
    parameter for the given ensemble estimator, according to the
    specified input parameters.
- `OneVsRestAdjClassifier`: One-Vs-Rest multiclass strategy.  The adjusted version is a custom 
    extension which overwrites the inherited `predict_proba` method with
    a more flexible method allowing custom normalization for the predicted probabilities. Any norm
    argument that can be passed directly to `sklearn.preprocessing.normalize` is allowed. Additionally,
    norm=None will skip the normalization step alltogeter. To mimick the inherited `OneVsRestClassfier`
    behavior, set norm='l2'. All other methods are inherited from `OneVsRestClassifier`.
    
### Scorers
- `TimeScorer`: Score using estimated prediction latency of estimator.
- `MemoryScorer`: Score using estimated memory of pickled estimator object.
- `CombinedScorer`: Score combining multiple scorers by averaging their scores.
- `cluster_distribution_score`: Scoring function which scores the resulting cluster distribution accross classes. 
    A more even distribution indicates a higher score.

### Authors

Evan Harris 

### License

This project is licensed under the MIT License - see the LICENSE file for details
