# scikit-ext : various scikit-learn extensions

### About
The `scikit_ext` package contains various scikit-learn extensions, built entirely on top of `sklearn` base classes.  The package is separated into two modules, `estimators` and `scorers`.  

### Estimators
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

### Authors

Evan Harris 

### License

This project is licensed under the MIT License - see the LICENSE file for details
