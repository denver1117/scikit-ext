"""
Various scikit-learn estimators and meta-estimators
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import rankdata 
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize
from sklearn.base import (
    BaseEstimator, ClassifierMixin, 
    is_classifier, clone)
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.model_selection._split import check_cv
from sklearn.model_selection._search import BaseSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.exceptions import NotFittedError
from sklearn.metrics import calinski_harabaz_score
from sklearn.pipeline import Pipeline

class PrunedPipeline(Pipeline):
    """
    A standard sklearn feature Pipeline with additional 
    pruning method. After fitting, the pruning method is 
    applied to the fitted pipeline. This applies the 
    feature selection directly to the fitted vocabulary
    (and idf values if applicable), removing all elements of
    these attributes that will not ultimately survive the  
    feature selection filter.

    The ``PrunedPipeline`` will make idential predictions 
    as a similarly trained ``Pipeline``. However, it will require
    less memory and will make faster predictions.

    Parameters
    ----------
    steps : list
        List of (name, transform) tuples (implementing fit/transform) that are
        chained, in the order in which they are chained, with the last object
        an estimator.
    memory : None, str or object with the joblib.Memory interface, optional
        Used to cache the fitted transformers of the pipeline. By default,
        no caching is performed. If a string is given, it is the path to
        the caching directory. Enabling caching triggers a clone of
        the transformers before fitting. Therefore, the transformer
        instance given to the pipeline cannot be inspected
        directly. Use the attribute ``named_steps`` or ``steps`` to
        inspect estimators within the pipeline. Caching the
        transformers is advantageous when fitting is time consuming.
    vectorizer_name : str, default vec
        Name of ``Pipeline`` step which performs feature extraction. Any
        transformer with a ``vocabulary_``dictionary can be the step
        with this name.
        Ideal transformers are of types
        sklearn.feature_extraction.text.CountVectorizer or
        sklearn.feature_extraction.text.TfidfVectorizer.
    selector_name : str, default select
        Name of ``Pipeline`` step which performs feature selection. Any
        transformer with a ``get_support`` method returning an iterable
        of booleans with length ``len(vocabulary_)`` can be the step with this name. 
        Ideal transformers are of type sklearn.feature_selection.univariate_selection._BaseFilter.

    Attributes
    ----------
    named_steps : bunch object, a dictionary with attribute access
        Read-only attribute to access any step parameter by user given name.
        Keys are step names and values are steps parameters.

    """

    def __init__(self, steps, memory=None, 
                 vectorizer_name="vec", 
                 selector_name="select"):
    
        Pipeline.__init__(
            self, steps, memory=memory)
        self.vectorizer_name=vectorizer_name
        self.selector_name=selector_name
        self._validate_prune()

    def fit(self, X, y=None, **fit_params):
        """
        Fit the model
        Fit all the transforms one after the other and transform the
        data, then fit the transformed data using the final estimator.
        Perform prune after standard pipeline fit.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.
        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.
        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        self : PrunedPipeline
            This estimator
        """

        # standard Pipeline fit method
        self._validate_prune()
        Xt, fit_params = self._fit(X, y, **fit_params)
        if self._final_estimator is not None:
            self._final_estimator.fit(Xt, y, **fit_params)

        # prune pipeline
        if self.selector_name and self.vectorizer_name:
            self._prune()

        return self

    def _validate_prune(self):
        """ Validate prune step inputs """

        names, estimators = zip(*self.steps)
        for name in [self.selector_name, self.vectorizer_name]:
            if name:
                if not name in names: 
                    raise ValueError(
                        "Name {0} should exist in steps".format(
                            name))
        self.selector_index = names.index(self.selector_name) 
        self.vectorizer_index = names.index(self.vectorizer_name)        

    def _prune(self):
        """
        Prune fitted ``Pipeline`` object. The pruner runs the 
        ``get_support`` method from the designated feature
        selector, returning the selector mask. Then the ``vocabulary_`` 
        (and optional ``idf_`` if exists) attribute is pruned
        to only contain elements who survive the selector mask. The
        selector step is then removed from the pipeline.
 
        Transform methods on the pipeline will then reflect these
        changes, reducing the size of the vectorizer and effectively
        skipping the selector step. 
      
        """

        # collect pipeline step data
        voc = self.steps[self.vectorizer_index][1].vocabulary_
        if hasattr(self.steps[self.vectorizer_index][1], "idf_"):
            idf = self.steps[self.vectorizer_index][1].idf_
        else:
            idf = None
        support = self.steps[self.selector_index][1].get_support()

        # restructure vocabulary
        terms = []
        indices = []
        for key, value in voc.iteritems():
            terms.append(key)
            indices.append(value)
        sort_mask = np.argsort(indices)
        terms = np.array(terms)[sort_mask]

        # rebuild vocabulary dictionary
        new_vocab = {}
        new_idf = []
        count = 0
        for index in range(len(terms)):
            if support[index]:
                new_vocab[terms[index]] = count
                if idf is not None:
                    new_idf.append(idf[index])
                count += 1

        # replace vocabulary 
        self.steps[self.vectorizer_index][1].vocabulary_ = new_vocab
        if idf is not None:
            self.steps[self.vectorizer_index][1]._tfidf._idf_diag = csr_matrix(np.diag(new_idf))
        removed_step = self.steps.pop(self.selector_index)

class MultiGridSearchCV(BaseSearchCV):
    """
    An iterator through multiple GridSearchCV
    models using various ``estimators`` and associated ``param_grids``.
    Providing two equal length iterables as required arguments 
    containing estimators and paraeter grids, as well as keyword arguments for 
    GridSearchCV, will then simply iterate through and fit multiple 
    GridSearchCV models, fitting them sequentially.

    Then the maximum ``best_score_`` is compared accross the 
    GridSearchCV models, and the best one is identified. The best
    estimator is set as an attribute, ``best_estimator_`` and 
    the best GridSearchCV model is set as an attribute, 
    ``best_grid_search_cv_``.
    """

    def __init__(self, estimators, param_grids, **kwargs):
   
        self.estimators=estimators
        self.param_grids=param_grids  
        self.gs_kwargs=kwargs
        BaseSearchCV.__init__(
            self, None, **kwargs)

    def fit(self, X, y=None):
        """
        Iterate through estimators and param_grids, fitting 
        each, and then chosing the best. 

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        """

        # Iterate through estimators fitting each
        models = []
        for index in range(len(self.estimators)):
            model = GridSearchCV(
                self.estimators[index], 
                self.param_grids[index],
                **self.gs_kwargs)
            model.fit(X, y)
            models.append(model)

        # Generate cross validation results
        cv_df = pd.DataFrame()
        for index in range(len(models)):
            tmpDf = pd.DataFrame(models[index].cv_results_)
            tmpDf["grid_search_index"] = index
            cv_df = cv_df.append(tmpDf)
        cv_df.index = range(len(cv_df))
        cv_df = cv_df[[c for c in cv_df.columns if "param_" not in c]]
        cv_df["rank_test_score"] = map(int, 
            (len(cv_df) + 1) - 
            rankdata(cv_df["mean_test_score"], method="ordinal"))
        self.cv_results_ = {}
        for col in cv_df.columns:
            self.cv_results_[col] = list(cv_df[col].values)
      
        # Find best model and set associated attributes
        self.scores_ = [x.best_score_ for x in models]
        self.best_index_ = np.argmax(self.scores_)
        self.best_score_ = models[self.best_index_].best_score_
        self.best_grid_search_cv_ = models[self.best_index_]
        self.best_estimator_ = models[self.best_index_].best_estimator_
        self.scorer_ = self.best_grid_search_cv_.scorer_
        self.multimetric_ = self.best_grid_search_cv_.multimetric_
        self.n_splits_ = self.best_grid_search_cv_.n_splits_
        return self

class IterRandomEstimator(BaseEstimator, ClassifierMixin):
    """
    Meta-Estimator intended primarily for unsupervised 
    estimators whose fitted model can be heavily dependent
    on an arbitrary random initialization state.  It is   
    best used for problems where a ``fit_predict`` method
    is intended, so the only data used for prediction will be
    the same data on which the model was fitted.

    The ``fit`` method will fit multiple iterations of the same
    base estimator, varying the ``random_state`` argument
    for each iteration.  The iterations will stop either 
    when ``max_iter`` is reached, or when the target
    score is obtained.

    The model does not use cross validation to find the best
    estimator.  It simply fits and scores on the entire input
    data set.  A hyperparaeter is not being optimized here,
    only random initialization states.  The idea is to find
    the best fitted model, and keep that exact model, rather 
    than to find the best hyperparameter set.
    """

    def __init__(self, estimator, target_score=None,
                 max_iter=10, random_state=None,
                 scoring=calinski_harabaz_score, 
                 fit_params=None, verbose=0):

        self.estimator=estimator
        self.target_score=target_score
        self.max_iter=max_iter
        self.random_state=random_state
        if not self.random_state:
            self.random_state = np.random.randint(100)
        self.fit_params=fit_params
        self.verbose=verbose
        self.scoring=scoring

    def fit(self, X, y=None, **fit_params):
        """
        Run fit on the estimator attribute multiple times 
        with various ``random_state`` arguments and choose
        the fitted estimator with the best score.

        Uses ``calinski_harabaz_score`` if no scoring is provided.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of the estimator
        """

        if self.fit_params is not None:
            warnings.warn('"fit_params" as a constructor argument was '
                          'deprecated in version 0.19 and will be removed '
                          'in version 0.21. Pass fit parameters to the '
                          '"fit" method instead.', DeprecationWarning)
            if fit_params:
                warnings.warn('Ignoring fit_params passed as a constructor '
                              'argument in favor of keyword arguments to '
                              'the "fit" method.', RuntimeWarning)
            else:
                fit_params = self.fit_params
        estimator = self.estimator
        estimator.verbose = self.verbose
        if self.verbose > 0:
            if not self.target_score:
                print("Fitting {0} estimators unless a target "
                      "score of {1} is reached".format(
                          self.max_iter, self.target_score))
            else:
                print("Fitting {0} estimators".format(
                          self.max_iter))
        count = 0
        scores = []
        estimators = []
        states = []
        random_state = self.random_state
        if not random_state:
            random_state = n
        while count < self.max_iter:
            estimator = clone(estimator)
            if random_state:
                random_state = random_state + 1
            estimator.random_state = random_state 
            estimator.fit(X, y, **fit_params) 
            labels = estimator.labels_
            score = self.scoring(X, labels)
            scores.append(score)
            estimators.append(estimator)
            states.append(random_state)
            if self.target_score is not None and score > self.target_score:
                break
            count += 1
        
        self.best_estimator_ = estimators[np.argmax(scores)]
        self.best_score_ = np.max(scores)
        self.best_index_ = np.argmax(scores)
        self.best_params_ = self.best_estimator_.get_params()
        self.scores_ = scores
        self.random_states_ = states

class OptimizedEnsemble(BaseSearchCV):
    """
    An optimized ensemble class. Will find the optimal ``n_estimators``
    parameter for the given ensemble estimator, according to the
    specified input parameters.

    The ``fit`` method will iterate through n_estimators options,
    starting with n_estimators_init, and using the step_function 
    reursively from there. Stop at max_iter or when the score 
    gain between iterations is less than threshold. 

    The OptimizedEnsemble class can then itself be used
    as an Estimator, or the ``best_estimator_`` attribute
    can be accessed directly, which is a fitted version of the input
    estimator with the optimal parameters.
    """

    def __init__(self, estimator, n_estimators_init=5,
                 threshold=0.01, max_iter=10, 
                 step_function=lambda x: x*2,
                 **kwargs):

        self.n_estimators_init=n_estimators_init
        self.threshold=threshold
        self.step_function=step_function
        self.max_iter=max_iter
        BaseSearchCV.__init__(
            self, estimator, **kwargs)

    def fit(self, X, y, **fit_params):
        """
        Find the optimal ``n_estimators`` parameter using a custom
        optimization routine. 

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of the estimator
        """

        if self.fit_params is not None:
            warnings.warn('"fit_params" as a constructor argument was '
                          'deprecated in version 0.19 and will be removed '
                          'in version 0.21. Pass fit parameters to the '
                          '"fit" method instead.', DeprecationWarning)
            if fit_params:
                warnings.warn('Ignoring fit_params passed as a constructor '
                              'argument in favor of keyword arguments to '
                              'the "fit" method.', RuntimeWarning)
            else:
                fit_params = self.fit_params
        estimator = self.estimator
        cv = check_cv(self.cv, y, classifier=is_classifier(estimator))
        n_splits = cv.get_n_splits(X, y, groups=None)
        if self.verbose > 0:
            print("Fitting {0} folds for each n_estimators candidate, "
                  "for a maximum of {1} candidates, totalling"
                  " a maximum of {2} fits".format(n_splits, 
                      self.max_iter, self.max_iter * n_splits))
        count = 0
        scores = []
        n_estimators = []
        n_est = self.n_estimators_init
        while count < self.max_iter:
            estimator = clone(estimator)
            estimator.n_estimators = n_est
            score = np.mean(cross_val_score(
                estimator, X, y, cv=self.cv, 
                scoring=self.scoring,
                fit_params=fit_params,
                verbose=self.verbose,
                n_jobs=self.n_jobs,
                pre_dispatch=self.pre_dispatch))
            scores.append(score)
            n_estimators.append(n_est)
            if (count > 0 and 
                    (scores[count] - scores[count - 1]) < self.threshold):
                break
            else:
                best_estimator = estimator
            count += 1
            n_est = self.step_function(n_est)
        self.scores_ = scores
        self.n_estimators_list_ = n_estimators

        if self.refit:
            self.best_estimator_ = clone(best_estimator)
            if y is not None:
                self.best_estimator_.fit(X, y, **fit_params)
            else:
                self.best_estimator_.fit(X, **fit_params)
            self.best_index_ = count - 1
            self.best_score_ = self.scores_[count - 1]
            self.best_n_estimators_ = self.n_estimators_list_[count - 1]
            self.best_params_ = self.best_estimator_.get_params()
        return self

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def score(self, X, y=None):
        """
        Call score on the estimator with the best found parameters.
        Only available if the underlying estimator supports ``score``.

        This uses the score defined by the ``best_estimator_.score`` method.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input data, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        Returns
        -------
        score : float
        """

        self._check_is_fitted('score')
        return self.best_estimator_.score(X, y)

class OneVsRestAdjClassifier(OneVsRestClassifier):
    """
    One-vs-the-rest (OvR) multiclass strategy

    Also known as one-vs-all, this strategy consists in fitting one classifier per class. 
    For each classifier, the class is fitted against all the other classes. 
    In addition to its computational efficiency (only n_classes classifiers are needed), 
    one advantage of this approach is its interpretability. 
    Since each class is represented by one and one classifier only, it is possible to gain 
    knowledge about the class by inspecting its corresponding classifier. 
    This is the most commonly used strategy for multiclass classification and is a fair default choice.

    The adjusted version is a custom extension which overwrites the inherited predict_proba() method with
    a more flexible method allowing custom normalization for the predicted probabilities. Any norm
    argument that can be passed directly to sklearn.preprocessing.normalize is allowed. Additionally,
    norm=None will skip the normalization step alltogeter. To mimick the inherited OneVsRestClassfier
    behavior, set norm='l2'. All other methods are inherited from OneVsRestClassifier.


    Parameters	
    ----------
    estimator : estimator object
        An estimator object implementing fit and one of decision_function or predict_proba.
    n_jobs : int, optional, default: 1
        The number of jobs to use for the computation. If -1 all CPUs are used. 
        If 1 is given, no parallel computing code is used at all, which is useful for debugging. 
        For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one are used.
    norm: str, optional, default: None
        Normalization method to be passed straight into sklearn.preprocessing.normalize as the norm 
        input. A value of None (default) will skip the normalization step.

    Attributes
    ----------
    estimators_ : list of n_classes estimators
        Estimators used for predictions.
    classes_ : array, shape = [n_classes]
        Class labels.
    label_binarizer_ : LabelBinarizer object
        Object used to transform multiclass labels to binary labels and vice-versa.
    multilabel_ : boolean
        Whether a OneVsRestClassifier is a multilabel classifier.
    """

    def __init__(self, estimator, norm=None, **kwargs):

        OneVsRestClassifier.__init__(
            self, estimator, **kwargs)
        self.norm = norm
 
    def predict_proba(self, X):
        """ 
        Probability estimates.

        The returned estimates for all classes are ordered by label of classes.

        Parameters	
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns  
        -------
        T : array-like, shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in the model, 
            where classes are ordered as they are in self.classes_.
        """

        probs = []
        for index in range(len(self.estimators_)):
            probs.append(self.estimators_[index].predict_proba(X)[:,1])            
        out = np.array([
            [probs[y][index] for y in range(len(self.estimators_))] 
            for index in range(len(probs[0]))])
        if self.norm:
            return normalize(out, norm=self.norm)
        else:
            return out
