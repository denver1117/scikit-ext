"""
Various scikit-learn extensions
"""

import numpy as np
import time
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize
from sklearn.metrics.scorer import _BaseScorer

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
        """ 
        Additional norm argument along with inheritance
        """

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

class _TimeScorer(_BaseScorer):
    def __call__(self, estimator, X, y_true=None, n_iter=1, unit=True):
        """
        Evaluate prediction latency.

        Parameters
        ----------
        estimator : object
            Trained estimator to use for scoring.
        X : array-like or sparse matrix
            Test data that will be fed to estimator.predict.
        y_true : array-like, default None
            Gold standard target values for X. Not necessary
            for _TimeScorer.
        n_iter : int, default 1
            Number of timing runs.
        unit : bool, default True
            Use per-unit latency or total latency.

        Returns
        -------
        score : float
            Average of timing iteratins applied to 
            prediction of estimator on X.
        """

        # overwrite kwargs from _kwargs
        if "n_iter" in self._kwargs.keys():
            n_iter = self._kwargs["n_iter"]
        if "unit" in self._kwargs.keys():
            unit = self._kwargs["unit"]

        # run timing iterations
        count = 0
        time_sum = 0
        while count < n_iter:
            count += 1
            if unit:
                time_sum += np.sum([
                    self._elapsed(estimator, [x]) 
                    for x in X])
            else:
                time_sum += self._elapsed(estimator, X)
        return 1 / ((time_sum / float(n_iter)) / float(len(X)))

    def _elapsed(self, estimator, X):
        """
        Return elapsed time for predict method of estimator
        on X.
        """
            
        start_time = time.time()
        y_pred = estimator.predict(X)
        end_time = time.time()
        return end_time - start_time

