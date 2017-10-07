"""
Various scikit-learn scorers and scoring functions
"""

import numpy as np
import time
import cPickle
from sklearn.metrics.scorer import _BaseScorer

class _TimeScorer(_BaseScorer):
    def __call__(self, estimator, X, y_true=None, n_iter=1, unit=True, scoring=None, tradeoff=None):
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
        scoring: scorer object, default None
            Scorer used for trade-off.
        tradeoff: float, default None
            Multiplier for tradeoff.

        Returns
        -------
        score : float
            Custom score combining scoring method (optional)
            and estimator prediction latency (ms).
        """

        # overwrite kwargs from _kwargs
        if "n_iter" in self._kwargs.keys():
            n_iter = self._kwargs["n_iter"]
        if "unit" in self._kwargs.keys():
            unit = self._kwargs["unit"]
        if "scoring" in self._kwargs.keys():
            scoring = self._kwargs["scoring"]
        if "tradeoff" in self._kwargs.keys():
            tradeoff = self._kwargs["tradeoff"]

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
        unit_time = 1000 * float((time_sum / float(n_iter)) / float(len(X)))
        if scoring and tradeoff:
            scoring_score = scoring(estimator, X, y_true)
            return scoring_score - (tradeoff * unit_time)
        else:
            return 1. / unit_time

    def _elapsed(self, estimator, X):
        """
        Return elapsed time for predict method of estimator
        on X.
        """
            
        start_time = time.time()
        y_pred = estimator.predict(X)
        end_time = time.time()
        return end_time - start_time

class _MemoryScorer(_BaseScorer):
    def __call__(self, estimator, X=None, y_true=None, scoring=None, tradeoff=None):
        """
        Score using estimated memory of pickled estimator object.

        Parameters
        ----------
        estimator : object
            Trained estimator to use for scoring.
        X : array-like or sparse matrix
            Test data that will be fed to estimator.predict.
            Not necessary for _MemoryScorer.
        y_true : array-like, default None
            Gold standard target values for X. Not necessary
            for _MemoryScorer.
        scoring: scorer object, default None
            Scorer used for trade-off.
        tradeoff: float, default None
            Multiplier for tradeoff.

        Returns
        -------
        score : float
            Custom score combining scoring method (optional)
            and estimator memory (MB).
        """

        # overwrite kwargs from _kwargs
        if "scoring" in self._kwargs.keys():
            scoring = self._kwargs["scoring"]
        if "tradeoff" in self._kwargs.keys():
            tradeoff = self._kwargs["tradeoff"]

        obj_size = (0.000001 * float(len(cPickle.dumps(estimator))))
        if scoring and tradeoff:
            scoring_score = scoring(estimator, X, y_true)
            return scoring_score - (tradeoff * obj_size)
        else:
            return 1. / obj_size

class _CombinedScorer(_BaseScorer):
    def __call__(self, estimator, X=None, y_true=None, scoring=None):
        """
        Score using estimated memory of pickled estimator object.

        Parameters
        ----------
        estimator : object
            Trained estimator to use for scoring.
        X : array-like or sparse matrix
            Test data that will be fed to estimator.predict.
            Not necessary for _MemoryScorer.
        y_true : array-like, default None
            Gold standard target values for X. Not necessary
            for _MemoryScorer.
        scoring: list of scorer objects, default None
            List of scorers to average.

        Returns
        -------
        score : float
            Custom score combining input scoring methods
            using the mean score..
        """

        # overwrite kwargs from _kwargs
        if "scoring" in self._kwargs.keys():
            scoring = self._kwargs["scoring"]

        if (not isinstance(scoring, list)) and (not isinstance(scoring, tuple)):
            scoring = [scoring]

        return np.mean([x(estimator, X, y_true) for x in scoring])

def cluster_distribution_score(X, labels):
    """
    Description

    Parameters
    ----------
    X : array-like, shape (``n_samples``, ``n_features``)
        List of ``n_features``-dimensional data points. Each row corresponds
        to a single data point.
    labels : array-like, shape (``n_samples``,)
        Predicted labels for each sample.

    Returns
    -------
    score : float
        The resulting Cluster Distribution score.
    """

    n_clusters = float(len(np.unique(labels)))
    max_count = float(np.max(np.bincount(labels)))
    return 1.0 / ((max_count / len(labels)) / (1.0 / n_clusters))

    
