"""
Test random iterations estimator 
"""

from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from scikit_ext.estimators import IterRandomEstimator
from scikit_ext.scorers import cluster_distribution_score

def main():

    # load sample data
    data = load_iris()
    X = data.data

    # initialize model
    scoring = cluster_distribution_score
    model = IterRandomEstimator(
        KMeans(n_clusters=4), 
        max_iter=5,
        verbose=0,
        scoring=scoring)

    # fit models
    model.fit(X)

    # print results
    print model.best_estimator_
    print model.best_score_
    print model.scores_

if __name__ == "__main__":
    main()
