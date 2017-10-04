"""
Test OptimizedEnsemble 
"""

from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from scikit_ext.extensions import IterRandomEstimator

def main():

    # load sample data
    data = load_iris()
    X = data.data

    # initialize model
    model = IterRandomEstimator(
        KMeans(n_clusters=4), 
        max_iter=5,
        verbose=0)

    # fit models
    model.fit(X)

    # print results
    print model.best_estimator_
    print model.best_score_
    print model.scores_

if __name__ == "__main__":
    main()
