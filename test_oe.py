"""
Test OptimizedEnsemble 
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from scikit_ext.extensions import OptimizedEnsemble
from sklearn.metrics import make_scorer, f1_score

def main():

    # load sample data
    data = load_iris()
    X = data.data
    y = data.target

    # initialize model
    model = OptimizedEnsemble(
        RandomForestClassifier(max_depth=1, random_state=5), 
        n_estimators_init=10,
        max_iter=2,
        threshold=0.01,
        verbose=True,
        step_function=lambda x: x*2)

    # fit models
    model.fit(X, y)

    # print results
    print model._estimator_type
    print model.best_estimator_
    print model.scores_
    print model.best_score_
    print model.best_index_
    print model.best_n_estimators_
    print model.classes_
    print model.n_estimators_list_
    print model.score(X,y)

if __name__ == "__main__":
    main()
