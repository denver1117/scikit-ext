"""
Test OneVsRestAdjClassifier 
"""

import pandas as pd
from scikit_ext.extensions import _TimeScorer, _MemoryScorer
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import make_scorer, accuracy_score

def main():

    # load sample data
    data = load_iris()
    X = data.data
    y = data.target

    # initialize model
    acc_scorer = make_scorer(accuracy_score)
    latency_score = _TimeScorer(
        None, 1, {"unit": False, "n_iter": 25, "scoring": acc_scorer, "tradeoff": 0.01})
    memory_score = _MemoryScorer(
        None, 1, {"scoring": acc_scorer, "tradeoff": 100})
    model = GridSearchCV(
        DecisionTreeClassifier(), cv=3, 
        param_grid={'max_depth': [None, 15, 10, 5, 2, 1]},
        scoring=latency_score)

    # fit models
    model.fit(X, y)

    # print results
    print pd.DataFrame(model.cv_results_)
    print model.best_score_
    print model.best_params_

if __name__ == "__main__":
    main()
