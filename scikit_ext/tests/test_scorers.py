"""
Test scorers module
"""

import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import make_scorer, accuracy_score

try:
    from scikit_ext import scorers
    _top_import_error = None
except Exception as e:
    _top_import_error = e

try:
    from scikit_ext.scorers import (
        TimeScorer, MemoryScorer, 
        CombinedScorer
        )
    _module_import_error = None
except Exception as e:
    _module_import_error = e

def test_top_import():
    assert _top_import_error == None

def test_module_import():
    assert _module_import_error == None

def test_time_scorer():
    # load sample data
    data = load_iris()
    X = data.data
    y = data.target

    # initialize model
    acc_scorer = make_scorer(accuracy_score)
    latency_score = TimeScorer(
        None, 1, {"unit": False, "n_iter": 25, "scoring": acc_scorer, "tradeoff": 0.01})

    # fit models
    model = GridSearchCV(
        DecisionTreeClassifier(random_state=5), 
        cv=5, param_grid={'max_depth': [5, 2]},
        scoring=latency_score
        )
    model.fit(X, y)
    assert round(model.best_score_,2) == 0.96

def test_mem_scorer():
    # load sample data
    data = load_iris()
    X = data.data
    y = data.target

    # initialize model
    acc_scorer = make_scorer(accuracy_score)
    memory_score = MemoryScorer(
        None, 1, {"scoring": acc_scorer, "tradeoff": 100})

    # fit models
    model = GridSearchCV(
        DecisionTreeClassifier(random_state=5),
        cv=5, param_grid={'max_depth': [5, 2]},
        scoring=memory_score
        )
    model.fit(X, y)
    assert round(model.best_score_,2) == 0.76

def test_combined_scorer():
    # load sample data
    data = load_iris()
    X = data.data
    y = data.target

    # initialize model
    acc_scorer = make_scorer(accuracy_score)
    latency_score = TimeScorer(
        None, 1, {"unit": False, "n_iter": 25, "scoring": acc_scorer, "tradeoff": 0.01})
    memory_score = MemoryScorer(
        None, 1, {"scoring": acc_scorer, "tradeoff": 100})
    combined_score = CombinedScorer(
        None, 1, {"scoring": [latency_score, memory_score]})

    # fit models
    model = GridSearchCV(
        DecisionTreeClassifier(random_state=5),
        cv=5, param_grid={'max_depth': [5, 2]},
        scoring=combined_score
        )
    model.fit(X, y)
    assert round(model.best_score_,2) == 0.85

