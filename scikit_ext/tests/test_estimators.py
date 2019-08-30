"""
Test scorers module
"""

from sklearn import feature_selection
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction import text

try:
    from scikit_ext import estimators
    _top_import_error = None
except Exception as e:
    _top_import_error = e

try:
    from scikit_ext.estimators import (
        IterRandomEstimator,
        MultiGridSearchCV,
        OptimizedEnsemble,
        OneVsRestAdjClassifier,
        PrunedPipeline,
        ZoomGridSearchCV     
        )
    _module_import_error = None
except Exception as e:
    _module_import_error = e

def test_top_import():
    assert _top_import_error == None

def test_module_import():
    assert _module_import_error == None

def test_iter_random():
    data = load_iris()
    X = data.data
    model = IterRandomEstimator(
        KMeans(n_clusters=4), 
        max_iter=5, random_state=5
        )
    model.fit(X)
    assert round(model.best_score_) == 531

def test_multi_gs():
    data = load_iris()
    X = data.data
    y = data.target
    estimators = [
        DecisionTreeClassifier(), 
        LogisticRegression(solver="liblinear", multi_class="auto")
        ]
    param_grids = [{"max_depth": [2,3]}, {"C": [0.1, 1.0, 10.0]}]
    model = MultiGridSearchCV(
        estimators, param_grids, 
        cv=5)
    model.fit(X, y)
    assert round(model.best_score_,2) == 0.97

def test_oe():
    data = load_iris()
    X = data.data
    y = data.target
    model = OptimizedEnsemble(
        RandomForestClassifier(max_depth=1, random_state=5), 
        n_estimators_init=10,
        max_iter=2,
        threshold=0.01,
        cv=5,
        step_function=lambda x: x*2)
    model.fit(X, y)
    assert round(model.best_score_,2) == 0.85

def test_ovr_adj():
    data = load_iris()
    X = data.data
    y = data.target
    dt = DecisionTreeClassifier(random_state=5)
    ovr_adj = OneVsRestAdjClassifier(dt)
    ovr_adj.fit(X,y)
    assert ovr_adj.predict(X)[0] == 0

def test_pruned():
    data = load_iris()
    y = data.target
    X = ["taco words words more zebra elephant" for index in range(len(y))]
    cv = text.CountVectorizer()
    select_feat = feature_selection.SelectKBest(feature_selection.chi2, k=2)
    dt = DecisionTreeClassifier(max_depth=1, random_state=5)
    pipeline = PrunedPipeline([
        ("vec", cv), ("select", select_feat), 
        ("clf", dt)])
    pipeline.fit(X, y)
    assert round(pipeline.score(X,y),2) == 0.33

def test_zoom():
    data = load_iris()
    X = data.data
    y = data.target
    estimator = DecisionTreeClassifier(random_state=5)
    param_grid = {
        "min_samples_split": [0.01, 0.05, 0.75], 
        "max_depth": [4,8,15,20,30], 
        "criterion": ["gini", "entropy"]}
    model = ZoomGridSearchCV(
        estimator, param_grid, 
        cv=5, n_iter=4)
    model.fit(X, y)
    assert round(model.best_score_,2) == 0.97
