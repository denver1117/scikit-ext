"""
Test MultiGridSearchCV class 
"""

from scikit_ext.estimators import MultiGridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

def main():

    # load sample data
    data = load_iris()
    X = data.data
    y = data.target

    # initialize model
    estimators = [DecisionTreeClassifier(), LogisticRegression()]
    param_grids = [{"max_depth": [2,3]}, {"C": [0.1, 1.0, 10.0]}]
    model = MultiGridSearchCV(
        estimators, param_grids, 
        cv=3)
    # fit models
    model.fit(X, y)

    # print results
    print(model.cv_results_)
    print(model.best_score_)
    print(model.best_estimator_)
    print(model.best_grid_search_cv_)

    # test predict
    print(model.predict(X[:10]))
    print(model.predict_proba(X[:10]))
    print(model.score(X, y))

if __name__ == "__main__":
    main()
