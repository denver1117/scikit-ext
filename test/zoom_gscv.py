"""
Test MultiGridSearchCV class 
"""

from scikit_ext.estimators import ZoomGridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

def main():

    # load sample data
    data = load_iris()
    X = data.data
    y = data.target

    # initialize model
    estimator = DecisionTreeClassifier()
    param_grid = {
        "min_samples_split": [0.01, 0.05, 0.75], 
        "max_depth": [4,8,15,20,30], 
        "criterion": ["gini", "entropy"]}
    model = ZoomGridSearchCV(
        estimator, param_grid, 
        cv=3, n_iter=4, verbose=1)
    # fit models
    model.fit(X, y)

    # print results
    print(model.cv_results_)
    print(model.best_estimator_)

    # test predict
    print(model.predict(X[:10]))
    print(model.predict_proba(X[:10]))
    print(model.score(X, y))

if __name__ == "__main__":
    main()
