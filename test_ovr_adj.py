"""
Test OneVsRestAdjClassifier 
"""

from scikit_ext.extensions import OneVsRestAdjClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.datasets import load_iris

def main():

    # load sample data
    data = load_iris()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1)

    # initialize models
    dt = DecisionTreeClassifier(max_depth=1)
    ovr_scikit = OneVsRestClassifier(dt)
    ovr_adj = OneVsRestAdjClassifier(dt)

    # fit models
    ovr_scikit.fit(X_train, y_train)
    ovr_adj.fit(X_train, y_train)

    # print scores (should be equal)
    print ovr_scikit.score(X_test, y_test)
    print ovr_adj.score(X_test, y_test)

    # get probs from predict_proba() methods
    probs_scikit = ovr_scikit.predict_proba(X_test)
    probs_adj = ovr_adj.predict_proba(X_test)

    # print types (should be equal)
    print type(probs_scikit)
    print type(probs_adj)

    # print probability matrices (should be slightly different)
    print probs_scikit
    print probs_adj

if __name__ == "__main__":
    main()
