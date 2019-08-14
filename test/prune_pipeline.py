"""
Test OneVsRestAdjClassifier 
"""

from scikit_ext.estimators import PrunedPipeline
from sklearn import feature_selection
from sklearn.feature_extraction import text
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

def main():

    # load sample data
    data = load_iris()
    y = data.target
    X = ["taco words words more zebra elephant" for index in range(len(y))]

    # initialize models
    cv = text.CountVectorizer()
    select_feat = feature_selection.SelectKBest(feature_selection.chi2, k=2)
    dt = DecisionTreeClassifier(max_depth=1)
    pipeline = PrunedPipeline([
        ("vec", cv), ("select", select_feat), 
        ("clf", dt)])

    pipeline.fit(X, y)
    print(pipeline.score(X,y))
    print(pipeline.predict(X)[:10])
    print(pipeline.predict_proba(X)[:10])
    print(pipeline.steps)
 
if __name__ == "__main__":
    main()
