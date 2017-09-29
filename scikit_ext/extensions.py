"""
Various scikit-learn extensions
"""

import numpy as np
from sklearn.multiclass import OneVsRestClassifier

class OneVsRestAdjClassifier(OneVsRestClassifier):
    """
    One-vs-the-rest (OvR) multiclass strategy

    Also known as one-vs-all, this strategy consists in fitting one classifier per class. 
    For each classifier, the class is fitted against all the other classes. 
    In addition to its computational efficiency (only n_classes classifiers are needed), 
    one advantage of this approach is its interpretability. 
    Since each class is represented by one and one classifier only, it is possible to gain 
    knowledge about the class by inspecting its corresponding classifier. 
    This is the most commonly used strategy for multiclass classification and is a fair default choice.

    The adjusted version is a custom extension which overwrites the inherited predict_proba() method 
    and returns the pre-normalized per-class predicted probabilities, rather than forcing them to sum
    to 1.  All other methods are inherited from OneVsRestClassifier.
    """
 
    def predict_proba(self, X):
        """ 
        Probability estimates.

        The returned estimates for all classes are ordered by label of classes.

        Parameters	
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns  
        -------
        T : array-like, shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in the model, 
            where classes are ordered as they are in self.classes_.
        """

        probs = []
        for index in range(len(self.estimators_)):
            probs.append(self.estimators_[index].predict_proba(X)[:,1])            
        return np.array([
            [probs[y][index] for y in range(len(self.estimators_))] 
            for index in range(len(probs[0]))])
