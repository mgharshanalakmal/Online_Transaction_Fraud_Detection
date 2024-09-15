from ml_model.preporocessing import OHEncoding
from ml_model.utils import execution_time

from sklearn.preprocessing import LabelEncoder

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.pipeline import _name_estimators

import numpy as np


class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, classifiers: list, vote: str = "classlable", weigths: list = None) -> None:
        """A majority vote ensemble classifier

        Args:
            classifiers (list):     Different classifiers for the ensemble.
            vote (str, optional):   If 'classlabel' the prediction is based on the argmax of class labels.
                                    Else if 'probability', the argmax of the sum of probabilities is used
                                    to predict the class label. Defaults to "classlable".
            weigths (list, optional): If a list of 'int' or 'float' values are provided, the classifiers are
                                    weighted by importance; Uses uniform weights if 'weights=None'. Defaults
                                    to None.
        """
        super(MajorityVoteClassifier, self).__init__()
        self.classifiers = classifiers
        self.named_classifiers = {key: value for key, value in _name_estimators(classifiers)}
        self.vote = vote
        self.weigths = weigths
        self.predictions = None

    @execution_time
    def fit(self, X, y):
        """_summary_

        Args:
            X (_type_): _description_
            y (_type_): _description_

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        if self.vote not in ("probability", "classlable"):
            raise ValueError(f"vote must be 'probability' or 'classlable'; got (vote={self.vote})")

        if self.weigths and len(self.weigths) != len(self.classifiers):
            raise ValueError(
                f"Number of classifiers and weights must be equal; got {len(self.weigths)} weights and {len(self.classifiers)} classifiers."
            )

        self.lableEnc_ = LabelEncoder()
        self.lableEnc_.fit(y)
        self.classes_ = self.lableEnc_.classes_

        self.oheEnc_ = OHEncoding(columns=["type"])
        self.oheEnc_.fit(X)
        self.oheEnc_.transform(X)

        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, self.lableEnc_.transform(y))
            self.classifiers_.append(fitted_clf)

        return self

    def predict(self, X):
        """_summary_

        Args:
            X (_type_): _description_

        Returns:
            _type_: _description_
        """

        if self.vote == "probability":
            maj_vote = np.argmax(self.predict_probabilities(X), axis=1)
            return maj_vote
        else:
            self.predictions = np.asarray([clf.predict(X) for clf in self.classifiers_]).T
            maj_vote = np.apply_along_axis(
                lambda x: np.argmax(np.bincount(x, weights=self.weigths)), axis=1, arr=self.predictions
            )

            maj_vote = self.lableEnc_.inverse_transform(maj_vote)

            return maj_vote

    def predict_probabilities(self, X):
        """Predict class probabilities for X.

        Args:
            X (array-like, sparse matrix):  Training vectors, where n_examples is the number of
                                            examples and n_features is the number of features.

        Returns:
            array-like: Weighted average probability for each class per example.
        """
        probas = np.asarray([clf.predict_proba(X) for clf in self.classifiers_])
        avg_proba = np.average(probas, axis=0, weights=self.weigths)

        return avg_proba

    def get_params(self, deep: bool = True) -> dict:
        if not deep:
            return super(MajorityVoteClassifier, self).get_params(deep=False)
        else:
            out = self.named_classifiers.copy()
            for name, step in self.named_classifiers.items():
                for key, value in step.get_params(deep=True).items():
                    out[f"{name}__{key}"] = value

            return out
