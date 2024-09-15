from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

import pandas as pd


class OHEncoding(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list) -> None:
        super(OHEncoding, self).__init__()
        self.columns = columns

    def fit(self, X, y=None):
        self.encoder = OneHotEncoder().fit(X[self.columns])
        return self

    def transform(self, X):
        encode_data = self.encoder.transform(X[self.columns]).toarray()
        encode_columns = self.encoder.get_feature_names_out()

        X[encode_columns] = encode_data
        X = X.drop(self.columns, axis=1)

        return X


class FeatureScaler(BaseEstimator, TransformerMixin):
    def __init__(self, scale_columns: list) -> None:
        super(FeatureScaler, self).__init__()
        self.scale_columns = scale_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        scaler = StandardScaler()
        X[self.scale_columns] = scaler.fit_transform(X[self.scale_columns])

        return X
