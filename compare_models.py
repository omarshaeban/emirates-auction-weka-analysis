"""Compare three classifiers, with baseline, correlation filtering, and PCA.

Evaluation uses stratified cross-validation and each preprocessing step is fit
on training folds only. No metrics are asserted without real input data.
"""
import argparse
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


class CorrelationDropper(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.90):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        corr = np.corrcoef(X, rowvar=False)
        corr = np.nan_to_num(np.atleast_2d(corr))
        excluded = set()
        for i in range(corr.shape[1]):
            for j in range(i + 1,corr.shape[1]):
                if j not in excluded and abs(corr[i,j]) > self.threshold:
                    excluded.add(j)
        self.keep_indices_ = [i for i in range(X.shape[1]) if i not in excluded]
        return self

    def transform(self,X):
        return np.asarray(X)[:,self.keep_indices_]


def new_encoder():
    try: return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError: return OneHotEncoder(handle_unknown="ignore", sparse=False)


def pipeline(model, variant):
    pre = ColumnTransformer([
        ("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),["bids","remaining_minutes"]),
        ("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",new_encoder())]),["vehicle_brand"]),
    ],sparse_threshold=0)
    steps=[("preprocess",pre)]
    if variant=="correlation":steps.append(("filter",CorrelationDropper(threshold=.90)))
    elif variant=="pca":steps.append(("pca",PCA(n_components=.95,svd_solver="full")))
    steps.append(("model",model))
    return Pipeline(steps)


def compare(frame: pd.DataFrame, folds:int=5) -> pd.DataFrame:
    if not {"vehicle_brand","bids","remaining_minutes","high_price"}.issubset(frame.columns):
        raise ValueError("Input must contain vehicle_brand, bids, remaining_minutes and high_price")
    X=frame[["vehicle_brand","bids","remaining_minutes"]].copy()
    y=frame["high_price"].astype(str)
    counts=y.value_counts()
    if len(counts)!=2 or counts.min()<folds:
        raise ValueError(f"At least {folds} records are required for each of two classes; counts={counts.to_dict()}")
    cv=StratifiedKFold(n_splits=folds,shuffle=True,random_state=42)
    algos={"Logistic Regression":LogisticRegression(max_iter=1000,random_state=42),
           "Random Forest":RandomForestClassifier(n_estimators=100,random_state=42),
           "SVM":SVC(kernel="rbf")}
    results=[]
    for variant in ["baseline","correlation","pca"]:
        for label,model in algos.items():
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                scored=cross_validate(pipeline(model,variant),X,y,cv=cv,scoring={"accuracy":"accuracy","precision":"precision_macro","recall":"recall_macro","f1":"f1_macro"},error_score='raise')
            results.append({"variant":variant,"algorithm":label,**{metric:round(float(np.mean(scored['test_'+metric])),4) for metric in ("accuracy","precision","recall","f1")}})
    return pd.DataFrame(results)


def main():
    a=argparse.ArgumentParser();a.add_argument('input');a.add_argument('--folds',type=int,default=5);a.add_argument('--output',default='model_comparison.csv');args=a.parse_args()
    result=compare(pd.read_csv(args.input),folds=args.folds)
    result.to_csv(args.output,index=False)
    print(result.to_string(index=False));print('Metrics are computed on supplied data; interpret sample data separately from real listings.')


if __name__=='__main__':main()
