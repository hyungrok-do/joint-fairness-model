
import numpy as np
import pandas as pd

from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score

from scipy.stats import hmean
from scipy.stats import gmean
from itertools import combinations


def mean_group_tpr_tnr(y_true, y_pred):
    A = y_true.index.values
    unique_A = np.unique(A)

    if type(y_true) == pd.DataFrame:
        _y_true = y_true.values.flatten()

    y_true_, y_pred_ = [], []
    for a in unique_A:
        y_true_.append(_y_true[A == a])
        y_pred_.append(y_pred[A == a])
        
    tprs, tnrs = [], []
    for yt, yp in zip(y_true_, y_pred_):
        tprs.append(recall_score(yt, yp, pos_label=0))
        tnrs.append(recall_score(yt, yp, pos_label=1))
        
    try:
        val = np.mean(tprs+tnrs)
    except:
        val = -1
    return val


def mean_group_tpr_tnr_minus_absolute_difference(y_true, y_pred):
    A = y_true.index.values
    unique_A = np.unique(A)

    if type(y_true) == pd.DataFrame:
        _y_true = y_true.values.flatten()

    y_true_, y_pred_ = [], []
    for a in unique_A:
        y_true_.append(_y_true[A == a])
        y_pred_.append(y_pred[A == a])
        
    tprs, tnrs = [], []
    for yt, yp in zip(y_true_, y_pred_):
        tprs.append(recall_score(yt, yp, pos_label=0))
        tnrs.append(recall_score(yt, yp, pos_label=1))
        
    try:
        val = np.mean([
            np.mean(tprs) - np.mean([np.abs(a-b) for a, b in combinations(tprs, 2)]),
            np.mean(tnrs) - np.mean([np.abs(a-b) for a, b in combinations(tnrs, 2)])])
    except:
        val = -1
    return val


def mean_group_tpr_tnr_minus_squared_difference(y_true, y_pred):
    A = y_true.index.values
    unique_A = np.unique(A)

    if type(y_true) == pd.DataFrame:
        _y_true = y_true.values.flatten()

    y_true_, y_pred_ = [], []
    for a in unique_A:
        y_true_.append(_y_true[A == a])
        y_pred_.append(y_pred[A == a])
        
    tprs, tnrs = [], []
    for yt, yp in zip(y_true_, y_pred_):
        tprs.append(recall_score(yt, yp, pos_label=0))
        tnrs.append(recall_score(yt, yp, pos_label=1))
        
    try:
        val = np.mean([
            np.mean(tprs) - np.mean([np.square(a-b) for a, b in combinations(tprs, 2)]),
            np.mean(tnrs) - np.mean([np.square(a-b) for a, b in combinations(tnrs, 2)])])
    except:
        val = -1
    return val     


def geometric_mean_group_tpr_tnr(y_true, y_pred):
    A = y_true.index.values
    unique_A = np.unique(A)

    if type(y_true) == pd.DataFrame:
        _y_true = y_true.values.flatten()

    y_true_, y_pred_ = [], []
    for a in unique_A:
        y_true_.append(_y_true[A == a])
        y_pred_.append(y_pred[A == a])
        
    tprs, tnrs = [], []
    for yt, yp in zip(y_true_, y_pred_):
        tprs.append(recall_score(yt, yp, pos_label=0))
        tnrs.append(recall_score(yt, yp, pos_label=1))
        
    try:
        val = gmean(tprs+tnrs)
    except:
        val = -1
    return val


def harmonic_mean_group_tpr_tnr(y_true, y_pred):
    A = y_true.index.values
    unique_A = np.unique(A)

    if type(y_true) == pd.DataFrame:
        _y_true = y_true.values.flatten()

    y_true_, y_pred_ = [], []
    for a in unique_A:
        y_true_.append(_y_true[A == a])
        y_pred_.append(y_pred[A == a])
        
    tprs, tnrs = [], []
    for yt, yp in zip(y_true_, y_pred_):
        tprs.append(recall_score(yt, yp, pos_label=0))
        tnrs.append(recall_score(yt, yp, pos_label=1))
        
    try:
        val = hmean(tprs+tnrs)
    except:
        val = -1
    return val


MeanGroupTPRTNR = make_scorer(mean_group_tpr_tnr, needs_proba=False)
MeanGroupTPRTNRMinusAbsDiff = make_scorer(mean_group_tpr_tnr_minus_absolute_difference, needs_proba=False)
MeanGroupTPRTNRMinusSqDiff = make_scorer(mean_group_tpr_tnr_minus_squared_difference, needs_proba=False)
GeometricMeanGroupTPRTNR = make_scorer(geometric_mean_group_tpr_tnr, needs_proba=False)
HarmonicMeanGroupTPRTNR = make_scorer(harmonic_mean_group_tpr_tnr, needs_proba=False)