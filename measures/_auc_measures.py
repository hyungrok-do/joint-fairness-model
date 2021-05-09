
import numpy as np
import pandas as pd

from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score

from scipy.stats import hmean
from scipy.stats import gmean
from itertools import combinations

def harmonic_mean_group_auc(y_true, p_pred):
    A = y_true.index.values
    unique_A = np.unique(A)

    if type(y_true) == pd.DataFrame:
        _y_true = y_true.values.flatten()

    y_true_, y_pred_ = [], []
    for a in unique_A:
        y_true_.append(_y_true[A == a])
        y_pred_.append(p_pred[A == a])

    aucs =[]
    for yt, yp in zip(y_true_, y_pred_):
        aucs.append(roc_auc_score(yt, yp))
    try:
        val = hmean(np.array(aucs))
    except:
        val = -1
    return val


def geometric_mean_group_auc(y_true, p_pred):
    A = y_true.index.values
    unique_A = np.unique(A)

    if type(y_true) == pd.DataFrame:
        _y_true = y_true.values.flatten()

    y_true_, y_pred_ = [], []
    for a in unique_A:
        y_true_.append(_y_true[A == a])
        y_pred_.append(p_pred[A == a])

    aucs =[]
    for yt, yp in zip(y_true_, y_pred_):
        aucs.append(roc_auc_score(yt, yp))
    try:
        val = gmean(np.array(aucs))
    except:
        val = -1
    return val


def overall_auc(y_true, p_pred):
    y_true_ = y_true.values.flatten()
    return roc_auc_score(y_true_, p_pred)


def mean_group_auc(y_true, p_pred):
    A = y_true.index.values
    unique_A = np.unique(A)

    if type(y_true) == pd.DataFrame:
        _y_true = y_true.values.flatten()

    y_true_, y_pred_ = [], []
    for a in unique_A:
        y_true_.append(_y_true[A == a])
        y_pred_.append(p_pred[A == a])

    aucs =[]
    for yt, yp in zip(y_true_, y_pred_):
        aucs.append(roc_auc_score(yt, yp))
    try:
        val = np.mean(np.array(aucs))
    except:
        val = -1
    return val


def mean_group_auc_minus_absolute_diff(y_true, p_pred):
    A = y_true.index.values
    unique_A = np.unique(A)

    if type(y_true) == pd.DataFrame:
        _y_true = y_true.values.flatten()

    y_true_, y_pred_ = [], []
    for a in unique_A:
        y_true_.append(_y_true[A == a])
        y_pred_.append(p_pred[A == a])

    aucs =[]
    for yt, yp in zip(y_true_, y_pred_):
        aucs.append(roc_auc_score(yt, yp))
    try:
        val = np.mean(np.array(aucs))
        val -= np.mean([np.abs(a-b) for a, b in combinations(aucs, 2)])
    except:
        val = -1
    return val


def mean_group_auc_minus_squared_diff(y_true, p_pred):
    A = y_true.index.values
    unique_A = np.unique(A)

    if type(y_true) == pd.DataFrame:
        _y_true = y_true.values.flatten()

    y_true_, y_pred_ = [], []
    for a in unique_A:
        y_true_.append(_y_true[A == a])
        y_pred_.append(p_pred[A == a])

    aucs =[]
    for yt, yp in zip(y_true_, y_pred_):
        aucs.append(roc_auc_score(yt, yp))
    try:
        val = np.mean(np.array(aucs))
        val -= np.mean([np.square(a-b) for a, b in combinations(val, 2)])
    except:
        val = -1
    return val


OverallAUC = make_scorer(overall_auc, needs_proba=True)
MeanGroupAUC = make_scorer(mean_group_auc, needs_proba=True)
MeanGroupAUCMinusAbsDiff = make_scorer(mean_group_auc_minus_absolute_diff, needs_proba=True)
MeanGroupAUCMinusSqDiff = make_scorer(mean_group_auc_minus_squared_diff, needs_proba=True)
HarmonicMeanGroupAUC = make_scorer(harmonic_mean_group_auc, needs_proba=True)
GeometricMeanGroupAUC = make_scorer(geometric_mean_group_auc, needs_proba=True)