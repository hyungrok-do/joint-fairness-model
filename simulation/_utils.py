# -*- coding: utf-8 -*-
"""
Created on Sun May  3 10:00:00 2021

@author: Hyungrok Do
         hyungrok.do11@gmail.com
         https://github.com/hyungrok-do/

Implementation of the paper Joint Fairness Model with Applications to Risk Predictions for Under-represented Populations

"""

import os
import argparse
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score

rmse = lambda y, y_: mean_squared_error(y, y_) ** .5

def Evaluate(true_y1, true_y2, pred_p1, pred_p2, true_b1, true_b2, pred_b1, pred_b2, threshold=0.5):
    pred_y1 = (pred_p1 > threshold).astype(true_y1.dtype)
    pred_y2 = (pred_p2 > threshold).astype(true_y2.dtype)

    true_y = np.concatenate([true_y1, true_y2])
    pred_p = np.concatenate([pred_p1, pred_p2])
    pred_y = np.concatenate([pred_y1, pred_y2])

    auc_all = roc_auc_score(true_y, pred_p)
    auc_1 = roc_auc_score(true_y1, pred_p1)
    auc_2 = roc_auc_score(true_y2, pred_p2)

    acc_all = accuracy_score(true_y, pred_y)
    tpr_all = recall_score(true_y, pred_y, pos_label=1)
    tnr_all = recall_score(true_y, pred_y, pos_label=0)

    acc_1 = accuracy_score(true_y1, pred_y1)
    acc_2 = accuracy_score(true_y2, pred_y2)
    tpr_1 = recall_score(true_y1, pred_y1, pos_label=1)
    tpr_2 = recall_score(true_y2, pred_y2, pos_label=1)
    tnr_1 = recall_score(true_y1, pred_y1, pos_label=0)
    tnr_2 = recall_score(true_y2, pred_y2, pos_label=0)

    tpr_diff = np.abs(tpr_1 - tpr_2)
    tnr_diff = np.abs(tnr_1 - tnr_2)
    auc_diff = np.abs(auc_1 - auc_2)

    true_b = np.concatenate([true_b1, true_b2])
    pred_b = np.concatenate([pred_b1, pred_b2])

    bias_all = mean_squared_error(true_b, pred_b)
    bias_1 = mean_squared_error(true_b1, pred_b1)
    bias_2 = mean_squared_error(true_b2, pred_b2)

    true_supp = np.where(true_b != 0)[0]
    true_supp_1 = np.where(true_b1 != 0)[0]
    true_supp_2 = np.where(true_b2 != 0)[0]
    pred_supp = np.where(pred_b != 0)[0]
    pred_supp_1 = np.where(pred_b1 != 0)[0]
    pred_supp_2 = np.where(pred_b2 != 0)[0]

    true_zero = np.where(true_b == 0)[0]
    true_zero_1 = np.where(true_b1 == 0)[0]
    true_zero_2 = np.where(true_b2 == 0)[0]
    pred_zero = np.where(pred_b == 0)[0]
    pred_zero_1 = np.where(pred_b1 == 0)[0]
    pred_zero_2 = np.where(pred_b2 == 0)[0]

    coef_tp_all = np.intersect1d(true_supp, pred_supp).__len__()
    coef_tp_1 = np.intersect1d(true_supp_1, pred_supp_1).__len__()
    coef_tp_2 = np.intersect1d(true_supp_2, pred_supp_2).__len__()

    sen_all = coef_tp_all / len(true_supp)
    sen_1 = coef_tp_1 / len(true_supp_1)
    sen_2 = coef_tp_2 / len(true_supp_2)

    coef_tn_all = np.intersect1d(true_zero, pred_zero).__len__()
    coef_tn_1 = np.intersect1d(true_zero_1, pred_zero_1).__len__()
    coef_tn_2 = np.intersect1d(true_zero_2, pred_zero_2).__len__()

    spe_all = coef_tn_all / len(true_zero)
    spe_1 = coef_tn_1 / len(true_zero_1)
    spe_2 = coef_tn_2 / len(true_zero_2)

    nzbias_all = mean_squared_error(true_b[true_b != 0], pred_b[true_b != 0])
    nzbias_1 = mean_squared_error(true_b1[true_b1 != 0], pred_b1[true_b1 != 0])
    nzbias_2 = mean_squared_error(true_b2[true_b2 != 0], pred_b2[true_b2 != 0])

    res = {'All-AUC': auc_all,
           'Group1-AUC': auc_1,
           'Group2-AUC': auc_2,
           'AUC-DIFF': auc_diff,
           'All-ACC': acc_all,
           'All-TPR-TNR': np.mean([tpr_all, tnr_all]),
           'All-TPR': tpr_all,
           'All-TNR': tnr_all,
           'Group1-ACC': acc_1,
           'Group1-TPR-TNR': np.mean([tpr_1, tnr_1]),
           'Group1-TPR': tpr_1,
           'Group1-TNR': tnr_1,
           'Group2-ACC': acc_2,
           'Group2-TPR-TNR': np.mean([tpr_2, tnr_2]),
           'Group2-TPR': tpr_2,
           'Group2-TNR': tnr_2,
           'TPR-DIFF': tpr_diff,
           'TNR-DIFF': tnr_diff,
           'TPR-TNR-DIFF': np.mean([tpr_diff, tnr_diff]),
           'All-BIAS': bias_all,
           'Group1-BIAS': bias_1,
           'Group2-BIAS': bias_2,
           'All-NZBIAS': nzbias_all,
           'Group1-NZBIAS': nzbias_1,
           'Group2-NZBIAS': nzbias_2,
           'All-COEFSEN': sen_all,
           'Group1-COEFSEN': sen_1,
           'Group2-COEFSEN': sen_2,
           'All-COEFSPE': spe_all,
           'Group1-COEFSPE': spe_1,
           'Group2-COEFSPE': spe_2}

    return res


def Evaluate_Prediction_Only(true_y1, true_y2, pred_p1, pred_p2, threshold=0.5):
    pred_y1 = (pred_p1 > threshold).astype(true_y1.dtype)
    pred_y2 = (pred_p2 > threshold).astype(true_y2.dtype)

    true_y = np.concatenate([true_y1, true_y2])
    pred_p = np.concatenate([pred_p1, pred_p2])
    pred_y = np.concatenate([pred_y1, pred_y2])

    auc_all = roc_auc_score(true_y, pred_p)
    auc_1 = roc_auc_score(true_y1, pred_p1)
    auc_2 = roc_auc_score(true_y2, pred_p2)

    acc_all = accuracy_score(true_y, pred_y)
    tpr_all = recall_score(true_y, pred_y, pos_label=1)
    tnr_all = recall_score(true_y, pred_y, pos_label=0)

    acc_1 = accuracy_score(true_y1, pred_y1)
    acc_2 = accuracy_score(true_y2, pred_y2)
    tpr_1 = recall_score(true_y1, pred_y1, pos_label=1)
    tpr_2 = recall_score(true_y2, pred_y2, pos_label=1)
    tnr_1 = recall_score(true_y1, pred_y1, pos_label=0)
    tnr_2 = recall_score(true_y2, pred_y2, pos_label=0)

    tpr_diff = np.abs(tpr_1 - tpr_2)
    tnr_diff = np.abs(tnr_1 - tnr_2)
    auc_diff = np.abs(auc_1 - auc_2)

    res = {'All-AUC': auc_all,
           'Group1-AUC': auc_1,
           'Group2-AUC': auc_2,
           'AUC-DIFF': auc_diff,
           'All-ACC': acc_all,
           'All-TPR-TNR': np.mean([tpr_all, tnr_all]),
           'All-TPR': tpr_all,
           'All-TNR': tnr_all,
           'Group1-ACC': acc_1,
           'Group1-TPR-TNR': np.mean([tpr_1, tnr_1]),
           'Group1-TPR': tpr_1,
           'Group1-TNR': tnr_1,
           'Group2-ACC': acc_2,
           'Group2-TPR-TNR': np.mean([tpr_2, tnr_2]),
           'Group2-TPR': tpr_2,
           'Group2-TNR': tnr_2,
           'TPR-DIFF': tpr_diff,
           'TNR-DIFF': tnr_diff,
           'TPR-TNR-DIFF': np.mean([tpr_diff, tnr_diff])}

    return res

def SimulationArgumentParser():
    parser = argparse.ArgumentParser(description='Run an experiment for simulation scenario')

    parser.add_argument('--save_path', default=os.getcwd(), type=str, help='Directory to save the experimental results')
    parser.add_argument('--name', default='untitled', type=str, help='Name of the experiment')
    parser.add_argument('--seed', default=55, type=int, help='Random seed for the experiment')
    parser.add_argument('--p', default=100, type=int, help='The number of covariates')
    parser.add_argument('--q', default=40, type=int, help='The number of non-zero covariates')
    parser.add_argument('--r', default=20, type=int, help='The position of the first non-zero coefficient for the second group')
    parser.add_argument('--n1', default=500, type=int, help='The sample size for the first group')
    parser.add_argument('--n2', default=200, type=int, help='The sample size for the second group')
    parser.add_argument('--b', default=-10, type=float, help='The intercept term for the true model of the second group')
    parser.add_argument('--t', default=0, type=int, help='The position of the last non-zero coefficient for the second group')
    
    return parser


