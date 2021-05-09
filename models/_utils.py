
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import PredefinedSplit


def get_max_lambda(x, y):
    pp = y.mean() - y
    return np.abs((x * pp[:,np.newaxis]).sum(0)).max() / len(y)


def GroupStratifiedKFold(y, A, n_folds=5, shuffle=True, random_state=None):
    te_idx = []
    for a in set(A):
        sub_y = y[A == a]
        sub_i = np.where(A == a)[0]
        cv = StratifiedKFold(n_splits=n_folds,
                             shuffle=shuffle,
                             random_state=random_state)
        te_idx.append([sub_i[te] for tr, te in cv.split(sub_i, sub_y)])

    test_fold = np.zeros(len(A), dtype=int)
    for split in range(n_folds):
        test_fold[np.concatenate([grp[split] for grp in te_idx])] = split

    return PredefinedSplit(test_fold)