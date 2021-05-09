
import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd

from models import LogisticJointFair
from models import get_max_lambda
from models import GroupStratifiedKFold
from simulation import Evaluate
from simulation import DataGenerator
from simulation import SimulationArgumentParser

from sklearn.model_selection import GridSearchCV

from measures import MeanGroupTPRTNR
from measures import MeanGroupTPRTNRMinusAbsDiff
from measures import MeanGroupTPRTNRMinusSqDiff
from measures import GeometricMeanGroupTPRTNR
from measures import HarmonicMeanGroupTPRTNR
from measures import OverallAUC
from measures import MeanGroupAUC
from measures import MeanGroupAUCMinusAbsDiff
from measures import MeanGroupAUCMinusSqDiff
from measures import GeometricMeanGroupAUC
from measures import HarmonicMeanGroupAUC

parser = SimulationArgumentParser()
args = parser.parse_args()

for arg, val in args._get_kwargs():
    print(arg, val)

save_dir = os.path.join(args.save_path, 'results-validation-measures')
os.makedirs(save_dir, exist_ok=True)

np.random.seed(args.seed)
generator = DataGenerator(args.p, args.q, args.r, args.b, args.t)
train_x, train_y = generator.generate(args.n1, args.n2, args.seed)
test_x, test_y = generator.generate(1000, 1000, args.seed*10+5)
train_A = train_x[:,0]

train_x1 = train_x[train_x[:,0] == 0]
train_x2 = train_x[train_x[:,0] == 1]
train_y1 = train_y[train_x[:,0] == 0]
train_y2 = train_y[train_x[:,0] == 1]

test_x1 = test_x[test_x[:,0] == 0]
test_x2 = test_x[test_x[:,0] == 1]
test_y1 = test_y[test_x[:,0] == 0]
test_y2 = test_y[test_x[:,0] == 1]

''' Modeling Part '''

max_lam_1 = get_max_lambda(train_x1[:,1:], train_y1)
max_lam_2 = get_max_lambda(train_x2[:,1:], train_y2)

model = LogisticJointFair()

param = {'lam1': np.exp(np.linspace(np.log(max_lam_1), np.log(max_lam_1 * 1e-6), 20)),
         'lam2': np.exp(np.linspace(np.log(max_lam_2), np.log(max_lam_2 * 1e-6), 20)),
         'lam3': 10. ** np.arange(-4, 2.1, 1),
         'lam4': 10. ** np.arange(-4, 2.1, 1)}

names = ['Overall Accuracy', 'Mean TPR/TNRs', 'Mean TPR/TNRs - Abs Diff', 'Mean TPR/TNRs - Sq Diff',
         'GMean TPR/TNRs', 'HMean TPR/TNRs',
         'Overall AUC', 'Mean AUCs', 'Mean AUCs - Abs Diff', 'Mean AUCs - Sq Diff',
         'GMean AUCs', 'HMean AUCs']

scorers = ['accuracy', MeanGroupTPRTNR, MeanGroupTPRTNRMinusAbsDiff, MeanGroupTPRTNRMinusSqDiff,
           GeometricMeanGroupTPRTNR, HarmonicMeanGroupTPRTNR,
           OverallAUC, MeanGroupAUC, MeanGroupAUCMinusAbsDiff, MeanGroupAUCMinusSqDiff,
           GeometricMeanGroupAUC, HarmonicMeanGroupAUC]

scorings = {}
for name, scorer in zip(names, scorers):
    scorings[name] = scorer

results_table = []
kf = GroupStratifiedKFold(train_y, train_A, n_folds=5, shuffle=True, random_state=args.seed)
cv = GridSearchCV(model, param, scoring=scorings, n_jobs=-1, cv=kf, refit=False)
cv.fit(train_x, pd.DataFrame(train_y, index=train_A))

for scorer in scorings:
    idx = np.where(cv.cv_results_['rank_test_'+scorer] == 1)[0][0]
    opt_param = cv.cv_results_['params'][idx]

    for key in opt_param:
        model.__setattr__(key, opt_param[key])

    model.fit(train_x, train_y)

    results_table.append(Evaluate(test_y1, test_y2,
                                  model.predict_proba(test_x1)[:, 1],
                                  model.predict_proba(test_x2)[:, 1]))

pd.DataFrame(results_table, index=names).to_csv(os.path.join(save_dir, f'valid-{args.name}.csv'))
