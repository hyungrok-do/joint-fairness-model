
import os
import numpy as np
import pandas as pd

from models import LogisticLasso
from models import LogisticSingleFair
from models import LogisticJointFair
from models import get_max_lambda
from models import GroupStratifiedKFold
from measures import HarmonicMeanGroupBrierScore, OverallBrierScore
from simulation import Evaluate
from simulation import DataGenerator
from simulation import SimulationArgumentParser

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

parser = SimulationArgumentParser()
args = parser.parse_args()

for arg, val in args._get_kwargs():
    print(arg, val)

save_dir = os.path.join(args.save_path, 'results-simulation')
os.makedirs(save_dir, exist_ok=True)

''' Generate Simulation Data '''

np.random.seed(args.seed)
Generator = DataGenerator(args.p, args.q, args.r, args.b, args.t)
train_x, train_y = Generator.generate(args.n1, args.n2, args.seed)
test_x, test_y = Generator.generate(1000, 1000, args.seed*10+5)
train_A = train_x[:,0]
test_A = test_x[:,0]

train_x1 = train_x[train_x[:,0] == 0]
train_x2 = train_x[train_x[:,0] == 1]
train_y1 = train_y[train_x[:,0] == 0]
train_y2 = train_y[train_x[:,0] == 1]

test_x1 = test_x[test_x[:,0] == 0]
test_x2 = test_x[test_x[:,0] == 1]
test_y1 = test_y[test_x[:,0] == 0]
test_y2 = test_y[test_x[:,0] == 1]


''' Fit Models '''

model_names = ['Group-separate',
               'Group-ignorant',
               'SFM', 'JFM-F', 'JFM-G']

max_lam = get_max_lambda(train_x, train_y)
max_lam_1 = get_max_lambda(train_x1, train_y1)
max_lam_2 = get_max_lambda(train_x2, train_y2)

models = [LogisticLasso(),
          LogisticLasso(),
          LogisticLasso(),
          LogisticSingleFair(),
          LogisticJointFair(similarity='fusion'),
          LogisticJointFair(similarity='group')]

param_grids = [
    {'lam': np.exp(np.linspace(np.log(max_lam_1), np.log(max_lam_1 * 1e-3), 50))},
    {'lam': np.exp(np.linspace(np.log(max_lam_2), np.log(max_lam_2 * 1e-3), 50))},

    {'lam': np.exp(np.linspace(np.log(max_lam), np.log(max_lam * 1e-3), 50))},

    {'lam1': np.exp(np.linspace(np.log(max_lam), np.log(max_lam * 1e-3), 50)),
     'lam2': 10. ** np.arange(-4, 2.1, 1)},

    {'lam1': np.exp(np.linspace(np.log(max_lam_1), np.log(max_lam_1 * 1e-3), 20)),
     'lam2': np.exp(np.linspace(np.log(max_lam_2), np.log(max_lam_2 * 1e-3), 20)),
     'lam3': 10. ** np.arange(-4, 2.1, 1),
     'lam4': 10. ** np.arange(-4, 2.1, 1)},

     {'lam1': np.exp(np.linspace(np.log(max_lam_1), np.log(max_lam_1 * 1e-3), 20)),
      'lam2': np.exp(np.linspace(np.log(max_lam_2), np.log(max_lam_2 * 1e-3), 20)),
      'lam3': 10. ** np.arange(-4, 2.1, 1),
      'lam4': 10. ** np.arange(-4, 2.1, 1)}
]

scorings = [
    OverallBrierScore,
    OverallBrierScore,
    OverallBrierScore,
    HarmonicMeanGroupBrierScore,
    HarmonicMeanGroupBrierScore,
    HarmonicMeanGroupBrierScore]

results_table = []

np.random.seed(args.seed)
kf = StratifiedKFold(n_splits=5, random_state=args.seed, shuffle=True)
cv = GridSearchCV(models[0], param_grids[0], scoring=scorings[0], verbose=0, n_jobs=-1, cv=kf)
cv.fit(train_x1[:,1:], train_y1)
best_model_group_1 = cv.best_estimator_

cv = GridSearchCV(models[1], param_grids[1], scoring=scorings[0], verbose=0, n_jobs=-1, cv=kf)
cv.fit(train_x2[:,1:], train_y2)
best_model_group_2 = cv.best_estimator_

results_table.append(Evaluate(test_y1, test_y2,
                              best_model_group_1.predict_proba(test_x1[:,1:])[:,1],
                              best_model_group_2.predict_proba(test_x2[:,1:])[:,1],
                              Generator.b1, Generator.b2,
                              best_model_group_1.coef_, best_model_group_2.coef_
                              ))

for param, model, scoring in zip(param_grids[2:], models[2:], scorings[2:]):
    np.random.seed(args.seed)
    kf = GroupStratifiedKFold(train_y, train_A, n_folds=5, shuffle=True, random_state=args.seed)

    cv = GridSearchCV(model, param, scoring=scoring, verbose=0, n_jobs=-1, cv=kf)
    cv.fit(train_x, pd.DataFrame(train_y, index=train_A))

    for key in cv.best_params_:
        print(key, cv.best_params_[key])

    try:
        pred_b1 = cv.best_estimator_.coef_0_
        pred_b2 = cv.best_estimator_.coef_1_
    except:
        pred_b1 = cv.best_estimator_.coef_[1:]
        pred_b2 = cv.best_estimator_.coef_[1:]
    results_table.append(Evaluate(test_y1, test_y2,
                         cv.best_estimator_.predict_proba(test_x1)[:,1],
                         cv.best_estimator_.predict_proba(test_x2)[:,1],
                         Generator.b1, Generator.b2,
                         pred_b1, pred_b2))

results_table = pd.DataFrame(results_table, index=model_names)
print(results_table)
results_table.to_csv(os.path.join(save_dir, f'results-{args.name}.csv'))
