
import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from time import time
from models import LogisticJointFair
from simulation import DataGenerator

save_dir = os.path.join(os.getcwd(), 'results-computation-time')
os.makedirs(save_dir, exist_ok=True)
print("save results at ", save_dir)

q = 40
b = -10
r = 20
n1 = 500
n2 = 200

lam = 1e-2

ps = np.arange(100, 5001, 100)

stack = []
for p in ps:
    computation_time = []
    for seed in np.arange(10):
        np.random.seed(seed)

        Simulator = DataGenerator(p, q, r, b)
        train_x, train_y = Simulator.generate(n1, n2, seed)

        model = LogisticJointFair(lam, lam, lam, lam, 0)

        start = time()
        model.fit(train_x, train_y)
        end = time()

        computation_time.append(end - start)
        print(p, seed, np.round(computation_time[-1], 3))

    summary = [
        np.mean(computation_time),
        np.median(computation_time),
        np.quantile(computation_time, 0.25),
        np.quantile(computation_time, 0.75),
        np.min(computation_time),
        np.max(computation_time),
        np.std(computation_time)
    ]
    stack.append(summary)
    print(p)

res = pd.DataFrame(stack, index=ps, columns=['mean', 'median', '1Q', '3Q', 'min', 'max', 'std'])

# print the results table
print(res)

# save the results table
res.to_csv(os.path.join(save_dir, 'computation-time-p.csv'))

# save the results figure
matplotlib.use('pdf')
fig = plt.figure(figsize=(12, 6))
ax = plt.subplot(111)
ax.set_xlabel('Number of Covariates (p)', fontsize=14)
ax.set_ylabel('Computation Time (Seconds)', fontsize=14)
ax.set_xticks(res.index)
ax.plot(res.index, res['mean'], label='Average', marker='s', c='black', lw=3)
ax.fill_between(res.index, res['mean']-res['std'], res['mean']+res['std'], color='gray', alpha=0.5)
ax.plot(res.index, res['mean']-res['std'], label='Std dev', c='gray', lw=1, ls='--')
ax.plot(res.index, res['mean']+res['std'], c='gray', lw=1, ls='--')
ax.legend()
plt.xticks(rotation=45)
fig.tight_layout()

fig.savefig(os.path.join(save_dir, 'computation-time-p.pdf'))