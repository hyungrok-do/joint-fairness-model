
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('pdf')
path = './results-validation-measures'
savepath = os.path.join(path, 'plots')
os.makedirs(savepath, exist_ok=True)
files = sorted(os.listdir(path))

colors=['black', 'darkgreen', 'orange', 'blue']
measure_names = ['Overall Accuracy',
                 'Avg of TPR/TNRs',
                 'Avg of TPR/TNRs-Abs Diff',
                 'Avg of TPR/TNRs-Sq Diff',
                 'GMean of TPR/TNRs',
                 'HMean of TPR/TNRs',
                 'Overall AUC',
                 'Avg of AUCs',
                 'Avg of AUCs-Abs Diff',
                 'Avg of AUCs-Sq Diff',
                 'GMean of AUCs',
                 'HMean of AUCs']


def make_plot(s, measure, title, dpi=400):
    filtered = sorted([file for file in files if file.startswith(f'valid-sim{s}')])
    info = np.array([(float(file.split('_')[1]), int(file.split('_')[2].replace('.csv',''))) for file in filtered])
    unique_param = np.unique(info[:,0])
    
    avg_stack = pd.DataFrame(columns=unique_param, index=measure_names)
    upp_stack = pd.DataFrame(columns=unique_param, index=measure_names)
    low_stack = pd.DataFrame(columns=unique_param, index=measure_names)

    for param in unique_param:
        params = np.array([file.split('_')[1] for file in files if file.startswith(f'valid-sim{s}')])
        params = params.astype(unique_param.dtype)
        target_files = sorted([_file for _param, _file in zip(params, filtered) if _param == param])
        
        param_stack = []
        for file in target_files:
            param_stack.append(pd.read_csv(os.path.join(path, file), index_col=0)[measure].values)
        avg_stack[param] = np.median(param_stack, 0)
        upp_stack[param] = np.quantile(param_stack, q=.75, axis=0) - np.median(param_stack, 0)
        low_stack[param] = np.median(param_stack, 0) - np.quantile(param_stack, q=.25, axis=0)
        
    avg_stack = avg_stack.T
    upp_stack = upp_stack.T
    low_stack = low_stack.T
    
    avg_stack.index = avg_stack.index.astype(float)
    upp_stack.index = upp_stack.index.astype(float)
    low_stack.index = low_stack.index.astype(float)
    
    if s == 1:
        avg_stack.index = (40 - avg_stack.index)/40. * 100
        xticks = avg_stack.index 
        xlab = 'Percentage of Shared Important Covariates (%)'
    if s == 2:
        avg_stack.index = np.arange(len(avg_stack))
        xlab = 'Baseline Prevalance of the Under-represented Group (%)'
        xticks = [10, 12, 15, 18, 22, 26, 30, 35, 40, 45, 50]
    if s == 3:
        xlab = 'Sample Size of the Under-represented Group'
        avg_stack.index = avg_stack.index.astype(int)
        xticks = avg_stack.index
    
    xticks_labels = [str(s) for s in xticks]
    xticks = np.arange(len(xticks))
    
    avg_stack.index = xticks
    upp_stack.index = xticks
    low_stack.index = xticks
    
    colors=['gray', 'red', 'black', 'darkgreen', 'orange', 'blue',
            'slateblue', 'yellowgreen', 'orangered', 'darkcyan', 'royalblue', 'darkviolet']
    markers=['*', 'x', 'h','^','o','s',
             'D', 'p', 'v', 'X', '8', 'P']
    loc_cal = np.linspace(-0.12, 0.12, 12)
    
    plt.figure(figsize=(16, 5))
    for i, model in enumerate(avg_stack.columns):
        model_avg = avg_stack[model]
        model_upp = upp_stack[model]
        model_low = low_stack[model]
        
        model_avg.index = model_avg.index + loc_cal[i]
        model_std = pd.concat([model_low, model_upp], axis=1).T.values
        ax = model_avg.plot(marker=markers[i], xticks=model_avg.index, color=colors[i], yerr=model_std, linewidth=1.6,
                            capsize=5, capthick=1.2, linestyle='-.', fontsize=16, markersize=10)
        ax.set_xlim((-0.5, len(model_avg)-0.5))
        ax.set_xticklabels(xticks_labels)
        
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height])
    ax.set_xlabel(xlab, fontsize=16)
    ax.legend(loc='center left', fancybox=True, framealpha=0.3, fontsize=16, bbox_to_anchor=(1., 0.5))
    plt.tight_layout()

    plt.savefig(os.path.join(savepath, f'Eval-Measure-Sim-{s}-{title}.pdf'), dpi=dpi)
    plt.close()


measures = ['All-AUC', 'Group1-AUC', 'Group2-AUC', 'AUC-DIFF']
titles = ['Overall AUC', 'AUC of the Over-represented Group', 'AUC of the Under-represented Group', 'Disparity of AUC']

for s in [1, 2, 3]:
    for measure, title in zip(measures, titles):
        make_plot(s, measure, title)
