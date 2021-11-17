
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('pdf')
path = './results-simulation'
savepath = os.path.join(path, 'plots')
os.makedirs(savepath, exist_ok=True)
files = sorted(os.listdir(path))

model_names = ['Group-separate', 'Group-ignorant', 'SFM', 'JFM']

def make_plot(scenario, measure, title, dpi=400):
    filtered = [file for file in files if file.startswith(f'results-sim{scenario}_')]
    info = np.array([(float(file.split('_')[1]), int(file.split('_')[2].replace('.csv',''))) for file in filtered])
    unique_param = np.unique(info[:,0])
    
    avg_stack = pd.DataFrame(columns=unique_param, index=model_names)
    upp_stack = pd.DataFrame(columns=unique_param, index=model_names)
    low_stack = pd.DataFrame(columns=unique_param, index=model_names)

    for param in unique_param:
        params = np.array([file.split('_')[1] for file in files if file.startswith(f'results-sim{scenario}_')])
        params = params.astype(unique_param.dtype)
        target_files = sorted([_file for _param, _file in zip(params, filtered) if _param == param])
        
        param_stack = []
        for file in target_files:
            param_stack.append(pd.read_csv(os.path.join(path, file), index_col=0)[measure].values)
            
        avg_stack[param] = np.median(param_stack, 0)
        upp_stack[param] = np.quantile(param_stack, q=.75, axis=0) - np.median(param_stack, 0)
        low_stack[param] = np.median(param_stack, axis=0) - np.quantile(param_stack, q=.25, axis=0)
        
    avg_stack = avg_stack.T
    upp_stack = upp_stack.T
    low_stack = low_stack.T
    
    avg_stack.index = avg_stack.index.astype(float)
    upp_stack.index = upp_stack.index.astype(float)
    low_stack.index = low_stack.index.astype(float)
    
    if scenario == 1:
        avg_stack.index = (40 - avg_stack.index)/40. * 100
        xticks = avg_stack.index 
        xlab = 'Percentage of Shared Important Covariates (%)'
        if 'DIFF' not in measure:
            if 'AUC' in measure:
                ylim_low = 0.6
                ylim_high = 1.0
            else:
                ylim_low = 0.5
                ylim_high = 1.0
    if scenario == 2:
        avg_stack.index = np.arange(len(avg_stack))
        xlab = 'Baseline Prevalance of the Under-represented Group (%)'
        xticks = [10, 12, 15, 18, 22, 26, 30, 35, 40, 45, 50]
        if 'DIFF' not in measure:
            if 'AUC' in measure:
                ylim_low = 0.75
                ylim_high = 1.0
            else:
                ylim_low = 0.6
                ylim_high = 1.0
    if scenario == 3:
        xlab = 'Sample Size of the Under-represented Group'
        avg_stack.index = avg_stack.index.astype(int)
        xticks = avg_stack.index
        if 'DIFF' not in measure:
            if 'AUC' in measure:
                ylim_low = 0.65
                ylim_high = 1.0
            else:
                ylim_low = 0.6
                ylim_high = 1.0
    if scenario == '1b':
        xlab = 'Number of Important Covariates for the Under-represented Group'
        avg_stack.index = 40 - avg_stack.index.astype(int)
        xticks = avg_stack.index
        if 'DIFF' not in measure:
            if 'AUC' in measure:
                ylim_low = 0.85
                ylim_high = 1.0
            else:
                ylim_low = 0.75
                ylim_high = 1.0
    if scenario == '2b':
        avg_stack.index = np.arange(len(avg_stack))
        xlab = 'Baseline Prevalance of the Under-represented Group (%)'
        xticks = [50, 55, 60, 65, 70, 74, 78, 82, 85, 88, 90]
        if 'DIFF' not in measure:
            if 'AUC' in measure:
                ylim_low = 0.75
                ylim_high = 1.0
            else:
                ylim_low = 0.6
                ylim_high = 1.0
    if scenario == '3b':
        xlab = 'Sample Size of the Over-represented Group'
        avg_stack.index = avg_stack.index.astype(int)
        xticks = avg_stack.index
        if 'DIFF' not in measure:
            ylim_low = 0.65
            ylim_high = 1.0
    if scenario == 4 or scenario == '4b':
        xlab = 'Number of Covariates'
        avg_stack.index = avg_stack.index.astype(int)
        xticks = avg_stack.index
        if 'DIFF' not in measure:
            ylim_low = 0.55
            ylim_high = 1.0
        
    xticks_labels = [str(s) for s in xticks]
    xticks = np.arange(len(xticks))
    
    avg_stack.index = xticks
    upp_stack.index = xticks
    low_stack.index = xticks

    colors=['orange', 'limegreen', 'violet', 'royalblue']
    markers=['s','^','o','h']
    loc_cal = np.linspace(-0.12, 0.12, 4)
    
    plt.figure(figsize=(10, 8))
    for i, model in enumerate(avg_stack.columns):
        model_avg = avg_stack[model].copy()
        model_upp = upp_stack[model].copy()
        model_low = low_stack[model].copy()
        
        model_avg.index = model_avg.index + loc_cal[i]
        model_std = pd.concat([model_low, model_upp], axis=1).T.values
        ax = model_avg.plot(marker=markers[i], xticks=model_avg.index, color=colors[i], yerr=model_std, linewidth=1.6,
                            capsize=5, capthick=1.2, linestyle='-.', fontsize=16, markersize=10)
        ax.set_xlim((-0.5, len(model_avg)-0.5))
        if 'DIFF' not in measure:
            ax.set_ylim((ylim_low, ylim_high))
        ax.set_xticklabels(xticks_labels)
    ax.set_xlabel(xlab, fontsize=16)
    ax.legend(loc='best', fancybox=True, framealpha=0.3, fontsize=16)
    plt.tight_layout()

    plt.savefig(os.path.join(savepath, f'Sim-{scenario}-{title}.pdf'), dpi=dpi)
    plt.close()



measures = ['All-AUC', 'Group1-AUC', 'Group2-AUC', 'AUC-DIFF',
            'All-NZBIAS', 'Group1-NZBIAS', 'Group2-NZBIAS',
            'All-COEFSEN', 'Group1-COEFSEN', 'Group2-COEFSEN',
            'All-COEFSPE', 'Group1-COEFSPE', 'Group2-COEFSPE']
titles = ['Overall AUC', 'AUC of the Over-represented Group', 'AUC of the Under-represented Group', 'Disparity of AUC',
          'Overall Coefficient Bias', 'Bias of the Over-represented Group', 'Bias of the Under-represented Group',
          'Overall Selection Sensitivity', 'Selection Sensitivity of the Over-represented Group', 'Selection Sensitivity of the Under-represented Group',
          'Overall Specificity Sensitivity', 'Selection Specificity of the Over-represented Group', 'Selection Specificity of the Under-represented Group']

for scenario in [1, 2, 3, 4, '1b', '2b', '3b', '4b']:
    for measure, title in zip(measures, titles):
        try:
            make_plot(scenario, measure, title)
            print(scenario, measure)
        except:
            print('something went wrong with', scenario, measure)
