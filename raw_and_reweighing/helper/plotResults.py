import sys
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
import seaborn as sns

############################ Common for ML and CNN pipelines #################################

def plot_result(df_results, x="val_metric", groupby="model", hue=None,
                groupby_only=[], beautify_io=True, legend=True,
                sorty_by=None):
    # df_results multiple cases handling
    if isinstance(df_results, (list, tuple)):
        df_results = pd.concat(df_results)
    
    df = df_results.copy()
    
    df['io'] = df.apply(lambda row: remap_io(row, beautify_io), axis=1)
    
    if not groupby_only:
        groupby_only = df[groupby].unique()
    else:
        df = df[df[groupby].isin(groupby_only)]
    ios = df["io"].unique()
    
    sns.set(style='whitegrid', context= 'paper')
    fig, axes = plt.subplots(1, len(groupby_only), sharex=True, sharey=True,
                             dpi=120, figsize=(4*len(groupby_only), 1+0.4*len(ios)))
    
    if not isinstance(axes, np.ndarray): axes = [axes]
    if (x == "val_metric"):
        fig.gca().set_xlim([0,100])
        # metric computation either not provided or permutation cases
        df['val_metric'] = df['val_metric'].apply(lambda x: x*100)
        
    if (x == "train_metric"):
        fig.gca().set_xlim([0,100])
        # metric computation either not provided or permutation cases
        df['train_metric'] = df['train_metric'].apply(lambda x: x*100)
    
    if (x == "test_metric"):
        fig.gca().set_xlim([0,100])
        # metric computation either not provided or permutation cases
        df['test_metric'] = df['test_metric'].apply(lambda x: x*100)
 
    for n, ((t, dfi), ax) in enumerate(zip(df.groupby(groupby), axes)):
        palette = sns.color_palette()
        ci, dodge, scale, errwidth, capsize = 95, 0.4, 0.4, 0.9, 0.08      
    
        ax = sns.pointplot(data=dfi, y='io', x=x, hue=hue, ax=ax,
                           ci=ci, errwidth=errwidth, capsize=capsize,
                           dodge=dodge, scale=scale, palette=palette, join=False)
        if hue:
            ax.legend_.remove()
        ax.set_title(r"{} ($n \approx{:.0f}$)".format(t.upper(), dfi["n_samples"].mean()))
        ax.set_xlabel(x)
        ax.set_ylabel("")
        
        if legend:
            # add legend: add models info and chance label
            handles, legends = ax.get_legend_handles_labels()
            if n == len(axes)-1:
                if ([] == legends) | (hue == None):
                    print("TBC")
                else:
                    leg1 = fig.legend(handles, legends, title=hue, bbox_to_anchor=[1, 1],
                                      loc="upper right", fancybox=True, frameon=True)

    return fig

def remap_io(row, beautify_io=True):
    if beautify_io:
        latex = []
        for target in [row.inp, row.out]:
            i = target.replace('_','\_')
            if target == 'X':
                i = "X"
            elif "confs" in row:
                i = f"c_{{{i}}}" if target in eval(row.confs) else f"y_{{{i}}}"
            else:
                i = f"y_{{{i}}}"
            latex.append(i)
        return f"${latex[0]} \longmapsto {latex[1]}$"
    else:
        return f"{row.inp}-{row.out}"

################################  CNNpipeline    #######################################

def plot_training_curves(df_results, metrics='all'):
    
    for (model,inp,out), dfi in df_results.groupby(['model','inp','out']):
        
        # if metrics='all' then plot all the learning curve metrics present in the results file
        if isinstance(metrics,(str)) and metrics.lower()=='all':
            metrics = {col.split('_curve_')[-1] for col in dfi.columns if '_curve_' in col}

        f, axes = plt.subplots(1, len(metrics), 
                               figsize=(5*len(metrics),4),
                               sharex=True, constrained_layout=True)
        axes = axes.ravel()
        f.suptitle(r"Model: {}     Task: ${} \rightarrow {}$".format(model, inp, out), fontsize=16)

        # use different colors for metrics and different line styles for different CV trials repeats
        metrics_cmap = cm.get_cmap('tab10')
        trial_linestyles = ['solid', 'dotted', 'dashed', 'dashdot', (0,(1,5)), (0,(5,10)), 
                            (0,(3,5,1,5)), (0,(1,10)), (0,(3,5,1,5,1,5)), (0,(3,1,1,1,1,1))]

        for i, metric  in enumerate(metrics):
            ax = axes[i]
            ax.set_title(metric.replace('_', ' ').title())
            ax.set_xlabel('Epochs')
            ax.grid(True)    
                    
            # if there are results from multiple CV trials 
            # then show them in a single plot using different line styles            
            leg_handles, leg_names = [], []

            for j, row in dfi.iterrows():
                ls = 'solid'
                if 'trial' in row:
                    ls = trial_linestyles[row['trial']]
                    # show the linestyles used as the legend
                    leg_handles.extend([Line2D([0,1],[0,1], ls=ls, color='gray')])
                    leg_names.extend([f"CV trial {row['trial']}"])
                    
                for k, (colname, vals) in enumerate(row.filter(like='_curve_'+metric).items()):
                    if not pd.isna(vals):
                        ax.plot(eval(vals), 
                                color=metrics_cmap(k),
                                ls=ls,
                                label=colname.split('_curve_')[0])
                    
                
                leg_data = ax.legend(loc='upper left')
                    
                if 'trial' in row: 
                    ax.legend(leg_handles, leg_names, loc='lower right')
                    # re add the legend showing the colors of the data split
                    ax.add_artist(leg_data)
                    
        plt.show()

################################   Permutation and Calculation    ######################################

# (TODO) Calculate p-value
def calc_p_values(df, x="test_score", viz=False):
    
    df["p_value"] = np.nan
    grp_order = ["io", "technique", "model"]
    if 'i_type' in df and len(df['i_type'].unique())>1:
        grp_order.insert(0, 'i_type')
    groups = df.dropna(subset=[x, f"permuted_{x}"]).groupby(grp_order)   
    n_models = len(df["model"].unique())
    
    if viz:
        sns.reset_orig()
        n_rows = len(groups)//n_models
#         fig, axes = plt.subplots(n_rows, n_models, 
#                                  sharex=True, sharey=False,
#                                  figsize=(20, n_models*n_rows))
        ## paper plot jugaad
        fig, axes = plt.subplots(1, 3, 
                                 sharex=True, sharey=True,
                                 figsize=(12, 4))
        axes = np.ravel(axes)
        plt.xlim([0,1])
        
    for i, (g, rows) in enumerate(groups):
        
        p_vals = [] 
        rows = rows.filter(like=x)
        if viz:
            permuted_scores = []    
            true_scores = rows[x]
            
        for _, r in rows.iterrows():
            p_scores = np.array(eval(r[f'permuted_{x}']))
            # calc p_value = (C + 1)/(n_permutations + 1) 
            # where C is permutations whose score >= true_score 
            true_score = r[x]
            C = np.sum(p_scores >= r[x])
            pi = (C+1)/(len(p_scores)+1)
            p_vals.extend([pi])
            
            if viz: permuted_scores.extend([*p_scores])        
#         if np.std(permuted_scores)>=1:
#             print("[WARN] the p-values for {} have high variance across each test-set (trial). \
# Simply averaging the p-values across trials in such a case is not recommended.".format(g))  

        df.loc[(df[grp_order]==g).all(axis=1), ["p_value", ""]] = np.mean(p_vals)  
        
        if viz:
            ax = axes[i]
#             ax.set_title("Model={}".format(g[-1]))
#             if i%n_models == 0:
#                 ax.set_ylabel("{} with {}".format(*g[-3:-1]))
            ax.hist(permuted_scores, bins='auto', alpha=0.8)
            for true_score in true_scores:
                ax.axvline(true_score, color='r')
            # draw chance lines 
            if x in ["test_score"]:
                # chance is 14% for site prediction and 50% for y and sex predictions
                chance = 0.5 if g[0][-1]!='c' else (1/7)
                ax.axvline(chance, color='grey', linestyle="--", lw=1.5)
                ax.set_xlim(0.,1.)
            ## paper plot jugaad
            ax.set_xticklabels([str(item) for item in range(0,120, 120//len(ax.get_xticklabels()))])
            inp = "X_{{{}}}".format(["14yr", "19yr", "22yr"][i])
            out = "y_{{{binge}}}"
            ax.set_title(r"${} \longmapsto {}$".format(inp,out))
#             if i==0: ax.set_ylabel("distribution / counts")
            if i==1: ax.set_xlabel("Balanced accuracy (%)")
            if i==0:
                from matplotlib.lines import Line2D
                custom_lines = [Line2D([0], [0], color="tab:blue", markerfacecolor="tab:blue", marker='o', markersize=5, lw=0),
                                Line2D([0], [0], color="tab:red", lw=1, linestyle="--")]
                ax.legend(custom_lines, ['permuted score', 'model score'], loc='lower left')
            
    sns.set(style='whitegrid')
    return df

# (TODO) 
def return_asterisks(pval):
    if pd.isna(pval):
        ast = ""
    elif pval <= 0.001:
        ast = '***'
    elif pval <= 0.01:
        ast = '**'
    elif pval <= 0.05:
        ast = '*'
    else:
        ast = 'n.s.'
    return ast

# (TODO) 
def combine_io_sample_size(df):
    df["io_n"] = ''
    # only estimate sample_size for the first technique to be plotted
    dfi = df.query("technique == '{}'".format(df.technique.unique()[0]))
    for io, dfii in dfi.groupby("io"):
        # set the same sample_size in io across all techniques to avoid different y_labels in subplots
        df.loc[(df["io"]==io), "io_n"] = "{}\n(n={:.0f})".format(io, dfii["n_samples"].mean())
    return df

## (TODO) METRICS ##
def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    return tn/(tn+fp)

def compute_metric(df, metric_name):
    
    metrics_map = {"recall_score":    metrics.recall_score,
                   "precision_score": metrics.precision_score,
                   "sensitivity":     metrics.recall_score,
                   "specificity":     specificity_score,
                   "accuracy_score":  metrics.accuracy_score,
                   "f1_score":        metrics.f1_score,
                  }
    metric=None
    for k in metrics_map.keys():
        if metric_name.lower() in k:
            metric = metrics_map[k] 
            break
    if metric is None:
        raise ValueError(
            "ERROR: Invalid 'x' metric requested. Allowed metric_names are {}".format(
                metrics_map.keys()))
    
    df['y_true'] = df["test_lbls"].apply(lambda lbls: np.array(eval(lbls), dtype=int))
    df['y_pred'] = df["test_probs"].apply(lambda probs: np.argmax(np.array(eval(probs)), axis=1))
    
    df[metric_name] = df.apply(lambda x: metric(y_true=x.y_true, y_pred=x.y_pred), axis=1)
    
    return df