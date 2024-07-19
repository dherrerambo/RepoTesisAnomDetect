import warnings
warnings.filterwarnings("ignore")

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np
import pandas as pd
import time, os

from lib_anomaly.utils import save_plot
from lib_anomaly.params import clf_order



### aux

def _add_error_bars(g, df:pd.DataFrame, error:str):
    x_coords = [p.get_x() + 0.5*p.get_width() for p in g.patches]
    y_coords = [p.get_height() for p in g.patches]
    g.errorbar(x=x_coords, y=y_coords, yerr=df[error], fmt="none", c= "k")
    return g

def _define_limits(S:pd.Series):
    ## limites
    ymin,ymax = S.min(),S.max()
    if ymin>0:
        ymin = ymin*0.9
    else:
        ymin = ymin*1.1
    if ymax>0:
        ymax = ymax*1.1
    else:
        ymax = ymax*0.9
    if ymax>1: ymax=1
    return ymin,ymax



## plots

def plot_pairplot(data:pd.DataFrame, target:str, show_title:bool=True, local_out_path_plots:str=None, fName:str=None, s3_out_path_plots:str=None, show_plot:bool=True, show_prints:bool=False, logger=None, **kwargs):
    fig = plt.figure(figsize=(10,10))
    sns.pairplot(data=data.drop([target],axis=1), hue=target+"Label", height=2, plot_kws={"s": 20, "alpha":0.6})
    title = f"pairplot by {target}"
    if show_title: plt.title(title,fontsize=10)
    if local_out_path_plots!=None: save_plot(fig=fig, local_out_path_plots=local_out_path_plots, fName=fName, s3_out_path_plots=s3_out_path_plots, show_title=show_title)
    if show_plot: plt.show()
    if logger!=None: logger.info(title)
    return


def plot_xy(data:pd.DataFrame, x:str, y:str, target:str, height:float=6, show_title:bool=True, local_out_path_plots:str=None, fName:str=None, s3_out_path_plots:str=None, show_plot:bool=True, logger=None, **kwargs):
    fig = plt.figure(figsize=(8,5))
    g = sns.lmplot(
                    data=data.sort_values(target)
                    , x=x, y=y
                    , hue=target+"Label"
                    , height=height, aspect=1.5
                    , markers=["x","*"]
                    , fit_reg=False
                    , scatter_kws={"alpha":0.6}
                    , legend=None
                );
    plt.legend(title=target, loc='upper right')
    title = f"Distribucion de {x=} y {y=}"
    if show_title: plt.title(title,fontsize=10)
    plt.tight_layout(pad=.4)
    if local_out_path_plots!=None: save_plot(fig=fig, local_out_path_plots=local_out_path_plots, fName=fName, s3_out_path_plots=s3_out_path_plots, show_title=show_title)
    if show_plot: plt.show()
    if logger!=None: logger.info(title)
    return


def plot_target_split_distribution(target:str, y, y_train, y_test, show_title:bool=True, local_out_path_plots:str=None, fName:str=None, s3_out_path_plots:str=None, show_plot:bool=True, logger=None, **kwargs):
    fig = plt.figure(figsize=(8,8))
    g = sns.kdeplot(y, label="y")
    g = sns.kdeplot(y_train, label="train")
    g = sns.kdeplot(y_test, label="test")
    plt.legend(["y","train","test"])
    title = f"Train and test by {target}"
    if show_title: plt.title(title,size=10)
    if local_out_path_plots!=None: save_plot(fig=fig, local_out_path_plots=local_out_path_plots, fName=fName, s3_out_path_plots=s3_out_path_plots, show_title=show_title)
    if show_plot: plt.show()
    if logger!=None: logger.info(title)
    return



def plot_metrics(results:pd.DataFrame, name_detector:str, ncol:int=6, xsize:int=8, show_plot:bool=True, show_title:bool=True, local_out_path_plots:str=None, s3_out_path_plots:str=None, logger=None, **kwargs):
    metrics = sorted(list(set(results["metric"])))
    metrics = [metrics[a:a+2] for a in range(0, len(metrics),2)]
    metrics = [[a[1],a[0]] for a in metrics]
    metrics = [item for row in metrics for item in row]
    cv_metrics = [a for a in metrics if a.startswith("cv_")]
    if len(cv_metrics)>0:
        metrics = cv_metrics + [a for a in metrics if a not in cv_metrics]
    if len(metrics)<ncol: ncol = len(metrics)
    nrow = int(np.ceil(len(metrics)/ncol))
    fig = plt.figure(figsize=(xsize, nrow*3))
    grid = gridspec.GridSpec(nrows=nrow, ncols=ncol)
    for i,_metric in enumerate(metrics):
        ax = plt.subplot(grid[i])
        ix_cols = ["metric","clf"]
        tmp = results[results["metric"]==_metric].copy()
        tmp = tmp[ix_cols +[name_detector]].melt(ix_cols).drop("metric",axis=1)
        # g = sns.barplot(data=tmp, x="clf", y="value", hue="variable", ax=ax)
        g = sns.barplot(data=tmp, x="clf", y="value", ax=ax)
        ## label bars
        for _ in g.containers:
            g.bar_label(_, fmt='%.2f', fontsize=6, padding=2, rotation=90)
        ## legend
        g.legend().remove()
        ## limites
        ymin,ymax = _define_limits(S=tmp["value"])
        g.set(ylim=(ymin, ymax), xlabel=None)
        ## size labels
        ax.set_xticklabels(g.get_xticklabels(), fontsize=8)
        ax.set_yticklabels(g.get_yticklabels(), fontsize=8)
        ax.set_ylabel('')
        ax.set_xlabel('')
        # plt.tight_layout()
        ax.set_title(_metric.replace("_train",""), fontsize=8)
    title = f"Metrics by clf models with anomaly detector [{name_detector}]"
    fName = f"metrics_{name_detector}"
    if show_title:
        plt.suptitle(title, fontsize=10)
    else:
        fName = fName+"__no_title"
    # fig.subplots_adjust(hspace=1, top=0.85)
    plt.tight_layout()
    if local_out_path_plots!=None: save_plot(fig=fig, local_out_path_plots=local_out_path_plots, fName=fName, s3_out_path_plots=s3_out_path_plots, show_title=show_title)
    if show_plot: plt.show()
    if logger!=None: logger.info(title)
    return


def plot_score_prob(results, show_title:bool=True, local_out_path_plots:str=None, fName:str=None, s3_out_path_plots:str=None, show_plot:bool=True, logger=None, **kwargs):
    fig,ax = plt.subplots(2,1, figsize=(10,5), sharex=True, sharey=False)
    g = sns.lineplot(results.sort_values("anom_score", ascending=False)["anom_score"].reset_index(drop=True), ax=ax[0], color="red")
    g = sns.lineplot(results.sort_values("anom_prob", ascending=False)["anom_prob"].reset_index(drop=True), ax=ax[1])
    g.set_xlabel("# registros")
    n_ = len(g.xaxis.get_ticklabels())-3
    s_ = 1/n_
    xticks = [-s_] + [a*s_ for a in range(n_+1)] + [1+s_]
    g.set(xticklabels=[f"{a:.1%}" for a in xticks])
    title = f"Records by score and probability"
    if show_title:
        plt.suptitle(title,fontsize=10)
    else:
        try:
            fName = fName + "__no_title"
        except:
            pass
    plt.tight_layout()
    if local_out_path_plots!=None: save_plot(fig=fig, local_out_path_plots=local_out_path_plots, fName=fName, s3_out_path_plots=s3_out_path_plots, show_title=show_title)
    if show_plot: plt.show()
    if logger!=None: logger.info(title)
    return



def plot_outlier_score(data:pd.DataFrame, var_x:str, var_y:str, size:tuple=(8,8), show_title:bool=True, local_out_path_plots:str=None, fName:str=None, s3_out_path_plots:str=None, show_plot:bool=True, logger=None, **kwargs):
    fig = plt.figure(figsize=size)
    g = sns.scatterplot(
                data=data.sort_values("anom_score")
                , x=var_x, y=var_y
                , s=20
                , hue="anom_score"
                , palette="rocket_r"
            )
    title = "Outliers distribution"
    if show_title:
        plt.title(title,fontsize=10)
    else:
        try:
            fName = fName + "__no_title"
        except:
            pass
    g.legend()
    plt.tight_layout()
    if local_out_path_plots!=None: save_plot(fig=fig, local_out_path_plots=local_out_path_plots, fName=fName, s3_out_path_plots=s3_out_path_plots, show_title=show_title)
    if show_plot: plt.show()
    if logger!=None: logger.info(title)
    return


def plot_proba_umbral(df_anom:pd.DataFrame, umbral:float, show_title:bool=True, local_out_path_plots:str=None, fName:str=None, s3_out_path_plots:str=None, show_plot:bool=True, logger=None, **kwargs):
    fig = plt.figure(figsize=(15,3))
    tmp = df_anom.copy()
    tmp = tmp.reset_index()
    tmp["index"] = tmp["index"] + 1
    filtro = tmp["anom_prob"]>=umbral
    g = sns.lineplot(data=tmp, x="index", y="anom_prob", zorder=1, alpha=0.5)
    g = plt.scatter(x=tmp[filtro]["index"], y=tmp[filtro]["anom_prob"], c="red", marker="*", zorder=1)
    plt.xlabel("record position in dataset")
    plt.ylabel("Anomaly Score")
    plt.xlim([0,len(tmp)+1])
    plt.xticks(list(range(0,len(tmp), int(len(tmp)/10)))[:-1]+[len(tmp)-1])
    title = f"Outliers detected by threshold={umbral:.1%} (# anomalities = {len(tmp[filtro])})"
    if show_title:
        plt.title(title,fontsize=10)
    else:
        try:
            fName = fName + "__no_title"
        except:
            pass
    plt.tight_layout()
    if local_out_path_plots!=None: save_plot(fig=fig, local_out_path_plots=local_out_path_plots, fName=fName, s3_out_path_plots=s3_out_path_plots, show_title=show_title)
    if show_plot: plt.show()
    if logger!=None: logger.info(title)
    return



def plot_cv(original_results:dict, sin_anomalias_results:dict, scoring_metric:str, local_out_path_plots:str=None, fName:str=None, s3_out_path_plots:str=None, show_title:bool=True, show_plot:bool=True, logger=None, **kwargs):
    fig = plt.figure(figsize=(8,4))
    cols = sorted([m for m in list(original_results.values())[0].keys() if m.startswith("cv_")])
    tmp = pd.concat([
        pd.DataFrame(original_results).loc[cols].T.assign(fuente="Before")
        , pd.DataFrame(sin_anomalias_results).loc[cols].T.assign(fuente="After")
        ]
    )
    g = sns.barplot(data=tmp.reset_index(), x="index", y=cols[0], hue="fuente")
    g = _add_error_bars(g=g, df=tmp, error=cols[1])
    g.set_xlabel("")
    g.set_ylabel(f"K-Fold mean {scoring_metric}")
    g.legend_.set_title(None)
    plt.ylim((tmp[cols[0]]-tmp[cols[1]]).min()*0.95,)
    title = f"Comparing results of applying K-Fold by metric='{scoring_metric}'"
    if show_title:
        plt.title(title,fontsize=10)
    else:
        try:
            fName = fName + "__no_title"
        except:
            pass
    plt.tight_layout()
    if local_out_path_plots!=None: save_plot(fig=fig, local_out_path_plots=local_out_path_plots, fName=fName, s3_out_path_plots=s3_out_path_plots, show_title=show_title)
    if show_plot: plt.show()
    if logger!=None: logger.info(title)
    return


def plot_anomalies_detected(data:pd.DataFrame, name_detector:str, contamination:float, show_title:bool=True, show_plot:bool=True, local_out_path_plots:str=None, s3_out_path_plots:str=None, logger=None, **kwargs):
    """
        Plot each of anomalies detected by each anomaly detecntion models
    """
    tmp = data[[name_detector, f"{name_detector}__prob"]].reset_index(drop=True).reset_index()
    tmp["index"] = tmp["index"]+1
    r = int(len(tmp)/10)
    fig = plt.figure(figsize=(8,3))
    g = sns.scatterplot(data=tmp, x="index", y=f"{name_detector}__prob", zorder=0, alpha=0.5, s=5, label="Normal")
    filtro = tmp[name_detector]>=1
    g = sns.scatterplot(data=tmp[filtro], x="index", y=f"{name_detector}__prob", color="red", zorder=1, marker="*", s=50, label="Anomalies")
    g.set(xlim=(1,len(tmp)), ylabel=f"{name_detector}__prob")
    # xticks
    xticks = range(1, tmp["index"].max(), r)
    xticks = list(xticks)
    if xticks[-1]<tmp["index"].max(): xticks =  xticks[:-1] + [tmp["index"].max()] 
    # g.set_xticks(xticks, fontsize=8)
    g.set_xticklabels(xticks, fontsize=8)
    g.set_yticklabels(g.get_yticklabels(), fontsize=8)
    g.set_xlabel('')
    g.set_ylabel(f"{name_detector}__prob", fontsize=8)
    g.legend(loc='upper right', fontsize=8, bbox_to_anchor=(1,1.2))
    title = f"Anomaly records with a {contamination=}"
    fName = f"anomalies_detected__{name_detector}"
    if show_title:
        plt.title(title, fontsize=10)
    else:
        try:
            fName =  fName+ "__no_title"
        except:
            pass
    if local_out_path_plots!=None: save_plot(fig=fig, local_out_path_plots=local_out_path_plots, fName=fName, s3_out_path_plots=s3_out_path_plots, show_title=show_title)
    plt.tight_layout()
    if show_plot: plt.show()
    if logger!=None: logger.info(title)
    return



def plot_results(results:pd.DataFrame, local_out_path_plots:str=None, s3_out_path_plots:str=None, show_title:bool=True, show_plot:bool=True, logger=None, **kwargs):
    """
        plot total results of metrics by clasification models and anomaly detection models
    """
    metrics = sorted(list(set(results["metric"])))
    metrics = [metrics[a:a+2] for a in range(0, len(metrics),2)]
    metrics = [[a[1],a[0]] for a in metrics]
    metrics = [item for row in metrics for item in row]
    fig = plt.figure(figsize=(10, 2*len(metrics)))
    grid = gridspec.GridSpec(nrows=len(metrics), ncols=2)
    for i,met in enumerate(metrics):
        cols = ["clf","base_line","lof","iforest","autoencoder","ensamble","ensamble_p_0_3"]
        tmp = results[results["metric"]==met][cols].melt("clf")
        ## plot
        ax = plt.subplot(grid[i])
        g = sns.barplot(data=tmp, x="clf", y="value", hue="variable", order=clf_order, ax=ax)
        for _ in g.containers:
            g.bar_label(_, fmt='%.2f', fontsize=6, padding=2, rotation=90)
        ## limits
        ymin,ymax = _define_limits(tmp["value"])
        g.set(ylabel="", xlabel="", ylim=(ymin,ymax))
        ## labels
        g.set_title(met.replace("_train",""), fontsize=8)
        # ax.set_xticklabels(g.get_xticklabels(), rotation=45, fontsize=8)
        g.set_yticklabels(g.get_yticklabels(), fontsize=8)
        ## legend
        if i==1:
            g.legend(
                    loc='upper right'
                    , fontsize=8
                    , borderaxespad=0.
                    , bbox_to_anchor=(1,1.7)
                    , ncol=3 #len(set(df["experiment"]))
                    # , frameon=False
                )
        else:
            g.legend().remove()
    ## title fig
    fName = "comparing_results"
    title = f"Comparing results by models and metrics"
    if show_title:
        plt.suptitle(title, x=0.3, y=0.965, fontsize=10)
    else:
        try:
            fName = fName + "__no_title"
        except:
            pass
    plt.tight_layout()
    if local_out_path_plots!=None: save_plot(fig=fig, local_out_path_plots=local_out_path_plots, fName=fName, show_title=show_title, s3_out_path_plots=s3_out_path_plots)
    if show_plot: plt.show()
    if logger!=None: logger.info(title)
    return


def plot_comparing_results(results:pd.DataFrame, name_detector:str, metric:str, show_title:bool=True, local_out_path_plots:str=None, s3_out_path_plots:str=None, show_plot:bool=True, logger=None, **kwargs):
    ix_cols = ["metric","clf"]
    metrics = [metric+"_train", metric+"_test"]
    tmp = results[results["metric"].isin(metrics)][ix_cols+['base_line',name_detector]].copy()
    tmp = tmp.melt(ix_cols)
    tmp["variable"] = tmp["variable"].map({"base_line":"Before",name_detector:"After"})
    fig,ax = plt.subplots(2,1,figsize=(8, 4))
    for i in range(len(metrics)):
        g = sns.barplot(data=tmp[tmp["metric"]==metrics[i]], x="clf", y="value", hue="variable", ax=ax[i])
        ## add value
        for _ in g.containers:
            g.bar_label(_, fmt='%0.4f', fontsize=6, padding=2)
        ## limites
        ymin,ymax = _define_limits(S=tmp["value"])
        g.set(ylim=(ymin,ymax), xlabel="", ylabel="")
        ## ajustar labels
        g.set_yticklabels(g.get_yticklabels(), fontsize=8)
        g.set_xticklabels(g.get_xticklabels(), fontsize=8)
        plt.tight_layout()
        ## legend
        if i!=0:
            g.legend().remove()
        else:
            g.legend(ncol=3, bbox_to_anchor=(1,1.3),loc='upper right',fontsize=8, borderaxespad=0.)
    title = f"""Comparing [{metric}] before and after\nanomalies removal by [{name_detector}]"""
    fName = f"before_after_{name_detector}_{metric}"
    if show_title:
        plt.suptitle(title, x=0.3, y=0.85, fontsize=10)
    else:
        try:
            fName = fName + "__no_title"
        except:
            pass
    plt.tight_layout()
    if local_out_path_plots!=None: save_plot(fig=fig, local_out_path_plots=local_out_path_plots, fName=fName, s3_out_path_plots=s3_out_path_plots, show_title=show_title)
    if show_plot: plt.show()
    if logger!=None: logger.info(title)
    return


def plot_comparing_anomaly_probability_score(data:pd.DataFrame, contamination:float, show_title:bool=True, show_plot:bool=True, local_out_path_plots:str=None, s3_out_path_plots:str=None, logger=None, **kwargs):
    cols1 = ["lof","iforest","autoencoder"]
    cols2 = [c for c in data if c.endswith("__prob") and c.split("__")[0] in cols1]
    tmp = data[cols1+cols2].reset_index().copy()
    tmp1 = tmp[["index"]+cols2].melt("index")
    tmp1["variable"] = tmp1["variable"].str.replace("__prob","")
    tmp2 = tmp.copy()
    for c in cols1:
        tmp2.loc[tmp2[c]==0,c+"__prob"] = None
    tmp2 = tmp2[["index"]+cols2].melt("index").dropna()
    tmp2["variable"] = tmp2["variable"].str.replace("__prob","")
    r = int(len(tmp)/10)
    ## plot
    fig = plt.figure(figsize=(10,3))
    g = sns.lineplot(data=tmp1, x="index", y="value", hue="variable", alpha=0.4, legend=cols1)
    g = sns.scatterplot(data=tmp2, x="index", y="value", hue="variable", zorder=1, marker="*", s=100, legend=None)
    g.set(xlim=(0,max(data.index)))
    xticks = range(1, tmp["index"].max(), r)
    xticks = list(xticks)
    if xticks[-1]<tmp["index"].max(): xticks =  xticks[:-1] + [tmp["index"].max()] 
    g.set_xticklabels(xticks, fontsize=8)
    g.set_yticklabels(g.get_yticklabels(), fontsize=8)
    g.set_xlabel('')
    g.set_ylabel(f"anomaly probability score", fontsize=8)
    g.legend(loc='upper right', ncol=len(cols1), fontsize=8, bbox_to_anchor=(1,1.2))
    title = f"Comparing anomaly score probability\nwith {contamination=}"
    fName = f"comparing_anomaly_score_prob"
    if show_title:
        plt.title(title, fontsize=10)
    else:
        try:
            fName =  fName+ "__no_title"
        except:
            pass
    if local_out_path_plots!=None: save_plot(fig=fig, local_out_path_plots=local_out_path_plots, fName=fName, s3_out_path_plots=s3_out_path_plots, show_title=show_title)
    plt.tight_layout()
    if show_plot: plt.show()
    if logger!=None: logger.info(title)
    return
