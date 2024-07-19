import time
import numpy as np
import pandas as pd

## metodos de deteccion de anomalias
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pyod.models.pca import PCA
from pyod.models.gmm import GMM
from pyod.models.lof import LOF
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.hbos import HBOS
from pyod.models.ecod import ECOD
from pyod.models.ocsvm import OCSVM


from lib_anomaly.models import get_metrics_report



def make_hidden_layers(no_features:int):
    nn_layers = [int(np.ceil(no_features*a) )for a in [1, 0.7, 0.3]]
    nn_layers = nn_layers + nn_layers[::-1][1:]
    return nn_layers


def anomaly_detector(data:pd.DataFrame, method:str, contamination:float, logger=None, **kwargs):
    results = dict()
    ## define method
    if method=="AutoEncoder":
        nn_layers = make_hidden_layers(no_features=data.shape[1])
        clf = eval(f"{method}(hidden_neurons={nn_layers}, verbose=0, contamination={contamination})")
    else:
        clf = eval(f"{method}(contamination={contamination})")
    ## fit method
    try:
        clf.fit(data, verbose=0)
    except:
        clf.fit(data)
    ## results
    results[method] = clf.labels_
    try:
        results[method+"_score"] = clf.decision_function(data, verbose=0)
    except:
        results[method+"_score"] = clf.decision_function(data)
    try:
        results[method+"_prob"] = clf.predict_proba(data, verbose=0)[:,1]
    except:
        results[method+"_prob"] = clf.predict_proba(data)[:,1]
    return results



def clean_anomalies(df_anom:pd.DataFrame, umbral:str, x_train, y_train, x_test, y_test, pipes_clf, show_prints:bool=False, logger=None, **kwargs):
    """
        Deja el DataFrame ´x_train´ sin las anomalias detectadas en el DataFrame ´df_anom´
    """
    filtro = eval(umbral)

    _anom = filtro[filtro==True].index
    _norm = filtro[filtro==False].index
    if show_prints==True: print(f"{len(_norm)=}, {len(_anom)=}, % reduccion = {(len(_anom)/len(x_train)):.1%}")
    
    ## results
    res_sin_anom = dict()
    for name,pipe in pipes_clf.items():
        start = time.time()
        if show_prints==True: print(f"{name}", "*"*50)
        try:
            pipe.fit(x_train.loc[_norm], y_train.loc[_norm], verbose=0)
        except:
            pipe.fit(x_train.loc[_norm], y_train.loc[_norm])
        try:
            y_pred = pipe.predict(x_test, verbose=0)
        except:
            y_pred = pipe.predict(x_test)
        metrics = get_metrics_report(y_true=y_test, y_pred=y_pred)
        res_sin_anom[name] = metrics
    if show_prints==True: print(f"\trun time = {round(time.time()-start,2)}s.\n")
    return res_sin_anom