# config
import random, time, datetime, gc, os
start = time.time()

# Cargar librerias
import numpy as np
import pandas as pd
import awswrangler as wr

from sklearn.pipeline           import Pipeline
from sklearn.impute             import SimpleImputer
from sklearn.preprocessing      import MinMaxScaler, OneHotEncoder
from sklearn.compose            import ColumnTransformer
from category_encoders.binary   import BinaryEncoder
from sklearn.model_selection    import train_test_split

from pyod.models.lof            import LOF
from pyod.models.iforest        import IForest
from pyod.models.auto_encoder   import AutoEncoder

from lib_anomaly.anomaly        import make_hidden_layers
from lib_anomaly.utils          import get_data_types, load_dataset, normalize_str, delete_empty_col
from lib_anomaly.models         import run_clasification_models
from lib_anomaly.plots          import plot_anomalies_detected, plot_metrics, plot_comparing_results, plot_results, plot_comparing_anomaly_probability_score




def config_experiment(ROOT_PATH:str, S3BUCKET:str, random_state:float=None, logger=None, **kwargs) -> dict:
    """
        preparando espacio segun el experimento
    """
    experiment = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    if logger!=None: logger.info(f"config_experiment: experiment id={experiment}")
    ## set same seed
    if random_state!=None:
        random.seed(random_state)
        np.random.seed(random_state)
    local_out_path = os.path.join(ROOT_PATH, "experimentos", "results", experiment)
    try:
        os.makedirs(local_out_path, exist_ok=True)
        if logger!=None: logger.info(f"""config_experiment: created local_out_path='{os.path.join("experimentos", "results", experiment)}'""")
    except Exception as e:
        pass
    ## path for plots
    try:
        local_out_path_plots = os.path.join(local_out_path,"plots")
        os.makedirs(local_out_path_plots, exist_ok=True)
        if logger!=None: logger.info(f"""config_experiment: created local_out_path_plots='{os.path.join("experimentos", "results", experiment)}'""")
    except:
        pass
    response = {
                "experiment": experiment
                , "random_state": random_state
                , "local_out_path": local_out_path+"/"
                , "s3_out_path": f"s3://{S3BUCKET}/experiments/"
                , "test_size": 0.1
            }
    return response


def make_result_experiment(results:dict, **kwargs) -> pd.DataFrame:
    df = pd.DataFrame()
    for k,v in results.items():
        tmp = pd.DataFrame(v).reset_index().assign(experiment=k)
        df = pd.concat([df, tmp], axis=0)
    df = df.melt(["experiment","index"])
    df.columns = ["anomaly_methods","metric","clf","value"]
    # df.head()
    ix_cols = ["metric","clf"]
    filtro = df["anomaly_methods"]=="Base Line"
    resp = df[filtro].drop("anomaly_methods", axis=1).rename(columns={"value": "Base Line"}).copy()
    for met in results.keys():
        if met!="Base Line":
            resp = resp.merge(
                df[df["anomaly_methods"]==met].drop("anomaly_methods", axis=1).rename(columns={"value": met})
                , how="left"
                , on=ix_cols
            )
    resp.columns = [normalize_str(c) for c in resp]
    cols_results = [
                    'metric','clf','dataset_name','contamination','random_state'
                    # ,'anomaly_methods'
                    ,'anomaly_threshold'
                    ,'base_line','lof','iforest','autoencoder','ensamble','ensamble_p_0_3'
                    ,'experiment'
                ]
    for c in cols_results:
        try:
            resp[c] = kwargs[c]
        except:
            pass
    ## save results
    return resp


def save_results(results:pd.DataFrame, local_out_path:str, experiment:str, s3_out_path:str, logger=None, **kwargs) -> dict:
    ## save results
    response = dict()
    fName = f'{local_out_path}/{experiment}.csv'
    results.to_csv(fName, index=False)
    if logger!=None: logger.info(f"save_results: {fName}")
    response["fName"] = fName
    try:
        ## to s3
        df = delete_empty_col(df=results)
        resp = wr.s3.to_parquet(
                        df=df
                        , path=s3_out_path
                        , index=False
                        , dataset=True
                        , partition_cols=["experiment"]
                    )
        if logger!=None: logger.info(f"save_results: parquet files to aws {s3_out_path}")
        response["s3out_path"] = s3_out_path
    except Exception as e:
        if logger!=None: logger.info(f"save_results: couldn't send files to aws ERROR={e}")
        print(f"Can't send to aws, Error\n{e}\n.")
    return response


def anomaly_detector(data, X_, y, name_detector:str, scoring_metric:str, contamination:float, random_state:float=None, test_size:float=0.1, logger=None, **kwargs):
    # if show_prints==True: print("-"*50, name_detector)
    if name_detector=="AutoEncoder":
        nn_layers = make_hidden_layers(no_features=X_.shape[1])
        detector = eval(f"{name_detector}(hidden_neurons={nn_layers}, verbose=0, contamination={contamination})")
    else:
        detector = eval(f"{name_detector}(contamination={contamination})")
    if logger!=None: logger.info(f"starting {name_detector}")
    detector.fit(X_)
    data[name_detector] = detector.labels_
    data[f"{name_detector}__score"] = detector.decision_function(X_)
    data[f"{name_detector}__prob"]  = detector.predict_proba(X_)[:,1]
    if logger!=None: logger.info(f"No of anomaly = {sum(data[name_detector])}({round(sum(data[name_detector])/len(data[name_detector])*100,4)}%)")
    tmp = pd.DataFrame(data[name_detector], columns=["anom"])
    if logger!=None: logger.info(f"anom distribution; {dict(tmp.value_counts().sort_index())}, {dict(tmp.value_counts(normalize=True).sort_index())}")
    ## Anomalias detectadas
    after_idx = data.reset_index(drop=True).reset_index()[data[name_detector]==0]['index']
    X_A,y_A, = X_[after_idx], y.iloc[after_idx]
    if logger!=None: logger.info(f"{X_A.shape=}")
    if logger!=None: logger.info(f"{y_A.shape=}")
    if logger!=None: logger.info(f"target distribution: {dict(y_A.value_counts().sort_index())}, {dict(y_A.value_counts(normalize=True).sort_index()*100)}")
    ## split
    x_trainA, x_testA, y_trainA, y_testA = train_test_split(
                                                        X_A
                                                        , y_A
                                                        , test_size=test_size
                                                        , stratify=y_A
                                                        , random_state=random_state
                                                    )
    if logger!=None: logger.info(f"split Anom: {x_trainA.shape=}, {x_testA.shape=}")
    ## clf
    results = run_clasification_models(
                                            x_train=x_trainA, y_train=y_trainA
                                            , x_test=x_testA, y_test=y_testA
                                            , scoring_metric=scoring_metric
                                            , random_state=random_state
                                        )

    return data, results

    
def make_ensamble_anomaly_detector(data:pd.DataFrame, X_, y, anomaly_methods:str, contamination:float, random_state:float, scoring_metric:str, test_size:float, logger=None, **kwargs):
    name_detector = "ensamble"
    if logger!=None: logger.info(f"starting {name_detector}")
    data[name_detector] = data[anomaly_methods].sum(axis=1)
    data[name_detector+"__prob"] = data[[c+"__prob" for c in anomaly_methods]].mean(axis=1)
    data[name_detector+"__prob_std"] = data[[c+"__prob" for c in anomaly_methods]].std(axis=1)
    # anom__prob = data[name_detector+"__prob"].reset_index(drop=True).values
    # anom_index = [i for i,a in enumerate(data[name_detector+"__prob"]) if a>data[name_detector+"__prob"].quantile(1-contamination)]
    norm_index = [i for i,a in enumerate(data[name_detector+"__prob"]) if a<=data[name_detector+"__prob"].quantile(1-contamination)]
    X_A,y_A, = X_[norm_index], y.iloc[norm_index]
    x_trainA, x_testA, y_trainA, y_testA = train_test_split(
                                                        X_A
                                                        , y_A
                                                        , test_size=test_size
                                                        , stratify=y_A
                                                        , random_state=random_state
                                                    )
    if logger!=None: logger.info(f"ensamble detector: split")
    results = run_clasification_models(
                                            x_train=x_trainA, y_train=y_trainA
                                            , x_test=x_testA, y_test=y_testA
                                            , scoring_metric=scoring_metric
                                            , random_state=random_state
                                            , logger=logger
                                        )
    if logger!=None: logger.info(f"ensamble detector: classifier")
    return data, results


def make_ensamble_anomaly_detector_by_threshold(data:pd.DataFrame, X_, y, anomaly_threshold:float, test_size:float, random_state:float, scoring_metric:str, logger=None, **kwargs):
    name_detector = f"Ensamble P({anomaly_threshold})"
    if logger!=None: logger.info(f"ensamble detector prob: start")
    name_detector = normalize_str(name_detector)
    # anom_index = [i for i,a in enumerate(data["anom__prob"]) if a>anomaly_threshold]
    # norm_index = [i for i,a in enumerate(data["anom__prob"]) if a<=anomaly_threshold]
    # logger.info(f"ensamble threshold: {len(anom_index)}: {anom_index}")
    # logger.info(f"ensamble threshold: {len(norm_index)}: {norm_index}")
    data[name_detector] = data["ensamble__prob"]>anomaly_threshold
    # data.iloc[norm_index, name_detector] = 0
    # data.iloc[anom_index, name_detector] = 1
    data[name_detector] = data[name_detector].astype(int)
    if logger!=None: logger.info(f"ensamble threshold: {dict(data[name_detector].value_counts().sort_index())}")
    norm_index = list(data.reset_index(drop=True)[data.reset_index(drop=True)[name_detector]==0].index)
    X_A,y_A, = X_[norm_index],y.iloc[norm_index]
    x_trainA, x_testA, y_trainA, y_testA = train_test_split(
                                                        X_A
                                                        , y_A
                                                        , test_size=test_size
                                                        , stratify=y_A
                                                        , random_state=random_state
                                                    )
    if logger!=None: logger.info(f"ensamble threshold: split")
    results = run_clasification_models(
                                            x_train=x_trainA, y_train=y_trainA
                                            , x_test=x_testA, y_test=y_testA
                                            , scoring_metric=scoring_metric
                                            , random_state=random_state
                                            , logger=logger
                                        )
    if logger!=None: logger.info(f"ensamble threshold: clasification")
    return data, results


def run_experiment(
                experiment:str
                , dataset_name:str
                , random_state:int
                , scoring_metric:str
                , contamination:float
                , test_size:float
                , local_out_path:str
                , s3_out_path:str
                , anomaly_threshold:float
                , logger=None
                , **kwargs
            ):

    ## vars
    results = dict()

    if logger!=None: logger.info(f"{local_out_path=}")
    if logger!=None: logger.info(f"{s3_out_path=}")
    

    # load dataset
    data, target, features = load_dataset(
                                    dataset_name=dataset_name
                                    , random_state=random_state
                                    , local_out_path=local_out_path
                                    , s3path=f"{s3_out_path}datasets/"
                                )
    
    # X and y datasets
    X = data.drop([target, target+"Label"], axis=1).copy()
    y = data[target]
    if logger!=None: logger.info(f"target distribution {dict(y.value_counts().sort_index())}")
    if logger!=None: logger.info(f"target distribution {dict(y.value_counts(normalize=True).sort_index()*100)}")
    
    # get columns types and fix data type
    data, cols_num, cols_cat_high, cols_cat_low = get_data_types(
                                        features=features
                                        , data=data
                                    )
    if len(cols_num)>0 and logger!=None: logger.info(f"{cols_num=}")
    if len(cols_cat_high)>0 and logger!=None: logger.info(f"{cols_cat_high=}")
    if len(cols_cat_low)>0 and logger!=None: logger.info(f"{cols_cat_low=}")

    ## pipeline
    process_num = Pipeline(steps=[
                            ("impute_mean", SimpleImputer(missing_values=np.nan, strategy="median")),
                            ("scale_minmax", MinMaxScaler((0,1)))
                        ]
                    )
    process_low_cat = Pipeline(steps=[
                            ("impute_constant", SimpleImputer(missing_values=np.nan, strategy="most_frequent"))
                            , ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore", ))
                        ]
                    )
    process_high_cat = Pipeline(steps=[
                            ("impute_constant", SimpleImputer(missing_values=np.nan, strategy="most_frequent"))
                            , ("binary", BinaryEncoder())
                            , ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore"))       ##agregado posterior
                        ]
                    )
    preprocess_vars = ColumnTransformer(
                        [
                            ("cat_low_dimension", process_low_cat, cols_cat_low)
                            , ("cat_high_dimension", process_high_cat, cols_cat_high)
                            , ("numerical", process_num, cols_num)
                        ]
                )
    
    ## preprocess X
    X_ = preprocess_vars.fit_transform(X)
    if logger!=None: logger.info(f"preprocess X, {X_.shape=}")
    # if show_prints==True: print(f"{X_.shape=}")

    # Dividir datasets en **Train** y **Test**
    x_train, x_test, y_train, y_test = train_test_split(
                                                        X_
                                                        , y
                                                        , test_size=test_size
                                                        , stratify=y
                                                        , random_state=random_state
                                                    )
    # if show_prints==True: print(f"{x_train.shape=}, {x_test.shape=}, {y_train.shape=}, {y_test.shape=}")
    if logger!=None: logger.info(f"{x_train.shape=}")
    if logger!=None: logger.info(f"{x_test.shape=}")
    if logger!=None: logger.info(f"{y_train.shape=}")
    if logger!=None: logger.info(f"{y_test.shape=}")

    ## Base Line
    name_detector = "Base Line"
    if logger!=None: logger.info(f"starting {name_detector}")
    results[name_detector] = run_clasification_models(
                                            x_train=x_train, y_train=y_train
                                            , x_test=x_test, y_test=y_test
                                            , scoring_metric=scoring_metric
                                            , random_state=random_state
                                            , logger=logger
                                        )
    
    ## run anomaly detection
    anomaly_methods = ["LOF", "IForest", "AutoEncoder"]

    for name_detector in anomaly_methods:

        data, results[name_detector] = anomaly_detector(
                                data=data
                                , X=X
                                , y=y
                                , X_=X_
                                , name_detector=name_detector
                                , scoring_metric=scoring_metric
                                , contamination=contamination
                                , random_state=random_state
                                , test_size=test_size
                                , logger=logger
                            )

    name_detector = "Ensamble"
    data, results[name_detector] = make_ensamble_anomaly_detector(
                                        data=data
                                        , X_=X_
                                        , y=y
                                        , anomaly_methods=anomaly_methods
                                        , contamination=contamination
                                        , random_state=random_state
                                        , scoring_metric=scoring_metric
                                        , test_size=test_size
                                        , logger=logger
                                    )
    
    name_detector = f"Ensamble P({anomaly_threshold})"
    data, results[name_detector] = make_ensamble_anomaly_detector_by_threshold(
                                        data=data
                                        , X_=X_
                                        , y=y
                                        , anomaly_threshold=anomaly_threshold
                                        , test_size=test_size
                                        , random_state=random_state
                                        , scoring_metric=scoring_metric
                                        , logger=logger
                                    )
    
    data.columns = data.columns.str.lower()
    
    return data, results


def make_plots(data:pd.DataFrame, results:pd.DataFrame, local_out_path:str, s3_out_path:str, contamination:float, show_title:bool, show_plots:bool, logger=None, **kwargs):

    plot_comparing_anomaly_probability_score(
                    data=data
                    , contamination=contamination
                    , show_title=show_title
                    , show_plot=show_plots
                    , local_out_path_plots=local_out_path
                    , s3_out_path_plots=s3_out_path
                    , logger=logger
            )
    
    for name_detector in ("lof","iforest","autoencoder","ensamble","ensamble_p_0_3"):

        if show_plots==True: print(":"*50, name_detector, ":"*50)

        if name_detector!='ensamble_p_0_3':
            # name_detector=='ensamble_p_0_3__prob' dose't have prob value it just filter values
            plot_anomalies_detected(
                    data=data
                    , name_detector=name_detector
                    , contamination=contamination
                    , show_title=show_title
                    , show_plot=show_plots
                    , local_out_path_plots=local_out_path
                    , s3_out_path_plots=s3_out_path
                    , logger=logger
            )
        
        plot_metrics(
                results=results
                , name_detector=name_detector
                , ncol=4
                , xsize=8
                , contamination=contamination
                , show_title=show_title
                , show_plot=show_plots
                , local_out_path_plots=local_out_path
                , s3_out_path_plots=s3_out_path
                , logger=logger
            )
        
        for metric in ["accuracy","mse","r2","precision","recall","roc_auc"]:
            plot_comparing_results(
                        results=results
                        , name_detector=name_detector
                        , metric=metric
                        , show_title=show_title
                        , show_plot=show_plots
                        , local_out_path_plots=local_out_path
                        , s3_out_path_plots=s3_out_path
                        , logger=logger
                )

    ## plot results
    if show_plots==True: print(":"*50, " RESULTS ", ":"*50)
    plot_results(
            results=results
            , show_title=show_title
            , show_plot=show_plots
            , local_out_path_plots=local_out_path
            , s3_out_path_plots=s3_out_path
            , logger=logger
        )

    return


def make_experiment(logger=None, **kwargs):

    response = pd.DataFrame()

    for i in range(kwargs["n_experiments"]):
        i = i+1
        params = kwargs.copy()
        params.update(config_experiment(**params))
        print(":"*50, i, params["dataset_name"], params['experiment'], ":"*50)
        if logger!=None: logger.info(f"loop = {i}")
        if logger!=None: logger.info(f"make_experiment: start runing experiment_id={params['experiment']}")
        ## data is the same df input with anomalies detected
        data, results = run_experiment(**params)
        results = make_result_experiment(results=results, **params)
        resp = save_results(results=results, **params)
        response = pd.concat([response, results])
        try:
            make_plots(data=data, results=results, **params)
        except Exception as e:
            if logger!=None: logger.error(f"make_plots: Error='{e}'")
            raise Exception(e)
    ## create table
    # result_table = "experiments_results"
    # wr.s3.store_parquet_metadata(path=)
    return data, results, params

