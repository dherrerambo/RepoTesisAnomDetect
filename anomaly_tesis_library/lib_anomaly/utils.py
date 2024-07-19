import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import seaborn as sns
import awswrangler as wr
import pickle, os, datetime, re, unidecode

from sklearn.preprocessing import MinMaxScaler


def normalize_str(S:str):
    """
        S = "modedeD##D#D#_D#D#_#-3#DG%y67g."
    """
    S = ' '.join(re.findall('[a-z0-9 ]+',unidecode.unidecode(S).lower().strip())).replace('  ',' ').strip().replace(' ','_')
    return S


def normalize_data(data:pd.DataFrame, cols:list=None, vmin:int=0, vmax:int=1, show_prints:bool=False, **kwargs):
    """
        normaliza un dataset de acuerdo a unos parametros
    """
    dataN = data.copy()
    if cols==None:
        cols = dataN.columns
    dataN[cols] = MinMaxScaler((vmin,vmax)).fit_transform(dataN[cols])
    if show_prints==True: print(dataN.describe().T.to_string())
    return dataN

def delete_empty_col(df:pd.DataFrame):
    cols_nulls = df.columns[(df.isnull().sum()==len(df)).values]
    return df.drop(cols_nulls, axis=1)


def load_dataset(dataset_name:str, random_state:float=None, s3path:str=None, logger=None, **kwargs):
    """
        Carga un dataset de acuerdo a un nombre dado
        La convencion del dataset es:
            target: las primeras 2 columnas
            features: el resto de columnas
    """

    try:
        ROOT_PATH = os.environ["ROOT_PATH"]
        file_path = os.path.join(ROOT_PATH, "experimentos","datasets")
        if logger!=None: logger.info(f"defining file_path for loading {dataset_name=}")
    except:
        if logger!=None: logger.info(f"error defining file_path for loading {dataset_name=}")
        pass

    if dataset_name=="iris":
        target = "species"
        try:
            data = wr.s3.read_parquet(path=f"{s3path}{dataset_name}.parquet")
        except:
            data = sns.load_dataset(dataset_name)
            data[target+"Label"] = data[target]
            data[target] = data[target].map(dict(zip(set(data[target]), range(len(set(data[target]))))))
            data = wr.s3.to_parquet(path=f"{s3path}{dataset_name}.parquet", index=False)
            if logger!=None: logger.info(f"{dataset_name=} send to aws")

    elif dataset_name=="breast_cancer":
        target = "diagnosis"
        try:
            data = wr.s3.read_parquet(path=f"{s3path}{dataset_name}.parquet")
        except:
            from sklearn.datasets import load_breast_cancer
            bc = load_breast_cancer()
            data = pd.DataFrame(bc['data'], columns=bc['feature_names'])
            cols = [normalize_str(c) for c in data]
            data.columns = cols
            data[target] = bc['target']
            data = data[cols+[target]]
            data[target+"Label"] = data[target].map(dict(enumerate(bc['target_names'])))
            data = pd.concat([data[[target,target+"Label"]], data[cols]], axis=1)
            data = wr.s3.to_parquet(path=f"{s3path}{dataset_name}.parquet", index=False)
            if logger!=None: logger.info(f"{dataset_name=} send to aws")
    
    elif dataset_name=="titanic":
        target = "survived"
        try:
            data = wr.s3.read_parquet(path=f"{s3path}{dataset_name}.parquet")
        except:
            data = sns.load_dataset(dataset_name)
            data = data.rename(columns={"alive":target+"Label"})
            data = pd.concat([data[[target, target+"Label"]], data.drop([target, target+"Label"],axis=1)], axis=1)
            data = wr.s3.to_parquet(path=f"{s3path}{dataset_name}.parquet", index=False)
            if logger!=None: logger.info(f"{dataset_name=} send to aws")


    elif dataset_name=="credit_card":
        target = "fraud"
        try:
            data = wr.s3.read_parquet(path=f"{s3path}{dataset_name}.parquet")
        except:
            fName = os.path.join(file_path,"creditcard")
            data = pd.read_csv(fName+".csv.gz")
            data.rename(columns={"Class":"fraud"}, inplace=True)
            data[target+"Label"] = data[target].map({0:"No",1:"Yes"})
            cols = [target, target+"Label"]
            data = data[cols + [c for c in data if c not in cols]]
            data = wr.s3.to_parquet(path=f"{s3path}{dataset_name}.parquet", index=False)
            if logger!=None: logger.info(f"{dataset_name=} send to aws")

    
    elif dataset_name=="heart_disease":
        target = "target"
        try:
            data = wr.s3.read_parquet(path=f"{s3path}{dataset_name}.parquet")
        except:
            data = pd.read_csv(file_path+"\heart_disease.csv")
            data[target+"Label"] = data[target].map({0:"Normal",1:"Heart Disease"})
            cols = [target,target+"Label"]
            data = data[cols + [c for c in data if c not in cols]]
            data = wr.s3.to_parquet(path=f"{s3path}{dataset_name}.parquet", index=False)
            if logger!=None: logger.info(f"{dataset_name=} send to aws")


    elif dataset_name=="heart_faliure":
        target = "HeartDisease"
        try:
            data = wr.s3.read_parquet(path=f"{s3path}{dataset_name}.parquet")
        except:
            data = pd.read_csv(file_path+"\heart_faliure.csv")
            data[target+"Label"] = data[target].map({0:"Normal",1:"Heart Disease"})
            cols = [target,target+"Label"]
            data = data[cols + [c for c in data if c not in cols]]
            data = wr.s3.to_parquet(path=f"{s3path}{dataset_name}.parquet", index=False)
            if logger!=None: logger.info(f"{dataset_name=} send to aws")

    elif dataset_name=="ecg":
        target = "Class"
        try:
            data = wr.s3.read_parquet(path=f"{s3path}{dataset_name}.parquet")
        except:
            data = pd.concat([
                    pd.read_csv(file_path+"ptbdb_normal.csv", header=None).assign(Class=0)
                    , pd.read_csv(file_path+"ptbdb_abnormal.csv", header=None).assign(Class=1)
            ]).reset_index(drop=True)
            cols = ["C_"+str(c).zfill(3) for c in data.columns if c!=target]+ [target]
            data.columns = cols
            data[target+"Label"] = data[target].map({0:"Normal", 1:"Anormal"})
            cols = list(data.drop([target, target+"Label"], axis=1).columns)
            data = data[[target, target+"Label"] + cols]
            data = data.rename(columns=dict(zip(cols, [f"V{c}" for c in cols])))
            data = wr.s3.to_parquet(path=f"{s3path}{dataset_name}.parquet", index=False)
            if logger!=None: logger.info(f"{dataset_name=} send to aws")

    elif dataset_name=="diabetes":
        target = "diabetes"
        try:
            data = wr.s3.read_parquet(path=f"{s3path}{dataset_name}.parquet")
        except:
            fName = os.path.join(file_path, "diabetes_prediction_dataset.csv")
            data = pd.read_csv(fName)
            data[target+"Label"] = data[target].map({0:"No", 1:"Yes"})
            cols = list(data.drop([target, target+"Label"], axis=1).columns)
            data = data[[target, target+"Label"] + cols]
            data = wr.s3.to_parquet(path=f"{s3path}{dataset_name}.parquet", index=False)
            if logger!=None: logger.info(f"{dataset_name=} send to aws")
    

    elif dataset_name=="air_quality":
        target = "air_quality"
        try:
            data = wr.s3.read_parquet(path=f"{s3path}{dataset_name}.parquet")
        except:
            fName = os.path.join(file_path, "AQI and Lat Long of Countries.csv")
            data = pd.read_csv(fName)
            data[target] = (data["AQI Category"]=="Good").map({True:1, False:0})
            data[target+"Label"] = data[target].map({0:"Bad", 1:"Good"})
            cols = list(data.drop([target, target+"Label", "AQI Category"], axis=1).columns)
            data = data[[target, target+"Label"] + cols]
            data = wr.s3.to_parquet(path=f"{s3path}{dataset_name}.parquet", index=False)
            if logger!=None: logger.info(f"{dataset_name=} send to aws")

    elif dataset_name=="credit_risk":
        target = "class"
        try:
            data = wr.s3.read_parquet(path=f"{s3path}{dataset_name}.parquet")
        except:
            fName = os.path.join(file_path, "credit_customers.csv")
            data = pd.read_csv(fName)
            data[target+"Label"] = data[target]
            data[target] = data[target].map({"good":0, "bad":1})
            cols = list(data.drop([target, target+"Label"], axis=1).columns)
            data = data[[target, target+"Label"] + cols]
            data = wr.s3.to_parquet(path=f"{s3path}{dataset_name}.parquet", index=False)
            if logger!=None: logger.info(f"{dataset_name=} send to aws")

    elif dataset_name=="customer_churn":
        target = "Exited"
        try:
            data = wr.s3.read_parquet(path=f"{s3path}{dataset_name}.parquet")
        except:
            fName = os.path.join(file_path, "Customer-Churn-Records.csv")
            data = pd.read_csv(fName)
            data[target+"Label"] = data[target].map({1:"Yes", 0:"No"})
            cols = list(data.drop([target, target+"Label", "RowNumber"], axis=1).columns)
            data = data[[target, target+"Label"] + cols]
            data = wr.s3.to_parquet(path=f"{s3path}{dataset_name}.parquet", index=False)
            if logger!=None: logger.info(f"{dataset_name=} send to aws")

    else:
        e = f"No dataset loaded\nwrong {dataset_name=}"
        raise Exception(e)

    col_bool = data.select_dtypes("bool").columns
    data[col_bool] = data[col_bool].astype(int)
    data = data.replace({pd.NA: None})
    data[target] = data[target].astype(int)

    ## shuffle
    data.columns = [c.strip().lower().replace(' ','_') for c in data]
    target = target.lower()
    data.rename(columns={target+"label": target+"Label"}, inplace=True)
    data = data.drop_duplicates().reset_index(drop=True)
    
    if random_state!=None:
        data = data.sample(frac=1, random_state=random_state)
    else:
        data = data.sample(frac=1)

    features = list(data.columns)[2:]

    if logger!=None: logger.info(f"{dataset_name=} {target=} features={len(features)} {data.shape=}")

    return data, target, features


## pickle
def save_pickle(obj, path:str, name:str, **kwargs):
    if path.endswith("/"): path = path[:-1]
    fName = f"{path}/{name}.pickle"""
    with open(fName, "wb") as fo:
        pickle.dump(obj, fo)
    return fName

def load_pickle(fName:str, **kwargs):
    with open(fName, "rb") as fo:
        obj = pickle.load(fo)
    return obj



## save
def save_plot(fig, local_out_path_plots:str, fName:str, s3_out_path_plots:str, file_formats:list=["png"], show_title:bool=True, logger=None, **kwargs):
    response = dict()
    fName = normalize_str(fName)
    if show_title==False: fName = f"{fName}__no_title"
    for ext in file_formats:        # ["png","eps"]
        fout = os.path.join(local_out_path_plots, f"{fName}.{ext}")
        if logger!=None: logger.info(f"{fout=}")
        fig.savefig(fout, dpi=300)
        response["fout"] = fout
        if logger!=None: logger.info(f"saved local={fout}")
        if s3_out_path_plots!=None:
            s3path = upload_file_aws(local_file=fout, s3_out_path=s3_out_path_plots)
            response["s3path"] = s3path
    return response


## aws
def upload_file_aws(local_file:str, s3_out_path:str, logger=None, **kwargs):
    fName = os.path.basename(local_file)
    if s3_out_path.endswith("/"): s3_out_path = s3_out_path[:-1]
    s3path = f"{s3_out_path}/{fName}"
    while wr.s3.does_object_exist(s3path):
        fName = f'{fName.split(".")[0]}__{datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")}.{fName.split(".")[1]}'
        s3path = f"{s3_out_path}/{fName}"
    response = wr.s3.upload(local_file=local_file, path=s3path)
    if logger!=None: logger.info(s3path)
    return s3path


def get_data_types(data:pd.DataFrame, features:list=None):
    if features==None:
        features = list(data.columns)
    else:
        if len(features)==0:
            features = list(data.columns)
        else:
            # use features
            pass
    # intentar convertir a numerico
    _cols_cat = list(data[features].select_dtypes(["object","category","string"]).columns)
    for c in _cols_cat:
        try:
            data[c] = pd.to_numeric(data[c])        ## intentar tranformar las columnas que sean categoricas a numericas
        except:
            pass
    ## definir los que sean categoricos
    _cols_cat = list(data[features].select_dtypes(["object","category","string"]).columns)
    ## distribucion de valores por cada variable categorica
    unique_values = data[_cols_cat].apply(lambda a: len(a.unique()))
    _cols_cat_high = list(unique_values[unique_values>30].index)
    _cols_cat_low = [c for c in _cols_cat if c not in _cols_cat_high]
    _cols_num = [c for c in features if c not in _cols_cat]

    return data, _cols_num, _cols_cat_high, _cols_cat_low


