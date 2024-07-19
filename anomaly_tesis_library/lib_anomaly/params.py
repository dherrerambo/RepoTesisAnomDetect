import os, pathlib, argparse
from dotenv import load_dotenv

from lib_anomaly.exceptions_handler import NoDataset, EmptyParams
# from exceptions_handler import NoDataset


## argumentos Constantes
args = [
        ("dataset_name","str",None)
        , ("scoring_metric","str","roc_auc")
        , ("anomaly_threshold","float",0.3)
        , ("random_state","int",None)
        , ("contamination","float",0.1)
        , ("make_plots","str",True)     ## definido como str
        , ("show_plots","str",True)     ## definido como str
        , ("show_title","str",True)     ## definido como str
        , ("n_experiments","int",1)
        , ("debug","str",False)         ## definido como str
    ]

clf_order = ["LR","SVC","KNN","DT","RF","AB"]



def __get_env():
    try:
        PROJECT_NAME = os.environ["PROJECT_NAME"]
        S3BUCKET = os.environ["S3BUCKET"]
    except:
        resp = load_dotenv()
        if resp==True:
            PROJECT_NAME = os.environ["PROJECT_NAME"]
            S3BUCKET = os.environ["S3BUCKET"]
        else:
            resp = load_dotenv(".env")
            PROJECT_NAME = os.environ["PROJECT_NAME"]
            S3BUCKET = os.environ["S3BUCKET"]
    return {"PROJECT_NAME":PROJECT_NAME, "S3BUCKET":S3BUCKET}


def __get_root_path(PROJECT_NAME:str, path_current_file:pathlib.Path, debug:bool=False):
    ROOT_PATH = path_current_file
    while ROOT_PATH.name!=PROJECT_NAME:
        if debug==True: print(ROOT_PATH)
        ROOT_PATH = ROOT_PATH.parent
    if ROOT_PATH=='':
        raise Exception('No ROOT_PATH founded.')
    if debug==True: print(ROOT_PATH)
    os.environ["ROOT_PATH"] = str(ROOT_PATH)
    return ROOT_PATH


def __get_params_from_argparse():
    parser = argparse.ArgumentParser(description='Run experiments')
    for var,var_type,_ in args:
        eval(f"""parser.add_argument(f"--{var}", type={var_type})""")
    args_parser = parser.parse_args()
    params = dict((k,v=='True' if v in ('True','False') else v) for k,v in vars(args_parser).items() if v!=None)
    if params["dataset_name"]==None:
        raise NoDataset
    return params


def get_params(path_current_file:pathlib.Path, logger=None, **kwargs):
    try:
        debug = kwargs["debug"]
    except:
        debug = False
    params = dict()
    # from argsparse
    try:
        env = get_ipython().__class__.__name__
    except Exception as e:
        params = __get_params_from_argparse()
        if logger!=None: logger.info("loaded from argparse")
    if len(kwargs)>0:
        params.update(kwargs)                                                       ## add kwargs
        if logger!=None: logger.info("updated with kwargs")
    default_args = dict((a[0],a[2]) for a in args if a[0] not in params)
    if len(default_args)>0:
        params.update(default_args)        ## add default values
        if logger!=None: logger.info("updated with default args")
    ## get PROJECT_NAME and S3BUCKET
    env_args = __get_env()
    if len(env_args)>0:
        params.update(env_args)
        if logger!=None: logger.info("updated with env_args")
    ## get ROOT_PATH
    ROOT_PATH = __get_root_path(
                        PROJECT_NAME=params["PROJECT_NAME"]
                        , path_current_file=path_current_file
                        , debug=debug
                    )
    params.update({"ROOT_PATH": ROOT_PATH})
    if logger!=None: logger.info(f"updated with {ROOT_PATH=}")
    ## complete arguments
    double_check = dict((a[0],a[2]) for a in args if a[0] not in params)
    params.update(double_check)
    return params
