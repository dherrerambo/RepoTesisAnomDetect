import os, sys, logging
from datetime import datetime


def getLogger(name:str=None, show:bool=False):
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    if name==None:
        name = datetime.now().strftime("%Y%m%d%H%M%S%f")
    log_filepath = os.path.join(log_dir,f"{name}.log")
    logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(funcName)s: %(message)s]"
    if show:
        logging.basicConfig(
                level= logging.INFO
                , format=logging_str
                , handlers=[
                        logging.FileHandler(log_filepath, encoding="utf8")
                        , logging.StreamHandler(sys.stdout)       ## imprime los losg en consola
                    ]
            )
    else:
        logging.basicConfig(
                level= logging.INFO
                , format=logging_str
                , handlers=[logging.FileHandler(log_filepath, encoding="utf8")]
            )
    return logging.getLogger(name=name)
