import time
start = time.time()

from lib_anomaly import logger
logger.info("starting run_experiment")

import pathlib
from lib_anomaly.params import get_params

params = get_params(
                path_current_file=pathlib.Path(__file__)
            )

print("*"*100)
print(f"{params=}")


from lib_anomaly.experiment import make_experiment


response = make_experiment(**params)

print(round((time.time()-start)/60,2),"min.")

## e.g.
# python .\experimentos\run_experiment.py --dataset_name="diabetes" --n_experiment=97 --show_plots=False --debug=False --make_plots=False
# python .\experimentos\run_experiment.py --dataset_name="credit_card" --n_experiment=10 --show_plots=False --debug=False --make_plots=False
# python .\experimentos\run_experiment.py --dataset_name=credit_risk --n_experiment=10 --contamination=0.01 --show_plots=False
# python .\experimentos\run_experiment.py --dataset_name=titanic --n_experiment=10 --contamination=0.01 --show_plots=False
