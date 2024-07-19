# Guia para correr los experimentos

Es necesario que la libreria se encuentre instalada o en su defecto si se usa local, se debe indicar la ruta para cargarla.

Los argumentos que se le pueden pasar al experimento son:

- dataset_name:str
    - iris
    - breast_cancer
    - titanic
    - credit_card
    - heart_disease
    - heart_faliure
    - ecg
    - diabetes
    - air_quality
    - credit_risk
    - customer_churn
- scoring_metric:str
    - accuracy
    - mse (mean square error)
    - precision
    - r2
    - recall
    - roc_auc (default)
- anomaly_threshold:float
    - default=***0.3***
- random_state:float
- contamination:float
    - default = ***0.1***
- make_plots:bool=***True***
- show_plots:bool=***True***
- debug:bool=***False***

Como se debe usar
~~~ powershell
cd experiments
python run_experiment.py --dataset_name="credit_risk" --n_experiment=100
python run_experiment.py --dataset_name="diabetes" --n_experiment=100
python run_experiment.py --dataset_name="credit_card" --n_experiment=100
python run_experiment.py --dataset_name="breast_cancer" --n_experiment=100
python run_experiment.py --dataset_name="heart_faliure" --n_experiment=100
python run_experiment.py --dataset_name="titanic" --n_experiment=100
~~~


Sin crear las imagenes y generando respuestas
~~~ powershell
python run_experiment.py --dataset_name="heart_faliure" --n_experiment=100 --make_plots=False debug=True
~~~
