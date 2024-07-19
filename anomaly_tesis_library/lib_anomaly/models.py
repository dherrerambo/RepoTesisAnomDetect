from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC

from sklearn.metrics            import classification_report, mean_squared_error, r2_score, accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection    import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold
from tqdm import tqdm




def load_classifiers_models():
    return {
        "LR": LogisticRegression()
        , "SVC": SVC()
        , "KNN": KNeighborsClassifier()
        , "DT": DecisionTreeClassifier()
        , "RF": RandomForestClassifier()
        , "AB": AdaBoostClassifier()
    }


def get_metrics_report(y_true, y_pred, show_prints:bool=False, logger=None, **kwargs):
    metrics = dict()
    for op in ["classification_report", "mean_squared_error", "r2_score", "accuracy_score", "roc_auc_score", "precision_score", "recall_score"]:
        if op=="classification_report":
            if show_prints==True: print(op,":\n", eval(f"{op}(y_true, y_pred)"))
            if show_prints==True: print("*"*50)
        else:
            metrics[op] = eval(f"{op}(y_true, y_pred)")
            if show_prints==True: print(f"{op}: {metrics[op]}")
    return metrics


def run_clasification_models(x_train, y_train, x_test, y_test, scoring_metric:str, random_state:float, logger=None, **kwargs):
    """
        Test clasification models by scoring metric
    """
    results = dict()
    classifiers = load_classifiers_models()
    try:
        pBar = tqdm(classifiers.items())
    except:
        pBar = classifiers.items()
    for name,clf in pBar:
        try:
            pBar.set_description(name)
        except:
            pass
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
        score = cross_val_score(
                                estimator=clf
                                , X=x_train
                                , y=y_train
                                , scoring=scoring_metric
                                , cv=cv
                            )
        if logger!=None: logger.info(f"clf={name}, cv_score")
        # if show_prints==True: print(f"cv_scores={score}")
        ## train
        y_pred_train = cross_val_predict(
                                estimator=clf
                                , X=x_train
                                , y=y_train
                                , cv=cv
                    )
        if logger!=None: logger.info(f"clf={name}, y_predict for train dataset")
        ## test
        y_pred_test = cross_val_predict(
                                estimator=clf
                                , X=x_test
                                , y=y_test
                                , cv=cv
                    )
        if logger!=None: logger.info(f"clf={name}, y_predict for test dataset")
        cla_report_train = get_metrics_report(y_true=y_train, y_pred=y_pred_train)
        cla_report = dict((k.replace("mean_squared_error","mse_train").replace("score","train"),v) for k,v in cla_report_train.items())
        # cla_report.update({f"cv_{scoring_metric}_mean":score.mean(),f"cv_{scoring_metric}_std":score.std()})
        cla_report_test = get_metrics_report(y_true=y_test, y_pred=y_pred_test)
        cla_report_test = dict((k.replace("mean_squared_error","mse_test").replace("score","test"),v) for k,v in cla_report_test.items())
        cla_report.update(cla_report_test)
        cla_report = dict((k,cla_report[k]) for k in sorted(cla_report.keys()))
        results[name] = cla_report
        if logger!=None: logger.info(f"clf={name}, metrics collected")
    return results