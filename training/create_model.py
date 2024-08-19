import lightgbm as lightgbm
import numpy as np
import optuna.integration.lightgbm as opt_lgb
import pandas as pd
from lightgbm import early_stopping, log_evaluation
from mord import LogisticAT, LogisticIT
from orf import OrderedForest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import (
    HuberRegressor,
    LassoCV,
    LinearRegression,
    QuantileRegressor,
    RidgeCV,
)
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, LinearSVR
from statsmodels.miscmodels.ordinal_model import OrderedModel

from training.constants import RANDOM_STATE


def get_fitted_model(
    classifier_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RidgeCV | GridSearchCV | lightgbm.Booster | OrderedModel:
    """
    Creates chosen model, performs tuning and fits\n
    :param X_train: train set with features to use during fitting
    :param y_train: train set with values to predict
    :param classifier_name: name of a chosen classifier:
            linear_regression or random_forest
    :return: trained classifier of a chosen type
    """
    if classifier_name == "lightgbm":
        return lightgbm_fit(X_train, y_train)

    if "ordered_model" in classifier_name:
        return fit_ordered_model(classifier_name, X_train, y_train)

    model = create_model(classifier_name)
    model.fit(X_train, y_train)

    return model


def create_model(classifier_name: str):
    """
    Creates chosen model\n
    :param classifier_name: name of a chosen classifier:
            linear_regression or random_forest
    :return: chosen classifier
    """
    match classifier_name:
        case "linear_regression":
            model = LinearRegression()
        case "linear_regression_ridge":
            model = RidgeCV(alphas=np.linspace(1e-3, 1, 10000))
        case "linear_regression_lasso":
            model = LassoCV(n_alphas=1000, random_state=0)
        case "lad_regression":
            hyper_params = [{"alpha": np.linspace(0.0, 1e-3, 100)}]

            reg_lad = QuantileRegressor(quantile=0.5, solver="highs")

            model = GridSearchCV(
                estimator=reg_lad,
                param_grid=hyper_params,
                scoring="neg_mean_absolute_error",
                verbose=2,
                return_train_score=True,
                n_jobs=-1,
            )
        case "huber_regression":
            huber = HuberRegressor(max_iter=1000)
            hyper_params = {"alpha": np.linspace(1e-3, 1, 10000)}

            model = GridSearchCV(
                estimator=huber,
                param_grid=hyper_params,
                scoring="neg_mean_absolute_error",
                verbose=2,
                return_train_score=True,
                n_jobs=-1,
            )
        case "linear_svm":
            clf_linear_svr = LinearSVR(
                loss="epsilon_insensitive", max_iter=10000, random_state=0
            )
            hyper_params = {"C": np.linspace(10, 30, num=20)}

            model = GridSearchCV(
                estimator=clf_linear_svr,
                param_grid=hyper_params,
                scoring="neg_mean_absolute_error",
                verbose=2,
                return_train_score=True,
                n_jobs=-1,
            )
        case "kernel_svm":
            svm = SVR(kernel="rbf", max_iter=10000)
            hyper_params = {"C": np.linspace(1, 10, num=100)}

            model = GridSearchCV(
                estimator=svm,
                param_grid=hyper_params,
                scoring="neg_mean_absolute_error",
                verbose=2,
                return_train_score=True,
                n_jobs=-1,
            )
        case "knn":
            logistic_model = KNeighborsRegressor()

            hyper_params = {
                "leaf_size": list(range(50, 100, 10)),
                "weights": ["uniform", "distance"],
                "metric": ["minkowski", "manhattan", "euclidean"],
                "n_neighbors": list(range(1, 51)),
            }

            model = GridSearchCV(
                estimator=logistic_model,
                param_grid=hyper_params,
                scoring="neg_mean_absolute_error",
                verbose=2,
                return_train_score=True,
                n_jobs=-1,
            )
        case "random_forest":
            rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
            hyper_params = {
                "max_features": [0.3],
                "n_estimators": [100, 200, 500],
                "criterion": ["squared_error", "absolute_error", "friedman_mse"],
            }
            model = GridSearchCV(
                estimator=rf,
                param_grid=hyper_params,
                scoring="neg_mean_absolute_error",
                return_train_score=True,
                n_jobs=-1,
            )
        case "ordered_random_forest":
            rf = OrderedForest(random_state=RANDOM_STATE, n_jobs=-1)
            hyper_params = {
                "max_features": [0.3],
                "min_samples_leaf": [i for i in range(2, 8)],
                "n_estimators": [100, 200, 500],
                # "honesty": [False],
                # "replace": [True]
            }
            model = GridSearchCV(
                estimator=rf,
                param_grid=hyper_params,
                scoring="neg_mean_absolute_error",
                return_train_score=True,
                n_jobs=-1,
            )
        case "logisticAT":
            hyper_params = [{"alpha": np.linspace(0.0, 1e-3, 100)}]

            logistic_model = LogisticAT()

            model = GridSearchCV(
                estimator=logistic_model,
                param_grid=hyper_params,
                scoring="neg_mean_absolute_error",
                verbose=2,
                return_train_score=True,
                n_jobs=-1,
            )
        case "logisticIT":
            hyper_params = [{"alpha": np.linspace(0.0, 1e-3, 100)}]

            logistic_model = LogisticIT()

            model = GridSearchCV(
                estimator=logistic_model,
                param_grid=hyper_params,
                scoring="neg_mean_absolute_error",
                verbose=2,
                return_train_score=True,
                n_jobs=-1,
            )
        case _:
            raise ValueError(f"Classifier {classifier_name} is unsupported")

    return model


def lightgbm_fit(X_train, y_train) -> lightgbm.Booster:
    """
    Performs tuning and fits lightgbm model\n
    :param X_train: train set with features to use during fitting
    :param y_train: train set with values to predict
    :return: trained lightgbm
    """
    lgb_train = opt_lgb.Dataset(X_train, y_train)
    params = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": "l2",
        "verbosity": -1,
    }
    tuner = opt_lgb.LightGBMTunerCV(
        params,
        lgb_train,
        folds=KFold(n_splits=5),
        num_boost_round=10000,
        callbacks=[early_stopping(100), log_evaluation(100)],
    )
    tuner.run()
    best_params = tuner.best_params

    lgb_tuned = lightgbm.train(
        best_params,
        lgb_train,
        num_boost_round=10000,
    )
    return lgb_tuned


def fit_ordered_model(
    model_name: str, X_train: pd.DataFrame, y_train: pd.Series
) -> OrderedModel:
    model_params = model_name.split("ordered_model_")[1]
    distr, method = model_params.split("_")

    model = OrderedModel(y_train, X_train, distr=distr)
    model = model.fit(method=method)

    return model
