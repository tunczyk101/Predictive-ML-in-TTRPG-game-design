import gpflow
import lightgbm as lightgbm
import numpy as np
import optuna.integration.lightgbm as opt_lgb
import pandas as pd
import torch
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
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, LinearSVR
from skorch import NeuralNet
from spacecutter.callbacks import AscensionCallback
from spacecutter.models import OrdinalLogisticModel
from statsmodels.miscmodels.ordinal_model import OrderedModel

from training.constants import NUM_CLASSES, RANDOM_STATE
from training.losses import CondorLoss, WeightedBCELoss
from training.models.baseline import BaselineModel
from training.models.condor import Condor, CondorNeuralNet
from training.models.coral_corn import CORAL_MLP, DEVICE, Corn, SkorchCORAL
from training.models.gpor import GPOR
from training.models.nn_rank import NeuralNetNNRank, NNRank
from training.models.or_cnn import ORCNN, NeuralNetORCNN
from training.models.ordered_models import LinearOrdinalModel
from training.models.simple_ordinal_classification import SimpleOrdinalClassification
from training.models.spacecutter.losses import CumulativeLinkLoss
from training.models.spacecutter.models import (
    SpacecutterGridSearchCV,
    get_spacecutter_predictor,
)
from training.score_functions import (
    orf_mean_absolute_error,
    spacecutter_mean_absolute_error,
)


def get_fitted_model(
    classifier_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_features: int = 53,
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

    model = create_model(classifier_name, n_features, y_train)
    model.fit(X_train, y_train)

    return model


def create_model(classifier_name: str, n_features: int, y_train: np.ndarray):
    """
    Creates chosen model\n
    :param classifier_name: name of a chosen classifier:
            linear_regression or random_forest
    :return: chosen classifier
    """
    match classifier_name:
        case "baseline":
            model = BaselineModel()
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
            hyper_params = {"alpha": np.linspace(1e-3, 1, 1000)}

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
            knn = KNeighborsRegressor()

            hyper_params = {
                "leaf_size": list(range(50, 100, 10)),
                "weights": ["uniform", "distance"],
                "metric": ["minkowski", "manhattan", "euclidean"],
                "n_neighbors": list(range(1, 51)),
            }

            model = GridSearchCV(
                estimator=knn,
                param_grid=hyper_params,
                scoring="neg_mean_absolute_error",
                verbose=2,
                return_train_score=True,
                n_jobs=-1,
            )
        case "random_forest":
            rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
            hyper_params = {
                "max_features": ["sqrt", 0.3],
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
                "honesty": [False],
                "replace": [True],
            }
            model = GridSearchCV(
                estimator=rf,
                param_grid=hyper_params,
                scoring=make_scorer(orf_mean_absolute_error, greater_is_better=False),
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
        case "linear_ordinal_model_probit":
            model = create_linear_ordinal_model("probit")
        case "linear_ordinal_model_logit":
            model = create_linear_ordinal_model("logit")
        case "simple_or":
            hyper_params = {
                "max_features": ["sqrt", 0.3],
                "n_estimators": [100, 200, 500],
                "criterion": ["gini", "entropy"],
            }
            model = GridSearchCV(
                estimator=SimpleOrdinalClassification(),
                param_grid=hyper_params,
                scoring="neg_mean_absolute_error",
                return_train_score=True,
                n_jobs=-1,
            )
        case "gpor":
            hyper_params = {
                "maxiter": [100],
                "kernel": [gpflow.kernels.ArcCosine()],
            }
            model = GridSearchCV(
                estimator=GPOR(),
                param_grid=hyper_params,
                scoring="neg_mean_absolute_error",
                return_train_score=True,
                n_jobs=-1,
            )
        case "coral":
            hyper_params = {
                "optimizer__weight_decay": [1e-3, 1e-2, 1e-1, 1],
                "lr": [1e-3, 1e-2, 1e-1],
            }
            model = GridSearchCV(
                estimator=SkorchCORAL(
                    module=CORAL_MLP,
                    module__input_size=n_features,
                    module__num_classes=23,
                    max_epochs=50,
                    lr=0.05,
                    optimizer=torch.optim.AdamW,
                    iterator_train__shuffle=True,
                    device=DEVICE,
                ),
                param_grid=hyper_params,
                scoring="neg_mean_absolute_error",
                return_train_score=True,
                n_jobs=-1,
            )
        case "corn":
            hyper_params = {
                "lambda_reg": [1e-3, 1e-2, 1e-1],
                "learning_rate": [1e-3, 1e-2, 1e-1],
                # "input_size": [53]
            }
            model = GridSearchCV(
                estimator=Corn(input_size=n_features),
                param_grid=hyper_params,
                scoring="neg_mean_absolute_error",
                return_train_score=True,
                n_jobs=-1,
            )
        case "spacecutter":
            hyper_params = {
                "lr": [1e-3, 1e-2, 1e-1, 1],
                "optimizer__weight_decay": [1e-3, 1e-2, 1e-1, 1],
            }
            predictor = get_spacecutter_predictor(n_features)

            estimator = NeuralNet(
                module=OrdinalLogisticModel,
                module__predictor=predictor,
                module__num_classes=NUM_CLASSES,
                criterion=CumulativeLinkLoss,
                optimizer=torch.optim.AdamW,
                device=DEVICE,
                max_epochs=100,
                callbacks=[
                    ("ascension", AscensionCallback()),
                ],
            )

            model = SpacecutterGridSearchCV(
                estimator=estimator,
                param_grid=hyper_params,
                scoring=make_scorer(
                    spacecutter_mean_absolute_error,
                    greater_is_better=False,
                    needs_proba=True,
                ),
                return_train_score=True,
                n_jobs=-1,
            )
        case "nn_rank":
            hyper_params = {
                "optimizer__weight_decay": [1e-3, 1e-2, 1e-1, 1],
                "optimizer__lr": [1e-3, 1e-2, 1e-1],
            }
            model = GridSearchCV(
                estimator=NeuralNetNNRank(
                    module=NNRank,
                    module__input_size=n_features,
                    criterion=torch.nn.BCELoss,
                    optimizer=torch.optim.AdamW,
                    device=DEVICE,
                    max_epochs=100,
                ),
                param_grid=hyper_params,
                scoring="neg_mean_absolute_error",
                return_train_score=True,
                n_jobs=-1,
            )
        case "condor":
            hyper_params = {
                "optimizer__weight_decay": [1e-3, 1e-2, 1e-1, 1],
                "optimizer__lr": [1e-3, 1e-2, 1e-1],
            }
            model = GridSearchCV(
                estimator=CondorNeuralNet(
                    module=Condor,
                    module__input_size=n_features,
                    criterion=CondorLoss,
                    optimizer=torch.optim.AdamW,
                    device=DEVICE,
                    max_epochs=100,
                ),
                param_grid=hyper_params,
                scoring="neg_mean_absolute_error",
                return_train_score=True,
                n_jobs=-1,
            )
        case "or_cnn":
            hyper_params = {
                "optimizer__weight_decay": [1e-3, 1e-2, 1e-1, 1],
                "optimizer__lr": [1e-3, 1e-2, 1e-1],
            }
            model = GridSearchCV(
                estimator=NeuralNetORCNN(
                    module=ORCNN,
                    module__input_size=n_features,
                    criterion__y_train=y_train,
                    criterion=WeightedBCELoss,
                    optimizer=torch.optim.AdamW,
                    device=DEVICE,
                    max_epochs=100,
                ),
                param_grid=hyper_params,
                scoring="neg_mean_absolute_error",
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


def create_linear_ordinal_model(distr: str) -> GridSearchCV:
    model = LinearOrdinalModel(distr=distr)
    hyper_params = {"offset": np.linspace(0.25, 1.25, 11)}
    model = GridSearchCV(
        estimator=model,
        param_grid=hyper_params,
        scoring="neg_mean_absolute_error",
        return_train_score=True,
        n_jobs=-1,
        verbose=10,
    )

    return model
