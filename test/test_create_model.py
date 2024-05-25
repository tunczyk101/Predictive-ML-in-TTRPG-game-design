import lightgbm as lightgbm
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import (
    HuberRegressor,
    LassoCV,
    LinearRegression,
    QuantileRegressor,
    RidgeCV,
)
from sklearn.svm import LinearSVR

from training.create_model import create_model, get_fitted_model


@pytest.fixture
def train_set() -> tuple[pd.DataFrame, pd.Series]:
    n = 10
    data = {
        "cha": [10 for _ in range(n)],
        "con": [10 for _ in range(n)],
        "dex": [10 for _ in range(n)],
    }

    X_train = pd.DataFrame(data=data)
    y_train = pd.Series(data=[10 for _ in range(n)])

    return X_train, y_train


def test_create_model(train_set):
    models_to_test = {
        "linear_regression": LinearRegression,
        "linear_regression_ridge": RidgeCV,
        "linear_regression_lasso": LassoCV,
    }

    X_train, y_train = train_set

    for name, model_type in models_to_test.items():
        model = get_fitted_model(name, X_train, y_train)
        assert type(model) == model_type


def test_create_model_gridsearch(train_set):
    models_to_test = {
        "lad_regression": QuantileRegressor,
        "huber_regression": HuberRegressor,
        "linear_svm": LinearSVR,
        "random_forest": RandomForestRegressor,
    }

    X_train, y_train = train_set

    for name, model_type in models_to_test.items():
        model = get_fitted_model(name, X_train, y_train)
        assert type(model.best_estimator_) == model_type


def test_create_lightgbm(train_set):
    X_train, y_train = train_set
    model = get_fitted_model("lightgbm", X_train, y_train)

    assert isinstance(model, lightgbm.Booster)


def test_wrong_classifier_name():
    with pytest.raises(ValueError):
        model = create_model("wrong_classifier_name")
