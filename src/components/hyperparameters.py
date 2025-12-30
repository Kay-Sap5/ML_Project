from catboost import CatBoostRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor
)
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor

models = {
        "LinearRegression":LinearRegression(),
        "XGBoostRegressor":XGBRegressor(),
        "DecisionTreeRegressor":DecisionTreeRegressor(),
        "RandomForestRegressor":RandomForestRegressor(),
        "AdaBoostRegressor":AdaBoostRegressor(),
        "GradientBoostingRegressor":GradientBoostingRegressor()}

param_grids = {

    "LinearRegression": {
        "fit_intercept": [True, False],
        "positive": [True, False]
    },

    "XGBoostRegressor": {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    },

    "DecisionTreeRegressor": {
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5]
    },

    "RandomForestRegressor": {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "bootstrap": [True, False]
    },

    "AdaBoostRegressor": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.05, 0.1, 1.0]
    },

    "GradientBoostingRegressor": {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5],
        "subsample": [0.8, 1.0]
    },
}
