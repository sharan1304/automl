from sklearn.cluster import (
    AgglomerativeClustering,
    DBSCAN,
    KMeans,
    MiniBatchKMeans,
)
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
    RidgeClassifier,
)
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

def detect_problem(y):
    unique_count = y.nunique(dropna=True)
    unique_ratio = unique_count / max(len(y), 1)

    if y.dtype == 'object' or str(y.dtype) == "category" or y.dtype == 'bool':
        return "classification"
    if unique_count <= 15 and unique_ratio < 0.05:
        return "classification"
    return "regression"


def _optional_xgboost_models(problem_type):
    try:
        from xgboost import XGBClassifier, XGBRegressor
    except Exception:
        return {}

    if problem_type == "classification":
        return {
            "XGBoost": XGBClassifier(
                eval_metric="logloss",
                random_state=42,
                n_estimators=150,
                max_depth=4,
                learning_rate=0.08,
            )
        }

    return {
        "XGBoost": XGBRegressor(
            random_state=42,
            n_estimators=150,
            max_depth=4,
            learning_rate=0.08,
            objective="reg:squarederror",
        )
    }


def _optional_lightgbm_models(problem_type):
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
    except Exception:
        return {}

    if problem_type == "classification":
        return {
            "LightGBM": LGBMClassifier(
                random_state=42,
                n_estimators=150,
                learning_rate=0.08,
                verbose=-1,
            )
        }

    return {
        "LightGBM": LGBMRegressor(
            random_state=42,
            n_estimators=150,
            learning_rate=0.08,
            verbose=-1,
        )
    }


def get_models(problem_type):
    if problem_type == "classification":
        models = {
            "Logistic Regression": LogisticRegression(max_iter=2000),
            "Ridge Classifier": RidgeClassifier(),
            "Support Vector Machine": SVC(),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB(),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "Extra Trees": ExtraTreesClassifier(random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=42),
            "AdaBoost": AdaBoostClassifier(random_state=42),
            "Neural Network": MLPClassifier(max_iter=700, random_state=42),
        }
        models.update(_optional_xgboost_models(problem_type))
        models.update(_optional_lightgbm_models(problem_type))
        return models

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(max_iter=5000),
        "ElasticNet": ElasticNet(max_iter=5000),
        "Support Vector Regression": SVR(),
        "K-Nearest Neighbors": KNeighborsRegressor(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42),
        "Extra Trees": ExtraTreesRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(random_state=42),
        "AdaBoost": AdaBoostRegressor(random_state=42),
        "Neural Network": MLPRegressor(max_iter=700, random_state=42),
    }
    models.update(_optional_xgboost_models(problem_type))
    models.update(_optional_lightgbm_models(problem_type))
    return models


def get_tuning_params(model_name):
    grids = {
        "Random Forest": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 8, 16],
            "model__min_samples_split": [2, 5],
        },
        "Extra Trees": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 8, 16],
        },
        "Gradient Boosting": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3],
        },
        "Hist Gradient Boosting": {
            "model__learning_rate": [0.05, 0.1],
            "model__max_iter": [100, 200],
            "model__max_leaf_nodes": [15, 31],
        },
        "XGBoost": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [3, 5],
            "model__learning_rate": [0.05, 0.1],
        },
        "LightGBM": {
            "model__n_estimators": [100, 200],
            "model__num_leaves": [15, 31],
            "model__learning_rate": [0.05, 0.1],
        },
        "Support Vector Machine": {
            "model__C": [0.5, 1, 2],
            "model__kernel": ["rbf", "linear"],
        },
        "Support Vector Regression": {
            "model__C": [0.5, 1, 2],
            "model__kernel": ["rbf", "linear"],
        },
        "K-Nearest Neighbors": {
            "model__n_neighbors": [3, 5, 7],
            "model__weights": ["uniform", "distance"],
        },
    }
    return grids.get(model_name, {})


def get_unsupervised_models(n_clusters=3):
    return {
        "KMeans": KMeans(n_clusters=n_clusters, random_state=42, n_init="auto"),
        "MiniBatch KMeans": MiniBatchKMeans(n_clusters=n_clusters, random_state=42),
        "Agglomerative Clustering": AgglomerativeClustering(n_clusters=n_clusters),
        "DBSCAN": DBSCAN(),
        "Gaussian Mixture": GaussianMixture(n_components=n_clusters, random_state=42),
        "Isolation Forest": IsolationForest(random_state=42),
    }
