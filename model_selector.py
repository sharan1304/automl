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
    if y.dtype == 'object':
        return "classification"
    if y.nunique() < 10:
        return "classification"
    return "regression"

def get_models(problem_type):
    if problem_type == "classification":
        return {
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

    return {
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


def get_unsupervised_models(n_clusters=3):
    return {
        "KMeans": KMeans(n_clusters=n_clusters, random_state=42, n_init="auto"),
        "MiniBatch KMeans": MiniBatchKMeans(n_clusters=n_clusters, random_state=42),
        "Agglomerative Clustering": AgglomerativeClustering(n_clusters=n_clusters),
        "DBSCAN": DBSCAN(),
        "Gaussian Mixture": GaussianMixture(n_components=n_clusters, random_state=42),
        "Isolation Forest": IsolationForest(random_state=42),
    }
