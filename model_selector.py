from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

def detect_problem(y):
    if y.dtype == 'object':
        return "classification"
    if y.nunique() < 10:
        return "classification"
    return "regression"

def get_models(problem_type):
    if problem_type == "classification":
        return {
            "Random Forest": RandomForestClassifier(),
            "Logistic Regression": LogisticRegression(max_iter=1000)
        }
    else:
        return {
            "Random Forest": RandomForestRegressor(),
            "Linear Regression": LinearRegression()
        }