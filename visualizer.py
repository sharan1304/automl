import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_confusion(cm):
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    path = "confusion.png"
    plt.savefig(path)
    plt.close()
    return path

def plot_regression(y_test, preds):
    plt.figure()
    plt.scatter(y_test, preds)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    path = "regression.png"
    plt.savefig(path)
    plt.close()
    return path

def plot_residuals(y_test, preds):
    plt.figure()
    residuals = y_test - preds
    plt.scatter(preds, residuals)
    plt.axhline(y=0)
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    path = "residual.png"
    plt.savefig(path)
    plt.close()
    return path

def plot_feature_importance(model, X):
    estimator = model
    feature_names = X.columns

    if hasattr(model, "named_steps"):
        preprocessor = model.named_steps.get("preprocessor")
        estimator = model.named_steps.get("model")
        if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
            try:
                feature_names = preprocessor.get_feature_names_out()
            except Exception:
                feature_names = X.columns

    if hasattr(estimator, "feature_importances_"):
        imp = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        imp = np.abs(estimator.coef_)
        if imp.ndim > 1:
            imp = imp.mean(axis=0)
    else:
        return None

    if len(feature_names) != len(imp):
        feature_names = [f"feature_{i}" for i in range(len(imp))]

    df = pd.DataFrame({"Feature": feature_names, "Importance": imp})
    df = df.sort_values("Importance").tail(20)

    plt.figure(figsize=(8, max(4, len(df) * 0.35)))
    df.plot.barh(x="Feature", y="Importance", legend=False)
    plt.tight_layout()

    path = "importance.png"
    plt.savefig(path)
    plt.close()
    return path
