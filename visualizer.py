import matplotlib.pyplot as plt
import seaborn as sns
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
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        df = pd.DataFrame({"Feature": X.columns, "Importance": imp})

        plt.figure()
        df.sort_values("Importance").plot.barh(x="Feature", y="Importance")

        path = "importance.png"
        plt.savefig(path)
        plt.close()
        return path
    return None