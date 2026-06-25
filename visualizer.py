import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd
import uuid

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


def plot_shap_explanations(model, X, max_samples=100):
    try:
        import shap
    except Exception:
        return None, None

    if not hasattr(model, "named_steps"):
        return None, None

    preprocessor = model.named_steps.get("preprocessor")
    estimator = model.named_steps.get("model")
    if preprocessor is None or estimator is None:
        return None, None

    try:
        sample_size = min(len(X), max_samples)
        X_sample = X.head(sample_size)
        transformed = preprocessor.transform(X_sample)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()

        try:
            feature_names = preprocessor.get_feature_names_out()
        except Exception:
            feature_names = [f"feature_{i}" for i in range(transformed.shape[1])]

        transformed_df = pd.DataFrame(transformed, columns=feature_names)
        transformed_df = transformed_df.apply(pd.to_numeric, errors="coerce").fillna(0)

        predicted_class_index = None
        if hasattr(estimator, "predict") and hasattr(estimator, "classes_"):
            first_prediction = estimator.predict(np.asarray(transformed_df.iloc[[0]]))[0]
            class_matches = np.where(estimator.classes_ == first_prediction)[0]
            if len(class_matches):
                predicted_class_index = int(class_matches[0])

        try:
            if hasattr(estimator, "feature_importances_"):
                explainer = shap.TreeExplainer(estimator)
                shap_values = explainer(transformed_df)
            else:
                background = transformed_df.head(min(30, sample_size))
                explain_rows = transformed_df.head(min(50, sample_size))
                if hasattr(estimator, "predict_proba"):
                    def predict_fn(data):
                        return estimator.predict_proba(np.asarray(data))
                else:
                    def predict_fn(data):
                        return estimator.predict(np.asarray(data))

                explainer = shap.Explainer(predict_fn, background)
                shap_values = explainer(explain_rows)
                transformed_df = explain_rows
        except Exception:
            background = transformed_df.head(min(30, sample_size))
            explain_rows = transformed_df.head(min(40, sample_size))
            explainer = shap.Explainer(estimator.predict, background)
            shap_values = explainer(explain_rows)
            transformed_df = explain_rows

        summary_values = shap_values
        if len(getattr(shap_values, "shape", ())) == 3:
            class_index = predicted_class_index
            if class_index is None:
                class_index = 1 if shap_values.shape[2] > 1 else 0
            summary_values = shap_values[:, :, class_index]

        artifacts_dir = ".automl_artifacts"
        os.makedirs(artifacts_dir, exist_ok=True)
        run_id = uuid.uuid4().hex[:8]

        plt.figure(figsize=(10, 6))
        shap.summary_plot(summary_values, transformed_df, show=False, max_display=20)
        summary_path = os.path.join(artifacts_dir, f"shap_summary_{run_id}.png")
        plt.tight_layout()
        plt.savefig(summary_path, bbox_inches="tight")
        plt.close()

        first_value = shap_values[0]
        if len(getattr(first_value, "shape", ())) == 2:
            class_index = predicted_class_index
            if class_index is None:
                class_index = 1 if first_value.shape[1] > 1 else 0
            first_value = first_value[:, class_index]

        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(first_value, show=False, max_display=15)
        waterfall_path = os.path.join(artifacts_dir, f"shap_waterfall_{run_id}.png")
        plt.tight_layout()
        plt.savefig(waterfall_path, bbox_inches="tight")
        plt.close()

        return summary_path, waterfall_path
    except Exception as e:
        print(f"SHAP explanation failed: {e}")
        plt.close("all")
        return None, None
