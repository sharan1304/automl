from data_preprocess import build_preprocessor
from model_selector import detect_problem, get_models, get_unsupervised_models
from llm_explainer import explain
from visualizer import (
    plot_confusion,
    plot_regression,
    plot_residuals,
    plot_feature_importance
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.base import clone
from sklearn.pipeline import Pipeline

import numpy as np
import pandas as pd


# ---- SMART TARGET SUGGESTION ----
def suggest_target(df):
    keywords = [
        'price', 'target', 'label', 'outcome',
        'class', 'result', 'diagnosis', 'output',
        'salary', 'sales', 'revenue', 'score', 'grade'
    ]

    for col in df.columns:
        if any(k in col.lower() for k in keywords):
            return col

    for col in reversed(df.columns):
        unique_ratio = df[col].nunique() / max(len(df), 1)
        if 0.05 < unique_ratio < 0.9:
            return col

    return df.columns[-1]


# ---- MAIN PIPELINE ----
def run_pipeline(df, target):

    # ---- Safety Checks ----
    if target not in df.columns:
        raise ValueError("Selected target column not found")

    if len(df) < 5:
        raise ValueError("Dataset too small for training")

    # ---- Dataset Info ----
    missing = df.isnull().sum().to_dict()
    shape = df.shape
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

    dataset_explanation = explain(f"""
    Dataset info:
    Shape: {shape[0]} rows and {shape[1]} columns
    Columns and types: {dtypes}
    Missing values per column: {missing}

    STRICT RULES:
    - If missing values = 0 → say NO missing values
    - Do NOT assume anything not present
    - No contradictions

    Explain:
    1. Dataset purpose
    2. Important features
    3. Problem type
    4. Key insights
    """)

    # ---- Features & Target ----
    X = df.drop(columns=[target])
    y = df[target]

    if X.shape[1] == 0:
        raise ValueError("No feature columns left after removing target")

    # ---- Problem Type ----
    problem_type = detect_problem(y)

    if problem_type == "regression" and not pd.api.types.is_numeric_dtype(y):
        raise ValueError("Regression target must be numeric")

    # ---- Reusable Preprocessing ----
    preprocessor, preprocessing_steps = build_preprocessor(X)

    # ---- Models ----
    models = get_models(problem_type)

    # ---- Train/Test Split (SAFE STRATIFY) ----
    if problem_type == "classification":
        min_class = y.value_counts().min()
        use_stratify = y if min_class >= 2 else None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=use_stratify
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

    results = {}
    model_scores = {}

    # ---- Train Supervised Models ----
    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("model", model),
        ])

        try:
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)
        except Exception as e:
            print(f"⚠️ Model {name} failed: {e}")
            continue

        if problem_type == "classification":
            acc = accuracy_score(y_test, preds)
            precision = precision_score(y_test, preds, average="weighted", zero_division=0)
            recall = recall_score(y_test, preds, average="weighted", zero_division=0)
            f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
            model_scores[name] = {
                "Accuracy": round(acc, 4),
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "F1": round(f1, 4),
            }
        else:
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)
            mae = mean_absolute_error(y_test, preds)

            model_scores[name] = {
                "RMSE": round(rmse, 2),
                "R2": round(r2, 4),
                "MAE": round(mae, 2)
            }

        results[name] = {
            "model": pipeline,
            "preds": preds,
        }

    # ---- Safety ----
    if not model_scores:
        raise ValueError("All models failed to train on this dataset")

    # ---- Best Model Selection ----
    if problem_type == "classification":
        best_model_name = max(
            model_scores,
            key=lambda x: model_scores[x]["Accuracy"]
        )
    else:
        best_model_name = min(
            model_scores,
            key=lambda x: model_scores[x]["RMSE"]
        )

    best_model = results[best_model_name]["model"]
    preds = results[best_model_name]["preds"]

    # ---- Graphs ----
    if problem_type == "classification":
        cm = confusion_matrix(y_test, preds)
        graph_path = plot_confusion(cm)
        residual_plot = None
    else:
        graph_path = plot_regression(y_test, preds)
        residual_plot = plot_residuals(y_test, preds)

    # ---- Feature Importance (NO LEAKAGE) ----
    importance_plot = plot_feature_importance(best_model, X_train)

    # Train the winning pipeline on every row before using it for unseen data.
    best_model.fit(X, y)

    # ---- Unsupervised Learning on Feature Space ----
    unsupervised_results = run_unsupervised_analysis(preprocessor, X)

    # ---- AI Model Explanation ----
    ai_explanation = explain(f"""
    Best model: {best_model_name}
    Problem type: {problem_type}
    Metrics: {model_scores[best_model_name]}
    Target: {target}
    Dataset size: {shape[0]} rows

    Explain:
    - Why model fits
    - Meaning of metrics
    - Performance quality
    - Suggestions to improve
    """)

    # ---- AI Graph Explanation ----
    graph_explanation = explain(f"""
    Graph type: {'Confusion Matrix' if problem_type == 'classification' else 'Regression + Residual Plot'}
    Model: {best_model_name}
    Metrics: {model_scores[best_model_name]}
    Unsupervised summary: {unsupervised_results}

    Explain:
    - What graph shows
    - Key insights
    - Model performance
    """)

    return {
        "dataset_explanation": dataset_explanation,
        "preprocessing_steps": preprocessing_steps,
        "problem_type": problem_type,
        "best_model": best_model_name,
        "best_pipeline": best_model,
        "feature_columns": X.columns.tolist(),
        "model_scores": model_scores,
        "unsupervised_results": unsupervised_results,
        "graph_path": graph_path,
        "residual_plot": residual_plot,
        "importance_plot": importance_plot,
        "ai_explanation": ai_explanation,
        "graph_explanation": graph_explanation
    }


def run_unsupervised_analysis(preprocessor, X):
    sample_count = len(X)
    if sample_count < 3:
        return {}

    n_clusters = min(3, sample_count - 1)
    X_unsupervised = clone(preprocessor).fit_transform(X)
    models = get_unsupervised_models(n_clusters=n_clusters)
    scores = {}

    for name, model in models.items():
        try:
            if hasattr(model, "fit_predict"):
                labels = model.fit_predict(X_unsupervised)
            else:
                labels = model.fit(X_unsupervised).predict(X_unsupervised)
        except Exception as e:
            print(f"⚠️ Unsupervised model {name} failed: {e}")
            continue

        labels = np.asarray(labels)
        unique_labels = sorted(set(labels.tolist()))
        cluster_count = len(unique_labels)
        non_noise_labels = [label for label in unique_labels if label != -1]
        cluster_sizes = {
            str(label): int((labels == label).sum())
            for label in unique_labels
        }

        score = None
        if len(non_noise_labels) >= 2 and cluster_count < sample_count:
            try:
                score = silhouette_score(X_unsupervised, labels)
            except Exception:
                score = None

        scores[name] = {
            "Clusters": len(non_noise_labels) if -1 in unique_labels else cluster_count,
            "Silhouette": None if score is None else round(score, 4),
            "Cluster Sizes": cluster_sizes,
        }

    return scores


def predict_unseen(result, unseen_df, target=None):
    if "best_pipeline" not in result:
        raise ValueError("No trained model found. Run AI Analysis first.")

    feature_columns = result["feature_columns"]
    data = unseen_df.copy()

    if target and target in data.columns:
        data = data.drop(columns=[target])

    missing_columns = [col for col in feature_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(
            "Unseen data is missing required feature columns: "
            + ", ".join(missing_columns)
        )

    X_unseen = data[feature_columns]
    predictions = result["best_pipeline"].predict(X_unseen)
    output = unseen_df.copy()
    output["Prediction"] = predictions
    return output
