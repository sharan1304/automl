from data_preprocess import preprocess_data
from model_selector import detect_problem, get_models
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
    mean_squared_error,
    r2_score,
    mean_absolute_error
)

import numpy as np


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

    # ---- Preprocessing ----
    df_processed, preprocessing_steps = preprocess_data(df)

    # ---- Features & Target ----
    X = df_processed.drop(columns=[target])
    y = df_processed[target]

    if X.shape[1] == 0:
        raise ValueError("No feature columns left after removing target")

    # ---- Problem Type ----
    problem_type = detect_problem(y)

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

    # ---- Train Models (SAFE + DEBUG) ----
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        except Exception as e:
            print(f"⚠️ Model {name} failed: {e}")
            continue

        if problem_type == "classification":
            acc = accuracy_score(y_test, preds)
            model_scores[name] = {"Accuracy": round(acc, 4)}
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
            "model": model,
            "preds": preds
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
        "model_scores": model_scores,
        "graph_path": graph_path,
        "residual_plot": residual_plot,
        "importance_plot": importance_plot,
        "ai_explanation": ai_explanation,
        "graph_explanation": graph_explanation
    }