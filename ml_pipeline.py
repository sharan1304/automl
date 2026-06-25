from data_preprocess import build_preprocessor
from model_selector import detect_problem, get_models, get_tuning_params, get_unsupervised_models
from llm_explainer import explain
from report_generator import generate_pdf_report
from visualizer import (
    plot_confusion,
    plot_regression,
    plot_residuals,
    plot_feature_importance,
    plot_shap_explanations,
)

from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_val_score, train_test_split
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

import joblib
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


def get_cv_strategy(y, problem_type, max_splits=5):
    sample_count = len(y)
    if sample_count < 3:
        return None

    if problem_type == "classification":
        min_class = y.value_counts().min()
        if min_class >= 2:
            splits = min(max_splits, int(min_class), sample_count)
            return StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)

    splits = min(max_splits, sample_count)
    if splits < 2:
        return None
    return KFold(n_splits=splits, shuffle=True, random_state=42)


def get_cv_score(pipeline, X, y, problem_type, cv):
    if cv is None:
        return None

    scoring = "accuracy" if problem_type == "classification" else "neg_root_mean_squared_error"
    try:
        scores = cross_val_score(
            pipeline,
            X,
            y,
            scoring=scoring,
            cv=cv,
            error_score=np.nan,
        )
    except Exception as e:
        print(f"Cross validation failed: {e}")
        return None

    scores = scores[~np.isnan(scores)]
    if len(scores) == 0:
        return None

    score = float(np.mean(scores))
    if problem_type == "regression":
        score = abs(score)
    return round(score, 4)


def tune_best_model(model_name, pipeline, X_train, y_train, problem_type, cv):
    params = get_tuning_params(model_name)
    if not params or cv is None:
        return pipeline, None

    scoring = "accuracy" if problem_type == "classification" else "neg_root_mean_squared_error"
    try:
        search = GridSearchCV(
            estimator=clone(pipeline),
            param_grid=params,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            error_score=np.nan,
        )
        search.fit(X_train, y_train)
    except Exception as e:
        print(f"Hyperparameter tuning skipped for {model_name}: {e}")
        return pipeline, None

    return search.best_estimator_, search.best_params_


def score_predictions(y_test, preds, problem_type):
    if problem_type == "classification":
        acc = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, average="weighted", zero_division=0)
        recall = recall_score(y_test, preds, average="weighted", zero_division=0)
        f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
        return {
            "Accuracy": round(acc, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1": round(f1, 4),
        }

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    return {
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
        "MAE": round(mae, 2)
    }


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
    cv = get_cv_strategy(y_train, problem_type)

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

        model_scores[name] = score_predictions(y_test, preds, problem_type)
        cv_score = get_cv_score(pipeline, X_train, y_train, problem_type, cv)
        if cv_score is not None:
            cv_metric = "CV Accuracy" if problem_type == "classification" else "CV RMSE"
            model_scores[name][cv_metric] = cv_score

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
    best_model, tuning_params = tune_best_model(
        best_model_name,
        best_model,
        X_train,
        y_train,
        problem_type,
        cv,
    )
    preds = best_model.predict(X_test)
    model_scores[best_model_name] = score_predictions(y_test, preds, problem_type)
    cv_score = get_cv_score(best_model, X_train, y_train, problem_type, cv)
    if cv_score is not None:
        cv_metric = "CV Accuracy" if problem_type == "classification" else "CV RMSE"
        model_scores[best_model_name][cv_metric] = cv_score
    if tuning_params:
        model_scores[best_model_name]["Tuned"] = "Yes"

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
    shap_summary_plot, shap_waterfall_plot = plot_shap_explanations(best_model, X_train)

    # Train the winning pipeline on every row before using it for unseen data.
    best_model.fit(X, y)
    model_path = "best_model.pkl"
    joblib.dump(best_model, model_path)

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

    result = {
        "dataset_explanation": dataset_explanation,
        "preprocessing_steps": preprocessing_steps,
        "problem_type": problem_type,
        "best_model": best_model_name,
        "best_pipeline": best_model,
        "model_path": model_path,
        "tuning_params": tuning_params,
        "feature_columns": X.columns.tolist(),
        "model_scores": model_scores,
        "unsupervised_results": unsupervised_results,
        "graph_path": graph_path,
        "residual_plot": residual_plot,
        "importance_plot": importance_plot,
        "shap_summary_plot": shap_summary_plot,
        "shap_waterfall_plot": shap_waterfall_plot,
        "ai_explanation": ai_explanation,
        "graph_explanation": graph_explanation
    }
    result["report_path"] = generate_pdf_report(result)
    return result


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
    pipeline = result["best_pipeline"]
    predictions = pipeline.predict(X_unseen)
    output = unseen_df.copy()
    output["Prediction"] = predictions

    if result.get("problem_type") == "classification" and hasattr(pipeline, "predict_proba"):
        try:
            probabilities = pipeline.predict_proba(X_unseen)
            classes = pipeline.classes_
            best_probability = probabilities.max(axis=1)
            output["Prediction_Confidence"] = np.round(best_probability, 4)
            for index, class_name in enumerate(classes):
                output[f"Probability_{class_name}"] = np.round(probabilities[:, index], 4)
        except Exception as e:
            print(f"Prediction probabilities unavailable: {e}")

    return output


def summarize_prediction_output(prediction_df, max_rows=5):
    rows = prediction_df.head(max_rows).astype(object).where(
        pd.notna(prediction_df.head(max_rows)),
        None,
    ).to_dict(orient="records")
    prediction_counts = {
        str(label): int(count)
        for label, count in prediction_df["Prediction"].value_counts(dropna=False).items()
    }
    confidence_summary = None
    if "Prediction_Confidence" in prediction_df.columns:
        confidence_summary = {
            "min": round(float(prediction_df["Prediction_Confidence"].min()), 4),
            "mean": round(float(prediction_df["Prediction_Confidence"].mean()), 4),
            "max": round(float(prediction_df["Prediction_Confidence"].max()), 4),
        }

    return {
        "row_count": int(len(prediction_df)),
        "prediction_counts": prediction_counts,
        "confidence_summary": confidence_summary,
        "sample_rows": rows,
    }
