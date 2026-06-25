import streamlit as st
import pandas as pd
from llm_explainer import (
    COMPARISON_LLM_MODELS,
    DEFAULT_LLM_MODEL,
    SYSTEM_PROMPT,
    explain,
    get_llm_call_history,
)
from ml_pipeline import (
    predict_unseen,
    run_pipeline,
    suggest_target,
    summarize_prediction_output,
)

st.set_page_config(page_title="AI Data Scientist", layout="wide")


def _clamp_score(value):
    return max(0, min(100, int(round(value))))


def build_llm_performance_metrics(result):
    if not result:
        return []

    explanations = [
        result.get("dataset_explanation", ""),
        result.get("ai_explanation", ""),
        result.get("graph_explanation", ""),
    ]
    combined_text = " ".join(explanations).lower()
    response_lengths = [len(text.split()) for text in explanations if text]
    avg_words = sum(response_lengths) / len(response_lengths) if response_lengths else 0

    best_model = str(result.get("best_model", "")).lower()
    problem_type = str(result.get("problem_type", "")).lower()
    score_terms = [
        str(metric).lower()
        for metrics in result.get("model_scores", {}).values()
        for metric in metrics.keys()
    ]
    context_terms = [best_model, problem_type] + score_terms
    context_hits = sum(1 for term in context_terms if term and term in combined_text)
    context_base = max(len([term for term in context_terms if term]), 1)

    history = get_llm_call_history()
    successful_calls = [call for call in history if call.get("success")]
    avg_response_time = (
        sum(call["response_time"] for call in successful_calls) / len(successful_calls)
        if successful_calls else None
    )

    unavailable_count = combined_text.count("unavailable")
    contradiction_terms = ["contradiction", "not provided", "cannot determine"]
    caution_count = sum(combined_text.count(term) for term in contradiction_terms)

    relevance = 65 + (context_hits / context_base) * 30
    explanation_quality = 55 + min(avg_words / 180, 1) * 35
    context_awareness = 60 + (context_hits / context_base) * 35
    consistency = 92 - (unavailable_count * 12) - (caution_count * 4)
    hallucination_rate = max(2, 18 - (context_hits / context_base) * 12 + caution_count * 2)

    return [
        {
            "name": "Response Relevance",
            "value": f"{_clamp_score(relevance)}%",
            "raw_value": _clamp_score(relevance),
            "score": _clamp_score(relevance),
            "caption": "Alignment with selected model, metrics, and dataset context.",
        },
        {
            "name": "Explanation Quality",
            "value": f"{_clamp_score(explanation_quality)}%",
            "raw_value": _clamp_score(explanation_quality),
            "score": _clamp_score(explanation_quality),
            "caption": "Depth and completeness of generated explanations.",
        },
        {
            "name": "Context Awareness",
            "value": f"{_clamp_score(context_awareness)}%",
            "raw_value": _clamp_score(context_awareness),
            "score": _clamp_score(context_awareness),
            "caption": "Use of known AutoML result details.",
        },
        {
            "name": "Response Consistency",
            "value": f"{_clamp_score(consistency)}%",
            "raw_value": _clamp_score(consistency),
            "score": _clamp_score(consistency),
            "caption": "Stability across dataset, model, and graph explanations.",
        },
        {
            "name": "Hallucination Rate",
            "value": f"{_clamp_score(hallucination_rate)}%",
            "raw_value": _clamp_score(hallucination_rate),
            "score": 100 - _clamp_score(hallucination_rate),
            "caption": "Estimated unsupported or uncertain content. Lower is better.",
        },
        {
            "name": "Response Time",
            "value": "N/A" if avg_response_time is None else f"{avg_response_time:.2f}s",
            "raw_value": None if avg_response_time is None else round(avg_response_time, 2),
            "score": 0 if avg_response_time is None else _clamp_score(100 - (avg_response_time * 10)),
            "caption": "Average time for completed LLM calls in this session.",
        },
    ]


def render_llm_metric_visualizations(metrics, history):
    st.divider()
    st.subheader("Metric Visualizations")

    score_df = pd.DataFrame([
        {
            "Metric": metric["name"],
            "Score": metric["score"],
            "Displayed Value": metric["value"],
        }
        for metric in metrics
    ])

    c1, c2 = st.columns(2)
    with c1:
        st.caption("Overall metric score")
        try:
            import plotly.express as px

            fig = px.bar(
                score_df,
                x="Metric",
                y="Score",
                text="Displayed Value",
                range_y=[0, 100],
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(xaxis_title="", yaxis_title="Score out of 100")
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.bar_chart(score_df.set_index("Metric")[["Score"]])

    with c2:
        st.caption("Percentage metrics")
        percent_df = pd.DataFrame([
            {
                "Metric": metric["name"],
                "Value": metric["raw_value"],
            }
            for metric in metrics
            if metric["name"] != "Response Time" and metric["raw_value"] is not None
        ])
        try:
            import plotly.express as px

            fig = px.line(percent_df, x="Metric", y="Value", markers=True, range_y=[0, 100])
            fig.update_layout(xaxis_title="", yaxis_title="Percent")
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.line_chart(percent_df.set_index("Metric")[["Value"]])

    if history:
        history_df = pd.DataFrame(history).reset_index()
        history_df["Call"] = history_df["index"] + 1
        st.caption("LLM response time by call")
        try:
            import plotly.express as px

            fig = px.line(
                history_df,
                x="Call",
                y="response_time",
                markers=True,
                color="success",
            )
            fig.update_layout(xaxis_title="Call", yaxis_title="Response Time (seconds)")
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.line_chart(history_df.set_index("Call")[["response_time"]])


def _llm_benchmark_terms(result):
    best_model = str(result.get("best_model", ""))
    problem_type = str(result.get("problem_type", ""))
    best_scores = result.get("model_scores", {}).get(best_model, {})
    terms = [best_model, problem_type]
    terms.extend(str(metric) for metric in best_scores.keys())
    terms.extend(str(value) for value in best_scores.values())
    return [term.lower() for term in terms if term]


def _score_llm_answer(answer, elapsed, required_terms):
    answer_lower = answer.lower()
    available = answer != "LLM explanation unavailable."
    if not available:
        return {
            "Context Score": 0,
            "Number Score": 0,
            "Consistency Score": 0,
            "Speed Score": 0,
            "Overall Score": 0,
        }

    context_hits = sum(1 for term in required_terms if term in answer_lower)
    context_score = (context_hits / max(len(required_terms), 1)) * 100

    numeric_terms = [term for term in required_terms if any(char.isdigit() for char in term)]
    numeric_hits = sum(1 for term in numeric_terms if term in answer_lower)
    number_score = (numeric_hits / max(len(numeric_terms), 1)) * 100 if numeric_terms else 80

    unsupported_terms = ["assume", "likely", "probably", "not provided", "cannot determine"]
    penalty = sum(answer_lower.count(term) for term in unsupported_terms) * 8
    consistency_score = max(0, 100 - penalty)
    speed_score = _clamp_score(100 - (elapsed * 12))
    overall = (
        context_score * 0.35
        + number_score * 0.25
        + consistency_score * 0.25
        + speed_score * 0.15
    )

    return {
        "Context Score": _clamp_score(context_score),
        "Number Score": _clamp_score(number_score),
        "Consistency Score": _clamp_score(consistency_score),
        "Speed Score": speed_score,
        "Overall Score": _clamp_score(overall),
    }


def build_llm_benchmark_prompts(result):
    best_model = result.get("best_model")
    best_scores = result.get("model_scores", {}).get(best_model, {})
    prediction_summary = st.session_state.get("latest_prediction_summary")

    prompts = [
        f"""
        You are being benchmarked on factual AutoML explanation.
        Problem type: {result.get('problem_type')}
        Best ML model: {best_model}
        Best model scores: {best_scores}
        Preprocessing: {result.get('preprocessing_steps')}

        Explain the model result in 5 bullet points. Use the exact metric numbers.
        """,
        f"""
        You are being benchmarked on graph interpretation.
        Graph explanation context: {result.get('graph_explanation')}
        Model scores: {result.get('model_scores')}

        Summarize what the user should trust or distrust about the result.
        Do not add facts outside this context.
        """,
        f"""
        You are being benchmarked on prediction understanding.
        Latest prediction output summary: {prediction_summary}
        Problem type: {result.get('problem_type')}
        Best ML model: {best_model}

        Explain the prediction output. If no prediction summary is present,
        say predictions have not been generated yet.
        """,
    ]
    return prompts


def run_llm_benchmark(result):
    import time

    required_terms = _llm_benchmark_terms(result)
    prompts = build_llm_benchmark_prompts(result)
    benchmark_rows = []

    for model in COMPARISON_LLM_MODELS:
        answers = []
        total_elapsed = 0
        aggregate = {
            "Context Score": 0,
            "Number Score": 0,
            "Consistency Score": 0,
            "Speed Score": 0,
            "Overall Score": 0,
        }

        for prompt in prompts:
            start = time.perf_counter()
            answer = explain(prompt, model=model)
            elapsed = time.perf_counter() - start
            scores = _score_llm_answer(answer, elapsed, required_terms)
            total_elapsed += elapsed
            answers.append(answer)
            for key in aggregate:
                aggregate[key] += scores[key]

        prompt_count = len(prompts)
        row = {
            "Model": model,
            "Avg Response Time": round(total_elapsed / prompt_count, 3),
            "Sample Answer": answers[0],
        }
        row.update({
            key: _clamp_score(value / prompt_count)
            for key, value in aggregate.items()
        })
        benchmark_rows.append(row)

    return pd.DataFrame(benchmark_rows).sort_values("Overall Score", ascending=False)


def render_llm_metrics_page():
    st.header("LLM Performance Metrics")
    result = st.session_state.get("automl_result")

    if not result:
        st.info("Run AI Analysis first to generate LLM explanations and calculate metrics.")
        return

    metrics = build_llm_performance_metrics(result)
    first_row = st.columns(3)
    second_row = st.columns(3)

    for index, metric in enumerate(metrics):
        column = first_row[index] if index < 3 else second_row[index - 3]
        with column:
            st.metric(metric["name"], metric["value"])
            st.progress(metric["score"] / 100)
            st.caption(metric["caption"])

    history = get_llm_call_history()
    render_llm_metric_visualizations(metrics, history)

    if history:
        st.divider()
        st.subheader("LLM Call Log")
        st.dataframe(pd.DataFrame(history), use_container_width=True)

    st.divider()
    st.subheader("LLM Model Benchmark")
    st.caption(
        f"Compares {DEFAULT_LLM_MODEL} against two other Groq models on the current dataset context."
    )
    if st.button("Run LLM Benchmark"):
        with st.spinner("Running same test prompts across all LLM models..."):
            benchmark_df = run_llm_benchmark(result)
            st.session_state["llm_benchmark_df"] = benchmark_df

    benchmark_df = st.session_state.get("llm_benchmark_df")
    if benchmark_df is not None:
        winner = benchmark_df.iloc[0]
        if winner["Model"] == DEFAULT_LLM_MODEL:
            st.success(
                f"{DEFAULT_LLM_MODEL} is best on this benchmark with "
                f"{winner['Overall Score']} overall score."
            )
        else:
            st.warning(
                f"{winner['Model']} scored highest on this run. "
                f"{DEFAULT_LLM_MODEL} score is shown below for comparison."
            )
        st.dataframe(
            benchmark_df.drop(columns=["Sample Answer"]),
            use_container_width=True,
        )
        with st.expander("Benchmark sample answers"):
            for _, row in benchmark_df.iterrows():
                st.markdown(f"**{row['Model']}**")
                st.write(row["Sample Answer"])

# ---- TITLE ----
st.title("🤖 AI Data Scientist")
st.caption("Upload any dataset → AI analyzes, trains, predicts and explains everything automatically")

st.divider()

page = st.sidebar.radio(
    "Page",
    ["AutoML Analysis", "LLM Performance Metrics"],
)

if page == "LLM Performance Metrics":
    render_llm_metrics_page()
    st.stop()

# ---- FILE UPLOAD ----
file = st.file_uploader("Upload Dataset (CSV)", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())
    st.caption(f"Dataset: {df.shape[0]} rows × {df.shape[1]} columns")

    st.divider()

    # ---- AI TARGET SUGGESTION ----
    suggested = suggest_target(df)
    if suggested not in df.columns:
        suggested = df.columns[-1]

    st.info(f"🤖 AI suggests predicting: **{suggested}**")

    target = st.selectbox(
        "🎯 Select Target Column",
        df.columns,
        index=list(df.columns).index(suggested)
    )

    st.divider()

    # ---- RUN BUTTON ----
    if st.button("🚀 Run AI Analysis"):
        with st.spinner("🔍 AI is analyzing your dataset..."):
            try:
                result = run_pipeline(df, target)
                st.session_state["automl_result"] = result
                st.session_state["automl_target"] = target

                st.success("✅ Analysis Completed")
                st.info(f"🔍 Problem Type Detected: **{result['problem_type']}**")

                st.divider()

                # ---- DATASET + PREPROCESSING ----
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("📊 Dataset Explanation")
                    st.write(result["dataset_explanation"])

                with col2:
                    st.subheader("⚙️ Preprocessing Steps")
                    if result["preprocessing_steps"]:
                        for step in result["preprocessing_steps"]:
                            st.write("•", step)
                    else:
                        st.write("✅ No preprocessing needed")

                st.divider()

                # ---- BEST MODEL ----
                st.subheader("🏆 Best Model")
                st.success(result["best_model"])
                if result.get("tuning_params"):
                    st.caption(f"Best tuning parameters: {result['tuning_params']}")

                # ---- MODEL COMPARISON ----
                st.subheader("📊 Model Comparison")
                df_scores = pd.DataFrame(result["model_scores"]).T
                if result["problem_type"] == "classification":
                    visible_metrics = [
                        "Accuracy",
                        "Precision",
                        "Recall",
                        "F1",
                        "CV Accuracy",
                        "Tuned",
                    ]
                    chart_metric = "Accuracy"
                else:
                    visible_metrics = [
                        "RMSE",
                        "R2",
                        "MAE",
                        "CV RMSE",
                        "Tuned",
                    ]
                    chart_metric = "RMSE"

                visible_metrics = [
                    metric for metric in visible_metrics
                    if metric in df_scores.columns
                ]
                df_scores = df_scores[visible_metrics]
                st.dataframe(df_scores)

                try:
                    import plotly.express as px

                    if chart_metric in df_scores.columns:
                        if chart_metric == "RMSE":
                            st.caption("⬇️ Lower RMSE = Better Model")
                        chart_df = df_scores.reset_index(names="Model")
                        fig = px.bar(chart_df, x="Model", y=chart_metric)
                        st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    if chart_metric in df_scores.columns:
                        if chart_metric == "RMSE":
                            st.caption("⬇️ Lower RMSE = Better Model")
                        st.bar_chart(df_scores[[chart_metric]])
                    elif result["problem_type"] == "regression" and "R2" in df_scores.columns:
                        st.caption("⬆️ Higher R2 = Better Model")
                        st.bar_chart(df_scores[["R2"]])

                if result["problem_type"] != "classification":
                    st.divider()

                    # ---- UNSUPERVISED LEARNING ----
                    st.subheader("🧩 Unsupervised Learning")
                    if result["unsupervised_results"]:
                        unsupervised_rows = []
                        for model_name, metrics in result["unsupervised_results"].items():
                            row = {"Model": model_name}
                            row.update(metrics)
                            unsupervised_rows.append(row)
                        st.dataframe(pd.DataFrame(unsupervised_rows))
                    else:
                        st.info("Dataset is too small for unsupervised analysis.")

                st.divider()

                # ---- VISUALIZATIONS ----
                if result["graph_path"] and result["importance_plot"]:
                    v1, v2 = st.columns(2)
                    with v1:
                        st.subheader("📉 Visualization")
                        st.image(result["graph_path"])
                    with v2:
                        st.subheader("📊 Feature Importance")
                        st.image(result["importance_plot"])
                else:
                    if result["graph_path"]:
                        st.subheader("📉 Visualization")
                        st.image(result["graph_path"])
                    if result["importance_plot"]:
                        st.subheader("📊 Feature Importance")
                        st.image(result["importance_plot"])

                if result["residual_plot"]:
                    st.subheader("📉 Residual Plot")
                    st.image(result["residual_plot"])

                if result.get("shap_summary_plot") or result.get("shap_waterfall_plot"):
                    st.divider()
                    st.subheader("🔎 SHAP Explainability")
                    s1, s2 = st.columns(2)
                    if result.get("shap_summary_plot"):
                        with s1:
                            st.image(result["shap_summary_plot"])
                    if result.get("shap_waterfall_plot"):
                        with s2:
                            st.image(result["shap_waterfall_plot"])

                st.divider()

                # ---- AI EXPLANATIONS ----
                st.subheader("🧠 Model Explanation")
                st.write(result["ai_explanation"])

                st.subheader("📘 Graph Explanation")
                st.write(result["graph_explanation"])

                st.divider()
                d1, d2 = st.columns(2)
                if result.get("model_path"):
                    with open(result["model_path"], "rb") as model_file:
                        d1.download_button(
                            "⬇️ Download Trained Model",
                            data=model_file,
                            file_name="best_model.pkl",
                            mime="application/octet-stream",
                        )
                if result.get("report_path"):
                    with open(result["report_path"], "rb") as report_file:
                        d2.download_button(
                            "⬇️ Download PDF Report",
                            data=report_file,
                            file_name="automl_report.pdf",
                            mime="application/pdf",
                        )

            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.exception(e)

    if "automl_result" in st.session_state:
        st.divider()
        st.subheader("🔮 Predict on Unseen Data")
        st.caption("Upload a new CSV with the same feature columns. If it contains the target column, it will be ignored.")

        unseen_file = st.file_uploader(
            "Upload Unseen/Test Dataset (CSV)",
            type=["csv"],
            key="unseen_file"
        )

        if unseen_file:
            unseen_df = pd.read_csv(unseen_file)
            st.write("Unseen data preview")
            st.dataframe(unseen_df.head())

            if st.button("🧪 Predict Unseen Data"):
                try:
                    prediction_df = predict_unseen(
                        st.session_state["automl_result"],
                        unseen_df,
                        st.session_state.get("automl_target")
                    )
                    st.session_state["latest_prediction_summary"] = summarize_prediction_output(
                        prediction_df
                    )

                    st.success("✅ Predictions generated")
                    st.dataframe(prediction_df)

                    csv_data = prediction_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download Predictions CSV",
                        data=csv_data,
                        file_name="unseen_predictions.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"❌ Prediction error: {e}")
                    st.exception(e)

        st.divider()
        st.subheader("✍️ Check One Prediction Manually")
        result = st.session_state["automl_result"]
        feature_columns = result["feature_columns"]

        with st.form("manual_prediction_form"):
            manual_values = {}
            for col in feature_columns:
                series = df[col] if col in df.columns else pd.Series(dtype="object")

                if pd.api.types.is_bool_dtype(series):
                    default_value = bool(series.mode(dropna=True).iloc[0]) if not series.mode(dropna=True).empty else False
                    manual_values[col] = st.checkbox(col, value=default_value)
                elif pd.api.types.is_numeric_dtype(series):
                    default_value = series.median()
                    if pd.isna(default_value):
                        default_value = 0
                    manual_values[col] = st.number_input(col, value=float(default_value))
                else:
                    clean_values = series.dropna().astype(str).unique().tolist()
                    if 0 < len(clean_values) <= 100:
                        manual_values[col] = st.selectbox(col, clean_values)
                    else:
                        default_value = series.dropna().astype(str).mode()
                        manual_values[col] = st.text_input(
                            col,
                            value="" if default_value.empty else default_value.iloc[0],
                        )

            manual_submit = st.form_submit_button("🔍 Predict This Input")

        if manual_submit:
            try:
                manual_df = pd.DataFrame([manual_values])
                manual_prediction = predict_unseen(result, manual_df)
                prediction_summary = summarize_prediction_output(manual_prediction)
                st.session_state["latest_prediction_summary"] = prediction_summary
                st.success(f"Prediction: {manual_prediction['Prediction'].iloc[0]}")
                st.dataframe(manual_prediction)
                st.subheader("🧠 Prediction Explanation")
                st.write(explain(f"""
                Explain this single prediction using only the provided values.
                Problem type: {result['problem_type']}
                Best model: {result['best_model']}
                Model scores: {result['model_scores'].get(result['best_model'], {})}
                Input row: {manual_values}
                Prediction output: {prediction_summary}

                Required:
                - State the exact prediction.
                - For classification, state confidence/probabilities only if present.
                - For regression, explain that the prediction is a numeric estimate.
                - Do not claim feature-level causes unless SHAP values for this row are provided.
                """))
            except Exception as e:
                st.error(f"❌ Manual prediction error: {e}")
                st.exception(e)

        st.divider()
        st.subheader("💬 Dataset Chatbot")
        result = st.session_state["automl_result"]
        if "dataset_chat_messages" not in st.session_state:
            st.session_state["dataset_chat_messages"] = []

        for message in st.session_state["dataset_chat_messages"]:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        question = st.chat_input(
            "Ask about this dataset, model performance, features, clusters, or predictions"
        )
        if question:
            st.session_state["dataset_chat_messages"].append({
                "role": "user",
                "content": question,
            })
            with st.chat_message("user"):
                st.write(question)

            context_prompt = f"""
            You are a dataset assistant for an AutoML result.
            Problem type: {result['problem_type']}
            Best ML model: {result['best_model']}
            Model scores: {result['model_scores']}
            Unsupervised results: {result['unsupervised_results']}
            Preprocessing: {result['preprocessing_steps']}
            Latest prediction output summary, if any: {st.session_state.get('latest_prediction_summary')}

            Answer directly using only this result context and the chat history.
            If the user asks about predictions, use the latest prediction output summary.
            If no prediction summary is available, say predictions have not been generated yet.
            Keep memory of earlier chat turns, but do not invent missing dataset facts.
            """
            chat_history = st.session_state["dataset_chat_messages"][-8:]
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": context_prompt},
            ]
            messages.extend(chat_history)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = explain(context_prompt + "\nUser question: " + question, messages=messages)
                    st.write(answer)

            st.session_state["dataset_chat_messages"].append({
                "role": "assistant",
                "content": answer,
            })
