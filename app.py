import streamlit as st
import pandas as pd
from ml_pipeline import run_pipeline, suggest_target

st.set_page_config(page_title="AI Data Scientist", layout="wide")

# ---- TITLE ----
st.title("🤖 AI Data Scientist")
st.caption("Upload any dataset → AI analyzes, trains, predicts and explains everything automatically")

st.divider()

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

                # ---- MODEL COMPARISON ----
                st.subheader("📊 Model Comparison")
                df_scores = pd.DataFrame(result["model_scores"]).T
                st.dataframe(df_scores)

                if "Accuracy" in df_scores.columns:
                    st.bar_chart(df_scores[["Accuracy"]])
                elif "R2" in df_scores.columns:
                    st.bar_chart(df_scores[["R2"]])
                elif "RMSE" in df_scores.columns:
                    st.caption("⬇️ Lower RMSE = Better Model")
                    st.bar_chart(df_scores[["RMSE"]])

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

                st.divider()

                # ---- AI EXPLANATIONS ----
                st.subheader("🧠 Model Explanation")
                st.write(result["ai_explanation"])

                st.subheader("📘 Graph Explanation")
                st.write(result["graph_explanation"])

            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.exception(e)