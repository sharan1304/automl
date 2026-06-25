# AI Data Scientist (AutoML)

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

An end-to-end AutoML platform that analyzes datasets, detects the problem type, trains multiple machine learning models, selects the best model, explains predictions using SHAP, generates visualizations, creates PDF reports, and provides an AI-powered dataset chatbot.

---

## Project Demo

### 1. Dataset Upload and Preview

![Dataset Upload](screenshots/01_home.png)

### 2. AI Dataset Analysis

![Dataset Analysis](screenshots/02_analysis.png)

### 3. Model Comparison and Best Model Selection

![Model Comparison](screenshots/03_model_comparison.png)

### 4. Visualizations and Feature Importance

![Visualization](screenshots/04_visualization.png)

### 5. SHAP Explainability

![SHAP Explainability](screenshots/05_shap_explainability.png)

### 6. Prediction and Explanation

![Prediction](screenshots/06_prediction.png)

---

## Features

| Category | Capability |
| --- | --- |
| Data Understanding | Upload CSV files, preview datasets, inspect rows and columns, and automatically suggest a target column. |
| AutoML | Detect classification or regression problems, preprocess data, train multiple models, and compare performance metrics. |
| Model Selection | Select the best model using validation scores and display model comparison results. |
| Explainability | Generate feature importance, SHAP summary plots, SHAP waterfall explanations, and natural-language model explanations. |
| Prediction | Enter custom feature values, generate predictions, show confidence scores, and explain prediction results. |
| AI Assistant | Ask dataset, model, feature, performance, and prediction questions through an integrated chatbot. |
| Reporting | Generate a PDF report with model results, visualizations, and explainability outputs. |

---

## Tech Stack

- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn
- FLAML
- XGBoost
- LightGBM
- SHAP
- Plotly, Matplotlib, and Seaborn
- Groq API for LLM-powered explanations
- ReportLab for PDF report generation

---

## Project Structure

```text
automl/
|-- app.py
|-- data_preprocess.py
|-- llm_explainer.py
|-- ml_pipeline.py
|-- model_selector.py
|-- report_generator.py
|-- visualizer.py
|-- requirements.txt
|-- Dockerfile
|-- deployment.yaml
|-- service.yaml
|-- screenshots/
|   |-- 01_home.png
|   |-- 02_analysis.png
|   |-- 03_model_comparison.png
|   |-- 04_visualization.png
|   |-- 05_shap_explainability.png
|   |-- 06_prediction.png
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/sharan1304/automl.git
cd automl
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file for the LLM features:

```bash
GROQ_API_KEY=your_groq_api_key
```

Run the Streamlit app:

```bash
streamlit run app.py
```

---

## How It Works

1. Upload a CSV dataset.
2. The app previews the data and suggests a target column.
3. The pipeline detects whether the task is classification or regression.
4. Numeric and categorical features are preprocessed automatically.
5. Multiple models are trained and compared.
6. The best model is selected based on performance.
7. Visualizations, feature importance, and SHAP explanations are generated.
8. Users can make predictions on custom input values.
9. The AI chatbot answers questions about the dataset, model, and results.
10. A PDF report can be generated for sharing.

---

## Example Results

In the demo workflow, the platform trains and compares several models on a diabetes classification dataset, selects the best-performing model, displays confusion matrix and feature importance plots, explains model behavior with SHAP, and provides prediction confidence for custom inputs.

---

## Future Improvements

- Add model export and reload controls from the UI.
- Add experiment tracking for previous runs.
- Add support for Excel and Parquet uploads.
- Add deployment templates for Streamlit Community Cloud and Hugging Face Spaces.
- Add more advanced LLM evaluation metrics.

---

## Author

**Sharan**

GitHub: [sharan1304](https://github.com/sharan1304)
