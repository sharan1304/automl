def _table_from_scores(model_scores):
    from reportlab.platypus import Table

    rows = [["Model"]]
    metric_names = sorted({
        metric
        for scores in model_scores.values()
        for metric in scores.keys()
    })
    rows[0].extend(metric_names)

    for model_name, scores in model_scores.items():
        rows.append([model_name] + [scores.get(metric, "") for metric in metric_names])

    return Table(rows, repeatRows=1)


def generate_pdf_report(result, path="automl_report.pdf"):
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer

    doc = SimpleDocTemplate(path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("AI Data Scientist Report", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Problem Type: {result['problem_type']}", styles["Normal"]))
    story.append(Paragraph(f"Best Model: {result['best_model']}", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Model Comparison", styles["Heading2"]))
    story.append(_table_from_scores(result["model_scores"]))
    story.append(Spacer(1, 12))

    for title, key in [
        ("Main Visualization", "graph_path"),
        ("Residual Plot", "residual_plot"),
        ("Feature Importance", "importance_plot"),
        ("SHAP Summary", "shap_summary_plot"),
        ("SHAP Waterfall", "shap_waterfall_plot"),
    ]:
        image_path = result.get(key)
        if image_path:
            story.append(Paragraph(title, styles["Heading2"]))
            story.append(Image(image_path, width=420, height=260, kind="proportional"))
            story.append(Spacer(1, 12))

    story.append(Paragraph("AI Model Explanation", styles["Heading2"]))
    story.append(Paragraph(result.get("ai_explanation", ""), styles["BodyText"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Graph Explanation", styles["Heading2"]))
    story.append(Paragraph(result.get("graph_explanation", ""), styles["BodyText"]))

    doc.build(story)
    return path
