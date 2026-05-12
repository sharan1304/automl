from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    steps = []
    df = df.copy()

    if df.isnull().sum().sum() > 0:
        df = df.fillna(df.mean(numeric_only=True))
        steps.append("Filled missing values using mean")

    for col in df.select_dtypes(include='object'):
        df[col] = LabelEncoder().fit_transform(df[col])
        steps.append(f"Encoded categorical column: {col}")

    return df, steps