from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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


def build_preprocessor(X, scale_numeric=True):
    numeric_columns = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()

    transformers = []

    if numeric_columns:
        numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric", Pipeline(numeric_steps), numeric_columns))

    if categorical_columns:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            categorical_columns,
        ))

    if not transformers:
        raise ValueError("No usable feature columns found")

    steps = []
    if numeric_columns:
        steps.append("Filled numeric missing values with median")
        if scale_numeric:
            steps.append("Scaled numeric features")
    if categorical_columns:
        steps.append("Filled categorical missing values with most frequent value")
        steps.append("One-hot encoded categorical columns")

    return ColumnTransformer(transformers=transformers), steps
