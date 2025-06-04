import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def extract_data(file_path):
    data = pd.read_csv(file_path)
    return data

def transform_data(df):
    X = df.drop("target", axis=1)
    y = df["target"]

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

    full_pipeline = Pipeline([
        ('preprocessing', preprocessor)
    ])

    X_processed = full_pipeline.fit_transform(X)
    return X_processed, y

def load_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pd.DataFrame(X_train.toarray() if hasattr(X_train, "toarray") else X_train).to_csv("X_train.csv", index=False)
    pd.DataFrame(X_test.toarray() if hasattr(X_test, "toarray") else X_test).to_csv("X_test.csv", index=False)
    y_train.to_csv("y_train.csv", index=False)
    y_test.to_csv("y_test.csv", index=False)

if __name__ == "__main__":
    df = extract_data("flights_data.csv")
    X, y = transform_data(df)
    load_data(X, y)
