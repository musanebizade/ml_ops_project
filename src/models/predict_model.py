import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from category_encoders import CatBoostEncoder
from sklearn.decomposition import PCA

def load_model(model_path):
    """Loads the trained model from a file."""
    model = joblib.load(model_path)
    return model

def preprocess_data_for_prediction(df):
    """Prepares the data for prediction (same preprocessing as during training)."""
    X = df.drop('target', axis=1)

    # Identify columns by types
    val_cols = [col for col in X.columns if col.startswith('val')]
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.difference(val_cols)
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    return X, val_cols, numeric_cols, categorical_cols

def create_preprocessor(val_cols, numeric_cols, categorical_cols):
    """Creates the preprocessing pipeline (same as during training)."""
    val_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95)),
    ])

    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', CatBoostEncoder(handle_unknown='ignore')),
    ])

    preprocessor = ColumnTransformer([
        ('val', val_pipeline, val_cols),
        ('num', numeric_pipeline, numeric_cols),
        ('cat', categorical_pipeline, categorical_cols),
    ])

    return preprocessor

def predict(model, X, preprocessor):
    """Makes predictions using the trained model and preprocessed data."""
    X_processed = preprocessor.fit_transform(X)
    predictions = model.predict(X_processed)
    return predictions

def save_predictions(predictions, output_path='predictions.csv'):
    """Saves the predictions to a CSV file."""
    output = pd.DataFrame({'Predictions': predictions})
    output.to_csv(output_path, index=False)
    print(f"Predictions saved as '{output_path}'.")

def main():
    # Load model
    model = load_model('best_model.pkl')

    # Load data (same format as training data)
    df = pd.read_parquet('/content/multisim_dataset.parquet')

    # Preprocess data for prediction
    X, val_cols, numeric_cols, categorical_cols = preprocess_data_for_prediction(df)

    # Create preprocessor
    preprocessor = create_preprocessor(val_cols, numeric_cols, categorical_cols)

    # Make predictions
    predictions = predict(model, X, preprocessor)

    # Save predictions
    save_predictions(predictions)

if __name__ == '__main__':
    main()
