import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from category_encoders import CatBoostEncoder
from scipy.stats import ttest_rel
import joblib

def load_data(file_path):
    """Loads the dataset from a given file path."""
    df = pd.read_parquet(file_path)
    return df

def preprocess_data(df):
    """Prepares the data for training."""
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Identify columns by types
    val_cols = [col for col in X_train.columns if col.startswith('val')]
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.difference(val_cols)
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    return X_train, X_test, y_train, y_test, val_cols, numeric_cols, categorical_cols

def create_preprocessor(val_cols, numeric_cols, categorical_cols):
    """Creates the preprocessing pipeline."""
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

def train_model(X_train, y_train, preprocessor, model_type='rf'):
    """Trains the model with the specified model type."""
    if model_type == 'rf':
        model = RandomForestClassifier(random_state=42)
    elif model_type == 'xgb':
        model = XGBClassifier(random_state=42, eval_metric='logloss')
    else:
        raise ValueError("Invalid model type. Choose 'rf' or 'xgb'.")

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model),
    ])

    # Cross-validation with accuracy
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    print(f"{model_type.upper()} CV Accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    return pipeline, scores

def hypothesis_testing(rf_scores, xgb_scores):
    """Performs hypothesis testing to compare the two models."""
    stat, p_value = ttest_rel(rf_scores, xgb_scores)
    print(f"Paired t-test p-value: {p_value:.4f}")

    if p_value < 0.05:
        better_model = 'Random Forest' if np.mean(rf_scores) > np.mean(xgb_scores) else 'XGBoost'
        print(f"Statistically significant difference. Better model: {better_model}")
    else:
        print("No statistically significant difference between models.")

def save_best_model(rf_pipeline, xgb_pipeline, rf_scores, xgb_scores):
    """Saves the best model after comparison."""
    best_model = rf_pipeline if np.mean(rf_scores) > np.mean(xgb_scores) else xgb_pipeline
    joblib.dump(best_model, 'best_model.pkl')
    print("Model saved as 'best_model.pkl'.")

def main():
    # Load data
    df = load_data('/content/multisim_dataset.parquet')

    # Preprocess data
    X_train, X_test, y_train, y_test, val_cols, numeric_cols, categorical_cols = preprocess_data(df)

    # Create preprocessor
    preprocessor = create_preprocessor(val_cols, numeric_cols, categorical_cols)

    # Train models
    rf_pipeline, rf_scores = train_model(X_train, y_train, preprocessor, model_type='rf')
    xgb_pipeline, xgb_scores = train_model(X_train, y_train, preprocessor, model_type='xgb')

    # Hypothesis testing
    hypothesis_testing(rf_scores, xgb_scores)

    # Save the best model
    save_best_model(rf_pipeline, xgb_pipeline, rf_scores, xgb_scores)

if __name__ == '__main__':
    main()