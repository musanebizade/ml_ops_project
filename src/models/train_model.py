import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from category_encoders import CatBoostEncoder
from sklearn.compose import ColumnTransformer
import joblib  # For saving the model

# Load data
def load_data(file_path):
    """
    Load the dataset from the provided file path.
    """
    df = pd.read_csv(file_path)
    return df

# Sample data and split
def sample_and_split_data(df, frac=0.2, random_state=42):
    """
    Sample the data and split it into training and test sets.
    """
    df_sample = df.sample(frac=frac, random_state=random_state)
    X = df_sample.drop('target', axis=1)
    y = df_sample['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    return X_train, X_test, y_train, y_test

# Feature selection based on column types
def feature_selection(X_train):
    val_cols = [col for col in X_train.columns if col.startswith('val')]
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.difference(val_cols)
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    return val_cols, numeric_cols, categorical_cols

# Define the pipeline for preprocessing and training
def create_pipelines(val_cols, numeric_cols, categorical_cols):
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

# Train the model
def train_model(X_train, y_train, preprocessor):
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestClassifier(random_state=42)),
    ])
    
    rf_pipeline.fit(X_train, y_train)
    return rf_pipeline

# Save the trained model
def save_model(model, filename='trained_model.joblib'):
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def main(file_path):
    # Load and prepare data
    df = load_data(file_path)
    X_train, X_test, y_train, y_test = sample_and_split_data(df)
    
    # Feature selection
    val_cols, numeric_cols, categorical_cols = feature_selection(X_train)
    
    # Create preprocessing pipeline
    preprocessor = create_pipelines(val_cols, numeric_cols, categorical_cols)
    
    # Train model
    model = train_model(X_train, y_train, preprocessor)
    
    # Save the trained model
    save_model(model)

if __name__ == "__main__":
    file_path = 'your_file.csv'  # Replace with your dataset's path
    main(file_path)
