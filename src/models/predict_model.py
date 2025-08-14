import pandas as pd
import joblib  # For loading the trained model

# Load the model
def load_model(filename='trained_model.joblib'):
    """
    Load the saved model from disk.
    """
    model = joblib.load(filename)
    return model

# Preprocessing for prediction
def preprocess_data(df, preprocessor):
    """
    Preprocess data before prediction.
    """
    df_processed = preprocessor.transform(df)
    return df_processed

# Make predictions
def make_predictions(model, df, preprocessor):
    """
    Make predictions using the trained model.
    """
    df_processed = preprocess_data(df, preprocessor)
    predictions = model.predict(df_processed)
    return predictions

def main(file_path):
    # Load the dataset for prediction
    df = pd.read_csv(file_path)
    
    # Load the trained model
    model = load_model()
    
    # Extract preprocessor from the model
    preprocessor = model.named_steps['preprocessor']
    
    # Make predictions
    predictions = make_predictions(model, df, preprocessor)
    print("Predictions:", predictions)

if __name__ == "__main__":
    file_path = 'your_file_to_predict.csv'  # Replace with the actual file path for prediction
    main(file_path)
