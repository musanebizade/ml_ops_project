import joblib
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from io import StringIO
import numpy as np

app = FastAPI()

# Load your trained model
try:
    model = joblib.load("xgb_model.pkl")
except FileNotFoundError:
    raise Exception("Model file 'xgb_model.pkl' not found. Make sure it's in the correct directory.")

# Columns expected by the model
EXPECTED_COLS = [
    "trf", "age", "gndr", "tenure", "age_dev", "dev_man", 
    "device_os_name", "dev_num", "is_dualsim", "simcard_type", "region"
]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        # Read the file content
        contents = await file.read()
        
        # Decode and read as CSV with robust parsing
        csv_content = contents.decode("utf-8")
        df = pd.read_csv(
            StringIO(csv_content),
            dtype=str,  # Read as strings first
            na_filter=False,  # Don't convert to NaN automatically
            skipinitialspace=True,
            encoding_errors='replace'
        )
        
        # Check if DataFrame is empty
        if df.empty:
            raise HTTPException(status_code=400, detail="The uploaded CSV file is empty")
        
        # Check for missing columns
        missing_cols = [col for col in EXPECTED_COLS if col not in df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {', '.join(missing_cols)}"
            )
        
        # Keep only the columns needed for the model (in the correct order)
        df_model = df[EXPECTED_COLS]
        
        # Check for any missing values that might cause issues
        if df_model.isnull().any().any():
            # You might want to handle missing values here
            # For now, we'll raise an error
            null_cols = df_model.columns[df_model.isnull().any()].tolist()
            raise HTTPException(
                status_code=400,
                detail=f"Missing values found in columns: {', '.join(null_cols)}"
            )
        
        # Perform prediction
        predictions = model.predict(df_model)
        
        # Convert numpy types to Python native types for JSON serialization
        if isinstance(predictions, np.ndarray):
            predictions = predictions.tolist()
        
        return {
            "predictions": predictions,
            "num_predictions": len(predictions)
        }
        
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="The CSV file is empty or invalid")
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail=f"Error parsing CSV file: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Data validation error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Health check endpoint
@app.get("/")
async def root():
    return {"message": "Model prediction API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}