import streamlit as st
import pandas as pd
import requests
import io

# Title of the Streamlit app
st.title("Prediction with Trained Model")

# File uploader widget
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Read the file content once and store it
        file_content = uploaded_file.read()
        
        # Create a StringIO object for reading the CSV with robust parsing
        try:
            # Try reading with different parameters to handle problematic CSVs
            df = pd.read_csv(
                io.StringIO(file_content.decode('utf-8')),
                dtype=str,  # Read all columns as strings initially
                na_filter=False,  # Don't convert to NaN
                skipinitialspace=True,  # Skip whitespace after delimiter
                encoding_errors='replace'  # Replace problematic characters
            )
        except UnicodeDecodeError:
            # Try different encoding if UTF-8 fails
            try:
                df = pd.read_csv(
                    io.StringIO(file_content.decode('latin-1')),
                    dtype=str,
                    na_filter=False,
                    skipinitialspace=True
                )
            except Exception as e:
                st.error(f"Error with encoding: {str(e)}")
                st.stop()
        except Exception as csv_error:
            st.error(f"Error parsing CSV: {str(csv_error)}")
            st.error("Please check that your CSV file is properly formatted")
            st.stop()
        
        # Preview the data
        st.write("Preview of the uploaded CSV file:")
        st.write(df.head())  # Show the first few rows
        
        # Check if the DataFrame contains the expected columns
        expected_cols = [
            "trf", "age", "gndr", "tenure", "age_dev", "dev_man", 
            "device_os_name", "dev_num", "is_dualsim", "simcard_type", "region"
        ]
        
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing columns in the CSV: {', '.join(missing_cols)}")
        else:
            # If all required columns are present, proceed to prediction
            if st.button("Get Prediction"):
                try:
                    # Create a new file-like object from the content for sending to API
                    files = {"file": ("uploaded_file.csv", file_content, "text/csv")}
                    
                    # Send the file to the FastAPI backend for prediction
                    response = requests.post(
                        "http://127.0.0.1:8000/predict",
                        files=files
                    )
                    
                    # Check if the request was successful
                    if response.status_code == 200:
                        result = response.json()
                        predictions = result.get("predictions", [])
                        st.write("Predictions:")
                        st.write(predictions)
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"An error occurred while getting predictions: {str(e)}")
    except Exception as e:
        st.error(f"Error reading the CSV file: {str(e)}")