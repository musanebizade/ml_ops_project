import pandas as pd
from sklearn.feature_selection import mutual_info_regression

def load_data(file_path):
    """
    Load the dataset from the provided file path.
    """
    df = pd.read_csv(file_path)
    return df

def encode_and_fill_data(df):
    """
    Encodes categorical columns and fills missing values with a placeholder.
    """
    # Encode object columns
    X_encoded = df.copy()
    for col in X_encoded.select_dtypes(include='object'):
        X_encoded[col] = X_encoded[col].astype('category').cat.codes

    # Fill NaNs with a safe placeholder
    X_encoded = X_encoded.fillna(-999)
    return X_encoded

def feature_selection_mutual_info(X, y):
    """
    Select features based on mutual information regression.
    """
    mi_scores = mutual_info_regression(X, y, random_state=42)
    mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
    return mi_series

def main(file_path):
    # Load dataset
    df = load_data(file_path)
    
    # Select features & target
    X = df.drop(columns=['target'])
    y = df['target']

    # Encode and fill missing values
    X_encoded = encode_and_fill_data(X)

    # Feature selection using mutual information
    mi_series = feature_selection_mutual_info(X_encoded, y)
    print(mi_series.head(20))

    # Handling categorical columns
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    df_cat_encoded = df[cat_cols].apply(lambda x: x.astype('category').cat.codes)
    df_cat_encoded['target'] = df['target']

    X_cat = df_cat_encoded.drop(columns=['target'])
    y = df_cat_encoded['target']

    # Feature selection for categorical data
    mi_series_cat = feature_selection_mutual_info(X_cat, y)
    print(mi_series_cat)

    # Set index and drop unnecessary columns
    df.set_index('telephone_number', inplace=True)
    df = df.drop(['is_smartphone', 'is_featurephone'], axis=1)

    # Show the first few rows of the final dataframe
    print(df.head())

if __name__ == "__main__":
    # Path to your dataset file
    file_path = 'your_file.csv'  # Replace with the actual path to your dataset
    main(file_path)