from typing import List
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from utils_and_constants import (
    DROP_COLNAMES,
    PROCESSED_DATASET,
    RAW_DATASET,
    TARGET_COLUMN,
)


def read_dataset(
    filename: str, drop_columns: List[str], target_column: str
) -> pd.DataFrame:
    df = pd.read_csv(filename, sep=";").drop(columns=drop_columns)
    
    df[target_column] = df[target_column].map({"yes": 1, "no": 0})
    
    # Drop rows if mapping failed (safety check)
    df = df.dropna(subset=[target_column])
    return df

def target_encode_categorical_features(
    df: pd.DataFrame, categorical_columns: List[str], target_column: str
) -> pd.DataFrame:
    encoded_data = df.copy()
    for col in categorical_columns:
        encoding_map = df.groupby(col)[target_column].mean().to_dict()
        encoded_data[col] = encoded_data[col].map(encoding_map)
    return encoded_data

def impute_and_scale_data(df_features: pd.DataFrame) -> pd.DataFrame:
    imputer = SimpleImputer(strategy="mean")
    X_preprocessed = imputer.fit_transform(df_features)

    scaler = StandardScaler()
    X_preprocessed = scaler.fit_transform(X_preprocessed)

    return pd.DataFrame(X_preprocessed, columns=df_features.columns)

def main():
    # 1. Read data
    banking_marketing = read_dataset(
        filename=RAW_DATASET, drop_columns=DROP_COLNAMES, target_column=TARGET_COLUMN
    )

    # 2. Target encode categorical columns
    categorical_columns = banking_marketing.select_dtypes(include=["object"]).columns.to_list()
    
    banking_marketing = target_encode_categorical_features(
        df=banking_marketing, 
        categorical_columns=categorical_columns, 
        target_column=TARGET_COLUMN
    )

    # 3. Impute and scale features
    banking_features_processed = impute_and_scale_data(
        banking_marketing.drop(columns=[TARGET_COLUMN], axis=1)
    )

    # 4. Write processed dataset
    banking_labels = banking_marketing[TARGET_COLUMN].reset_index(drop=True)
    banking_marketing_final = pd.concat([banking_features_processed, banking_labels], axis=1)
    banking_marketing_final.to_csv(PROCESSED_DATASET, index=None)
    print(f"Processed dataset saved to {PROCESSED_DATASET}")

if __name__ == "__main__":
    main()