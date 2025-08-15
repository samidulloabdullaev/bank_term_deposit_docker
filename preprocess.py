"""This file contains preprocessing functions to create new features from the dataset."""

import pandas as pd
import numpy as np 

from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from collections import Counter

# Convert object columns to category type
def convert_object_to_category(df:pd.DataFrame) -> pd.DataFrame:

    object_cols = df.select_dtypes(include=['object', 'category']).columns

    if object_cols.empty:
        raise ValueError("No object columns found in the DataFrame.")
    
    for col in object_cols:
        df[col] = df[col].astype('category')

    return df

# Function to preprocess categorical columns
def transform_categorical_cols(df:pd.DataFrame, cat_cols:list[str], encoder_type:str='onehot') -> pd.DataFrame:
    # Defensive check to ensure the input DataFrame is not None
    if df is None:
        raise ValueError("Input DataFrame for categorical preprocessing is None. "
                         "Please check the return value of the preceding function.")

    # Select the encoder based on the encoder_type argument
    if encoder_type == 'onehot':
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoded_cols = encoder.fit_transform(df[cat_cols])
        feature_names = encoder.get_feature_names_out(cat_cols)
        encoded_df = pd.DataFrame(encoded_cols, columns=feature_names, index=df.index)
    elif encoder_type == 'ordinal':
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        encoded_cols = encoder.fit_transform(df[cat_cols])
        feature_names = encoder.get_feature_names_out(cat_cols)
        encoded_df = pd.DataFrame(encoded_cols, columns=feature_names, index=df.index)
    elif encoder_type == 'label':
        encoded_df = df[cat_cols].copy()
        for col in cat_cols:
            le = LabelEncoder()
            encoded_df[col + "_le"] = le.fit_transform(df[col].astype(str))
        encoded_df = encoded_df[[col + "_le" for col in cat_cols]]
        feature_names = encoded_df.columns
    else:
        raise ValueError("Invalid encoder_type. Choose 'onehot', 'ordinal', or 'label'.")

    # Select the remaining columns
    remainder_cols = df.drop(columns=cat_cols)

    # Concatenate the encoded columns with the remaining columns
    processed_df = pd.concat([encoded_df, remainder_cols], axis=1)

    return processed_df


# Preprocess numerical columns
def transform_num_cols(train:pd.DataFrame, test:pd.DataFrame, numerical_cols:list[str], scaler:str="standard" ) -> pd.DataFrame:
    # Defensive check to ensure the input DataFrame is not None
    if train is None and test is None:
        raise ValueError("Input DataFrames for numerical preprocessing are None. "
                         "Please check the return value of the preceding function.")

    if scaler == 'standard':
        scaler = StandardScaler()
    elif scaler == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Invalid scaler type. Choose 'standard' or 'minmax'.")
    

    # Standardize numerical columns
    for col in numerical_cols:
        if col not in train.columns or col not in test.columns:
            raise ValueError(f"Column {col} is missing in one of the DataFrames.")
        
        train[col] = scaler.fit_transform(train[[col]])
        test[col] = scaler.transform(test[[col]])
    
    return train, test


def outlier_removal_iqr(df:pd.DataFrame, numerical_cols:list[str]) -> pd.DataFrame:

    """Preprocess numerical columns by clipping outliers."""
    df = df.copy()

    for col in numerical_cols:
        lower_bound = df[col].quantile(0.005)
        upper_bound = df[col].quantile(0.995)
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    return df


# Target encoding for categorical columns with differtent methods
# target encoding
def stratified_target_encode_train_test(
        train_df:pd.DataFrame, 
        test_df:pd.DataFrame, 
        cat_cols:list[str], 
        target_col:str, 
        n_splits:int=5, 
        method:str="mean", 
        smoothing:float=0.3, 
        random_state:int=42
    ):

    train_encoded = train_df.copy()
    test_encoded = test_df.copy()
    global_stat = None

    for col in cat_cols:
        train_encoded[col + "_te"] = np.nan
        test_encoded[col + "_te"] = np.nan

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for train_idx, valid_idx in skf.split(train_df, train_df[target_col]):
            fold_train, fold_valid = train_df.iloc[train_idx], train_df.iloc[valid_idx]

            if method == "mean":
                stats = fold_train.groupby(col)[target_col].mean()
            elif method == "median":
                stats = fold_train.groupby(col)[target_col].median()
            elif method == "mode":
                stats = fold_train.groupby(col)[target_col].agg(lambda x: Counter(x).most_common(1)[0][0])
            else:
                raise ValueError("Invalid method: choose from 'mean', 'median', 'mode'")

            train_encoded.loc[valid_idx, col + "_te"] = fold_valid[col].map(stats).astype(float)

        if method == "mean":
            global_stat = train_df[target_col].mean()
        elif method == "median":
            global_stat = train_df[target_col].median()
        elif method == "mode":
            global_stat = Counter(train_df[target_col]).most_common(1)[0][0]

        train_encoded[col + "_te"].fillna(global_stat, inplace=True)

        full_stats = train_df.groupby(col)[target_col].mean() if method == "mean" else \
                     train_df.groupby(col)[target_col].median() if method == "median" else \
                     train_df.groupby(col)[target_col].agg(lambda x: Counter(x).most_common(1)[0][0])

        test_encoded[col + "_te"] = test_encoded[col].map(full_stats).astype(float)
        test_encoded[col + "_te"].fillna(global_stat, inplace=True)

    return train_encoded, test_encoded
