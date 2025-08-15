"""This file contains feature engineering functions to create new features from the dataset."""

import pandas as pd
import numpy as np 

from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, PrecisionRecallDisplay

def new_numerical_cols(df:pd.DataFrame, numerical_cols:list[str]) -> pd.DataFrame:
    
    if df is None or len(df) == 0:
        raise ValueError("Input DataFrame is None or empty. Please provide a valid DataFrame.")
    
    if df[numerical_cols].shape[1] == 0:
        raise ValueError("No numerical columns found in the DataFrame. Please provide valid numerical columns.")

    df['duration_log'] = np.log1p(df['duration'])
    df['campaign_log'] = np.log1p(df['campaign'])
    df['pdays_log'] = np.log1p(df['pdays'] + 2)
    df['previous_log'] = np.log1p(df['previous'] + 1)

    new_numerical_cols = ['duration_log', 'campaign_log', 'pdays_log', 'previous_log']
    
    # calculate correlation between new generated cols and original cols. if correlation >= 90 drop new generated col
    drop_cols = set()
    for new_cols in new_numerical_cols:
        for num_cols in numerical_cols:
            correlation = df[num_cols].corr(df[new_cols])
            if correlation >= 0.90 or correlation <= -0.90:
                print(f"There is high correlation - {correlation:.2f} between {num_cols} and {new_cols}. {new_cols} has been dropped to avoid multicollinarity!")
                drop_cols.add(new_cols)
                
    drop_cols = ['id'] + list(drop_cols)
    
    return df[[col for col in df.columns if col not in drop_cols]]


def new_features(df:pd.DataFrame, num_cols:list[str]) -> pd.DataFrame:
    
    df = df.copy()
    df['duration_balance'] = df['duration'] * df['balance']
    df['duration_age'] = df['duration'] * df['age']
    df['duration_age_balance'] = df['duration'] * df['age'] * df['balance']
    df['duration_day'] = df['duration'] * df['day']
    df['duration_age_day'] = df['duration'] * df['age'] * df['day']

    for col in num_cols:
        if col != 'duration':
            df[f"{col}_duration_weight"] = df['duration'] / (df[col] + 0.1)

    df['balance_log'] = np.log1p(df['balance']).clip(lower=0)
    df['job_edu'] = df['job'].astype(str) + "_" + df['education'].astype(str)
    df['contacted_before'] = (df['pdays'] != -1).astype(int)
    df['age_squared'] = df['age'] ** 2

    df['duration_sin'] = np.sin(2*np.pi * df['duration'] / 400)
    df['duration_cos'] = np.cos(2*np.pi * df['duration'] / 400)

    month_map = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
        'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
        'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    df['month_num'] = df['month'].map(month_map).astype('int')

    df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)

    df.drop('month_num',axis=1,inplace=True)

    return df