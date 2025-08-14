"""This module provides utility functions throughout the project."""

import numpy as np
import pandas as pd
import warnings


def select_sample_data(df: pd.DataFrame, sample_size: int = 100) -> pd.DataFrame:
    """
    Select a random sample of data from the given DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame from which to select a sample.
    sample_size (int): The number of samples to select. Default is 100.

    Returns:
    pd.DataFrame: A DataFrame containing the sampled data.
    """
    if sample_size <= 0:
        raise ValueError("Sample size must be a positive integer.")
    
    if len(df) == 0:
        raise ValueError("Input DataFrame is empty. Cannot sample from an empty DataFrame.")
    
    # Ensure sample size does not exceed the number of available rows
    if sample_size > len(df):
        raise ValueError(f"Sample size exceeds the number of available {len(df)} rows in the DataFrame.")
    
    # Sample the DataFrame
    np.random.seed(42)  # For reproducibility
    return df.sample(n=sample_size, random_state=42).reset_index(drop=True)


def reduce_mem_usage(df: pd.DataFrame, verbose:bool = True):
    """    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.
    Args:       
        df (pd.DataFrame): The dataframe to optimize.
        verbose (bool): If True, prints memory usage before and after optimization. 
    Returns:
        pd.DataFrame: The dataframe with optimized data types.
    """

    mem_before = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f"Memory usage of dataframe is {mem_before:.2f} MB")
    
    # Ignore warnings related to pandas downcasting
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        for col in df.columns:
            col_type = df[col].dtype
            
            if str(col_type)[:3] == 'int':
                c_min = df[col].min()
                c_max = df[col].max()
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)

                        
            elif str(col_type)[:5] == 'float':
                c_min = df[col].min()
                c_max = df[col].max()
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float64).min and c_max < np.finfo(np.float64).max:
                    df[col] = df[col].astype(np.float64)

    
    mem_after = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f"Memory usage after optimization is: {mem_after:.2f} MB")
        print(f"Decreased by {(100 * (mem_before - mem_after) / mem_before):.1f}%")
        
    return df
