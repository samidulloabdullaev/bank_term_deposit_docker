"""This file serves as the main execution point for preprocessing tasks."""

import pandas as pd
from helpers import select_sample_data, reduce_mem_usage
from engineering import new_numerical_cols, new_features
from preprocess import (
    convert_object_to_category, 
    transform_categorical_cols, 
    transform_num_cols,
    outlier_removal_iqr,
    stratified_target_encode_train_test,
)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import pandas as pd
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Custom transformers
class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_cols:list[str]):
        self.numerical_cols = numerical_cols
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return outlier_removal_iqr(X, self.numerical_cols)

class NumericalFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_cols:list[str]):
        self.numerical_cols = numerical_cols
    
    def fit(self, X:pd.DataFrame, y=None) -> 'NumericalFeatureEngineer':
        return self
    
    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        X = new_features(X, num_cols=self.numerical_cols)

        # update the numerical_cols after feature engineering
        self.numerical_cols = X.select_dtypes(include=['number']).columns.tolist()
        X = new_numerical_cols(df=X, numerical_cols=self.numerical_cols)
        return X

class ScalerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, scaler_type:str='standard'):
        self.scaler_type = scaler_type
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("Invalid scaler type. Choose 'standard' or 'minmax'.")
    
    def fit(self, X:pd.DataFrame, y=None) -> 'ScalerTransformer':
        self.scaler.fit(X)
        return self
    
    def transform(self, X:pd.DataFrame,) -> pd.DataFrame:
        return pd.DataFrame(self.scaler.transform(X), columns=X.columns, index=X.index)


class CategoricalFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, cat_cols:list[str], encoder_type:str='onehot'):
        self.cat_cols = cat_cols
        self.encoder_type = encoder_type
    
    def fit(self, X:pd.DataFrame, y=None) -> 'CategoricalFeatureEngineer':
        return self
    
    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        X = new_features(X, num_cols=self.cat_cols)
        
        # update the cat_cols after feature engineering
        self.cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        X = convert_object_to_category(df=X)
        X = transform_categorical_cols(df=X, cat_cols=self.cat_cols, encoder_type=self.encoder_type)
        return X


def main():
    logging.info("Reading dataset...")
    train = pd.read_csv('data/train.csv').drop(columns=['id'])
    test = pd.read_csv('data/test.csv').drop(columns=['id'])

    target_y = train['y'].copy()
    train = train.drop(columns=['y'], errors='ignore')
    # # separate numerical and categorical columns
    # num_cols = train.select_dtypes(include=['number']).columns.tolist()
    # cat_cols = train.select_dtypes(exclude=['number']).columns.tolist()

    # # Define pipeline steps for numerical and categorical columns
    # numerical_pipeline = Pipeline([
    #     ('outlier', OutlierRemover(num_cols)),
    #     ('feature_eng', NumericalFeatureEngineer(num_cols)),
    #     ('scaler', ScalerTransformer(scaler_type='standard')),
    # ])

    # categorical_pipeline = Pipeline([
    #     ('transform', CategoricalFeatureEngineer(cat_cols=cat_cols, encoder_type='onehot')),
    # ])

    # preprocessor = ColumnTransformer([
    #     ('num', numerical_pipeline, num_cols),
    #     ('cat', categorical_pipeline, cat_cols)
    # ])

    # logging.info("Fitting pipeline on training data...")
    # train_processed = preprocessor.fit_transform(train)
    # test_processed = preprocessor.transform(test)

    # # Convert processed arrays back to DataFrames
    # num_features = preprocessor.named_transformers_['num'].named_steps['feature_eng'].numerical_cols
    # cat_features = preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(cat_cols)
    
    # columns = list(num_features) + list(cat_features)
    
    # train_processed = pd.DataFrame(train_processed, columns=columns)
    # test_processed = pd.DataFrame(test_processed, columns=columns)

    # logging.info("Reducing memory usage...")
    # train_processed = reduce_mem_usage(train_processed)
    # test_processed = reduce_mem_usage(test_processed)

    # logging.info("Preprocessing completed successfully.")
    

    logging.info("Start feature engineering...")
    num_cols = train.select_dtypes(include=['number']).columns.tolist()
    cat_cols = train.select_dtypes(include=['object', 'category']).columns.tolist()

    logging.info("Basic Outlier Removal...")
    train_new = outlier_removal_iqr(df=train, numerical_cols=num_cols)
    test_new = outlier_removal_iqr(df=test, numerical_cols=num_cols)

    # logging.info("Creating new numerical features...")
    # train_new = new_numerical_cols(train_new, numerical_cols=num_cols)
    # test_new = new_numerical_cols(test_new, numerical_cols=num_cols)

    # logging.info("Creating new features...")
    # train_new = new_features(df=train_new, num_cols=num_cols)
    # test_new = new_features(df=test_new, num_cols=num_cols)

    logging.info("Converting object columns to category type...")
    train_new = convert_object_to_category(df=train_new)
    test_new = convert_object_to_category(df=test_new)

    logging.info("Preprocessing categorical columns...")
    # train_new = transform_categorical_cols(df=train_new, cat_cols=cat_cols, encoder_type='label')
    # test_new = transform_categorical_cols(df=test_new, cat_cols=cat_cols, encoder_type='label')

    # logging.info("Preprocessing numerical columns...")
    # train_new, test_new = transform_num_cols(train=train_new, test=test_new, numerical_cols=num_cols, scaler='standard')

    # add target column back to train set
    train_new['y'] = target_y

    logging.info("Reducing memory usage...")
    train_processed = reduce_mem_usage(train_new)
    test_processed = reduce_mem_usage(test_new)

    logging.info("Preprocessing completed successfully.")

    return train_processed, test_processed

if __name__ == "__main__":

    train_processed, test_processed = main()
    train_processed.to_csv('data/train_processed.csv', index=False)
    test_processed.to_csv('data/test_processed.csv', index=False)
    logging.info("Processed datasets saved.")