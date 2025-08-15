"""This file servers as the training script for the LightGBM model.
    It includes model initialization, training with cross-validation,
    and evaluation of the model's performance.
"""

import numpy as np 
import pandas as pd
import logging
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import log_loss, roc_auc_score
import joblib
from preprocess import convert_object_to_category

import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SEED = 42

def main() -> None:
    # read the preprocessed data
    logging.info("Loading preprocessed data...")
    train = pd.read_csv('data/train_processed.csv')
    test = pd.read_csv('data/test_processed.csv')

    # convert object columns to category
    logging.info("Converting object columns to category...")
    train = convert_object_to_category(train)
    test = convert_object_to_category(test)

    X = train.drop(columns=['y'])
    y = train['y']

    # model initialization
    logging.info("Initializing LightGBM models...")
    model_lgb = lgb.LGBMClassifier(
        max_depth=6,
        n_estimators=20000,
        learning_rate=0.03,
        reg_alpha=1.8,
        reg_lambda=3.5,
        colsample_bytree=0.5,
        subsample=0.8,
        max_bin=4523,
        objective= 'binary',
        metric= 'binary_logloss',
        boosting_type= 'gbdt',
        n_jobs= -1,
        random_state= SEED,
        verbose= -1,
    )

    model_lgb2 = lgb.LGBMClassifier(
        n_estimators = 40000,
        learning_rate = 0.0358306214515723,
        num_leaves = 228,
        max_depth = 6,
        min_child_samples = 83,
        subsample = 0.8788304820753131,
        colsample_bytree = 0.6169349166144594,
        reg_alpha = 3.700714656885025,
        reg_lambda = 4.709578317972932,
        objective = "binary",
        metric = "binary_logloss",
        n_jobs= -1,
        random_state= SEED,
        verbose= -1,
        max_bin=255,
    )


    # stratified k-fold cross-validation
    logging.info("Setting up Stratified K-Fold cross-validation...")
    n_splits = 5
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    # Out-of-fold (OOF) predictions for blending on the training data
    oof_lgb = np.zeros(X.shape[0])
    oof_lgb2 = np.zeros(X.shape[0])  

    logloss_lgb = np.zeros(n_splits)
    logloss_lgb2 = np.zeros(n_splits)

    auc_score_lgb = np.zeros(n_splits)
    auc_score_lgb2 = np.zeros(n_splits)

    # Predictions on the test set
    lgb_preds = np.zeros(test.shape[0])
    lgb_preds2 = np.zeros(test.shape[0]) 

    # Cross-validation training
    logging.info("Starting cross-validation training...")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        logging.info(f"\nTraining fold {fold + 1}/{n_splits}")
        
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        model_lgb.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(period=1000)]
        )
        
        oof_lgb[val_idx] = model_lgb.predict_proba(X_val)[:, 1]
        lgb_preds += model_lgb.predict_proba(test)[:, 1] / n_splits
        
        # save log loss and roc auc score for each fold
        logloss_lgb[fold] = log_loss(y_val, oof_lgb[val_idx])
        auc_score_lgb[fold] = roc_auc_score(y_val, oof_lgb[val_idx])

        model_lgb2.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(period=1000)]
        )
        
        oof_lgb2[val_idx] = model_lgb2.predict_proba(X_val)[:, 1]
        lgb_preds2 += model_lgb2.predict_proba(test)[:, 1] / n_splits
        
        # save log loss and roc auc score for each fold
        logloss_lgb2[fold] = log_loss(y_val, oof_lgb2[val_idx])
        auc_score_lgb2[fold] = roc_auc_score(y_val, oof_lgb2[val_idx])

    # Final evaluation on the training data
    logging.info("Final evaluation on the training data...")
    logging.info(f"OOF LightGBM mean log loss: {np.mean(logloss_lgb):.4f}")
    logging.info(f"OOF LightGBM2 mean log loss: {np.mean(logloss_lgb2):.4f}")

    logging.info(f"OOF LightGBM mean ROC AUC: {np.mean(auc_score_lgb):.4f}")
    logging.info(f"OOF LightGBM2 mean ROC AUC: {np.mean(auc_score_lgb2):.4f}")
    
    # Save the models
    logging.info("Saving the trained models...")    
    
    # Save the predictions
    logging.info("Saving best model predictions...")

    if np.mean(logloss_lgb) < np.mean(logloss_lgb2):
        logging.info("Using model_lgb for predictions.")
        joblib.dump(model_lgb, 'models/best_model.pkl')
        lgb_preds = lgb_preds
    else:
        logging.info("Using model_lgb2 for predictions.")
        joblib.dump(model_lgb2, 'models/best_model.pkl')
        lgb_preds = lgb_preds2
    

    sample_submission = pd.read_csv('data/sample_submission.csv')
    sample_submission['y'] = lgb_preds

    sample_submission.to_csv('data/submission.csv', index=False)
    logging.info("Predictions saved successfully.")


if __name__ == "__main__":
    main()
    logging.info("Training script completed successfully.")

