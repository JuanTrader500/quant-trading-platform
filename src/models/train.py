import pandas as pd
import numpy as np
import os
import logging
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, processed_data_path="data/processed/processed_features.csv", model_dir="models/artifacts"):
        self.processed_data_path = processed_data_path
        self.model_dir = model_dir
        
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def prepare_data(self, df):
        """
        Split data temporally and normalize features.
        """
        logger.info("Preparing data for training...")
        
        # Define features and target
        # Based on FinalModel.ipynb logic
        target_col = 'log_target'
        feature_cols = [col for col in df.columns if col not in [target_col, 'date']]
        
        X = df[feature_cols]
        y = df[target_col]
        
        # Temporal split (80% train, 20% test)
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Normalization
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Save scaler
        joblib.dump(scaler, os.path.join(self.model_dir, "scaler.pkl"))
        
        return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols

    def train_and_evaluate(self):
        """
        Train multiple models and find the best one.
        """
        if not os.path.exists(self.processed_data_path):
            logger.error("Processed data not found. Please run the data preparation pipeline first.")
            return

        df = pd.read_csv(self.processed_data_path)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
        X_train, X_test, y_train, y_test, features = self.prepare_data(df)
        
        models_config = {
            'Ridge': {
                'model': Ridge(),
                'params': {'alpha': [0.1, 1, 10, 100], 'solver': ['svd', 'cholesky']}
            },
            'Lasso': {
                'model': Lasso(),
                'params': {'alpha': [0.0001, 0.001, 0.01, 0.1]}
            },
            'ElasticNet': {
                'model': ElasticNet(),
                'params': {'alpha': [0.001, 0.01, 0.1], 'l1_ratio': [0.1, 0.5, 0.9]}
            },
            'RandomForest': {
                'model': RandomForestRegressor(),
                'params': {'n_estimators': [50, 100], 'max_depth': [3, 5, 10]}
            },
            'XGBoost': {
                'model': XGBRegressor(),
                'params': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [3, 5]}
            },
            'LightGBM': {
                'model': LGBMRegressor(),
                'params': {'n_estimators': [100], 'learning_rate': [0.05], 'num_leaves': [15, 31]}
            }
        }
        
        results = []
        best_rmse = float('inf')
        best_model_name = None

        for name, config in models_config.items():
            logger.info(f"Training {name}...")
            grid = GridSearchCV(config['model'], config['params'], cv=5, scoring='neg_root_mean_squared_error')
            grid.fit(X_train, y_train)
            
            best_model = grid.best_estimator_
            preds = best_model.predict(X_test)
            
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            
            results.append({
                'Model': name,
                'RMSE_Test': rmse,
                'MAE_Test': mae,
                'R2_Test': r2,
                'Best_Params': grid.best_params_
            })
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_model_name = name
                joblib.dump(best_model, os.path.join(self.model_dir, "best_model.pkl"))
                logger.info(f"New best model found: {name} with RMSE {rmse:.6f}")

        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(self.model_dir, "model_comparison_results.csv"), index=False)
        logger.info(f"Final best model: {best_model_name}")
        return results_df

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train_and_evaluate()
