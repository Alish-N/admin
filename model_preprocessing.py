import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold
from xgboost import XGBRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                           explained_variance_score, precision_score, recall_score,
                           accuracy_score)
import joblib

class ISPCustomerSatisfaction:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.model = None
        self.feature_importance = None
        
    def preprocess_data(self, df):
        """Preprocess the input data"""
        df_processed = df.copy()
        
        # Define feature columns
        numeric_columns = [
            'monthly_fee',
            'data_usage_gb',
            'avg_speed_mbps',
            'promised_speed',
            'uptime_percentage',
            'active_tickets'
        ]
        
        categorical_columns = [
            'service_plan',
            'connection_type',
            'payment_status',
            'ticket_type',
            'ticket_status'
        ]
        
        # Process numeric columns
        for col in numeric_columns:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                
                # Handle outliers using IQR method
                Q1 = df_processed[col].quantile(0.25)
                Q3 = df_processed[col].quantile(0.75)
                IQR = Q3 - Q1
                df_processed[col] = df_processed[col].clip(
                    lower=Q1 - 1.5 * IQR,
                    upper=Q3 + 1.5 * IQR
                )
        
        # Add engineered features
        if all(col in df_processed.columns for col in ['avg_speed_mbps', 'promised_speed']):
            df_processed['speed_ratio'] = df_processed['avg_speed_mbps'] / df_processed['promised_speed']
        
        if all(col in df_processed.columns for col in ['monthly_fee', 'data_usage_gb']):
            df_processed['cost_per_gb'] = df_processed['monthly_fee'] / df_processed['data_usage_gb']
        
        # Process categorical columns
        for col in categorical_columns:
            if col in df_processed.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    self.label_encoders[col].fit(df_processed[col].astype(str))
                df_processed[col] = self.label_encoders[col].transform(df_processed[col].astype(str))
        
        # Select final features
        features = numeric_columns + ['speed_ratio', 'cost_per_gb'] + categorical_columns
        df_processed = df_processed[features].fillna(0)
        
        return df_processed
    
    def train_model(self, data_path):
        """Train the model"""
        try:
            # Load and validate data
            df = pd.read_csv(data_path)
            if 'customer_satisfaction' not in df.columns:
                raise ValueError("Target column 'customer_satisfaction' not found")
            
            # Preprocess features
            X = self.preprocess_data(df)
            y = df['customer_satisfaction']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Create validation set
            X_train_final, X_val, y_train_final, y_val = train_test_split(
                X_train_scaled, y_train, test_size=0.2, random_state=42
            )
            
            # Initialize model
            self.model = XGBRegressor(
                n_estimators=100,
                learning_rate=0.03,
                max_depth=3,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.7,
                objective='reg:squarederror',
                random_state=42
            )
            
            # Train model with eval set
            eval_set = [(X_val, y_val)]
            self.model.fit(
                X_train_final,
                y_train_final,
                eval_set=eval_set,
                verbose=False
            )
            
            # Store feature importance
            self.feature_importance = dict(zip(X.columns, self.model.feature_importances_))
            
            # Calculate predictions
            train_pred = self.model.predict(X_train_scaled)
            test_pred = self.model.predict(X_test_scaled)
            
            # Calculate metrics
            metrics = {
                'train_metrics': {
                    'regression': self.calculate_regression_metrics(y_train, train_pred, X_train),
                    'classification': self.calculate_classification_metrics(y_train, train_pred)
                },
                'test_metrics': {
                    'regression': self.calculate_regression_metrics(y_test, test_pred, X_test),
                    'classification': self.calculate_classification_metrics(y_test, test_pred)
                },
                'feature_importance': self.feature_importance
            }
            
            # Perform cross-validation
            cv_scores = self.perform_cross_validation(X_train_scaled, y_train)
            metrics['cross_validation'] = {
                'r2_mean': float(cv_scores.mean()),
                'r2_std': float(cv_scores.std())
            }
            
            return metrics
            
        except Exception as e:
            print(f"Error in train_model: {str(e)}")
            raise
    
    def predict(self, input_data):
        """Make predictions"""
        try:
            processed_data = self.preprocess_data(input_data)
            scaled_data = self.scaler.transform(processed_data)
            prediction = self.model.predict(scaled_data)
            
            return {
                'prediction': prediction[0],
                'feature_importance': self.feature_importance
            }
        except Exception as e:
            print(f"Error in predict: {str(e)}")
            raise
    
    def perform_cross_validation(self, X, y, n_splits=5):
        """Perform manual cross-validation"""
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = []
        
        # Convert to numpy arrays if they aren't already
        X_array = X if isinstance(X, np.ndarray) else X.to_numpy()
        y_array = y if isinstance(y, np.ndarray) else y.to_numpy()
        
        for train_idx, val_idx in kf.split(X_array):
            # Split data using numpy indexing
            X_train_cv, X_val_cv = X_array[train_idx], X_array[val_idx]
            y_train_cv, y_val_cv = y_array[train_idx], y_array[val_idx]
            
            # Train model with basic parameters
            model_cv = XGBRegressor(
                n_estimators=1000,
                learning_rate=0.01,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='reg:squarederror',
                random_state=42
            )
            
            # Fit model
            model_cv.fit(X_train_cv, y_train_cv)
            
            # Predict and calculate R²
            y_pred_cv = model_cv.predict(X_val_cv)
            cv_scores.append(r2_score(y_val_cv, y_pred_cv))
        
        return np.array(cv_scores)
    
    def calculate_classification_metrics(self, y_true, y_pred, threshold=7.0):
        """Calculate classification metrics using a satisfaction threshold"""
        # Convert continuous predictions to binary (satisfied vs not satisfied)
        y_true_binary = (y_true >= threshold).astype(int)
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        return {
            'accuracy': accuracy_score(y_true_binary, y_pred_binary),
            'precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
            'recall': recall_score(y_true_binary, y_pred_binary, zero_division=0)
        }
    
    def calculate_regression_metrics(self, y_true, y_pred, X):
        """Calculate regression metrics"""
        n = len(y_true)
        p = X.shape[1]  # number of predictors
        
        # Calculate R² score
        r2 = r2_score(y_true, y_pred)
        
        # Calculate adjusted R²
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        
        # Calculate explained variance score
        exp_var = explained_variance_score(y_true, y_pred)
        
        return {
            'r2': r2,
            'adjusted_r2': adjusted_r2,
            'explained_variance': exp_var,
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
        }
    
    def save_model(self, path):
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_importance': self.feature_importance
        }
        joblib.dump(model_data, path)
    
    def load_model(self, path):
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_importance = model_data.get('feature_importance', None)