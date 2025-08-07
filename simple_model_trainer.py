import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

class SimpleFloodPredictionModel:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_importance = None
        self.X_test = None
        self.y_test = None
        
    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """
        Train XGBoost model
        """
        # Convert to numpy arrays if needed
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        
        # Initialize XGBoost classifier with default parameters
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )
        
        # Train the model
        xgb_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = xgb_model.predict(X_test)
        y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Feature importance
        feature_importance = xgb_model.feature_importances_
        
        return {
            'model': xgb_model,
            'accuracy': accuracy,
            'auc_score': auc_score,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'feature_importance': feature_importance,
            'best_params': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1}
        }
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """
        Train Random Forest model
        """
        # Initialize Random Forest classifier
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # Train the model
        rf_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = rf_model.predict(X_test)
        y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Feature importance
        feature_importance = rf_model.feature_importances_
        
        return {
            'model': rf_model,
            'accuracy': accuracy,
            'auc_score': auc_score,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'feature_importance': feature_importance,
            'best_params': {'n_estimators': 100, 'max_depth': 10}
        }
    
    def train_logistic_regression(self, X_train, y_train, X_test, y_test):
        """
        Train Logistic Regression model
        """
        # Initialize Logistic Regression classifier
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        
        # Train the model
        lr_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = lr_model.predict(X_test)
        y_pred_proba = lr_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Feature importance (coefficients for logistic regression)
        feature_importance = np.abs(lr_model.coef_[0])
        
        return {
            'model': lr_model,
            'accuracy': accuracy,
            'auc_score': auc_score,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'feature_importance': feature_importance,
            'best_params': {'max_iter': 1000}
        }
    
    def train_all_models(self, X_train, y_train, X_test, y_test, feature_names):
        """
        Train all models and compare performance
        """
        # Store test data for detailed metrics
        self.X_test = X_test
        self.y_test = y_test
        
        print("Training XGBoost...")
        xgb_results = self.train_xgboost(X_train, y_train, X_test, y_test)
        self.models['XGBoost'] = xgb_results
        
        print("Training Random Forest...")
        rf_results = self.train_random_forest(X_train, y_train, X_test, y_test)
        self.models['Random Forest'] = rf_results
        
        print("Training Logistic Regression...")
        lr_results = self.train_logistic_regression(X_train, y_train, X_test, y_test)
        self.models['Logistic Regression'] = lr_results
        
        # Find best model
        best_accuracy = 0
        for model_name, results in self.models.items():
            if results['accuracy'] > best_accuracy:
                best_accuracy = results['accuracy']
                self.best_model = results['model']
                self.best_model_name = model_name
                self.feature_importance = results['feature_importance']
        
        # Create feature importance DataFrame
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df
    
    def get_model_comparison(self):
        """
        Get comparison of all models
        """
        comparison = {}
        
        for model_name, results in self.models.items():
            comparison[model_name] = {
                'accuracy': results['accuracy'],
                'auc_score': results['auc_score'],
                'best_params': results['best_params']
            }
        
        return comparison
    
    def get_detailed_metrics(self, model_name):
        """
        Get detailed metrics for a specific model
        """
        if model_name not in self.models:
            return None
        
        if self.X_test is None or self.y_test is None:
            return None
        
        results = self.models[model_name]
        y_pred = results['predictions']
        y_test = self.y_test
        
        # Calculate detailed metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm
        }
    
    def predict(self, X):
        """
        Make predictions using the best model
        """
        if self.best_model is None:
            raise ValueError("No model has been trained yet")
        
        return self.best_model.predict(X)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities using the best model
        """
        if self.best_model is None:
            raise ValueError("No model has been trained yet")
        
        return self.best_model.predict_proba(X)
