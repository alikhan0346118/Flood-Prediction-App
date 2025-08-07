import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class CSVDataLoader:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
    def load_data(self, filepath):
        """
        Load data from CSV file
        """
        try:
            data = pd.read_csv(filepath)
            print(f"Data loaded successfully: {data.shape[0]} rows, {data.shape[1]} columns")
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def preprocess_data(self, data):
        """
        Preprocess the flood prediction data
        """
        df = data.copy()
        
        # Check for missing values
        print("Missing values:")
        print(df.isnull().sum())
        
        # Handle missing values if any
        df = df.dropna()
        
        # Separate features and target first
        target_column = 'Flooded'
        if target_column not in df.columns:
            print(f"Target column '{target_column}' not found. Available columns: {df.columns.tolist()}")
            return None, None, None, None
        
        feature_columns = [col for col in df.columns if col != target_column]
        X = df[feature_columns]
        y = df[target_column].astype(int)  # Ensure target is integer
        
        # Encode categorical variables in features only
        categorical_columns = X.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
        
        self.feature_names = feature_columns
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features only (not the target)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Ensure target variables are integers and not scaled
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Features: {feature_columns}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        print(f"y_train unique values: {np.unique(y_train)}")
        print(f"y_test unique values: {np.unique(y_test)}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def get_feature_names(self):
        """
        Get feature names
        """
        return self.feature_names
    
    def transform_new_data(self, data):
        """
        Transform new data using fitted scaler and encoders
        """
        df = data.copy()
        
        # Encode categorical variables
        for col, le in self.label_encoders.items():
            if col in df.columns:
                df[col] = le.transform(df[col].astype(str))
        
        # Scale features
        df_scaled = self.scaler.transform(df)
        
        return df_scaled
    
    def get_data_summary(self, data):
        """
        Get summary statistics of the data
        """
        summary = {
            'total_samples': len(data),
            'features': len(data.columns) - 1,  # Excluding target
            'flooded_count': data['Flooded'].sum(),
            'non_flooded_count': len(data) - data['Flooded'].sum(),
            'flood_rate': data['Flooded'].mean() * 100,
            'feature_stats': data.describe().to_dict()
        }
        return summary
