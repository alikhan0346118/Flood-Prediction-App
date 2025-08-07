#!/usr/bin/env python3
"""
Test script for Flood Prediction App
This script tests the core functionality without running the full Streamlit app
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def test_data_loading():
    """Test if data can be loaded and processed"""
    print("Testing data loading...")
    try:
        # Load data
        df = pd.read_csv('dataset/synthetic_flood_data_pakistan.csv')
        print(f"✅ Data loaded successfully: {len(df)} records")
        
        # Check columns
        expected_columns = ['Rainfall_mm', 'Elevation_m', 'Distance_to_River_km', 
                          'Land_Cover_Type', 'Population_Density', 'Flooded']
        missing_columns = [col for col in expected_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ Missing columns: {missing_columns}")
            return False
        else:
            print("✅ All expected columns present")
        
        # Check data types
        print(f"✅ Data types: {df.dtypes.to_dict()}")
        
        # Basic statistics
        print(f"✅ Flood events: {df['Flooded'].sum()} out of {len(df)} ({df['Flooded'].mean()*100:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False

def test_model_training():
    """Test if models can be trained"""
    print("\nTesting model training...")
    try:
        # Load and prepare data
        df = pd.read_csv('dataset/synthetic_flood_data_pakistan.csv')
        
        # Encode categorical variables
        le = LabelEncoder()
        df['Land_Cover_Type_Encoded'] = le.fit_transform(df['Land_Cover_Type'])
        
        # Prepare features
        feature_columns = ['Rainfall_mm', 'Elevation_m', 'Distance_to_River_km', 
                          'Land_Cover_Type_Encoded', 'Population_Density']
        X = df[feature_columns]
        y = df['Flooded']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"✅ Data split: {len(X_train)} training, {len(X_test)} testing")
        
        # Test models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss', n_estimators=50),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            if name == 'Logistic Regression':
                # Scale features for logistic regression
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            results[name] = {'accuracy': accuracy, 'auc': auc}
            print(f"✅ {name}: Accuracy={accuracy:.3f}, AUC={auc:.3f}")
        
        # Check if models perform reasonably well
        avg_accuracy = np.mean([r['accuracy'] for r in results.values()])
        avg_auc = np.mean([r['auc'] for r in results.values()])
        
        print(f"\n✅ Average Accuracy: {avg_accuracy:.3f}")
        print(f"✅ Average AUC: {avg_auc:.3f}")
        
        if avg_accuracy > 0.5 and avg_auc > 0.5:
            print("✅ Models performing reasonably well")
            return True
        else:
            print("⚠️ Models performing below expected threshold")
            return False
            
    except Exception as e:
        print(f"❌ Error training models: {e}")
        return False

def test_prediction():
    """Test if predictions work"""
    print("\nTesting prediction functionality...")
    try:
        # Load and prepare data
        df = pd.read_csv('dataset/synthetic_flood_data_pakistan.csv')
        le = LabelEncoder()
        df['Land_Cover_Type_Encoded'] = le.fit_transform(df['Land_Cover_Type'])
        
        # Prepare features
        feature_columns = ['Rainfall_mm', 'Elevation_m', 'Distance_to_River_km', 
                          'Land_Cover_Type_Encoded', 'Population_Density']
        X = df[feature_columns]
        y = df['Flooded']
        
        # Train a simple model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        
        # Test prediction
        test_input = np.array([[100.0, 300.0, 2.0, 1, 1000.0]])  # Sample input
        prediction = model.predict(test_input)[0]
        probability = model.predict_proba(test_input)[0][1]
        
        print(f"✅ Prediction test successful")
        print(f"✅ Sample input prediction: {prediction} (probability: {probability:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing prediction: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Flood Prediction App Components")
    print("=" * 50)
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Model Training", test_model_training),
        ("Prediction", test_prediction)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! The app should work correctly.")
        print("\nTo run the app, use: streamlit run app.py")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    return passed == len(results)

if __name__ == "__main__":
    main()
