# Configuration file for Flood Prediction App

# App Configuration
APP_TITLE = "Flood Prediction App"
APP_ICON = "🌊"
PAGE_LAYOUT = "wide"

# Model Configuration
MODELS = {
    'Random Forest': {
        'n_estimators': 100,
        'random_state': 42,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1
    },
    'XGBoost': {
        'random_state': 42,
        'eval_metric': 'logloss',
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1
    },
    'Logistic Regression': {
        'random_state': 42,
        'max_iter': 1000,
        'C': 1.0,
        'solver': 'lbfgs'
    }
}

# Data Configuration
FEATURE_COLUMNS = [
    'Rainfall_mm',
    'Elevation_m', 
    'Distance_to_River_km',
    'Land_Cover_Type_Encoded',
    'Population_Density'
]

TARGET_COLUMN = 'Flooded'

# Training Configuration
TEST_SIZE = 0.2
RANDOM_STATE = 42
STRATIFY = True

# Risk Assessment Thresholds
RISK_THRESHOLDS = {
    'HIGH': 0.7,
    'MODERATE': 0.4,
    'LOW': 0.0
}

# Visualization Configuration
PLOT_COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e', 
    'success': '#2ca02c',
    'warning': '#ffc107',
    'danger': '#dc3545'
}

# UI Configuration
SIDEBAR_STATE = "expanded"
MAX_DISPLAY_ROWS = 10

# File Paths
DATASET_PATH = 'dataset/synthetic_flood_data_pakistan.csv'

# Recommendations by Risk Level
RECOMMENDATIONS = {
    'HIGH': [
        "Immediate evacuation may be necessary",
        "Monitor weather conditions closely", 
        "Prepare emergency supplies",
        "Contact local authorities"
    ],
    'MODERATE': [
        "Stay alert to weather updates",
        "Prepare emergency plan",
        "Monitor water levels", 
        "Keep emergency contacts ready"
    ],
    'LOW': [
        "Continue normal activities",
        "Stay informed about weather",
        "Maintain emergency preparedness",
        "Follow local guidelines"
    ]
}
