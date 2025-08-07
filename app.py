import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Flood Prediction App",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the flood dataset"""
    try:
        df = pd.read_csv('dataset/synthetic_flood_data_pakistan.csv')
        
        # Create a copy to avoid modifying the original data
        df_processed = df.copy()
        
        # Encode categorical variables
        le = LabelEncoder()
        df_processed['Land_Cover_Type_Encoded'] = le.fit_transform(df_processed['Land_Cover_Type'])
        
        return df_processed, le
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

def prepare_features(df):
    """Prepare features for modeling"""
    # Select features for modeling
    feature_columns = ['Rainfall_mm', 'Elevation_m', 'Distance_to_River_km', 
                      'Land_Cover_Type_Encoded', 'Population_Density']
    
    X = df[feature_columns]
    y = df['Flooded']
    
    return X, y

def train_models(X, y):
    """Train multiple ML models"""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    # Train models
    trained_models = {}
    results = {}
    
    for name, model in models.items():
        if name == 'Logistic Regression':
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
        
        trained_models[name] = {
            'model': model,
            'scaler': scaler if name == 'Logistic Regression' else None,
            'accuracy': accuracy,
            'auc': auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'y_test': y_test
        }
        
        results[name] = {
            'accuracy': accuracy,
            'auc': auc
        }
    
    return trained_models, results, X_test, y_test

def plot_model_comparison(results):
    """Plot model performance comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy comparison
    models = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in models]
    
    bars1 = ax1.bar(models, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # AUC comparison
    aucs = [results[model]['auc'] for model in models]
    bars2 = ax2.bar(models, aucs, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax2.set_title('Model AUC Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('AUC')
    ax2.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, auc in zip(bars2, aucs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{auc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_roc_curves(trained_models):
    """Plot ROC curves for all models"""
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, (name, model_data) in enumerate(trained_models.items()):
        fpr, tpr, _ = roc_curve(model_data['y_test'], model_data['y_pred_proba'])
        auc = model_data['auc']
        
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name=f'{name} (AUC = {auc:.3f})',
            line=dict(color=colors[i], width=2)
        ))
    
    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        name='Random Classifier',
        line=dict(color='gray', width=1, dash='dash')
    ))
    
    fig.update_layout(
        title='ROC Curves Comparison',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=800,
        height=500
    )
    
    return fig

def plot_feature_importance(trained_models, feature_names):
    """Plot feature importance for tree-based models"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Random Forest feature importance
    rf_model = trained_models['Random Forest']['model']
    rf_importance = rf_model.feature_importances_
    
    axes[0].barh(feature_names, rf_importance, color='#1f77b4')
    axes[0].set_title('Random Forest Feature Importance', fontweight='bold')
    axes[0].set_xlabel('Importance')
    
    # XGBoost feature importance
    xgb_model = trained_models['XGBoost']['model']
    xgb_importance = xgb_model.feature_importances_
    
    axes[1].barh(feature_names, xgb_importance, color='#ff7f0e')
    axes[1].set_title('XGBoost Feature Importance', fontweight='bold')
    axes[1].set_xlabel('Importance')
    
    plt.tight_layout()
    return fig

def plot_confusion_matrices(trained_models):
    """Plot confusion matrices for all models"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, (name, model_data) in enumerate(trained_models.items()):
        cm = confusion_matrix(model_data['y_test'], model_data['y_pred'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'{name} Confusion Matrix', fontweight='bold')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    plt.tight_layout()
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🌊 Flood Prediction App</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Advanced machine learning models to predict flood risk and help make informed decisions
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading and processing data..."):
        df, label_encoder = load_data()
    
    if df is None:
        st.error("Failed to load data. Please check if the dataset file exists.")
        return
    
    # Sidebar
    st.sidebar.markdown("## 🎯 Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["📊 Data Overview", "🤖 Model Training", "📈 Model Comparison", "🔮 Predict Flood Risk", "📋 About"]
    )
    
    if page == "📊 Data Overview":
        show_data_overview(df)
    elif page == "🤖 Model Training":
        show_model_training(df)
    elif page == "📈 Model Comparison":
        show_model_comparison(df)
    elif page == "🔮 Predict Flood Risk":
        show_prediction_interface(df, label_encoder)
    elif page == "📋 About":
        show_about_page()

def show_data_overview(df):
    st.markdown('<h2 class="sub-header">📊 Dataset Overview</h2>', unsafe_allow_html=True)
    
    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Flood Events", df['Flooded'].sum())
    with col3:
        st.metric("No Flood Events", (df['Flooded'] == 0).sum())
    with col4:
        flood_rate = (df['Flooded'].sum() / len(df)) * 100
        st.metric("Flood Rate", f"{flood_rate:.1f}%")
    
    # Data preview
    st.subheader("📋 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Statistical summary
    st.subheader("📈 Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)
    
    # Visualizations
    st.subheader("📊 Data Visualizations")
    
    # Distribution plots
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(df, x='Rainfall_mm', color='Flooded', 
                          title='Rainfall Distribution by Flood Status',
                          color_discrete_map={0: '#1f77b4', 1: '#ff7f0e'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(df, x='Elevation_m', color='Flooded',
                          title='Elevation Distribution by Flood Status',
                          color_discrete_map={0: '#1f77b4', 1: '#ff7f0e'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("🔥 Correlation Analysis")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(correlation_matrix, 
                    title='Feature Correlation Matrix',
                    color_continuous_scale='RdBu',
                    aspect='auto')
    st.plotly_chart(fig, use_container_width=True)
    
    # Land cover distribution
    st.subheader("🌍 Land Cover Distribution")
    land_cover_counts = df['Land_Cover_Type'].value_counts()
    fig = px.pie(values=land_cover_counts.values, 
                 names=land_cover_counts.index,
                 title='Distribution of Land Cover Types')
    st.plotly_chart(fig, use_container_width=True)

def show_model_training(df):
    st.markdown('<h2 class="sub-header">🤖 Model Training</h2>', unsafe_allow_html=True)
    
    if st.button("🚀 Train Models", type="primary"):
        with st.spinner("Training models... This may take a few moments."):
            X, y = prepare_features(df)
            trained_models, results, X_test, y_test = train_models(X, y)
            
            # Store results in session state
            st.session_state.trained_models = trained_models
            st.session_state.results = results
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            
            st.success("✅ Models trained successfully!")
            
            # Display results
            st.subheader("📊 Training Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Random Forest Accuracy", f"{results['Random Forest']['accuracy']:.3f}")
            with col2:
                st.metric("XGBoost Accuracy", f"{results['XGBoost']['accuracy']:.3f}")
            with col3:
                st.metric("Logistic Regression Accuracy", f"{results['Logistic Regression']['accuracy']:.3f}")
            
            # Show detailed classification reports
            st.subheader("📋 Detailed Classification Reports")
            
            for name, model_data in trained_models.items():
                with st.expander(f"📊 {name} Classification Report"):
                    report = classification_report(model_data['y_test'], model_data['y_pred'])
                    st.text(report)
    
    else:
        st.info("👆 Click the button above to train the machine learning models.")

def show_model_comparison(df):
    st.markdown('<h2 class="sub-header">📈 Model Comparison</h2>', unsafe_allow_html=True)
    
    if 'trained_models' not in st.session_state:
        st.warning("⚠️ Please train the models first in the 'Model Training' section.")
        return
    
    trained_models = st.session_state.trained_models
    results = st.session_state.results
    
    # Model comparison metrics
    st.subheader("📊 Performance Metrics")
    
    comparison_df = pd.DataFrame(results).T
    st.dataframe(comparison_df, use_container_width=True)
    
    # Visualizations
    st.subheader("📈 Performance Visualizations")
    
    # Model comparison plots
    fig = plot_model_comparison(results)
    st.pyplot(fig)
    
    # ROC curves
    st.subheader("🔄 ROC Curves")
    roc_fig = plot_roc_curves(trained_models)
    st.plotly_chart(roc_fig, use_container_width=True)
    
    # Feature importance
    st.subheader("🎯 Feature Importance")
    feature_names = ['Rainfall_mm', 'Elevation_m', 'Distance_to_River_km', 
                    'Land_Cover_Type_Encoded', 'Population_Density']
    importance_fig = plot_feature_importance(trained_models, feature_names)
    st.pyplot(importance_fig)
    
    # Confusion matrices
    st.subheader("🔍 Confusion Matrices")
    cm_fig = plot_confusion_matrices(trained_models)
    st.pyplot(cm_fig)
    
    # Best model identification
    st.subheader("🏆 Best Model Analysis")
    
    best_model_acc = max(results.items(), key=lambda x: x[1]['accuracy'])
    best_model_auc = max(results.items(), key=lambda x: x[1]['auc'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="success-box">
            <h4>🏆 Best Model by Accuracy</h4>
            <p><strong>{best_model_acc[0]}</strong></p>
            <p>Accuracy: {best_model_acc[1]['accuracy']:.3f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="success-box">
            <h4>🏆 Best Model by AUC</h4>
            <p><strong>{best_model_auc[0]}</strong></p>
            <p>AUC: {best_model_auc[1]['auc']:.3f}</p>
        </div>
        """, unsafe_allow_html=True)

def show_prediction_interface(df, label_encoder):
    st.markdown('<h2 class="sub-header">🔮 Flood Risk Prediction</h2>', unsafe_allow_html=True)
    
    if 'trained_models' not in st.session_state:
        st.warning("⚠️ Please train the models first in the 'Model Training' section.")
        return
    
    trained_models = st.session_state.trained_models
    
    st.markdown("""
    <div class="warning-box">
        <h4>⚠️ Important Note</h4>
        <p>This prediction tool is for educational and research purposes. 
        Always consult with local authorities and experts for actual flood risk assessment.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Input form
    st.subheader("📝 Enter Environmental Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        rainfall = st.slider("🌧️ Rainfall (mm)", 
                           min_value=float(df['Rainfall_mm'].min()), 
                           max_value=float(df['Rainfall_mm'].max()),
                           value=float(df['Rainfall_mm'].mean()),
                           step=0.1)
        
        elevation = st.slider("⛰️ Elevation (m)", 
                            min_value=float(df['Elevation_m'].min()), 
                            max_value=float(df['Elevation_m'].max()),
                            value=float(df['Elevation_m'].mean()),
                            step=0.1)
        
        distance_to_river = st.slider("🏞️ Distance to River (km)", 
                                    min_value=float(df['Distance_to_River_km'].min()), 
                                    max_value=float(df['Distance_to_River_km'].max()),
                                    value=float(df['Distance_to_River_km'].mean()),
                                    step=0.01)
    
    with col2:
        land_cover = st.selectbox("🌍 Land Cover Type", 
                                 options=df['Land_Cover_Type'].unique())
        
        population_density = st.slider("👥 Population Density", 
                                     min_value=float(df['Population_Density'].min()), 
                                     max_value=float(df['Population_Density'].max()),
                                     value=float(df['Population_Density'].mean()),
                                     step=1.0)
    
    # Model selection
    st.subheader("🤖 Select Model for Prediction")
    selected_model = st.selectbox("Choose a model:", list(trained_models.keys()))
    
    if st.button("🔮 Predict Flood Risk", type="primary"):
        # Prepare input data
        land_cover_encoded = label_encoder.transform([land_cover])[0]
        
        input_data = np.array([[rainfall, elevation, distance_to_river, 
                               land_cover_encoded, population_density]])
        
        # Get model and make prediction
        model_data = trained_models[selected_model]
        model = model_data['model']
        
        if selected_model == 'Logistic Regression':
            # Scale input for logistic regression
            input_scaled = model_data['scaler'].transform(input_data)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]
        else:
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]
        
        # Display results
        st.subheader("📊 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction == 1:
                st.markdown("""
                <div class="warning-box">
                    <h4>🚨 HIGH FLOOD RISK</h4>
                    <p>Flood conditions are likely based on the input parameters.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="success-box">
                    <h4>✅ LOW FLOOD RISK</h4>
                    <p>Flood conditions are unlikely based on the input parameters.</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.metric("Flood Probability", f"{probability:.1%}")
        
        with col3:
            st.metric("Model Confidence", f"{max(probability, 1-probability):.1%}")
        
        # Risk assessment
        st.subheader("📋 Risk Assessment")
        
        if probability >= 0.7:
            risk_level = "HIGH"
            risk_color = "#dc3545"
            recommendations = [
                "Immediate evacuation may be necessary",
                "Monitor weather conditions closely",
                "Prepare emergency supplies",
                "Contact local authorities"
            ]
        elif probability >= 0.4:
            risk_level = "MODERATE"
            risk_color = "#ffc107"
            recommendations = [
                "Stay alert to weather updates",
                "Prepare emergency plan",
                "Monitor water levels",
                "Keep emergency contacts ready"
            ]
        else:
            risk_level = "LOW"
            risk_color = "#28a745"
            recommendations = [
                "Continue normal activities",
                "Stay informed about weather",
                "Maintain emergency preparedness",
                "Follow local guidelines"
            ]
        
        st.markdown(f"""
        <div style="border-left: 4px solid {risk_color}; padding-left: 1rem; margin: 1rem 0;">
            <h4>Risk Level: {risk_level}</h4>
            <p>Based on the current environmental conditions, the flood risk is classified as <strong>{risk_level}</strong>.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("💡 Recommendations")
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"{i}. {rec}")
        
        # Feature importance for this prediction
        st.subheader("🎯 Key Factors")
        
        if selected_model in ['Random Forest', 'XGBoost']:
            feature_names = ['Rainfall_mm', 'Elevation_m', 'Distance_to_River_km', 
                           'Land_Cover_Type_Encoded', 'Population_Density']
            
            if selected_model == 'Random Forest':
                importances = model.feature_importances_
            else:
                importances = model.feature_importances_
            
            # Create feature importance dataframe
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=True)
            
            fig = px.bar(importance_df, x='Importance', y='Feature', 
                        orientation='h', title=f'{selected_model} Feature Importance')
            st.plotly_chart(fig, use_container_width=True)

def show_about_page():
    st.markdown('<h2 class="sub-header">📋 About This App</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 🌊 Flood Prediction App
    
    This application uses advanced machine learning techniques to predict flood risk based on environmental parameters.
    
    ### 🎯 Features
    
    - **Multiple ML Models**: XGBoost, Random Forest, and Logistic Regression
    - **Model Comparison**: Compare performance across different algorithms
    - **Interactive Predictions**: Real-time flood risk assessment
    - **Comprehensive Analysis**: Feature importance, ROC curves, and confusion matrices
    - **User-Friendly Interface**: Intuitive design for easy navigation
    
    ### 🔬 Technical Details
    
    **Dataset**: Synthetic flood data for Pakistan containing:
    - Rainfall measurements (mm)
    - Elevation data (m)
    - Distance to nearest river (km)
    - Land cover classification
    - Population density
    - Historical flood events
    
    **Models Used**:
    - **Random Forest**: Ensemble method using multiple decision trees
    - **XGBoost**: Gradient boosting with optimized hyperparameters
    - **Logistic Regression**: Linear model for binary classification
    
    **Evaluation Metrics**:
    - Accuracy: Overall prediction correctness
    - AUC-ROC: Model discrimination ability
    - Classification Report: Precision, recall, and F1-score
    
    ### ⚠️ Important Disclaimer
    
    This application is designed for educational and research purposes. 
    For actual flood risk assessment and emergency planning:
    
    - Always consult with local meteorological authorities
    - Follow official evacuation orders
    - Use multiple data sources for decision-making
    - Consider local geographical and environmental factors
    
    ### 🛠️ Built With
    
    - **Streamlit**: Web application framework
    - **Scikit-learn**: Machine learning library
    - **XGBoost**: Gradient boosting library
    - **Pandas & NumPy**: Data manipulation
    - **Plotly & Matplotlib**: Data visualization
    
    ### 📊 Model Performance
    
    The models are trained on historical flood data and validated using cross-validation techniques.
    Performance may vary based on:
    - Data quality and completeness
    - Geographic region characteristics
    - Seasonal variations
    - Climate change impacts
    
    ### 🔄 Continuous Improvement
    
    This application can be enhanced by:
    - Adding more environmental variables
    - Incorporating real-time weather data
    - Using ensemble methods
    - Adding temporal analysis
    - Including satellite imagery data
    """)

if __name__ == "__main__":
    main()
