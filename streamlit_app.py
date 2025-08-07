import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from sklearn.metrics import classification_report
warnings.filterwarnings('ignore')

# Import our custom modules
from csv_data_loader import CSVDataLoader
from simple_model_trainer import SimpleFloodPredictionModel

# Page configuration
st.set_page_config(
    page_title="Flood Prediction App - CSV Dataset",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-high {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
    }
    .prediction-low {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
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

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'data_loader' not in st.session_state:
    st.session_state.data_loader = None
if 'training_results' not in st.session_state:
    st.session_state.training_results = None
if 'data_summary' not in st.session_state:
    st.session_state.data_summary = None

def main():
    # Header
    st.markdown('<h1 class="main-header">🌊 Flood Prediction App - CSV Dataset</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "📊 Data Analysis", "🤖 Model Training", "🎯 Predictions", "📈 Model Performance"]
    )
    
    # Add reset buttons in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Actions")
    
    if st.session_state.data_loaded:
        if st.sidebar.button("🔄 Reload Dataset"):
            st.session_state.data_loaded = False
            st.session_state.data = None
            st.session_state.data_loader = None
            st.session_state.data_summary = None
            st.session_state.model_trained = False
            st.session_state.model = None
            st.session_state.training_results = None
            st.rerun()
    
    if st.session_state.model_trained:
        if st.sidebar.button("🔄 Retrain Models"):
            st.session_state.model_trained = False
            st.session_state.model = None
            st.session_state.training_results = None
            st.rerun()
    
    # Show status in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Status")
    
    if st.session_state.data_loaded:
        st.sidebar.success("✅ Dataset Loaded")
    else:
        st.sidebar.warning("⚠️ Dataset Not Loaded")
    
    if st.session_state.model_trained:
        st.sidebar.success("✅ Models Trained")
    else:
        st.sidebar.warning("⚠️ Models Not Trained")
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Data Analysis":
        show_data_analysis_page()
    elif page == "🤖 Model Training":
        show_model_training_page()
    elif page == "🎯 Predictions":
        show_predictions_page()
    elif page == "📈 Model Performance":
        show_model_performance_page()

def show_home_page():
    st.markdown("""
    ## Welcome to the Flood Prediction App! 🌊
    
    This application uses machine learning to predict flood-prone areas based on the Pakistan flood dataset. 
    The app analyzes various environmental factors to assess flood risk and provides interactive visualizations.
    
    ### Features:
    - 📊 **Data Analysis**: Explore the flood dataset with interactive visualizations
    - 🤖 **Model Training**: Train multiple machine learning models
    - 🎯 **Predictions**: Make predictions on new data
    - 📈 **Performance**: Compare model performance and metrics
    
    ### Dataset Information:
    The dataset contains information about various environmental factors that contribute to flood risk:
    - Rainfall (mm)
    - Elevation (m)
    - Distance to River (km)
    - Land Cover Type
    - Population Density
    - Flood Status (Target Variable)
    """)
    
    # Show data status
    if st.session_state.data_loaded:
        st.success("✅ Dataset is loaded and ready!")
        
        # Show data summary if available
        if st.session_state.data_summary:
            summary = st.session_state.data_summary
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Samples", summary['total_samples'])
            with col2:
                st.metric("Features", summary['features'])
            with col3:
                st.metric("Flooded Areas", summary['flooded_count'])
            with col4:
                st.metric("Flood Rate", f"{summary['flood_rate']:.1f}%")
    else:
        # Load data button
        if st.button("🚀 Load Dataset", type="primary"):
            load_dataset()

def load_dataset():
    """Load the CSV dataset"""
    with st.spinner("Loading dataset..."):
        try:
            # Initialize data loader
            data_loader = CSVDataLoader()
            
            # Load data
            data = data_loader.load_data("dataset/synthetic_flood_data_pakistan.csv")
            
            if data is not None:
                st.session_state.data = data
                st.session_state.data_loader = data_loader
                st.session_state.data_loaded = True
                
                # Get and store data summary
                summary = data_loader.get_data_summary(data)
                st.session_state.data_summary = summary
                
                st.success("✅ Dataset loaded successfully!")
                
                # Show data summary
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Samples", summary['total_samples'])
                with col2:
                    st.metric("Features", summary['features'])
                with col3:
                    st.metric("Flooded Areas", summary['flooded_count'])
                with col4:
                    st.metric("Flood Rate", f"{summary['flood_rate']:.1f}%")
            else:
                st.error("❌ Failed to load dataset")
        except Exception as e:
            st.error(f"❌ Error loading dataset: {e}")

def show_data_analysis_page():
    st.markdown("## 📊 Data Analysis")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load the dataset first from the Home page.")
        return
    
    data = st.session_state.data
    
    # Data overview
    st.subheader("Dataset Overview")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(data.head(10))
    
    with col2:
        st.write("**Dataset Info:**")
        st.write(f"- Shape: {data.shape}")
        st.write(f"- Columns: {list(data.columns)}")
        st.write(f"- Missing values: {data.isnull().sum().sum()}")
    
    # Feature distributions
    st.subheader("Feature Distributions")
    
    # Select feature to visualize
    feature_cols = [col for col in data.columns if col != 'Flooded']
    selected_feature = st.selectbox("Select feature to visualize:", feature_cols)
    
    if selected_feature:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Histogram
        ax1.hist(data[selected_feature], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title(f'Distribution of {selected_feature}')
        ax1.set_xlabel(selected_feature)
        ax1.set_ylabel('Frequency')
        
        # Box plot by flood status
        if data['Flooded'].nunique() > 1:
            data.boxplot(column=selected_feature, by='Flooded', ax=ax2)
            ax2.set_title(f'{selected_feature} by Flood Status')
            ax2.set_xlabel('Flooded')
            ax2.set_ylabel(selected_feature)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Correlation matrix
    st.subheader("Feature Correlations")
    numeric_data = data.select_dtypes(include=[np.number])
    correlation_matrix = numeric_data.corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
    plt.title('Correlation Matrix')
    st.pyplot(fig)
    
    # Target distribution
    st.subheader("Target Variable Distribution")
    flood_counts = data['Flooded'].value_counts()
    
    fig = px.pie(
        values=flood_counts.values,
        names=['Not Flooded', 'Flooded'],
        title='Flood Status Distribution'
    )
    st.plotly_chart(fig, use_container_width=True)

def show_model_training_page():
    st.markdown("## 🤖 Model Training")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load the dataset first from the Home page.")
        return
    
    # Show training status
    if st.session_state.model_trained:
        st.success("✅ Models are trained and ready!")
        
        # Show stored training results
        if st.session_state.training_results:
            show_training_results(st.session_state.training_results)
    else:
        # Train models button
        if st.button("🚀 Train Models", type="primary"):
            train_models()

def train_models():
    """Train the machine learning models"""
    with st.spinner("Training models..."):
        try:
            data = st.session_state.data
            data_loader = st.session_state.data_loader
            
            # Preprocess data
            X_train, X_test, y_train, y_test = data_loader.preprocess_data(data)
            
            if X_train is None:
                st.error("❌ Failed to preprocess data")
                return
            
            # Initialize model trainer
            model_trainer = SimpleFloodPredictionModel()
            
            # Store data_loader reference for feature names access
            model_trainer.data_loader = data_loader
            
            # Train all models
            results = model_trainer.train_all_models(
                X_train, y_train, X_test, y_test, 
                data_loader.get_feature_names()
            )
            
            st.session_state.model = model_trainer
            st.session_state.model_trained = True
            
            # Store training results
            training_results = {
                'comparison': model_trainer.get_model_comparison(),
                'best_model_name': model_trainer.best_model_name,
                'feature_importance': results
            }
            st.session_state.training_results = training_results
            
            st.success("✅ Models trained successfully!")
            
            # Show training results
            show_training_results(training_results)
            
        except Exception as e:
            st.error(f"❌ Error training models: {e}")

def show_training_results(training_results):
    """Display training results"""
    st.subheader("Model Performance Comparison")
    comparison = training_results['comparison']
    
    # Create comparison chart
    models = list(comparison.keys())
    accuracies = [comparison[model]['accuracy'] for model in models]
    auc_scores = [comparison[model]['auc_score'] for model in models]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=models,
        y=accuracies,
        name='Accuracy',
        marker_color='lightblue'
    ))
    fig.add_trace(go.Bar(
        x=models,
        y=auc_scores,
        name='AUC Score',
        marker_color='lightcoral'
    ))
    
    fig.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Models',
        yaxis_title='Score',
        barmode='group'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show best model
    best_model_name = training_results['best_model_name']
    st.info(f"🏆 Best Model: {best_model_name}")
    
    # Show feature importance
    if training_results['feature_importance'] is not None:
        st.subheader("Feature Importance")
        feature_importance_df = training_results['feature_importance']
        
        fig = px.bar(
            feature_importance_df,
            x='importance',
            y='feature',
            orientation='h',
            title='Feature Importance'
        )
        st.plotly_chart(fig, use_container_width=True)

def show_predictions_page():
    st.markdown("## 🎯 Make Predictions")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load the dataset first from the Home page.")
        return
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the models first from the Model Training page.")
        return
    
    # Input form for predictions
    st.subheader("Enter Prediction Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0)
        elevation = st.number_input("Elevation (m)", min_value=-500.0, max_value=1000.0, value=200.0)
        distance_to_river = st.number_input("Distance to River (km)", min_value=0.0, max_value=20.0, value=2.0)
    
    with col2:
        land_cover_options = ['Forest', 'Agriculture', 'Urban', 'Barren']
        land_cover = st.selectbox("Land Cover Type", land_cover_options)
        population_density = st.number_input("Population Density", min_value=0.0, max_value=10000.0, value=1000.0)
    
    if st.button("🔮 Predict Flood Risk", type="primary"):
        make_prediction(rainfall, elevation, distance_to_river, land_cover, population_density)

def make_prediction(rainfall, elevation, distance_to_river, land_cover, population_density):
    """Make a prediction using the trained model"""
    try:
        # Create input data
        input_data = pd.DataFrame({
            'Rainfall_mm': [rainfall],
            'Elevation_m': [elevation],
            'Distance_to_River_km': [distance_to_river],
            'Land_Cover_Type': [land_cover],
            'Population_Density': [population_density]
        })
        
        # Transform input data
        data_loader = st.session_state.data_loader
        model_trainer = st.session_state.model
        
        input_scaled = data_loader.transform_new_data(input_data)
        
        # Make prediction
        prediction = model_trainer.predict(input_scaled)
        probability = model_trainer.best_model.predict_proba(input_scaled)[0][1]
        
        # Display results
        st.subheader("Prediction Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction[0] == 1:
                st.markdown('<div class="prediction-high">', unsafe_allow_html=True)
                st.markdown("### 🚨 HIGH FLOOD RISK")
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown('<div class="prediction-low">', unsafe_allow_html=True)
                st.markdown("### ✅ LOW FLOOD RISK")
                st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.metric("Flood Probability", f"{probability:.1%}")
            st.metric("Risk Level", "High" if probability > 0.5 else "Low")
        
        # Risk assessment
        st.subheader("Risk Assessment")
        if probability > 0.7:
            st.error("🚨 Very High Risk: Immediate action required!")
        elif probability > 0.5:
            st.warning("⚠️ High Risk: Monitor closely and prepare for evacuation")
        elif probability > 0.3:
            st.info("📊 Moderate Risk: Stay alert and monitor conditions")
        else:
            st.success("✅ Low Risk: Normal conditions")
        
        # Feature importance
        if hasattr(model_trainer.best_model, 'feature_importances_'):
            st.subheader("Feature Importance")
            feature_importance = model_trainer.best_model.feature_importances_
            feature_names = data_loader.get_feature_names()
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=True)
            
            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Feature Importance'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ Error making prediction: {e}")

def show_model_performance_page():
    st.markdown("## 📈 Model Performance")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the models first from the Model Training page.")
        return
    
    model_trainer = st.session_state.model
    
    # Model comparison
    st.subheader("Model Comparison")
    comparison = model_trainer.get_model_comparison()
    
    # Create comparison table
    comparison_df = pd.DataFrame(comparison).T
    st.dataframe(comparison_df)
    
    # Detailed metrics for best model
    st.subheader("Best Model Detailed Metrics")
    best_model_name = model_trainer.best_model_name
    detailed_metrics = model_trainer.get_detailed_metrics(best_model_name)
    
    if detailed_metrics is not None:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{detailed_metrics['accuracy']:.3f}")
        with col2:
            st.metric("Precision", f"{detailed_metrics['precision']:.3f}")
        with col3:
            st.metric("Recall", f"{detailed_metrics['recall']:.3f}")
        with col4:
            st.metric("F1-Score", f"{detailed_metrics['f1_score']:.3f}")
        
        # Confusion matrix
        st.subheader("Confusion Matrix")
        cm = detailed_metrics['confusion_matrix']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        st.pyplot(fig)
        
        # Classification report
        st.subheader("Classification Report")
        if hasattr(model_trainer, 'X_test') and hasattr(model_trainer, 'y_test'):
            y_pred = model_trainer.best_model.predict(model_trainer.X_test)
            report = classification_report(model_trainer.y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)
    else:
        st.warning("⚠️ Detailed metrics not available. Please retrain the models.")
    
    # Show feature importance if available
    if hasattr(model_trainer, 'feature_importance') and model_trainer.feature_importance is not None:
        st.subheader("Feature Importance")
        if hasattr(model_trainer, 'data_loader') and model_trainer.data_loader is not None:
            feature_names = model_trainer.data_loader.get_feature_names()
        else:
            feature_names = [f"Feature_{i}" for i in range(len(model_trainer.feature_importance))]
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model_trainer.feature_importance
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Feature Importance'
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
