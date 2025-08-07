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
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/flood-prediction-app',
        'Report a bug': 'https://github.com/your-repo/flood-prediction-app/issues',
        'About': '### Flood Prediction App\nA machine learning application for predicting flood-prone areas using environmental data.'
    }
)

# Set dark theme
st.markdown("""
<script>
    // Force dark theme
    document.documentElement.style.setProperty('--background-color', '#0f0f23');
    document.documentElement.style.setProperty('--text-color', '#e8e8e8');
</script>
""", unsafe_allow_html=True)

# Custom CSS
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Global Styles */
    .main {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        color: #e8e8e8;
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    /* Header Styles */
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 0 4px 8px rgba(0,0,0,0.3);
        font-family: 'Inter', sans-serif;
        letter-spacing: -0.02em;
    }
    
    .sub-header {
        font-size: 2rem;
        font-weight: 600;
        color: #a8b2d1;
        margin-bottom: 1.5rem;
        font-family: 'Inter', sans-serif;
    }
    
    /* Card Styles */
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.4);
    }
    
    .prediction-high {
        background: linear-gradient(135deg, rgba(239,68,68,0.2) 0%, rgba(220,38,38,0.1) 100%);
        border: 1px solid rgba(239,68,68,0.3);
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(239,68,68,0.2);
        backdrop-filter: blur(10px);
    }
    
    .prediction-low {
        background: linear-gradient(135deg, rgba(34,197,94,0.2) 0%, rgba(22,163,74,0.1) 100%);
        border: 1px solid rgba(34,197,94,0.3);
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(34,197,94,0.2);
        backdrop-filter: blur(10px);
    }
    
    .success-box {
        background: linear-gradient(135deg, rgba(34,197,94,0.15) 0%, rgba(22,163,74,0.1) 100%);
        border: 1px solid rgba(34,197,94,0.3);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }
    
    /* Sidebar Styles */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    .css-1lcbmhc {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Button Styles */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(102,126,234,0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(102,126,234,0.4);
        background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
    }
    
    /* Metric Styles */
    .css-1wivap2 {
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1rem;
        backdrop-filter: blur(10px);
    }
    
    /* Dataframe Styles */
    .dataframe {
        background: rgba(255,255,255,0.05);
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Plotly Chart Styles */
    .js-plotly-plot {
        background: rgba(255,255,255,0.02) !important;
        border-radius: 12px;
        padding: 1rem;
    }
    
    /* Text Styles */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        color: #e8e8e8;
    }
    
    p, div, span {
        font-family: 'Inter', sans-serif;
        color: #cbd5e1;
    }
    
    /* Code Blocks */
    .stCodeBlock {
        background: rgba(0,0,0,0.3);
        border-radius: 8px;
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Alert Styles */
    .stAlert {
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }
    
    /* Selectbox and Input Styles */
    .stSelectbox > div > div {
        background: rgba(255,255,255,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 8px;
        color: #e8e8e8;
    }
    
    .stNumberInput > div > div > input {
        background: rgba(255,255,255,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 8px;
        color: #e8e8e8;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255,255,255,0.1);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
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
    # Header with enhanced styling
    st.markdown('<h1 class="main-header">🌊 Flood Prediction Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #a8b2d1; font-size: 1.2rem; margin-bottom: 2rem;">Advanced Machine Learning for Flood Risk Assessment</p>', unsafe_allow_html=True)
    st.markdown('<div style="height: 2px; background: linear-gradient(90deg, transparent, #667eea, transparent); margin: 2rem 0;"></div>', unsafe_allow_html=True)
    
    # Sidebar with enhanced styling
    st.sidebar.markdown('<div style="background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%); padding: 1rem; border-radius: 12px; margin-bottom: 1rem;">', unsafe_allow_html=True)
    st.sidebar.markdown('<h3 style="color: #667eea; margin-bottom: 1rem;">🧭 Navigation</h3>', unsafe_allow_html=True)
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "📊 Data Analysis", "🤖 Model Training", "🎯 Predictions", "📈 Model Performance"],
        label_visibility="collapsed"
    )
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Add reset buttons in sidebar
    st.sidebar.markdown('<div style="height: 1px; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent); margin: 1rem 0;"></div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div style="background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%); padding: 1rem; border-radius: 12px; margin-bottom: 1rem;">', unsafe_allow_html=True)
    st.sidebar.markdown('<h4 style="color: #667eea; margin-bottom: 1rem;">⚡ Actions</h4>', unsafe_allow_html=True)
    
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
    
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Show status in sidebar
    st.sidebar.markdown('<div style="height: 1px; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent); margin: 1rem 0;"></div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div style="background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%); padding: 1rem; border-radius: 12px;">', unsafe_allow_html=True)
    st.sidebar.markdown('<h4 style="color: #667eea; margin-bottom: 1rem;">📊 Status</h4>', unsafe_allow_html=True)
    
    if st.session_state.data_loaded:
        st.sidebar.markdown('<div style="color: #22c55e; font-weight: 500;">✅ Dataset Loaded</div>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<div style="color: #f59e0b; font-weight: 500;">⚠️ Dataset Not Loaded</div>', unsafe_allow_html=True)
    
    if st.session_state.model_trained:
        st.sidebar.markdown('<div style="color: #22c55e; font-weight: 500;">✅ Models Trained</div>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<div style="color: #f59e0b; font-weight: 500;">⚠️ Models Not Trained</div>', unsafe_allow_html=True)
    
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
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
    st.markdown('<h2 class="sub-header">🚀 Welcome to Flood Prediction Analytics</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%); 
                backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.1); 
                padding: 2rem; border-radius: 16px; margin-bottom: 2rem;">
        <p style="font-size: 1.1rem; line-height: 1.6; color: #cbd5e1; margin-bottom: 1.5rem;">
            This advanced analytics platform leverages cutting-edge machine learning algorithms to predict flood-prone areas 
            based on comprehensive environmental data from Pakistan. Our sophisticated models analyze multiple factors 
            to provide accurate flood risk assessments with interactive visualizations.
        </p>
        
        <h3 style="color: #667eea; margin-bottom: 1rem;">🎯 Key Features</h3>
        <ul style="color: #cbd5e1; line-height: 1.8;">
            <li><strong>📊 Advanced Data Analysis:</strong> Interactive visualizations and statistical insights</li>
            <li><strong>🤖 Multi-Model Training:</strong> XGBoost, Random Forest, and Logistic Regression</li>
            <li><strong>🎯 Real-time Predictions:</strong> Instant flood risk assessment for new data</li>
            <li><strong>📈 Performance Analytics:</strong> Comprehensive model comparison and metrics</li>
        </ul>
        
        <h3 style="color: #667eea; margin-top: 1.5rem; margin-bottom: 1rem;">📋 Dataset Information</h3>
        <p style="color: #cbd5e1; margin-bottom: 1rem;">Our comprehensive dataset includes critical environmental factors:</p>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
            <div style="background: rgba(255,255,255,0.05); padding: 0.75rem; border-radius: 8px;">
                <strong style="color: #667eea;">🌧️ Rainfall (mm)</strong>
            </div>
            <div style="background: rgba(255,255,255,0.05); padding: 0.75rem; border-radius: 8px;">
                <strong style="color: #667eea;">⛰️ Elevation (m)</strong>
            </div>
            <div style="background: rgba(255,255,255,0.05); padding: 0.75rem; border-radius: 8px;">
                <strong style="color: #667eea;">🏞️ Distance to River (km)</strong>
            </div>
            <div style="background: rgba(255,255,255,0.05); padding: 0.75rem; border-radius: 8px;">
                <strong style="color: #667eea;">🌳 Land Cover Type</strong>
            </div>
            <div style="background: rgba(255,255,255,0.05); padding: 0.75rem; border-radius: 8px;">
                <strong style="color: #667eea;">👥 Population Density</strong>
            </div>
            <div style="background: rgba(255,255,255,0.05); padding: 0.75rem; border-radius: 8px;">
                <strong style="color: #667eea;">🎯 Flood Status (Target)</strong>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show data status
    if st.session_state.data_loaded:
        st.markdown('<div style="background: linear-gradient(135deg, rgba(34,197,94,0.2) 0%, rgba(22,163,74,0.1) 100%); border: 1px solid rgba(34,197,94,0.3); padding: 1.5rem; border-radius: 16px; margin-bottom: 2rem;">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: #22c55e; margin-bottom: 1rem;">✅ Dataset Successfully Loaded</h3>', unsafe_allow_html=True)
        st.markdown('<p style="color: #cbd5e1;">Your dataset is ready for analysis and model training.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show data summary if available
        if st.session_state.data_summary:
            summary = st.session_state.data_summary
            st.markdown('<h3 style="color: #667eea; margin-bottom: 1rem;">📊 Dataset Overview</h3>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="color: #667eea; margin-bottom: 0.5rem;">📈 Total Samples</h4>
                    <p style="font-size: 2rem; font-weight: 700; color: #e8e8e8; margin: 0;">{summary['total_samples']:,}</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="color: #667eea; margin-bottom: 0.5rem;">🔧 Features</h4>
                    <p style="font-size: 2rem; font-weight: 700; color: #e8e8e8; margin: 0;">{summary['features']}</p>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="color: #667eea; margin-bottom: 0.5rem;">🌊 Flooded Areas</h4>
                    <p style="font-size: 2rem; font-weight: 700; color: #e8e8e8; margin: 0;">{summary['flooded_count']:,}</p>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="color: #667eea; margin-bottom: 0.5rem;">📊 Flood Rate</h4>
                    <p style="font-size: 2rem; font-weight: 700; color: #e8e8e8; margin: 0;">{summary['flood_rate']:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        # Load data button
        st.markdown('<div style="text-align: center; margin: 3rem 0;">', unsafe_allow_html=True)
        if st.button("🚀 Load Dataset", type="primary", use_container_width=True):
            load_dataset()
        st.markdown('</div>', unsafe_allow_html=True)

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
                
                st.markdown('<div style="background: linear-gradient(135deg, rgba(34,197,94,0.2) 0%, rgba(22,163,74,0.1) 100%); border: 1px solid rgba(34,197,94,0.3); padding: 1.5rem; border-radius: 16px; margin-bottom: 2rem;">', unsafe_allow_html=True)
                st.markdown('<h3 style="color: #22c55e; margin-bottom: 1rem;">✅ Dataset Loaded Successfully!</h3>', unsafe_allow_html=True)
                st.markdown('<p style="color: #cbd5e1;">Your dataset is now ready for analysis and model training.</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.rerun()
            else:
                st.markdown('<div style="background: linear-gradient(135deg, rgba(239,68,68,0.2) 0%, rgba(220,38,38,0.1) 100%); border: 1px solid rgba(239,68,68,0.3); padding: 1.5rem; border-radius: 16px;">', unsafe_allow_html=True)
                st.markdown('<h3 style="color: #ef4444; margin-bottom: 1rem;">❌ Failed to Load Dataset</h3>', unsafe_allow_html=True)
                st.markdown('<p style="color: #cbd5e1;">Please check the file path and try again.</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.markdown('<div style="background: linear-gradient(135deg, rgba(239,68,68,0.2) 0%, rgba(220,38,38,0.1) 100%); border: 1px solid rgba(239,68,68,0.3); padding: 1.5rem; border-radius: 16px;">', unsafe_allow_html=True)
            st.markdown(f'<h3 style="color: #ef4444; margin-bottom: 1rem;">❌ Error Loading Dataset</h3>', unsafe_allow_html=True)
            st.markdown(f'<p style="color: #cbd5e1;">Error: {e}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

def show_data_analysis_page():
    st.markdown('<h2 class="sub-header">📊 Data Analysis & Exploration</h2>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.markdown('<div style="background: linear-gradient(135deg, rgba(245,158,11,0.2) 0%, rgba(217,119,6,0.1) 100%); border: 1px solid rgba(245,158,11,0.3); padding: 1.5rem; border-radius: 16px;">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: #f59e0b; margin-bottom: 1rem;">⚠️ Dataset Required</h3>', unsafe_allow_html=True)
        st.markdown('<p style="color: #cbd5e1;">Please load the dataset first from the Home page to begin analysis.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    data = st.session_state.data
    
    # Data overview
    st.markdown('<h3 style="color: #667eea; margin-bottom: 1rem;">📋 Dataset Overview</h3>', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div style="background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.1); padding: 1rem; border-radius: 12px;">', unsafe_allow_html=True)
        st.dataframe(data.head(10), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div style="background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 12px;">', unsafe_allow_html=True)
        st.markdown('<h4 style="color: #667eea; margin-bottom: 1rem;">📊 Dataset Info</h4>', unsafe_allow_html=True)
        st.markdown(f'<p style="color: #cbd5e1; margin-bottom: 0.5rem;"><strong>Shape:</strong> {data.shape}</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="color: #cbd5e1; margin-bottom: 0.5rem;"><strong>Columns:</strong> {len(data.columns)}</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="color: #cbd5e1; margin-bottom: 0.5rem;"><strong>Missing Values:</strong> {data.isnull().sum().sum()}</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="color: #cbd5e1; margin-bottom: 0.5rem;"><strong>Memory Usage:</strong> {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Feature distributions
    st.markdown('<h3 style="color: #667eea; margin: 2rem 0 1rem 0;">📈 Feature Distributions</h3>', unsafe_allow_html=True)
    
    # Select feature to visualize
    feature_cols = [col for col in data.columns if col != 'Flooded']
    selected_feature = st.selectbox("Select feature to visualize:", feature_cols, label_visibility="collapsed")
    
    if selected_feature:
        st.markdown('<div style="background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem;">', unsafe_allow_html=True)
        
        # Create enhanced visualizations with dark theme
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.patch.set_facecolor('#0f0f23')
        
        # Histogram with enhanced styling
        ax1.hist(data[selected_feature], bins=30, alpha=0.8, color='#667eea', edgecolor='#4c63d2', linewidth=1.5)
        ax1.set_title(f'Distribution of {selected_feature}', color='#e8e8e8', fontsize=14, fontweight=600, pad=20)
        ax1.set_xlabel(selected_feature, color='#cbd5e1', fontsize=12)
        ax1.set_ylabel('Frequency', color='#cbd5e1', fontsize=12)
        ax1.grid(True, alpha=0.2, color='#4a5568')
        ax1.set_facecolor('#1a1a2e')
        ax1.tick_params(colors='#cbd5e1')
        
        # Box plot by flood status with enhanced styling
        if data['Flooded'].nunique() > 1:
            box_data = [data[data['Flooded'] == 0][selected_feature], data[data['Flooded'] == 1][selected_feature]]
            box_plot = ax2.boxplot(box_data, labels=['Not Flooded', 'Flooded'], patch_artist=True)
            
            # Color the boxes
            colors = ['#22c55e', '#ef4444']
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax2.set_title(f'{selected_feature} by Flood Status', color='#e8e8e8', fontsize=14, fontweight=600, pad=20)
            ax2.set_xlabel('Flood Status', color='#cbd5e1', fontsize=12)
            ax2.set_ylabel(selected_feature, color='#cbd5e1', fontsize=12)
            ax2.grid(True, alpha=0.2, color='#4a5568')
            ax2.set_facecolor('#1a1a2e')
            ax2.tick_params(colors='#cbd5e1')
        
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Correlation matrix
    st.markdown('<h3 style="color: #667eea; margin: 2rem 0 1rem 0;">🔗 Feature Correlations</h3>', unsafe_allow_html=True)
    numeric_data = data.select_dtypes(include=[np.number])
    correlation_matrix = numeric_data.corr()
    
    st.markdown('<div style="background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem;">', unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#0f0f23')
    
    # Create enhanced heatmap
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, 
                mask=mask,
                annot=True, 
                cmap='RdYlBu_r', 
                center=0, 
                ax=ax,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": .8},
                annot_kws={"size": 10, "color": "#1a1a2e", "weight": "bold"})
    
    ax.set_title('Feature Correlation Matrix', color='#e8e8e8', fontsize=16, fontweight=600, pad=20)
    ax.set_facecolor('#1a1a2e')
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right', color='#cbd5e1')
    plt.yticks(rotation=0, color='#cbd5e1')
    
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Target distribution
    st.markdown('<h3 style="color: #667eea; margin: 2rem 0 1rem 0;">🎯 Target Variable Distribution</h3>', unsafe_allow_html=True)
    flood_counts = data['Flooded'].value_counts()
    
    st.markdown('<div style="background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 12px;">', unsafe_allow_html=True)
    
    # Create enhanced pie chart
    fig = px.pie(
        values=flood_counts.values,
        names=['Not Flooded', 'Flooded'],
        title='Flood Status Distribution',
        color_discrete_sequence=['#22c55e', '#ef4444']
    )
    
    # Update layout for dark theme
    fig.update_layout(
        title_font_color='#e8e8e8',
        title_font_size=16,
        title_font_family='Inter',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#cbd5e1', family='Inter'),
        showlegend=True,
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(255,255,255,0.1)',
            borderwidth=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

def show_model_training_page():
    st.markdown('<h2 class="sub-header">🤖 Model Training & Optimization</h2>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.markdown('<div style="background: linear-gradient(135deg, rgba(245,158,11,0.2) 0%, rgba(217,119,6,0.1) 100%); border: 1px solid rgba(245,158,11,0.3); padding: 1.5rem; border-radius: 16px;">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: #f59e0b; margin-bottom: 1rem;">⚠️ Dataset Required</h3>', unsafe_allow_html=True)
        st.markdown('<p style="color: #cbd5e1;">Please load the dataset first from the Home page to begin model training.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Show training status
    if st.session_state.model_trained:
        st.markdown('<div style="background: linear-gradient(135deg, rgba(34,197,94,0.2) 0%, rgba(22,163,74,0.1) 100%); border: 1px solid rgba(34,197,94,0.3); padding: 1.5rem; border-radius: 16px; margin-bottom: 2rem;">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: #22c55e; margin-bottom: 1rem;">✅ Models Successfully Trained</h3>', unsafe_allow_html=True)
        st.markdown('<p style="color: #cbd5e1;">Your machine learning models are ready for predictions and analysis.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show stored training results
        if st.session_state.training_results:
            show_training_results(st.session_state.training_results)
    else:
        # Train models button
        st.markdown('<div style="text-align: center; margin: 3rem 0;">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: #667eea; margin-bottom: 1.5rem;">🚀 Ready to Train Models</h3>', unsafe_allow_html=True)
        st.markdown('<p style="color: #cbd5e1; margin-bottom: 2rem;">Click the button below to train XGBoost, Random Forest, and Logistic Regression models on your dataset.</p>', unsafe_allow_html=True)
        if st.button("🤖 Train Models", type="primary", use_container_width=True):
            train_models()
        st.markdown('</div>', unsafe_allow_html=True)

def train_models():
    """Train the machine learning models"""
    with st.spinner("Training models..."):
        try:
            data = st.session_state.data
            data_loader = st.session_state.data_loader
            
            # Preprocess data
            X_train, X_test, y_train, y_test = data_loader.preprocess_data(data)
            
            if X_train is None:
                st.markdown('<div style="background: linear-gradient(135deg, rgba(239,68,68,0.2) 0%, rgba(220,38,38,0.1) 100%); border: 1px solid rgba(239,68,68,0.3); padding: 1.5rem; border-radius: 16px;">', unsafe_allow_html=True)
                st.markdown('<h3 style="color: #ef4444; margin-bottom: 1rem;">❌ Data Preprocessing Failed</h3>', unsafe_allow_html=True)
                st.markdown('<p style="color: #cbd5e1;">Please check your dataset and try again.</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
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
            
            st.markdown('<div style="background: linear-gradient(135deg, rgba(34,197,94,0.2) 0%, rgba(22,163,74,0.1) 100%); border: 1px solid rgba(34,197,94,0.3); padding: 1.5rem; border-radius: 16px; margin-bottom: 2rem;">', unsafe_allow_html=True)
            st.markdown('<h3 style="color: #22c55e; margin-bottom: 1rem;">✅ Models Trained Successfully!</h3>', unsafe_allow_html=True)
            st.markdown('<p style="color: #cbd5e1;">All models have been trained and are ready for predictions and analysis.</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show training results
            show_training_results(training_results)
            
        except Exception as e:
            st.markdown('<div style="background: linear-gradient(135deg, rgba(239,68,68,0.2) 0%, rgba(220,38,38,0.1) 100%); border: 1px solid rgba(239,68,68,0.3); padding: 1.5rem; border-radius: 16px;">', unsafe_allow_html=True)
            st.markdown('<h3 style="color: #ef4444; margin-bottom: 1rem;">❌ Training Error</h3>', unsafe_allow_html=True)
            st.markdown(f'<p style="color: #cbd5e1;">Error: {e}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

def show_training_results(training_results):
    """Display training results"""
    st.markdown('<h3 style="color: #667eea; margin: 2rem 0 1rem 0;">📊 Model Performance Comparison</h3>', unsafe_allow_html=True)
    comparison = training_results['comparison']
    
    # Create comparison chart
    models = list(comparison.keys())
    accuracies = [comparison[model]['accuracy'] for model in models]
    auc_scores = [comparison[model]['auc_score'] for model in models]
    
    st.markdown('<div style="background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem;">', unsafe_allow_html=True)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=models,
        y=accuracies,
        name='Accuracy',
        marker_color='#667eea',
        marker_line_color='#4c63d2',
        marker_line_width=1.5,
        opacity=0.8
    ))
    fig.add_trace(go.Bar(
        x=models,
        y=auc_scores,
        name='AUC Score',
        marker_color='#ef4444',
        marker_line_color='#dc2626',
        marker_line_width=1.5,
        opacity=0.8
    ))
    
    fig.update_layout(
        title='Model Performance Comparison',
        title_font_color='#e8e8e8',
        title_font_size=16,
        title_font_family='Inter',
        xaxis_title='Models',
        yaxis_title='Score',
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#cbd5e1', family='Inter'),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.1)'
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.1)'
        ),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(255,255,255,0.1)',
            borderwidth=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Show best model
    best_model_name = training_results['best_model_name']
    st.markdown('<div style="background: linear-gradient(135deg, rgba(34,197,94,0.2) 0%, rgba(22,163,74,0.1) 100%); border: 1px solid rgba(34,197,94,0.3); padding: 1.5rem; border-radius: 16px; margin-bottom: 2rem;">', unsafe_allow_html=True)
    st.markdown(f'<h3 style="color: #22c55e; margin-bottom: 1rem;">🏆 Best Performing Model</h3>', unsafe_allow_html=True)
    st.markdown(f'<p style="color: #cbd5e1; font-size: 1.2rem; font-weight: 600;">{best_model_name}</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Show feature importance
    if training_results['feature_importance'] is not None:
        st.markdown('<h3 style="color: #667eea; margin: 2rem 0 1rem 0;">🔍 Feature Importance Analysis</h3>', unsafe_allow_html=True)
        feature_importance_df = training_results['feature_importance']
        
        st.markdown('<div style="background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 12px;">', unsafe_allow_html=True)
        
        fig = px.bar(
            feature_importance_df,
            x='importance',
            y='feature',
            orientation='h',
            title='Feature Importance Ranking',
            color='importance',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            title_font_color='#e8e8e8',
            title_font_size=16,
            title_font_family='Inter',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#cbd5e1', family='Inter'),
            xaxis=dict(
                gridcolor='rgba(255,255,255,0.1)',
                zerolinecolor='rgba(255,255,255,0.1)'
            ),
            yaxis=dict(
                gridcolor='rgba(255,255,255,0.1)',
                zerolinecolor='rgba(255,255,255,0.1)'
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

def show_predictions_page():
    st.markdown('<h2 class="sub-header">🎯 Real-time Flood Risk Assessment</h2>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.markdown('<div style="background: linear-gradient(135deg, rgba(245,158,11,0.2) 0%, rgba(217,119,6,0.1) 100%); border: 1px solid rgba(245,158,11,0.3); padding: 1.5rem; border-radius: 16px;">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: #f59e0b; margin-bottom: 1rem;">⚠️ Dataset Required</h3>', unsafe_allow_html=True)
        st.markdown('<p style="color: #cbd5e1;">Please load the dataset first from the Home page to make predictions.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    if not st.session_state.model_trained:
        st.markdown('<div style="background: linear-gradient(135deg, rgba(245,158,11,0.2) 0%, rgba(217,119,6,0.1) 100%); border: 1px solid rgba(245,158,11,0.3); padding: 1.5rem; border-radius: 16px;">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: #f59e0b; margin-bottom: 1rem;">⚠️ Models Not Trained</h3>', unsafe_allow_html=True)
        st.markdown('<p style="color: #cbd5e1;">Please train the models first from the Model Training page.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Input form for predictions
    st.markdown('<h3 style="color: #667eea; margin: 2rem 0 1rem 0;">📝 Enter Prediction Parameters</h3>', unsafe_allow_html=True)
    
    st.markdown('<div style="background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.1); padding: 2rem; border-radius: 16px; margin-bottom: 2rem;">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h4 style="color: #667eea; margin-bottom: 1rem;">🌧️ Environmental Factors</h4>', unsafe_allow_html=True)
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0, label_visibility="collapsed")
        elevation = st.number_input("Elevation (m)", min_value=-500.0, max_value=1000.0, value=200.0, label_visibility="collapsed")
        distance_to_river = st.number_input("Distance to River (km)", min_value=0.0, max_value=20.0, value=2.0, label_visibility="collapsed")
    
    with col2:
        st.markdown('<h4 style="color: #667eea; margin-bottom: 1rem;">🏞️ Geographic & Demographic</h4>', unsafe_allow_html=True)
        land_cover_options = ['Forest', 'Agriculture', 'Urban', 'Barren']
        land_cover = st.selectbox("Land Cover Type", land_cover_options, label_visibility="collapsed")
        population_density = st.number_input("Population Density", min_value=0.0, max_value=10000.0, value=1000.0, label_visibility="collapsed")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("🔮 Predict Flood Risk", type="primary", use_container_width=True):
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
        st.markdown('<h3 style="color: #667eea; margin: 2rem 0 1rem 0;">🎯 Prediction Results</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction[0] == 1:
                st.markdown('<div class="prediction-high">', unsafe_allow_html=True)
                st.markdown('<h3 style="color: #ef4444; margin-bottom: 1rem;">🚨 HIGH FLOOD RISK</h3>', unsafe_allow_html=True)
                st.markdown('<p style="color: #cbd5e1; font-size: 1.1rem;">Immediate action required. Consider evacuation and emergency protocols.</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="prediction-low">', unsafe_allow_html=True)
                st.markdown('<h3 style="color: #22c55e; margin-bottom: 1rem;">✅ LOW FLOOD RISK</h3>', unsafe_allow_html=True)
                st.markdown('<p style="color: #cbd5e1; font-size: 1.1rem;">Normal conditions. Continue monitoring weather updates.</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div style="background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 12px;">', unsafe_allow_html=True)
            st.markdown('<h4 style="color: #667eea; margin-bottom: 1rem;">📊 Risk Metrics</h4>', unsafe_allow_html=True)
            st.markdown(f'<p style="color: #cbd5e1; font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem;">Flood Probability: <span style="color: {"#ef4444" if probability > 0.5 else "#22c55e"}">{probability:.1%}</span></p>', unsafe_allow_html=True)
            st.markdown(f'<p style="color: #cbd5e1; font-size: 1.2rem; margin-bottom: 0.5rem;">Risk Level: <span style="color: {"#ef4444" if probability > 0.5 else "#22c55e"}">{"High" if probability > 0.5 else "Low"}</span></p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Risk assessment
        st.markdown('<h3 style="color: #667eea; margin: 2rem 0 1rem 0;">⚠️ Risk Assessment & Recommendations</h3>', unsafe_allow_html=True)
        
        if probability > 0.7:
            st.markdown('<div style="background: linear-gradient(135deg, rgba(239,68,68,0.2) 0%, rgba(220,38,38,0.1) 100%); border: 1px solid rgba(239,68,68,0.3); padding: 1.5rem; border-radius: 16px;">', unsafe_allow_html=True)
            st.markdown('<h4 style="color: #ef4444; margin-bottom: 1rem;">🚨 Very High Risk: Immediate Action Required!</h4>', unsafe_allow_html=True)
            st.markdown('<ul style="color: #cbd5e1; line-height: 1.8;"><li>Initiate emergency evacuation procedures</li><li>Activate flood warning systems</li><li>Deploy emergency response teams</li><li>Monitor water levels continuously</li></ul>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        elif probability > 0.5:
            st.markdown('<div style="background: linear-gradient(135deg, rgba(245,158,11,0.2) 0%, rgba(217,119,6,0.1) 100%); border: 1px solid rgba(245,158,11,0.3); padding: 1.5rem; border-radius: 16px;">', unsafe_allow_html=True)
            st.markdown('<h4 style="color: #f59e0b; margin-bottom: 1rem;">⚠️ High Risk: Monitor Closely</h4>', unsafe_allow_html=True)
            st.markdown('<ul style="color: #cbd5e1; line-height: 1.8;"><li>Prepare for potential evacuation</li><li>Monitor weather conditions</li><li>Check flood barriers and drainage</li><li>Alert local authorities</li></ul>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        elif probability > 0.3:
            st.markdown('<div style="background: linear-gradient(135deg, rgba(59,130,246,0.2) 0%, rgba(37,99,235,0.1) 100%); border: 1px solid rgba(59,130,246,0.3); padding: 1.5rem; border-radius: 16px;">', unsafe_allow_html=True)
            st.markdown('<h4 style="color: #3b82f6; margin-bottom: 1rem;">📊 Moderate Risk: Stay Alert</h4>', unsafe_allow_html=True)
            st.markdown('<ul style="color: #cbd5e1; line-height: 1.8;"><li>Monitor weather forecasts</li><li>Check local flood warnings</li><li>Prepare emergency supplies</li><li>Stay informed of updates</li></ul>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="background: linear-gradient(135deg, rgba(34,197,94,0.2) 0%, rgba(22,163,74,0.1) 100%); border: 1px solid rgba(34,197,94,0.3); padding: 1.5rem; border-radius: 16px;">', unsafe_allow_html=True)
            st.markdown('<h4 style="color: #22c55e; margin-bottom: 1rem;">✅ Low Risk: Normal Conditions</h4>', unsafe_allow_html=True)
            st.markdown('<ul style="color: #cbd5e1; line-height: 1.8;"><li>Continue normal operations</li><li>Monitor weather updates</li><li>Maintain flood preparedness</li><li>Regular safety checks</li></ul>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Feature importance
        if hasattr(model_trainer.best_model, 'feature_importances_'):
            st.markdown('<h3 style="color: #667eea; margin: 2rem 0 1rem 0;">🔍 Feature Importance Analysis</h3>', unsafe_allow_html=True)
            feature_importance = model_trainer.best_model.feature_importances_
            feature_names = data_loader.get_feature_names()
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=True)
            
            st.markdown('<div style="background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 12px;">', unsafe_allow_html=True)
            
            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Feature Importance for This Prediction',
                color='Importance',
                color_continuous_scale='Viridis'
            )
            
            fig.update_layout(
                title_font_color='#e8e8e8',
                title_font_size=16,
                title_font_family='Inter',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#cbd5e1', family='Inter'),
                xaxis=dict(
                    gridcolor='rgba(255,255,255,0.1)',
                    zerolinecolor='rgba(255,255,255,0.1)'
                ),
                yaxis=dict(
                    gridcolor='rgba(255,255,255,0.1)',
                    zerolinecolor='rgba(255,255,255,0.1)'
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    except Exception as e:
        st.markdown('<div style="background: linear-gradient(135deg, rgba(239,68,68,0.2) 0%, rgba(220,38,38,0.1) 100%); border: 1px solid rgba(239,68,68,0.3); padding: 1.5rem; border-radius: 16px;">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: #ef4444; margin-bottom: 1rem;">❌ Prediction Error</h3>', unsafe_allow_html=True)
        st.markdown(f'<p style="color: #cbd5e1;">Error: {e}</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def show_model_performance_page():
    st.markdown('<h2 class="sub-header">📈 Model Performance Analytics</h2>', unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.markdown('<div style="background: linear-gradient(135deg, rgba(245,158,11,0.2) 0%, rgba(217,119,6,0.1) 100%); border: 1px solid rgba(245,158,11,0.3); padding: 1.5rem; border-radius: 16px;">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: #f59e0b; margin-bottom: 1rem;">⚠️ Models Not Trained</h3>', unsafe_allow_html=True)
        st.markdown('<p style="color: #cbd5e1;">Please train the models first from the Model Training page to view performance analytics.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    model_trainer = st.session_state.model
    
    # Model comparison
    st.markdown('<h3 style="color: #667eea; margin: 2rem 0 1rem 0;">📊 Model Comparison Matrix</h3>', unsafe_allow_html=True)
    comparison = model_trainer.get_model_comparison()
    
    # Create comparison table
    comparison_df = pd.DataFrame(comparison).T
    st.markdown('<div style="background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem;">', unsafe_allow_html=True)
    st.dataframe(comparison_df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Detailed metrics for best model
    st.markdown('<h3 style="color: #667eea; margin: 2rem 0 1rem 0;">🏆 Best Model Detailed Metrics</h3>', unsafe_allow_html=True)
    best_model_name = model_trainer.best_model_name
    detailed_metrics = model_trainer.get_detailed_metrics(best_model_name)
    
    if detailed_metrics is not None:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: #667eea; margin-bottom: 0.5rem;">🎯 Accuracy</h4>
                <p style="font-size: 2rem; font-weight: 700; color: #e8e8e8; margin: 0;">{detailed_metrics['accuracy']:.3f}</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: #667eea; margin-bottom: 0.5rem;">📏 Precision</h4>
                <p style="font-size: 2rem; font-weight: 700; color: #e8e8e8; margin: 0;">{detailed_metrics['precision']:.3f}</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: #667eea; margin-bottom: 0.5rem;">🔄 Recall</h4>
                <p style="font-size: 2rem; font-weight: 700; color: #e8e8e8; margin: 0;">{detailed_metrics['recall']:.3f}</p>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: #667eea; margin-bottom: 0.5rem;">⚡ F1-Score</h4>
                <p style="font-size: 2rem; font-weight: 700; color: #e8e8e8; margin: 0;">{detailed_metrics['f1_score']:.3f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Confusion matrix
        st.markdown('<h3 style="color: #667eea; margin: 2rem 0 1rem 0;">🔍 Confusion Matrix</h3>', unsafe_allow_html=True)
        cm = detailed_metrics['confusion_matrix']
        
        st.markdown('<div style="background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem;">', unsafe_allow_html=True)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('#0f0f23')
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                   cbar_kws={"shrink": .8}, annot_kws={"size": 14, "color": "#1a1a2e", "weight": "bold"})
        plt.title('Confusion Matrix', color='#e8e8e8', fontsize=16, fontweight=600, pad=20)
        plt.ylabel('True Label', color='#cbd5e1', fontsize=12)
        plt.xlabel('Predicted Label', color='#cbd5e1', fontsize=12)
        ax.set_facecolor('#1a1a2e')
        
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Classification report
        st.markdown('<h3 style="color: #667eea; margin: 2rem 0 1rem 0;">📋 Classification Report</h3>', unsafe_allow_html=True)
        if hasattr(model_trainer, 'X_test') and hasattr(model_trainer, 'y_test'):
            y_pred = model_trainer.best_model.predict(model_trainer.X_test)
            report = classification_report(model_trainer.y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            
            st.markdown('<div style="background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem;">', unsafe_allow_html=True)
            st.dataframe(report_df, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="background: linear-gradient(135deg, rgba(245,158,11,0.2) 0%, rgba(217,119,6,0.1) 100%); border: 1px solid rgba(245,158,11,0.3); padding: 1.5rem; border-radius: 16px;">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: #f59e0b; margin-bottom: 1rem;">⚠️ Detailed Metrics Not Available</h3>', unsafe_allow_html=True)
        st.markdown('<p style="color: #cbd5e1;">Please retrain the models to view detailed performance metrics.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Show feature importance if available
    if hasattr(model_trainer, 'feature_importance') and model_trainer.feature_importance is not None:
        st.markdown('<h3 style="color: #667eea; margin: 2rem 0 1rem 0;">🔍 Global Feature Importance</h3>', unsafe_allow_html=True)
        if hasattr(model_trainer, 'data_loader') and model_trainer.data_loader is not None:
            feature_names = model_trainer.data_loader.get_feature_names()
        else:
            feature_names = [f"Feature_{i}" for i in range(len(model_trainer.feature_importance))]
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model_trainer.feature_importance
        }).sort_values('Importance', ascending=True)
        
        st.markdown('<div style="background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 12px;">', unsafe_allow_html=True)
        
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Global Feature Importance Ranking',
            color='Importance',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            title_font_color='#e8e8e8',
            title_font_size=16,
            title_font_family='Inter',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#cbd5e1', family='Inter'),
            xaxis=dict(
                gridcolor='rgba(255,255,255,0.1)',
                zerolinecolor='rgba(255,255,255,0.1)'
            ),
            yaxis=dict(
                gridcolor='rgba(255,255,255,0.1)',
                zerolinecolor='rgba(255,255,255,0.1)'
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
