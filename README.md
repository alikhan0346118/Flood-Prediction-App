# 🌊 Geospatial Flood Prediction App

A comprehensive machine learning application for predicting flood-prone areas using geospatial data analysis. This app combines advanced data processing, feature engineering, and multiple machine learning algorithms to provide accurate flood risk assessments.

## 🚀 Features

### 📊 Data Processing & Analysis
- **Geospatial Data Generation**: Creates realistic sample data for flood prediction
- **Feature Engineering**: Advanced feature creation including interaction terms and categorical encoding
- **Data Visualization**: Comprehensive charts and graphs for data exploration
- **Correlation Analysis**: Heatmaps and statistical analysis of feature relationships

### 🤖 Machine Learning Models
- **XGBoost**: Gradient boosting with hyperparameter tuning
- **Random Forest**: Ensemble learning with feature importance analysis
- **Logistic Regression**: Linear model with regularization
- **Model Comparison**: Performance metrics and accuracy calculations
- **Cross-validation**: 5-fold cross-validation for robust evaluation

### 🎯 Prediction Capabilities
- **Real-time Predictions**: Interactive form for new location predictions
- **Risk Factor Analysis**: Detailed breakdown of contributing factors
- **Probability Scores**: Confidence levels for predictions
- **Batch Predictions**: Predictions for entire datasets

### 🗺️ Geospatial Visualization
- **Interactive Maps**: Folium-based maps with flood risk visualization
- **Point Clustering**: Color-coded points based on flood risk
- **Prediction Maps**: Visual representation of model predictions
- **3D Scatter Plots**: Multi-dimensional data visualization

### 📈 Performance Analysis
- **ROC Curves**: Model performance comparison
- **Confusion Matrices**: Detailed classification analysis
- **Feature Importance**: Ranking of influential variables
- **Classification Reports**: Precision, recall, and F1-scores

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone or download the project files**

2. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** and navigate to the URL shown in the terminal (usually `http://localhost:8501`)

## 📁 Project Structure

```
geospatial-flood-prediction/
├── app.py                 # Main Streamlit application
├── data_processor.py      # Data processing and feature engineering
├── model_trainer.py       # Machine learning model training
├── visualization.py       # Charts, graphs, and maps
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🎮 How to Use

### 1. Home Page
- Overview of the application features
- Quick statistics and navigation guide

### 2. Data Analysis
- **Dataset Overview**: View sample data and statistics
- **Feature Distributions**: Histograms and density plots
- **Correlation Analysis**: Heatmap of feature relationships
- **Box Plots**: Distribution analysis by flood risk
- **Categorical Analysis**: Land use and soil type distributions
- **Interactive Scatter Plots**: Customizable 2D visualizations
- **3D Scatter Plots**: Multi-dimensional data exploration

### 3. Model Training
- **Select Models**: Choose which algorithms to train
- **Training Configuration**: View training parameters
- **Model Comparison**: Compare accuracy and AUC scores
- **Feature Importance**: See which factors most influence predictions

### 4. Predictions
- **Input Form**: Enter location and environmental parameters
- **Real-time Prediction**: Get instant flood risk assessment
- **Risk Factor Analysis**: Detailed breakdown of contributing factors
- **Confidence Scores**: Probability and confidence metrics

### 5. Geospatial Maps
- **Data Distribution Map**: Visualize actual flood data
- **Prediction Map**: See model predictions on the map
- **Interactive Controls**: Filter and customize map display
- **Statistics**: Summary of prediction results

### 6. Model Performance
- **Performance Metrics**: Detailed model comparison
- **ROC Curves**: Model discrimination analysis
- **Confusion Matrices**: Classification performance
- **Feature Importance**: Variable ranking for each model

## 🔧 Technical Details

### Data Features
- **Geographic**: Latitude, longitude
- **Topographic**: Elevation, slope
- **Climatic**: Rainfall
- **Hydrological**: Distance to rivers
- **Environmental**: Soil type, land use
- **Demographic**: Population density

### Feature Engineering
- **Interaction Features**: Elevation × slope, rainfall × elevation
- **Categorical Encoding**: Label encoding for categorical variables
- **Binary Features**: Near river, low elevation, high rainfall flags
- **Categorical Binning**: Elevation and rainfall categories

### Model Parameters
- **XGBoost**: n_estimators, max_depth, learning_rate, subsample, colsample_bytree
- **Random Forest**: n_estimators, max_depth, min_samples_split, min_samples_leaf
- **Logistic Regression**: C (regularization), penalty, solver

### Evaluation Metrics
- **Accuracy**: Overall prediction accuracy
- **AUC Score**: Area under ROC curve
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

## 📊 Sample Results

The application typically achieves:
- **Accuracy**: 85-95%
- **AUC Score**: 0.85-0.95
- **Training Time**: 2-5 minutes for all models
- **Prediction Time**: < 1 second for new locations

## 🎯 Use Cases

### Urban Planning
- Identify flood-prone areas for infrastructure development
- Plan evacuation routes and emergency response
- Design drainage systems and flood control measures

### Insurance & Risk Assessment
- Evaluate flood risk for property insurance
- Assess vulnerability of different regions
- Price insurance policies based on risk levels

### Environmental Monitoring
- Track changes in flood risk over time
- Monitor impact of climate change on flooding
- Assess effectiveness of flood mitigation measures

### Emergency Management
- Early warning systems for flood events
- Resource allocation for disaster response
- Community preparedness planning

## 🔮 Future Enhancements

- **Real-time Data Integration**: Connect to live weather and hydrological data
- **Time Series Analysis**: Historical flood pattern analysis
- **Deep Learning Models**: Neural networks for complex pattern recognition
- **Mobile App**: iOS/Android application for field use
- **API Integration**: RESTful API for third-party applications
- **Multi-language Support**: Internationalization for global use

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Bug fixes
- New features
- Documentation improvements
- Performance optimizations

## 📄 License

This project is open source and available under the MIT License.

## 📞 Support

For questions, issues, or support:
- Create an issue in the project repository
- Contact the development team
- Check the documentation and examples

---

**Note**: This application uses synthetic data for demonstration purposes. For real-world applications, replace the sample data generation with actual geospatial datasets from reliable sources. 