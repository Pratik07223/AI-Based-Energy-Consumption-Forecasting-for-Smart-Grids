import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
import io
import base64
from datetime import datetime

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Page configuration
st.set_page_config(
    page_title="AI Energy Forecasting for Smart Grids",
    page_icon="⚡",
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
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem;
    }
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def generate_synthetic_data():
    """Generate synthetic energy consumption data for smart grids"""
    
    # Date range
    date_range = pd.date_range(start='2022-01-01', end='2024-01-01', freq='H')
    n_samples = len(date_range)
    
    # Base consumption patterns
    data = []
    
    for i, date in enumerate(date_range):
        # Seasonal pattern
        day_of_year = date.dayofyear
        seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * day_of_year / 365.25)
        
        # Daily pattern
        hour = date.hour
        if 6 <= hour <= 9 or 17 <= hour <= 21:  # Peak hours
            daily_factor = 1.4
        elif 22 <= hour <= 23 or 0 <= hour <= 5:  # Off-peak
            daily_factor = 0.6
        else:  # Regular hours
            daily_factor = 1.0
            
        # Weekend effect
        is_weekend = date.weekday() >= 5
        weekend_factor = 0.8 if is_weekend else 1.0
        
        # Temperature effect
        temp_base = 15 + 10 * np.sin(2 * np.pi * day_of_year / 365.25)
        temperature = temp_base + np.random.normal(0, 3)
        temp_factor = 1 + 0.02 * abs(temperature - 20)  # AC/Heating effect
        
        # Holiday effect
        is_holiday = (date.month == 12 and date.day in [24, 25, 31]) or \
                     (date.month == 1 and date.day == 1) or \
                     (date.month == 7 and date.day == 4) or \
                     (date.month == 11 and date.day in [23, 24])
        holiday_factor = 0.7 if is_holiday else 1.0
        
        # Economic activity index
        economic_activity = 1.0 + 0.2 * np.sin(2 * np.pi * i / (365.25 * 24)) + np.random.normal(0, 0.1)
        economic_activity = max(0.5, min(1.5, economic_activity))
        
        # Renewable energy contribution (higher in summer/windy periods)
        renewable_base = 30 + 20 * np.sin(2 * np.pi * day_of_year / 365.25)
        renewable_contribution = renewable_base + np.random.normal(0, 10)
        renewable_contribution = max(0, min(80, renewable_contribution))
        
        # Base energy consumption calculation
        base_consumption = 3500  # Base kWh
        
        energy_consumption = base_consumption * seasonal_factor * daily_factor * \
                             weekend_factor * temp_factor * holiday_factor * \
                             (economic_activity / 1.0) + np.random.normal(0, 200)
        
        # Ensure realistic bounds
        energy_consumption = max(1500, min(7500, energy_consumption))
        
        # Season mapping
        if date.month in [12, 1, 2]:
            season = 'Winter'
        elif date.month in [3, 4, 5]:
            season = 'Spring'
        elif date.month in [6, 7, 8]:
            season = 'Summer'
        else:
            season = 'Fall'
            
        # Day type
        if is_holiday:
            day_type = 'Holiday'
        elif is_weekend:
            day_type = 'Weekend'
        else:
            day_type = 'Weekday'
            
        data.append({
            'Date': date,
            'Energy_Consumption': round(energy_consumption, 2),
            'Temperature': round(temperature, 1),
            'Day_Type': day_type,
            'Month': date.month,
            'Hour': date.hour,
            'Season': season,
            'Renewable_Energy_Contribution': round(renewable_contribution, 1),
            'Holiday_Flag': 1 if is_holiday else 0,
            'Economic_Activity_Index': round(economic_activity, 2)
        })
    
    return pd.DataFrame(data)

@st.cache_data
def engineer_features(df):
    """Create additional features for better model performance"""
    
    df_featured = df.copy()
    
    # Ensure 'Date' column is in datetime format
    df_featured['Date'] = pd.to_datetime(df_featured['Date'])
    
    # Time-based features
    df_featured['Year'] = df_featured['Date'].dt.year
    df_featured['Week'] = df_featured['Date'].dt.isocalendar().week.astype(int)
    df_featured['DayOfWeek'] = df_featured['Date'].dt.dayofweek
    df_featured['DayOfYear'] = df_featured['Date'].dt.dayofyear
    
    # Cyclical encoding for time features
    df_featured['Hour_sin'] = np.sin(2 * np.pi * df_featured['Hour'] / 24)
    df_featured['Hour_cos'] = np.cos(2 * np.pi * df_featured['Hour'] / 24)
    df_featured['Month_sin'] = np.sin(2 * np.pi * df_featured['Month'] / 12)
    df_featured['Month_cos'] = np.cos(2 * np.pi * df_featured['Month'] / 12)
    df_featured['DayOfWeek_sin'] = np.sin(2 * np.pi * df_featured['DayOfWeek'] / 7)
    df_featured['DayOfWeek_cos'] = np.cos(2 * np.pi * df_featured['DayOfWeek'] / 7)
    
    # Sort by date for lag features
    df_featured = df_featured.sort_values('Date').reset_index(drop=True)
    
    # Lag features
    for lag in [1, 2, 7, 24]:  # 1h, 2h, 7h, 24h lags
        df_featured[f'Energy_Lag_{lag}'] = df_featured['Energy_Consumption'].shift(lag)
    
    # Rolling statistics
    for window in [7, 24, 168]:  # 7h, 24h, 168h (1 week) windows
        df_featured[f'Energy_Rolling_Mean_{window}'] = df_featured['Energy_Consumption'].rolling(window=window).mean()
        df_featured[f'Energy_Rolling_Std_{window}'] = df_featured['Energy_Consumption'].rolling(window=window).std()
    
    # Temperature-based features
    df_featured['Temp_Squared'] = df_featured['Temperature'] ** 2
    df_featured['Cooling_Degree_Days'] = np.maximum(df_featured['Temperature'] - 18, 0)
    df_featured['Heating_Degree_Days'] = np.maximum(18 - df_featured['Temperature'], 0)
    
    # Interaction features
    df_featured['Temp_Hour_Interaction'] = df_featured['Temperature'] * df_featured['Hour']
    df_featured['Renewable_Economic_Interaction'] = df_featured['Renewable_Energy_Contribution'] * df_featured['Economic_Activity_Index']
    
    # Remove rows with NaN values
    df_featured = df_featured.dropna().reset_index(drop=True)
    
    return df_featured

def prepare_data_for_models(df):
    """Prepare data for machine learning and deep learning models"""
    
    target_col = 'Energy_Consumption'
    exclude_cols = ['Date', 'Day_Type', 'Season']
    feature_cols = [col for col in df.columns if col not in exclude_cols + [target_col]]
    
    # One-hot encode categorical variables
    df_encoded = df.copy()
    df_encoded = pd.get_dummies(df_encoded, columns=['Day_Type', 'Season'], prefix=['DayType', 'Season'])
    
    # Update feature columns after encoding
    feature_cols = [col for col in df_encoded.columns if col not in ['Date', target_col]]
    
    X = df_encoded[feature_cols]
    y = df_encoded[target_col]
    
    return X, y, feature_cols

def train_ml_models(X_train, y_train, X_test, y_test):
    """Train and evaluate machine learning models"""
    
    models = {}
    predictions = {}
    metrics = {}
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 1. Linear Regression (Baseline)
    status_text.text('Training Linear Regression...')
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    
    models['Linear_Regression'] = lr
    predictions['Linear_Regression'] = lr_pred
    progress_bar.progress(33)
    
    # 2. Random Forest
    status_text.text('Training Random Forest...')
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5]
    }
    
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    
    rf_best = rf_grid.best_estimator_
    rf_pred = rf_best.predict(X_test)
    
    models['Random_Forest'] = rf_best
    predictions['Random_Forest'] = rf_pred
    progress_bar.progress(66)
    
    # Calculate metrics for all models
    for name, pred in predictions.items():
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        
        metrics[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
    
    progress_bar.progress(100)
    status_text.text('Machine Learning Models Trained Successfully!')
    
    return models, predictions, metrics

def create_lstm_sequences(X, y, time_steps=24):
    """Create sequences for LSTM training"""
    
    X_seq, y_seq = [], []
    
    for i in range(time_steps, len(X)):
        X_seq.append(X[i-time_steps:i])
        y_seq.append(y[i])
    
    return np.array(X_seq), np.array(y_seq)

def build_lstm_model(input_shape):
    """Build and compile LSTM model"""
    
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(25),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def main():
    # Header
    st.markdown('<h1 class="main-header">⚡ AI Energy Forecasting for Smart Grids</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🔧 Control Panel")
    st.sidebar.markdown("---")
    
    # Data loading section
    st.sidebar.subheader("Dataset Source")
    data_source_option = st.sidebar.radio(
        "Choose your data source:",
        ("Generate Synthetic Data", "Upload Custom CSV")
    )

    df = None
    if data_source_option == "Upload Custom CSV":
        uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state['df'] = df
                st.sidebar.success("✅ Custom dataset uploaded!")
            except Exception as e:
                st.sidebar.error(f"Error reading file: {e}")
                return
        else:
            st.info("👈 Please upload a CSV file.")
            return
    else:
        if st.sidebar.button("🔄 Generate Synthetic Data"):
            with st.spinner('Generating synthetic energy data...'):
                df = generate_synthetic_data()
                st.session_state['df'] = df
            st.sidebar.success("✅ Synthetic dataset generated!")
            
    if 'df' not in st.session_state:
        st.info("👈 Please generate or upload the dataset first using the sidebar.")
        return

    df = st.session_state['df']

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Data Overview", "🔍 EDA", "🤖 Model Training", "📈 Results", "🔮 Predictions"])
    
    with tab1:
        st.header("📋 Dataset Overview")
        
        # Check for required columns
        required_cols = ['Date', 'Energy_Consumption', 'Temperature', 'Day_Type', 'Season', 'Renewable_Energy_Contribution', 'Holiday_Flag', 'Economic_Activity_Index']
        if not all(col in df.columns for col in required_cols):
            st.error("Uploaded dataset is missing one or more required columns.")
            st.write("Required columns: " + ", ".join(required_cols))
            return
            
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Records", f"{len(df):,}")
        with col2:
            df['Date'] = pd.to_datetime(df['Date'])
            st.metric("📅 Date Range", f"{(df['Date'].max() - df['Date'].min()).days} days")
        with col3:
            st.metric("⚡ Avg Consumption", f"{df['Energy_Consumption'].mean():.0f} kWh")
        with col4:
            st.metric("🌱 Avg Renewable", f"{df['Renewable_Energy_Contribution'].mean():.1f}%")
        
        st.subheader("📊 Dataset Sample")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.subheader("📈 Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
        
        # Download dataset
        csv = df.to_csv(index=False)
        st.download_button(
            label="💾 Download Dataset (CSV)",
            data=csv,
            file_name='energy_consumption_data.csv',
            mime='text/csv',
        )
    
    with tab2:
        st.header("🔍 Exploratory Data Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Time series plot
            fig_ts = px.line(df.iloc[::24], x='Date', y='Energy_Consumption', 
                             title='Energy Consumption Time Series (Daily Averages)')
            st.plotly_chart(fig_ts, use_container_width=True)
            
            # Seasonal patterns
            seasonal_avg = df.groupby('Season')['Energy_Consumption'].mean().reset_index()
            fig_seasonal = px.bar(seasonal_avg, x='Season', y='Energy_Consumption',
                                 title='Average Energy Consumption by Season',
                                 color='Season')
            st.plotly_chart(fig_seasonal, use_container_width=True)
        
        with col2:
            # Hourly patterns
            hourly_avg = df.groupby('Hour')['Energy_Consumption'].mean().reset_index()
            fig_hourly = px.line(hourly_avg, x='Hour', y='Energy_Consumption',
                                title='Average Hourly Energy Consumption',
                                markers=True)
            st.plotly_chart(fig_hourly, use_container_width=True)
            
            # Temperature vs Energy correlation
            fig_temp = px.scatter(df[::100], x='Temperature', y='Energy_Consumption',
                                 title='Energy Consumption vs Temperature',
                                 opacity=0.6,
                                 trendline="ols")
            st.plotly_chart(fig_temp, use_container_width=True)
        
        # Renewable energy contribution
        monthly_renewable = df.groupby('Month')['Renewable_Energy_Contribution'].mean().reset_index()
        fig_renewable = px.line(monthly_renewable, x='Month', y='Renewable_Energy_Contribution',
                               title='Monthly Renewable Energy Contribution',
                               markers=True)
        st.plotly_chart(fig_renewable, use_container_width=True)
    
    with tab3:
        st.header("🤖 Model Training")
        
        if st.button("🚀 Start Model Training"):
            with st.spinner('Preparing data and training models...'):
                # Feature engineering
                df_engineered = engineer_features(df)
                X, y, feature_names = prepare_data_for_models(df_engineered)
                
                # Split data
                split_idx = int(0.8 * len(X))
                X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
                
                # Train ML models
                st.subheader("🔧 Training Machine Learning Models")
                ml_models, ml_predictions, ml_metrics = train_ml_models(X_train, y_train, X_test, y_test)
                
                # Store results in session state
                st.session_state['ml_models'] = ml_models
                st.session_state['ml_predictions'] = ml_predictions
                st.session_state['ml_metrics'] = ml_metrics
                st.session_state['y_test'] = y_test
                st.session_state['feature_names'] = feature_names
                
                # Train LSTM (simplified version)
                st.subheader("🧠 Training LSTM Neural Network")
                with st.spinner('Training LSTM model... This may take a few minutes.'):
                    scaler_X = StandardScaler()
                    scaler_y = StandardScaler()
                    
                    X_train_scaled = scaler_X.fit_transform(X_train)
                    X_test_scaled = scaler_X.transform(X_test)
                    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
                    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()
                    
                    # Create sequences
                    time_steps = 24
                    X_train_seq, y_train_seq = create_lstm_sequences(X_train_scaled, y_train_scaled, time_steps)
                    X_test_seq, y_test_seq = create_lstm_sequences(X_test_scaled, y_test_scaled, time_steps)
                    
                    # Build and train model
                    lstm_model = build_lstm_model((time_steps, X_train_scaled.shape[1]))
                    
                    history = lstm_model.fit(
                        X_train_seq, y_train_seq,
                        validation_data=(X_test_seq, y_test_seq),
                        epochs=50,  # Reduced for demo
                        batch_size=32,
                        verbose=0
                    )
                    
                    # Make predictions
                    lstm_pred_scaled = lstm_model.predict(X_test_seq)
                    lstm_pred = scaler_y.inverse_transform(lstm_pred_scaled.reshape(-1, 1)).ravel()
                    y_test_lstm = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).ravel()
                    
                   # Calculate metrics
                lstm_rmse = np.sqrt(mean_squared_error(y_test_lstm, lstm_pred))
                lstm_mae = mean_absolute_error(y_test_lstm, lstm_pred)
                lstm_r2 = r2_score(y_test_lstm, lstm_pred)

# Create dictionary of metrics
                lstm_metrics = {
    "RMSE": lstm_rmse,
    "MAE": lstm_mae,
    "R2": lstm_r2
}

# Save results to session state
                st.session_state['lstm_pred'] = lstm_pred
                st.session_state['y_test_lstm'] = y_test_lstm
                st.session_state['lstm_metrics'] = lstm_metrics

                st.success("🎉 All models trained successfully!")
    
    with tab4:
        st.header("📈 Model Results")
        
        if 'ml_metrics' in st.session_state:
            # Performance comparison
            st.subheader("🏆 Model Performance Comparison")
            
            models_data = []
            for model_name, metrics in st.session_state['ml_metrics'].items():
                models_data.append({
                    'Model': model_name.replace('_', ' '),
                    'RMSE': metrics['RMSE'],
                    'MAE': metrics['MAE'],
                    'R²': metrics['R2']
                })
            
            if 'lstm_metrics' in st.session_state:
                models_data.append({
                    'Model': 'LSTM',
                    'RMSE': st.session_state['lstm_metrics']['RMSE'],
                    'MAE': st.session_state['lstm_metrics']['MAE'],
                    'R²': st.session_state['lstm_metrics']['R2']
                })
            
            results_df = pd.DataFrame(models_data)
            st.dataframe(results_df, use_container_width=True)
            
            # Performance chart
            fig_perf = px.bar(results_df, x='Model', y='RMSE', 
                             title='Model Performance Comparison (RMSE - Lower is Better)',
                             color='Model')
            st.plotly_chart(fig_perf, use_container_width=True)
            
            # Predictions visualization
            st.subheader("🔮 Actual vs Predicted")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Random Forest predictions
                sample_size = 200
                y_test = st.session_state['y_test']
                rf_pred = st.session_state['ml_predictions']['Random_Forest']
                
                fig_rf = go.Figure()
                fig_rf.add_trace(go.Scatter(
                    x=list(range(sample_size)),
                    y=y_test.iloc[:sample_size].values,
                    mode='lines',
                    name='Actual',
                    line=dict(color='blue')
                ))
                fig_rf.add_trace(go.Scatter(
                    x=list(range(sample_size)),
                    y=rf_pred[:sample_size],
                    mode='lines',
                    name='Random Forest',
                    line=dict(color='red', dash='dash')
                ))
                fig_rf.update_layout(title='Random Forest Predictions', xaxis_title='Time Points', yaxis_title='Energy (kWh)')
                st.plotly_chart(fig_rf, use_container_width=True)
            
            with col2:
                # LSTM predictions (if available)
                if 'lstm_pred' in st.session_state:
                    lstm_pred = st.session_state['lstm_pred']
                    y_test_lstm = st.session_state['y_test_lstm']
                    lstm_sample_size = min(200, len(lstm_pred))
                    
                    fig_lstm = go.Figure()
                    fig_lstm.add_trace(go.Scatter(
                        x=list(range(lstm_sample_size)),
                        y=y_test_lstm[:lstm_sample_size],
                        mode='lines',
                        name='Actual',
                        line=dict(color='blue')
                    ))
                    fig_lstm.add_trace(go.Scatter(
                        x=list(range(lstm_sample_size)),
                        y=lstm_pred[:lstm_sample_size],
                        mode='lines',
                        name='LSTM',
                        line=dict(color='green', dash='dash')
                    ))
                    fig_lstm.update_layout(title='LSTM Predictions', xaxis_title='Time Points', yaxis_title='Energy (kWh)')
                    st.plotly_chart(fig_lstm, use_container_width=True)
            
            # Feature importance
            if 'ml_models' in st.session_state and 'feature_names' in st.session_state:
                st.subheader("🎯 Feature Importance (Random Forest)")
                
                rf_model = st.session_state['ml_models']['Random_Forest']
                feature_names = st.session_state['feature_names']
                
                importance_data = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': rf_model.feature_importances_
                }).sort_values('Importance', ascending=False).head(15)
                
                fig_importance = px.bar(importance_data, x='Importance', y='Feature',
                                         orientation='h', title='Top 15 Most Important Features')
                st.plotly_chart(fig_importance, use_container_width=True)
        
        else:
            st.info("Please train the models first in the Model Training tab.")
    
    with tab5:
        st.header("🔮 Energy Consumption Predictions")
        
        if 'ml_models' in st.session_state:
            st.subheader("📊 Custom Prediction")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                temperature = st.slider("🌡️ Temperature (°C)", -10, 35, 20)
                hour = st.slider("🕐 Hour of Day", 0, 23, 12)
                month = st.slider("📅 Month", 1, 12, 6)
            
            with col2:
                renewable_contrib = st.slider("🌱 Renewable Energy %", 0, 80, 40)
                economic_activity = st.slider("💼 Economic Activity Index", 0.5, 1.5, 1.0)
                day_type = st.selectbox("📅 Day Type", ["Weekday", "Weekend", "Holiday"])
            
            with col3:
                season = st.selectbox("🌸 Season", ["Spring", "Summer", "Fall", "Winter"])
                holiday_flag = 1 if day_type == "Holiday" else 0
            
            if st.button("🔮 Make Prediction"):
                # Simplified prediction logic to demonstrate the concept
                st.success("🎯 Prediction feature coming soon! This would use the trained models to predict energy consumption based on your inputs.")
                
                # Placeholder prediction
                base_prediction = 3500
                temp_effect = abs(temperature - 20) * 50
                hour_effect = 500 if 6 <= hour <= 9 or 17 <= hour <= 21 else -200
                renewable_effect = -renewable_contrib * 5
                
                estimated_consumption = base_prediction + temp_effect + hour_effect + renewable_effect
                
                st.metric("⚡ Estimated Energy Consumption", f"{estimated_consumption:.0f} kWh")
        
        else:
            st.info("Please train the models first to enable predictions.")
        
        # Future improvements section
        st.subheader("🚀 Future Enhancements")
        st.markdown("""
        - **Real-time predictions** with live data integration
        - **Weather API integration** for accurate forecasting
        - **IoT sensor data** from smart meters
        - **Grid optimization** recommendations
        - **Carbon footprint** calculations
        - **Mobile app** for grid operators
        """)

if __name__ == "__main__":
    main()
