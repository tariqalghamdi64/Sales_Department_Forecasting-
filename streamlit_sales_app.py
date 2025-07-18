import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from prophet import Prophet
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configuration - Update these URLs as needed
GOOGLE_DRIVE_FILE_ID = "1s7T9_2Z68w8jUkHv-XeiRCqz88BXK-PF"  # From your Google Drive link
GITHUB_STORE_URL = "https://raw.githubusercontent.com/tariqalghamdi64/Sales_Department_Forecasting-/main/store.csv"  # Your GitHub repository

# Page configuration
st.set_page_config(
    page_title="📊 Sales Department Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark mode and styling
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stApp {
        background-color: #0e1117;
    }
    .css-1d391kg {
        background-color: #262730;
    }
    .stSelectbox > div > div > div {
        background-color: #262730;
        color: #fafafa;
    }
    .stButton > button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #ff3333;
        color: white;
    }
    .metric-card {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #4a4a4a;
        margin: 10px 0;
    }
    .title-text {
        color: #ff4b4b;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 30px;
    }
    .subtitle-text {
        color: #fafafa;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading functions
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_sales_data():
    """Load and cache sales training data from Google Drive"""
    try:
        # Google Drive file ID from the URL
        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}&export=download"
        
        sales_data = pd.read_csv(url)
        return sales_data
    except Exception as e:
        st.error(f"❌ Error loading sales data from Google Drive: {e}")
        st.info("💡 Please ensure the Google Drive link is accessible and the file is publicly shared.")
        return None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_store_data():
    """Load and cache store information data from GitHub"""
    try:
        store_data = pd.read_csv(GITHUB_STORE_URL)
        return store_data
    except Exception as e:
        st.error(f"❌ Error loading store data from GitHub: {e}")
        st.info("💡 Please update the GITHUB_STORE_URL in the configuration section with your actual repository URL.")
        return None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def preprocess_data(sales_data, store_data):
    """Preprocess and merge data with caching"""
    if sales_data is None or store_data is None:
        return None, None, None
    
    # Filter open stores only
    sales_df = sales_data[sales_data['Open'] == 1].copy()
    sales_df.drop(columns=['Open'], axis=1, inplace=True)
    
    # Convert Date column to datetime in sales_df
    sales_df['Date'] = pd.to_datetime(sales_df['Date'])
    
    # Handle missing values in store data
    store_df = store_data.copy()
    
    # Fill missing values
    str_cols = ['Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval', 
                'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear']
    for col in str_cols:
        if col in store_df.columns:
            store_df[col] = store_df[col].fillna(0)
    
    # Fill CompetitionDistance with mean
    if 'CompetitionDistance' in store_df.columns:
        store_df['CompetitionDistance'].fillna(store_df['CompetitionDistance'].mean(), inplace=True)
    
    # Merge data
    merged_df = pd.merge(sales_df, store_df, on='Store', how='left')
    
    # Add date features
    merged_df['Year'] = merged_df['Date'].dt.year
    merged_df['Month'] = merged_df['Date'].dt.month
    merged_df['Day'] = merged_df['Date'].dt.day
    
    return sales_df, store_df, merged_df

@st.cache_data(ttl=3600)  # Cache for 1 hour
def create_holidays_data(merged_df):
    """Create holidays data for Prophet forecasting"""
    if merged_df is None:
        return None
    
    # School holidays
    school_holidays = merged_df[merged_df['SchoolHoliday'] == 1]['Date'].values
    school_holidays_df = pd.DataFrame({
        'ds': pd.to_datetime(school_holidays),
        'holiday': 'school_holiday'
    })
    
    # State holidays
    state_holidays_mask = (merged_df['StateHoliday'] == 'a') | \
                         (merged_df['StateHoliday'] == 'b') | \
                         (merged_df['StateHoliday'] == 'c')
    state_holidays = merged_df[state_holidays_mask]['Date'].values
    state_holidays_df = pd.DataFrame({
        'ds': pd.to_datetime(state_holidays),
        'holiday': 'state_holiday'
    })
    
    # Combine holidays
    all_holidays = pd.concat([state_holidays_df, school_holidays_df])
    return all_holidays

@st.cache_data(ttl=1800)  # Cache for 30 minutes (forecasts change more frequently)
def generate_forecast(store_id, merged_df, holidays_df, periods, include_holidays=True):
    """Generate sales forecast for a specific store"""
    if merged_df is None:
        return None, None
    
    # Filter data for specific store
    store_data = merged_df[merged_df['Store'] == store_id].copy()
    if store_data.empty:
        return None, None
    
    # Prepare data for Prophet
    prophet_data = store_data[['Date', 'Sales']].rename(columns={'Date': 'ds', 'Sales': 'y'})
    prophet_data = prophet_data.sort_values('ds')
    
    # Create and fit model
    if include_holidays and holidays_df is not None:
        model = Prophet(holidays=holidays_df)
    else:
        model = Prophet()
    
    model.fit(prophet_data)
    
    # Make future predictions
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    
    return forecast, model

def create_plotly_forecast_plot(forecast, model, store_id):
    """Create interactive Plotly forecast plot"""
    if forecast is None or model is None:
        return None
    
    # Create figure
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=forecast['ds'][:-len(forecast)+len(model.history)],
        y=model.history['y'],
        mode='markers+lines',
        name='Historical Sales',
        line=dict(color='#ff4b4b', width=2),
        marker=dict(size=4)
    ))
    
    # Add forecast
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        mode='lines',
        name='Forecast',
        line=dict(color='#00ff00', width=2, dash='dash')
    ))
    
    # Add confidence intervals
    fig.add_trace(go.Scatter(
        x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
        y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(0,255,0,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence Interval'
    ))
    
    fig.update_layout(
        title=f'📈 Sales Forecast for Store {store_id}',
        xaxis_title='Date',
        yaxis_title='Sales',
        template='plotly_dark',
        height=500,
        showlegend=True
    )
    
    return fig

def create_components_plot(forecast, model):
    """Create Prophet components plot using Plotly"""
    if forecast is None or model is None:
        return None
    
    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=('Trend', 'Yearly Seasonality', 'Weekly Seasonality', 'Holidays'),
        vertical_spacing=0.08
    )
    
    # Trend component
    fig.add_trace(
        go.Scatter(x=forecast['ds'], y=forecast['trend'], 
                  mode='lines', name='Trend', line=dict(color='#ff4b4b')),
        row=1, col=1
    )
    
    # Yearly seasonality
    if 'yearly' in forecast.columns:
        fig.add_trace(
            go.Scatter(x=forecast['ds'], y=forecast['yearly'], 
                      mode='lines', name='Yearly', line=dict(color='#00ff00')),
            row=2, col=1
        )
    
    # Weekly seasonality
    if 'weekly' in forecast.columns:
        fig.add_trace(
            go.Scatter(x=forecast['ds'], y=forecast['weekly'], 
                      mode='lines', name='Weekly', line=dict(color='#0080ff')),
            row=3, col=1
        )
    
    # Holidays
    if 'holidays' in forecast.columns:
        fig.add_trace(
            go.Scatter(x=forecast['ds'], y=forecast['holidays'], 
                      mode='lines', name='Holidays', line=dict(color='#ff8000')),
            row=4, col=1
        )
    
    fig.update_layout(
        title='🔍 Forecast Components Analysis',
        template='plotly_dark',
        height=800,
        showlegend=False
    )
    
    return fig

# Main app
def main():
    st.markdown('<h1 class="title-text">📊 Sales Department Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("## 🎛️ Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Overview", "📈 Data Analysis", "🔮 Sales Forecasting", "📊 Store Analytics"]
    )
    
    # Data status and cache management
    st.sidebar.markdown("## 📊 Data Status")
    
    if 'data_loaded' in st.session_state and st.session_state.data_loaded:
        st.sidebar.success("✅ Data loaded from cache")
    else:
        st.sidebar.info("🔄 Loading data...")
    
    # Debug: Clear cache button
    if st.sidebar.button("🔄 Clear Cache"):
        st.cache_data.clear()
        if 'data_loaded' in st.session_state:
            del st.session_state.data_loaded
        st.success("✅ Cache cleared! Data will be reloaded on next refresh.")
        st.rerun()
    
    # Load data with status indicators
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    if not st.session_state.data_loaded:
        with st.spinner("🔄 Loading data for the first time..."):
            sales_data = load_sales_data()
            store_data = load_store_data()
            if sales_data is not None and store_data is not None:
                st.success("✅ Data loaded successfully! Cached for future use.")
                st.session_state.data_loaded = True
            else:
                st.error("❌ Failed to load data. Please check your data sources.")
                return
    else:
        # Load cached data silently
        sales_data = load_sales_data()
        store_data = load_store_data()
    
    # Preprocess data (this is also cached)
    sales_df, store_df, merged_df = preprocess_data(sales_data, store_data)
    holidays_df = create_holidays_data(merged_df)
    
    if merged_df is None:
        st.error("❌ Failed to load data. Please check your data files.")
        return
    
    # Page routing
    if page == "🏠 Overview":
        show_overview(merged_df, sales_df, store_df, sales_data, store_data)
    elif page == "📈 Data Analysis":
        show_data_analysis(merged_df)
    elif page == "🔮 Sales Forecasting":
        show_forecasting(merged_df, holidays_df)
    elif page == "📊 Store Analytics":
        show_store_analytics(merged_df)

def show_overview(merged_df, sales_df, store_df, sales_data, store_data):
    """Overview page with key metrics and insights"""
    st.markdown('<h2 class="subtitle-text">🏠 Dashboard Overview</h2>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🏪 Total Stores", len(merged_df['Store'].unique()))
    
    with col2:
        start_date = merged_df['Date'].min().strftime('%Y-%m-%d')
        end_date = merged_df['Date'].max().strftime('%Y-%m-%d')
        
        st.markdown("### 📅 Date Range")
        st.markdown(f"**From:** {start_date}")
        st.markdown(f"**To:** {end_date}")
    
    with col3:
        st.metric("💰 Total Sales", f"${merged_df['Sales'].sum():,.0f}")
    
    with col4:
        st.metric("👥 Total Customers", f"{merged_df['Customers'].sum():,.0f}")
    
    # Data summary
    st.markdown("### 📋 Data Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Sales Data Info")
        # Create a summary of sales data
        sales_summary = pd.DataFrame({
            'Metric': ['Total Rows', 'Total Columns', 'Date Range', 'Unique Stores', 'Total Sales', 'Total Customers'],
            'Value': [
                len(sales_df),
                len(sales_df.columns),
                f"{sales_df['Date'].min().strftime('%Y-%m-%d')} to {sales_df['Date'].max().strftime('%Y-%m-%d')}",
                sales_df['Store'].nunique(),
                f"${sales_df['Sales'].sum():,.0f}",
                f"{sales_df['Customers'].sum():,.0f}"
            ]
        })
        st.dataframe(sales_summary, use_container_width=True)
        
        # Show data types
        st.markdown("**Data Types:**")
        dtype_info = pd.DataFrame({
            'Column': sales_df.columns,
            'Data Type': sales_df.dtypes.astype(str),
            'Non-Null Count': sales_df.count(),
            'Null Count': sales_df.isnull().sum()
        })
        st.dataframe(dtype_info, use_container_width=True)
    
    with col2:
        st.markdown("#### Store Data Info")
        # Create a summary of store data
        store_summary = pd.DataFrame({
            'Metric': ['Total Stores', 'Total Columns', 'Store Types', 'Assortment Types', 'Avg Competition Distance'],
            'Value': [
                len(store_df),
                len(store_df.columns),
                store_df['StoreType'].nunique(),
                store_df['Assortment'].nunique(),
                f"{store_df['CompetitionDistance'].mean():.1f} km"
            ]
        })
        st.dataframe(store_summary, use_container_width=True)
        
        # Show data types
        st.markdown("**Data Types:**")
        dtype_info = pd.DataFrame({
            'Column': store_df.columns,
            'Data Type': store_df.dtypes.astype(str),
            'Non-Null Count': store_df.count(),
            'Null Count': store_df.isnull().sum()
        })
        st.dataframe(dtype_info, use_container_width=True)
    
    # Missing data visualization
    st.markdown("### 🔍 Missing Data Analysis")
    
    # Before preprocessing
    st.markdown("#### 📊 Before Preprocessing")
    col1, col2 = st.columns(2)
    
    with col1:
        # Original sales data missing values
        # Use the already loaded data instead of calling load_sales_data again
        if sales_data is not None:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(sales_data.isnull(), cbar=False, cmap='viridis', ax=ax)
            plt.title('Missing Data - Original Sales Data')
            st.pyplot(fig)
            plt.close()
            
            # Missing values summary
            missing_sales = sales_data.isnull().sum()
            if missing_sales.sum() > 0:
                st.markdown("**Missing Values in Sales Data:**")
                missing_df = pd.DataFrame({
                    'Column': missing_sales.index,
                    'Missing Count': missing_sales.values,
                    'Missing Percentage': (missing_sales.values / len(sales_data) * 100).round(2)
                })
                st.dataframe(missing_df[missing_df['Missing Count'] > 0], use_container_width=True)
            else:
                st.success("✅ No missing values in original sales data!")
    
    with col2:
        # Original store data missing values
        if store_data is not None:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(store_data.isnull(), cbar=False, cmap='viridis', ax=ax)
            plt.title('Missing Data - Original Store Data')
            st.pyplot(fig)
            plt.close()
            
            # Missing values summary
            missing_store = store_data.isnull().sum()
            if missing_store.sum() > 0:
                st.markdown("**Missing Values in Store Data:**")
                missing_df = pd.DataFrame({
                    'Column': missing_store.index,
                    'Missing Count': missing_store.values,
                    'Missing Percentage': (missing_store.values / len(store_data) * 100).round(2)
                })
                st.dataframe(missing_df[missing_df['Missing Count'] > 0], use_container_width=True)
            else:
                st.success("✅ No missing values in original store data!")
    
    # After preprocessing
    st.markdown("#### 🧹 After Preprocessing")
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(sales_df.isnull(), cbar=False, cmap='viridis', ax=ax)
        plt.title('Missing Data - Processed Sales Data')
        st.pyplot(fig)
        plt.close()
        
        # Check if any missing values remain
        missing_sales_processed = sales_df.isnull().sum()
        if missing_sales_processed.sum() > 0:
            st.warning("⚠️ Some missing values remain in processed sales data")
            missing_df = pd.DataFrame({
                'Column': missing_sales_processed.index,
                'Missing Count': missing_sales_processed.values
            })
            st.dataframe(missing_df[missing_df['Missing Count'] > 0], use_container_width=True)
        else:
            st.success("✅ All missing values handled in processed sales data!")
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(store_df.isnull(), cbar=False, cmap='viridis', ax=ax)
        plt.title('Missing Data - Processed Store Data')
        st.pyplot(fig)
        plt.close()
        
        # Check if any missing values remain
        missing_store_processed = store_df.isnull().sum()
        if missing_store_processed.sum() > 0:
            st.warning("⚠️ Some missing values remain in processed store data")
            missing_df = pd.DataFrame({
                'Column': missing_store_processed.index,
                'Missing Count': missing_store_processed.values
            })
            st.dataframe(missing_df[missing_df['Missing Count'] > 0], use_container_width=True)
        else:
            st.success("✅ All missing values handled in processed store data!")

def show_data_analysis(merged_df):
    """Data analysis page with interactive visualizations"""
    st.markdown('<h2 class="subtitle-text">📈 Data Analysis & Insights</h2>', unsafe_allow_html=True)
    
    # Correlation analysis
    st.markdown("### 🔗 Correlation Analysis")
    correlations = merged_df.corr(numeric_only=True)['Sales'].sort_values(ascending=False)
    
    fig = px.bar(
        x=correlations.index,
        y=correlations.values,
        title="📊 Correlation with Sales",
        labels={'x': 'Features', 'y': 'Correlation'},
        color=correlations.values,
        color_continuous_scale='RdBu'
    )
    fig.update_layout(template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)
    
    # Time series analysis
    st.markdown("### 📅 Time Series Analysis")
    
    # Monthly trends
    monthly_sales = merged_df.groupby('Month')['Sales'].mean().reset_index()
    fig = px.line(
        monthly_sales,
        x='Month',
        y='Sales',
        title="📈 Average Sales by Month",
        markers=True
    )
    fig.update_layout(template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)
    
    # Day of week analysis
    dow_sales = merged_df.groupby('DayOfWeek')['Sales'].mean().reset_index()
    fig = px.bar(
        dow_sales,
        x='DayOfWeek',
        y='Sales',
        title="📊 Average Sales by Day of Week",
        color='Sales',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)
    
    # Store type analysis
    st.markdown("### 🏪 Store Type Analysis")
    store_type_sales = merged_df.groupby('StoreType')['Sales'].mean().reset_index()
    fig = px.pie(
        store_type_sales,
        values='Sales',
        names='StoreType',
        title="🥧 Average Sales by Store Type"
    )
    fig.update_layout(template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)
    
    # Promo analysis
    st.markdown("### 🎉 Promotional Impact")
    promo_analysis = merged_df.groupby('Promo')[['Sales', 'Customers']].mean().reset_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            promo_analysis,
            x='Promo',
            y='Sales',
            title="💰 Sales by Promo Status",
            color='Promo',
            color_discrete_map={0: '#ff4b4b', 1: '#00ff00'}
        )
        fig.update_layout(template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            promo_analysis,
            x='Promo',
            y='Customers',
            title="👥 Customers by Promo Status",
            color='Promo',
            color_discrete_map={0: '#ff4b4b', 1: '#00ff00'}
        )
        fig.update_layout(template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)

def show_forecasting(merged_df, holidays_df):
    """Sales forecasting page with Prophet models"""
    st.markdown('<h2 class="subtitle-text">🔮 Sales Forecasting</h2>', unsafe_allow_html=True)
    
    # Store selection
    available_stores = sorted(merged_df['Store'].unique())
    selected_store = st.selectbox("🏪 Select Store ID:", available_stores)
    
    # Forecasting parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        forecast_periods = st.slider("📅 Forecast Periods (days):", 30, 365, 90)
    
    with col2:
        include_holidays = st.checkbox("🎉 Include Holidays", value=True)
    
    with col3:
        if st.button("🚀 Generate Forecast"):
            with st.spinner("🔮 Generating forecast..."):
                forecast, model = generate_forecast(
                    selected_store, merged_df, holidays_df, 
                    forecast_periods, include_holidays
                )
                
                if forecast is not None and model is not None:
                    st.success(f"✅ Forecast generated for Store {selected_store}!")
                    
                    # Display forecast plot
                    st.markdown("### 📈 Sales Forecast")
                    forecast_plot = create_plotly_forecast_plot(forecast, model, selected_store)
                    if forecast_plot:
                        st.plotly_chart(forecast_plot, use_container_width=True)
                    
                    # Display components plot
                    st.markdown("### 🔍 Forecast Components")
                    components_plot = create_components_plot(forecast, model)
                    if components_plot:
                        st.plotly_chart(components_plot, use_container_width=True)
                    
                    # Display forecast data
                    st.markdown("### 📊 Forecast Data")
                    forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(20)
                    forecast_display.columns = ['Date', 'Predicted Sales', 'Lower Bound', 'Upper Bound']
                    st.dataframe(forecast_display, use_container_width=True)
                    
                    # Download forecast data
                    csv = forecast_display.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Forecast Data",
                        data=csv,
                        file_name=f"forecast_store_{selected_store}.csv",
                        mime="text/csv"
                    )
                else:
                    st.error(f"❌ No data available for Store {selected_store}")

def show_store_analytics(merged_df):
    """Store-specific analytics page"""
    st.markdown('<h2 class="subtitle-text">📊 Store Analytics</h2>', unsafe_allow_html=True)
    
    # Store selection
    available_stores = sorted(merged_df['Store'].unique())
    selected_store = st.selectbox("🏪 Select Store for Analysis:", available_stores)
    
    if selected_store:
        store_data = merged_df[merged_df['Store'] == selected_store].copy()
        
        if not store_data.empty:
            # Store metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("💰 Total Sales", f"${store_data['Sales'].sum():,.0f}")
            
            with col2:
                st.metric("👥 Total Customers", f"{store_data['Customers'].sum():,.0f}")
            
            with col3:
                st.metric("📈 Average Daily Sales", f"${store_data['Sales'].mean():,.0f}")
            
            with col4:
                st.metric("📅 Days of Data", len(store_data))
            
            # Store performance over time
            st.markdown("### 📈 Store Performance Over Time")
            
            fig = px.line(
                store_data,
                x='Date',
                y=['Sales', 'Customers'],
                title=f"📊 Store {selected_store} Performance",
                labels={'value': 'Count', 'variable': 'Metric'}
            )
            fig.update_layout(template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
            
            # Store details
            st.markdown("### 🏪 Store Information")
            store_info = store_data[['StoreType', 'Assortment', 'CompetitionDistance', 
                                   'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear']].iloc[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Store Type:**", store_info['StoreType'])
                st.write("**Assortment:**", store_info['Assortment'])
                st.write("**Competition Distance:**", f"{store_info['CompetitionDistance']:.1f} km")
            
            with col2:
                st.write("**Competition Open Since:**", 
                        f"{store_info['CompetitionOpenSinceMonth']}/{store_info['CompetitionOpenSinceYear']}")
            
            # Store data table
            st.markdown("### 📋 Store Data")
            st.dataframe(store_data.head(20), use_container_width=True)
            
            # Download store data
            csv = store_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Store Data",
                data=csv,
                file_name=f"store_{selected_store}_data.csv",
                mime="text/csv"
            )
        else:
            st.warning(f"⚠️ No data available for Store {selected_store}")

if __name__ == "__main__":
    main() 
