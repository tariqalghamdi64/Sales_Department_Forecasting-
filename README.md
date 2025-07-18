# 📊 Sales Department Analytics Dashboard

A comprehensive Streamlit web application for analyzing sales data and generating forecasts using Prophet models.

## 🚀 Features

- **🏠 Overview Dashboard**: Key metrics and data summary
- **📈 Data Analysis**: Interactive visualizations and insights
- **🔮 Sales Forecasting**: Prophet-based forecasting with holiday effects
- **📊 Store Analytics**: Individual store performance analysis
- **🎨 Dark Mode**: Beautiful dark theme with emojis
- **💾 Data Caching**: Optimized performance with proper caching
- **📥 Data Export**: Download forecasts and store data

## 📋 Prerequisites

The app automatically loads data from:
- **Google Drive**: Sales training data (`train.csv`) - Large file hosted on Google Drive
- **GitHub**: Store information data (`store.csv`) - Smaller file hosted on GitHub

### 🔧 Configuration
Update the following URLs in `streamlit_sales_app.py` if needed:
- `GOOGLE_DRIVE_FILE_ID`: Your Google Drive file ID for the train dataset
- `GITHUB_STORE_URL`: Your GitHub raw URL for the store dataset

## 🛠️ Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Streamlit app:**
   ```bash
   streamlit run streamlit_sales_app.py
   ```

3. **Open your browser** and navigate to the URL shown in the terminal (usually `http://localhost:8501`)

## 📱 App Navigation

### 🏠 Overview Page
- Key metrics dashboard
- Data summary and information
- Missing data analysis with heatmaps

### 📈 Data Analysis Page
- Correlation analysis with sales
- Time series analysis (monthly, daily trends)
- Store type performance comparison
- Promotional impact analysis

### 🔮 Sales Forecasting Page
- Select any store for forecasting
- Adjustable forecast periods (30-365 days)
- Option to include/exclude holidays
- Interactive forecast plots with confidence intervals
- Component analysis (trend, seasonality, holidays)
- Download forecast data

### 📊 Store Analytics Page
- Individual store performance metrics
- Store-specific visualizations
- Store information and details
- Download store-specific data

## 🎨 Features

- **Responsive Design**: Works on desktop and mobile
- **Interactive Plots**: Hover effects and zoom capabilities
- **Real-time Updates**: Instant response to user interactions
- **Error Handling**: Graceful handling of missing data
- **Performance Optimized**: Efficient data caching and processing

## 🔧 Technical Details

- **Framework**: Streamlit
- **Visualization**: Plotly (interactive) + Matplotlib/Seaborn
- **Forecasting**: Facebook Prophet
- **Data Processing**: Pandas, NumPy
- **Caching**: Streamlit's built-in caching for optimal performance

## 📊 Data Requirements

The app expects the following columns in your data files:

**train.csv:**
- Store, Date, Sales, Customers, Open, Promo, StateHoliday, SchoolHoliday, DayOfWeek

**store.csv:**
- Store, StoreType, Assortment, CompetitionDistance, CompetitionOpenSinceMonth, CompetitionOpenSinceYear, Promo2, Promo2SinceWeek, Promo2SinceYear, PromoInterval

## 🐛 Troubleshooting

1. **Data Loading Issues**: Ensure your CSV files are in the correct location
2. **Missing Dependencies**: Run `pip install -r requirements.txt`
3. **Prophet Installation**: If Prophet fails to install, try: `pip install prophet --no-deps`
4. **Memory Issues**: The app uses caching to optimize memory usage

## 📈 Future Enhancements

- Multiple store comparison
- Advanced forecasting models
- Real-time data integration
- Custom date range selection
- Export to different formats (Excel, PDF)

---

**Happy Analyzing! 📊✨** 