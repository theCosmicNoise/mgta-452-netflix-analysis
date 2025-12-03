# Netflix Business Economics Dashboard

## The Business Behind the Stream: Economics of Global Entertainment Platforms

A comprehensive dashboard analyzing Netflix's business economics, including revenue streams, content spending strategies, subscriber growth patterns, and profitability trends.

## Project Overview

This project examines the business economics of global streaming and film production, focusing on how platforms like Netflix earn, spend, and sustain profitability. Using publicly available financial and content data, we analyze:

- **Financial Performance**: Revenue, profit margins, and content investment trends
- **Regional Analysis**: Market performance across UCAN, EMEA, LATAM, and APAC
- **Content Economics**: ROI analysis of film production budgets and box office performance
- **Subscriber Dynamics**: Growth patterns, acquisition costs, and ARPU trends
- **Predictive Analysis**: Revenue and subscriber forecasting with scenario modeling

## Features

### 6 Interactive Analysis Sections:

1. **Project Overview**: Introduction and project process
2. **Overview**: Executive summary with key metrics and high-level trends
3. **Financial Analysis**: Revenue, profit, content spend analysis
4. **Subscriber Metrics**: Growth trends and ARPU analysis
5. **Regional Performance**: Revenue and subscriber breakdown by region
6. **Content Strategy**: ROI and economics of content production
7. **Predictive Models**: Forecasting models and scenario planning
8. **Statistical Analysis**: Advanced statistical methods and insights
9. **Content Economics**: Deep dive into content production economics and ROI

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Prepare Data Files

Ensure all the following CSV files are in the directory:
- `Revenue.csv`
- `Profit.csv`
- `ContentSpend.csv`
- `NumSubscribers.csv`
- `RevenueByRegion.csv`
- `NumSubscribersByRegion.csv`
- `netflix_revenue_updated.csv`
- `movies.csv`

### Step 3: Run the Dashboard

```bash
streamlit run new-net.py
```

The dashboard will open automatically in your default web browser at `http://localhost:8501`

## Data Sources

The dashboard analyzes the following datasets:

1. **Annual Financial Data**: https://www.kaggle.com/datasets/mauryansshivam/netflix-ott-revenue-and-subscribers-csv-file
   - Revenue (billions USD)
   - Profit (billions USD)
   - Content Spend (billions USD)
   - Total Subscribers (millions)

2. **Quarterly Regional Data**: https://www.kaggle.com/datasets/adnananam/netflix-revenue-and-usage-statistics
   - Revenue by region (UCAN, EMEA, LATAM, APAC)
   - Subscribers by region
   - ARPU (Average Revenue Per User) by region

3. **Movies Dataset**: https://raw.githubusercontent.com/danielgrijalva/movie-stats/master/movies.csv
   - Production budgets
   - Box office gross
   - Genre, ratings, and other metadata
   - ROI calculations

## Key Insights

### Financial Highlights
- Netflix's quarterly revenue has grown significantly from Q1 2019 to Q3 2023
- Profit margins have improved as the company scales
- Content spend represents approximately 60-80% of revenue

### Regional Performance
- UCAN (US & Canada) shows highest ARPU but slower growth
- APAC (Asia-Pacific) is the fastest-growing region
- International markets driving majority of new subscriber growth

### Content Strategy
- Lower-budget films often achieve higher ROI percentages
- Genre diversification reduces risk
- Portfolio approach balances blockbusters with cost-efficient productions

### Subscriber Growth
- Market maturation in developed regions
- Focus shifting from pure growth to ARPU optimization
- Regional expansion remains key growth driver

## Technical Stack

- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive visualizations
- **Scikit-learn**: Machine learning for forecasting
- **NumPy**: Numerical computing

## Visualization Types

The dashboard includes:
- Line charts for time-series trends
- Bar charts for comparisons
- Pie charts for distribution analysis
- Scatter plots for correlation analysis
- Stacked area charts for regional breakdown
- Heatmaps for correlation matrices
- Dual-axis charts for multi-metric analysis

## Dashboard Features

- **Responsive Design**: Adapts to different screen sizes
- **Interactive Filters**: Navigate between analysis sections
- **Real-time Calculations**: Dynamic metric updates
- **Scenario Planning**: Adjustable parameters for forecasting
- **Export-Ready**: High-quality charts suitable for presentations

## Usage Tips

1. **Navigation**: Use the sidebar to switch between analysis sections
2. **Interactivity**: Hover over charts to see detailed values
3. **Zoom**: Click and drag on charts to zoom into specific periods
4. **Scenario Analysis**: Adjust sliders in the Predictive Analysis section to model different business scenarios

### Key Metrics Explained:
- **ARPU**: Average Revenue Per User (monthly)
- **QoQ Growth**: Quarter-over-Quarter growth rate
- **ROI**: Return on Investment ((Gross - Budget) / Budget × 100)
- **CAGR**: Compound Annual Growth Rate
- **Content Spend %**: Content spending as percentage of revenue

## Contributing

This project is built as the final deliverable for MGTA 452: Collecting and Analyzing Large Data course at Rady School of Management, UCSD. Contributors are Anjali Roy, Kai Ni and Prabhlin Kaur Matta.

---
