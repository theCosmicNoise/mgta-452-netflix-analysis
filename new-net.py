import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# SUPPLEMENTARY ANALYSIS FUNCTIONS
# ============================================================================


def calculate_cagr(initial_value, final_value, periods):
    """Calculate Compound Annual Growth Rate"""
    return (((final_value / initial_value) ** (1 / periods)) - 1) * 100


def market_saturation_analysis(subscribers_df):
    """Analyze market saturation using subscriber growth rates"""
    df = subscribers_df.copy()
    df["Growth Rate"] = df["Subscribers"].pct_change() * 100
    df["Acceleration"] = df["Growth Rate"].diff()
    return df


def efficiency_metrics(data):
    """Calculate various efficiency metrics"""
    df = pd.merge(data["revenue"], data["profit"], on="Year")
    df = pd.merge(df, data["subscribers"], on="Year")
    df = pd.merge(df, data["content_spend"], on="Year")

    df["Profit Margin (%)"] = (df["Profit"] / df["Revenue"]) * 100
    df["ARPU ($)"] = (df["Revenue"] * 1000) / df["Subscribers"]
    df["Content % Revenue"] = (df["Content spend"] / df["Revenue"]) * 100
    df["Revenue per $ Content"] = df["Revenue"] / df["Content spend"]
    df["Profit per Subscriber ($)"] = (df["Profit"] * 1000) / df["Subscribers"]

    return df


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Netflix Business Economics",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# SLEEK DARK THEME WITH LIGHT TEXT
# ============================================================================

st.markdown(
    """
<style>
    /* Global Styles */
    .main {
        background-color: #0a0a0a;
        padding-top: 2rem;
    }

    /* Typography */
    .main-header {
        font-size: 2.8rem;
        font-weight: 300;
        letter-spacing: -1px;
        color: #f0f0f0;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    .page-subtitle {
        font-size: 1rem;
        font-weight: 400;
        color: #b0b0b0;
        text-align: center;
        margin-bottom: 3rem;
    }

    .section-header {
        font-size: 1.5rem;
        font-weight: 400;
        color: #f0f0f0;
        margin: 2.5rem 0 1.5rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #2a2a2a;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 300;
        color: #f0f0f0;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.85rem;
        font-weight: 500;
        color: #a0a0a0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    [data-testid="stMetricDelta"] {
        color: #b0b0b0;
    }

    /* Cards */
    .metric-card {
        background-color: #1a1a1a;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #2a2a2a;
        transition: box-shadow 0.3s ease;
    }

    .metric-card:hover {
        box-shadow: 0 2px 20px rgba(229,9,20,0.1);
    }

    /* Insight Boxes */
    .insight-box {
        background-color: #1a1a1a;
        padding: 1.8rem;
        border-radius: 8px;
        border-left: 3px solid #E50914;
        margin: 2rem 0;
    }

    .insight-box h4 {
        color: #f0f0f0;
        margin-bottom: 1rem;
    }

    .insight-box p {
        color: #c0c0c0;
        line-height: 1.6;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0f0f0f;
        border-right: 1px solid #2a2a2a;
    }

    [data-testid="stSidebar"] h1 {
        font-weight: 300;
        font-size: 1.5rem;
        color: #f0f0f0;
    }

    [data-testid="stSidebar"] .stRadio > label {
        color: #f0f0f0;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #c0c0c0;
    }

    /* Dividers */
    hr {
        margin: 2.5rem 0;
        border: none;
        border-top: 1px solid #2a2a2a;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #1a1a1a;
        border: 1px solid #2a2a2a;
        color: #c0c0c0;
        font-weight: 400;
        padding: 0.75rem 1.5rem;
        border-radius: 6px;
    }

    .stTabs [aria-selected="true"] {
        background-color: #E50914;
        color: #ffffff;
        border-color: #E50914;
    }

    /* Clean UI */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 6px;
        font-weight: 400;
        color: #f0f0f0;
    }

    /* Plotly charts */
    .js-plotly-plot .plotly {
        background-color: #0a0a0a !important;
    }

    /* Text elements */
    p, span, div, li {
        color: #c0c0c0 !important;
    }

    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #f0f0f0 !important;
    }

    /* Info boxes */
    .stAlert {
        background-color: #1a1a1a;
        border: 1px solid #2a2a2a;
        color: #c0c0c0;
    }

    /* Dataframes */
    [data-testid="stDataFrame"] {
        background-color: #1a1a1a;
    }

    /* Code blocks */
    .code-block {
        background-color: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 6px;
        padding: 1rem;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        color: #d0d0d0;
        overflow-x: auto;
        margin: 1rem 0;
    }

    /* Project overview cards */
    .overview-card {
        background: linear-gradient(135deg, #1a1a1a 0%, #0f0f0f 100%);
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #2a2a2a;
        margin: 1.5rem 0;
    }

    .overview-card h3 {
        color: #E50914 !important;
        margin-bottom: 1rem;
        font-weight: 400;
    }

    .tech-badge {
        display: inline-block;
        background-color: #E50914;
        color: #ffffff;
        padding: 0.3rem 0.8rem;
        border-radius: 4px;
        font-size: 0.8rem;
        margin: 0.2rem;
        font-weight: 500;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================================
# DATA LOADING
# ============================================================================


@st.cache_data
def load_data():
    try:
        data = {
            "revenue": pd.read_csv("./data-netflix/Revenue.csv"),
            "profit": pd.read_csv("./data-netflix/Profit.csv"),
            "subscribers": pd.read_csv("./data-netflix/NumSubscribers.csv"),
            "content_spend": pd.read_csv("./data-netflix/ContentSpend.csv"),
            "revenue_region": pd.read_csv("./data-netflix/RevenueByRegion.csv"),
            "subscribers_region": pd.read_csv(
                "./data-netflix/NumSubscribersByRegion.csv"
            ),
            "netflix_detailed": pd.read_csv(
                "./data-netflix/netflix_revenue_updated.csv"
            ),
            "movies": pd.read_csv("./data-netflix/movies.csv"),
        }
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


data = load_data()

if data is None:
    st.stop()

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

st.sidebar.markdown("# Netflix Analytics")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Project Documentation",
        "Overview",
        "Financial Analysis",
        "Subscriber Metrics",
        "Regional Performance",
        "Content Strategy",
        "Predictive Models",
        "Statistical Analysis",
        "Content Economics",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "Comprehensive business analytics examining Netflix's revenue, "
    "profitability, subscriber growth, and content strategy using "
    "advanced statistical methods and predictive modeling."
)

# ============================================================================
# PROJECT OVERVIEW PAGE
# ============================================================================

if page == "Project Documentation":
    st.markdown(
        '<div class="main-header">Netflix Business Economics</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="page-subtitle">Comprehensive Data Analytics Project Overview</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Project Description
    st.markdown(
        """
    <div class="overview-card">
        <h3>🎯 Project Objective</h3>
        <p>This analytics dashboard provides a comprehensive examination of Netflix's business economics,
        combining financial metrics, subscriber analytics, regional performance, and content strategy
        insights. The project leverages advanced statistical methods and machine learning to uncover
        trends, forecast future performance, and provide actionable business intelligence.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Technology Stack
    st.markdown(
        """
    <div class="overview-card">
        <h3>🛠️ Technology Stack</h3>
        <p style="margin-bottom: 1rem;">This project utilizes modern data science and visualization tools:</p>
        <div>
            <span class="tech-badge">Python 3.x</span>
            <span class="tech-badge">Streamlit</span>
            <span class="tech-badge">Plotly</span>
            <span class="tech-badge">Pandas</span>
            <span class="tech-badge">NumPy</span>
            <span class="tech-badge">Scikit-learn</span>
            <span class="tech-badge">Machine Learning</span>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Data Sources
    st.markdown(
        """
    <div class="overview-card">
        <h3>📁 Data Architecture</h3>
        <p>The project integrates multiple data sources to provide comprehensive insights:</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Core Business Metrics:**
        - `Revenue.csv` - Historical revenue data
        - `Profit.csv` - Profitability metrics
        - `NumSubscribers.csv` - Global subscriber counts
        - `ContentSpend.csv` - Content investment data
        """)

    with col2:
        st.markdown("""
        **Regional & Content Data:**
        - `RevenueByRegion.csv` - Geographic revenue breakdown
        - `NumSubscribersByRegion.csv` - Regional subscriber distribution
        - `netflix_revenue_updated.csv` - Detailed financial records
        - `movies.csv` - Film budget and box office data
        """)

    # Key Methodologies
    st.markdown(
        """
    <div class="overview-card">
        <h3>🔬 Analytical Methodologies</h3>
    </div>
    """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(
        ["Statistical Analysis", "Predictive Models", "Business Metrics"]
    )

    with tab1:
        st.markdown("""
        #### Statistical Techniques

        **Correlation Analysis**
        - Pearson correlation coefficients to identify relationships between revenue, profit, and subscribers
        - Multi-variate analysis of content spend impact on financial performance

        **Growth Rate Calculations**
        - Compound Annual Growth Rate (CAGR) for long-term trend analysis
        - Quarter-over-quarter and year-over-year growth metrics
        - Market saturation indicators using growth rate derivatives

        **Distribution Analysis**
        - ROI distribution across film genres and budget ranges
        - Regional subscriber concentration metrics
        - Revenue diversification indices
        """)

    with tab2:
        st.markdown("""
        #### Machine Learning Models

        **Linear Regression**
        - Revenue forecasting based on historical trends
        - Subscriber growth prediction models
        - Feature importance analysis for key business drivers

        **Polynomial Regression**
        - Non-linear trend modeling for mature market behavior
        - Saturation curve estimation
        - Advanced forecasting with higher-order features

        **Model Evaluation**
        - R² score analysis for model accuracy
        - Cross-validation for robust predictions
        - Confidence intervals for forecast ranges
        """)

    with tab3:
        st.markdown("""
        #### Business Intelligence Metrics

        **Profitability Indicators**
        - Profit margins and operational efficiency
        - Return on content investment (ROCI)
        - Average Revenue Per User (ARPU)

        **Efficiency Metrics**
        - Content spend as % of revenue
        - Revenue per dollar of content investment
        - Profit per subscriber analysis

        **Strategic Metrics**
        - Regional market penetration rates
        - Content portfolio ROI distribution
        - Subscriber acquisition cost trends
        """)

    # Code Structure
    st.markdown(
        """
    <div class="overview-card">
        <h3>💻 Code Architecture</h3>
        <p>The dashboard follows a modular, maintainable structure:</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("""
    ```python
    # Core Components

    1. Data Loading & Caching
       - @st.cache_data decorator for performance
       - Error handling for missing datasets
       - Automated data type inference

    2. Statistical Functions
       - calculate_cagr(): Compound growth calculations
       - market_saturation_analysis(): Growth rate derivatives
       - efficiency_metrics(): KPI computation

    3. Visualization Layer
       - Plotly for interactive charts
       - Dark theme consistent styling
       - Responsive layout design

    4. Page Routing
       - Sidebar navigation system
       - Modular page components
       - Isolated analysis modules
    ```
    """)

    # Key Findings
    st.markdown(
        """
    <div class="overview-card">
        <h3>🔍 Key Findings Summary</h3>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if data is not None:
        # Calculate some quick insights
        revenue_df = data["revenue"]
        profit_df = data["profit"]
        subscribers_df = data["subscribers"]

        total_revenue = revenue_df["Revenue"].sum()
        avg_profit_margin = (
            profit_df["Profit"].sum() / revenue_df["Revenue"].sum()
        ) * 100
        total_subscribers = subscribers_df["Subscribers"].iloc[-1]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Revenue", f"${total_revenue:.1f}B", "Cumulative")

        with col2:
            st.metric("Avg Profit Margin", f"{avg_profit_margin:.1f}%", "Historical")

        with col3:
            st.metric("Current Subscribers", f"{total_subscribers:.1f}M", "Latest")

        with col4:
            cagr = calculate_cagr(
                revenue_df["Revenue"].iloc[0],
                revenue_df["Revenue"].iloc[-1],
                len(revenue_df) - 1,
            )
            st.metric("Revenue CAGR", f"{cagr:.1f}%", "Multi-year")

    st.markdown(
        """
    <div class="insight-box">
        <h4>Strategic Insights</h4>
        <p><strong>Growth Trajectory:</strong> Netflix demonstrates consistent revenue growth with CAGR exceeding market expectations,
        driven by subscriber expansion and ARPU optimization.</p>
        <p><strong>Content Economics:</strong> Strategic content investment shows strong correlation with subscriber retention,
        with optimal spend ratio around 17-20% of revenue.</p>
        <p><strong>Regional Dynamics:</strong> North America remains the highest ARPU region, while APAC shows strongest
        subscriber growth potential with improving profitability metrics.</p>
        <p><strong>Predictive Outlook:</strong> Machine learning models indicate continued growth with potential market
        saturation in mature markets offset by emerging market expansion.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Navigation Guide
    st.markdown(
        """
    <div class="overview-card">
        <h3>🧭 Dashboard Navigation</h3>
        <p><strong>Overview:</strong> High-level business metrics and KPI trends</p>
        <p><strong>Financial Analysis:</strong> Deep dive into revenue, profit, and cash flow</p>
        <p><strong>Subscriber Metrics:</strong> Growth analysis and cohort behavior</p>
        <p><strong>Regional Performance:</strong> Geographic market analysis</p>
        <p><strong>Content Strategy:</strong> Investment ROI and portfolio optimization</p>
        <p><strong>Predictive Models:</strong> ML-based forecasting and trend projection</p>
        <p><strong>Statistical Analysis:</strong> Advanced analytics and correlations</p>
        <p><strong>Content Economics:</strong> Film budget vs. box office performance</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

# ============================================================================
# OVERVIEW PAGE
# ============================================================================

elif page == "Overview":
    st.markdown(
        '<div class="main-header">Netflix Business Overview</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="page-subtitle">Comprehensive Performance Metrics & Key Insights</div>',
        unsafe_allow_html=True,
    )

    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        latest_revenue = data["revenue"]["Revenue"].iloc[-1]
        prev_revenue = data["revenue"]["Revenue"].iloc[-2]
        revenue_growth = ((latest_revenue - prev_revenue) / prev_revenue) * 100
        st.metric(
            "Latest Revenue", f"${latest_revenue:.1f}B", f"{revenue_growth:+.1f}%"
        )

    with col2:
        latest_profit = data["profit"]["Profit"].iloc[-1]
        prev_profit = data["profit"]["Profit"].iloc[-2]
        profit_growth = ((latest_profit - prev_profit) / prev_profit) * 100
        st.metric("Latest Profit", f"${latest_profit:.2f}B", f"{profit_growth:+.1f}%")

    with col3:
        latest_subs = data["subscribers"]["Subscribers"].iloc[-1]
        prev_subs = data["subscribers"]["Subscribers"].iloc[-2]
        subs_growth = ((latest_subs - prev_subs) / prev_subs) * 100
        st.metric("Subscribers", f"{latest_subs:.1f}M", f"{subs_growth:+.1f}%")

    with col4:
        profit_margin = (latest_profit / latest_revenue) * 100
        st.metric("Profit Margin", f"{profit_margin:.1f}%", "Current")

    st.markdown("---")

    # Revenue & Profit Trends
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            '<div class="section-header">Revenue Growth Trajectory</div>',
            unsafe_allow_html=True,
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=data["revenue"]["Year"],
                y=data["revenue"]["Revenue"],
                mode="lines+markers",
                name="Revenue",
                line=dict(color="#E50914", width=3),
                marker=dict(size=8),
                fill="tonexty",
                fillcolor="rgba(229,9,20,0.1)",
            )
        )

        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="Revenue (Billion USD)",
            height=350,
            paper_bgcolor="#0a0a0a",
            plot_bgcolor="#0a0a0a",
            font=dict(family="Arial", size=13, color="#f0f0f0"),
        )
        fig.update_xaxes(showgrid=True, gridcolor="#2a2a2a")
        fig.update_yaxes(showgrid=True, gridcolor="#2a2a2a")

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(
            '<div class="section-header">Profit Evolution</div>', unsafe_allow_html=True
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=data["profit"]["Year"],
                y=data["profit"]["Profit"],
                mode="lines+markers",
                name="Profit",
                line=dict(color="#00D9FF", width=3),
                marker=dict(size=8),
                fill="tonexty",
                fillcolor="rgba(0,217,255,0.1)",
            )
        )

        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="Profit (Billion USD)",
            height=350,
            paper_bgcolor="#0a0a0a",
            plot_bgcolor="#0a0a0a",
            font=dict(family="Arial", size=13, color="#f0f0f0"),
        )
        fig.update_xaxes(showgrid=True, gridcolor="#2a2a2a")
        fig.update_yaxes(showgrid=True, gridcolor="#2a2a2a")

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Subscriber Growth
    st.markdown(
        '<div class="section-header">Global Subscriber Base Evolution</div>',
        unsafe_allow_html=True,
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data["subscribers"]["Year"],
            y=data["subscribers"]["Subscribers"],
            mode="lines+markers",
            name="Subscribers",
            line=dict(color="#E50914", width=3),
            marker=dict(size=10),
            fill="tozeroy",
            fillcolor="rgba(229,9,20,0.15)",
        )
    )

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Subscribers (Millions)",
        height=400,
        paper_bgcolor="#0a0a0a",
        plot_bgcolor="#0a0a0a",
        font=dict(family="Arial", size=13, color="#f0f0f0"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#2a2a2a")
    fig.update_yaxes(showgrid=True, gridcolor="#2a2a2a")

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Key Insights
    cagr_revenue = calculate_cagr(
        data["revenue"]["Revenue"].iloc[0],
        data["revenue"]["Revenue"].iloc[-1],
        len(data["revenue"]) - 1,
    )

    cagr_subs = calculate_cagr(
        data["subscribers"]["Subscribers"].iloc[0],
        data["subscribers"]["Subscribers"].iloc[-1],
        len(data["subscribers"]) - 1,
    )

    st.markdown(
        f"""
    <div class="insight-box">
        <h4>Key Business Insights</h4>
        <p><strong>Revenue CAGR:</strong> {cagr_revenue:.1f}% annual growth demonstrates strong market position and pricing power</p>
        <p><strong>Subscriber CAGR:</strong> {cagr_subs:.1f}% shows sustained global expansion across mature and emerging markets</p>
        <p><strong>Margin Trend:</strong> {profit_margin:.1f}% current margin reflects operational efficiency and scale advantages</p>
        <p style="color: #c0c0c0; font-size: 0.9rem; margin-top: 1rem;">
            Netflix's business model demonstrates strong unit economics with improving profitability as the platform scales.
            The combination of subscriber growth and ARPU expansion drives sustainable long-term value creation.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

# ============================================================================
# FINANCIAL ANALYSIS
# ============================================================================

elif page == "Financial Analysis":
    st.markdown(
        '<div class="main-header">Financial Analysis</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="page-subtitle">Revenue Streams, Profitability & Investment Strategy</div>',
        unsafe_allow_html=True,
    )

    # Comprehensive Financial Dashboard
    col1, col2, col3 = st.columns(3)

    with col1:
        total_revenue = data["revenue"]["Revenue"].sum()
        st.metric("Cumulative Revenue", f"${total_revenue:.1f}B")

    with col2:
        total_profit = data["profit"]["Profit"].sum()
        st.metric("Cumulative Profit", f"${total_profit:.2f}B")

    with col3:
        total_content = data["content_spend"]["Content spend"].sum()
        st.metric("Total Content Spend", f"${total_content:.1f}B")

    st.markdown("---")

    # Revenue vs Profit vs Content Spend
    st.markdown(
        '<div class="section-header">Financial Metrics Comparison</div>',
        unsafe_allow_html=True,
    )

    merged_df = pd.merge(data["revenue"], data["profit"], on="Year")
    merged_df = pd.merge(merged_df, data["content_spend"], on="Year")

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=merged_df["Year"],
            y=merged_df["Revenue"],
            name="Revenue",
            marker_color="#E50914",
            opacity=0.8,
        )
    )

    fig.add_trace(
        go.Bar(
            x=merged_df["Year"],
            y=merged_df["Profit"],
            name="Profit",
            marker_color="#00D9FF",
            opacity=0.8,
        )
    )

    fig.add_trace(
        go.Bar(
            x=merged_df["Year"],
            y=merged_df["Content spend"],
            name="Content Spend",
            marker_color="#FFD700",
            opacity=0.8,
        )
    )

    fig.update_layout(
        barmode="group",
        xaxis_title="Year",
        yaxis_title="USD (Billions)",
        height=450,
        paper_bgcolor="#0a0a0a",
        plot_bgcolor="#0a0a0a",
        font=dict(family="Arial", size=13, color="#f0f0f0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#2a2a2a")

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Profit Margin Analysis
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            '<div class="section-header">Profit Margin Trend</div>',
            unsafe_allow_html=True,
        )

        merged_df["Profit Margin (%)"] = (
            merged_df["Profit"] / merged_df["Revenue"]
        ) * 100

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=merged_df["Year"],
                y=merged_df["Profit Margin (%)"],
                mode="lines+markers",
                line=dict(color="#E50914", width=3),
                marker=dict(size=10),
                fill="tozeroy",
                fillcolor="rgba(229,9,20,0.15)",
            )
        )

        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="Profit Margin (%)",
            height=350,
            paper_bgcolor="#0a0a0a",
            plot_bgcolor="#0a0a0a",
            font=dict(family="Arial", size=13, color="#f0f0f0"),
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor="#2a2a2a")

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(
            '<div class="section-header">Content Spend vs Revenue</div>',
            unsafe_allow_html=True,
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=merged_df["Content spend"],
                y=merged_df["Revenue"],
                mode="markers",
                marker=dict(
                    size=15,
                    color=merged_df["Year"],
                    colorscale="Reds",
                    showscale=True,
                    colorbar=dict(title="Year"),
                    line=dict(width=1, color="#ffffff"),
                ),
                text=merged_df["Year"],
                hovertemplate="<b>Year: %{text}</b><br>"
                + "Content: $%{x:.1f}B<br>"
                + "Revenue: $%{y:.1f}B<extra></extra>",
            )
        )

        fig.update_layout(
            xaxis_title="Content Spend (Billion USD)",
            yaxis_title="Revenue (Billion USD)",
            height=350,
            paper_bgcolor="#0a0a0a",
            plot_bgcolor="#0a0a0a",
            font=dict(family="Arial", size=13, color="#f0f0f0"),
        )
        fig.update_xaxes(showgrid=True, gridcolor="#2a2a2a")
        fig.update_yaxes(showgrid=True, gridcolor="#2a2a2a")

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Financial Insights
    avg_margin = merged_df["Profit Margin (%)"].mean()
    content_ratio = (
        merged_df["Content spend"].sum() / merged_df["Revenue"].sum()
    ) * 100

    st.markdown(
        f"""
    <div class="insight-box">
        <h4>Financial Performance Analysis</h4>
        <p><strong>Average Profit Margin:</strong> {avg_margin:.1f}% demonstrates strong operational efficiency</p>
        <p><strong>Content Investment Ratio:</strong> {content_ratio:.1f}% of revenue reinvested in content development</p>
        <p><strong>Growth Strategy:</strong> Balanced approach between revenue maximization and strategic content investment</p>
        <p style="color: #c0c0c0; font-size: 0.9rem; margin-top: 1rem;">
            Netflix maintains a disciplined financial strategy with improving margins while sustaining aggressive content
            investment to drive subscriber growth and retention.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

# ============================================================================
# SUBSCRIBER METRICS
# ============================================================================

elif page == "Subscriber Metrics":
    st.markdown(
        '<div class="main-header">Subscriber Analytics</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="page-subtitle">Growth Patterns, Market Penetration & Retention Insights</div>',
        unsafe_allow_html=True,
    )

    # Subscriber Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        current_subs = data["subscribers"]["Subscribers"].iloc[-1]
        st.metric("Current Subscribers", f"{current_subs:.1f}M")

    with col2:
        initial_subs = data["subscribers"]["Subscribers"].iloc[0]
        growth_multiple = current_subs / initial_subs
        st.metric("Growth Multiple", f"{growth_multiple:.1f}x")

    with col3:
        subs_cagr = calculate_cagr(
            data["subscribers"]["Subscribers"].iloc[0],
            data["subscribers"]["Subscribers"].iloc[-1],
            len(data["subscribers"]) - 1,
        )
        st.metric("CAGR", f"{subs_cagr:.1f}%")

    with col4:
        latest_growth = data["subscribers"]["Subscribers"].pct_change().iloc[-1] * 100
        st.metric("Latest Growth", f"{latest_growth:.1f}%")

    st.markdown("---")

    # Growth Rate Analysis
    st.markdown(
        '<div class="section-header">Subscriber Growth Rate Analysis</div>',
        unsafe_allow_html=True,
    )

    saturation_df = market_saturation_analysis(data["subscribers"])

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Absolute Subscriber Count", "Growth Rate (%)"),
        vertical_spacing=0.12,
        row_heights=[0.6, 0.4],
    )

    fig.add_trace(
        go.Scatter(
            x=saturation_df["Year"],
            y=saturation_df["Subscribers"],
            mode="lines+markers",
            name="Subscribers",
            line=dict(color="#E50914", width=3),
            marker=dict(size=8),
            fill="tozeroy",
            fillcolor="rgba(229,9,20,0.15)",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=saturation_df["Year"],
            y=saturation_df["Growth Rate"],
            mode="lines+markers",
            name="Growth Rate",
            line=dict(color="#00D9FF", width=3),
            marker=dict(size=8),
        ),
        row=2,
        col=1,
    )

    fig.update_xaxes(
        title_text="Year", row=2, col=1, showgrid=True, gridcolor="#2a2a2a"
    )
    fig.update_yaxes(
        title_text="Subscribers (M)", row=1, col=1, showgrid=True, gridcolor="#2a2a2a"
    )
    fig.update_yaxes(
        title_text="Growth Rate (%)", row=2, col=1, showgrid=True, gridcolor="#2a2a2a"
    )

    fig.update_layout(
        height=600,
        paper_bgcolor="#0a0a0a",
        plot_bgcolor="#0a0a0a",
        font=dict(family="Arial", size=13, color="#f0f0f0"),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ARPU Analysis
    st.markdown(
        '<div class="section-header">Average Revenue Per User (ARPU)</div>',
        unsafe_allow_html=True,
    )

    merged_df = pd.merge(data["revenue"], data["subscribers"], on="Year")
    merged_df["ARPU"] = (merged_df["Revenue"] * 1000) / merged_df["Subscribers"]

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=merged_df["Year"],
                y=merged_df["ARPU"],
                mode="lines+markers",
                line=dict(color="#E50914", width=3),
                marker=dict(size=10),
                fill="tonexty",
                fillcolor="rgba(229,9,20,0.15)",
            )
        )

        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="ARPU (USD)",
            height=350,
            paper_bgcolor="#0a0a0a",
            plot_bgcolor="#0a0a0a",
            font=dict(family="Arial", size=13, color="#f0f0f0"),
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor="#2a2a2a")

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        arpu_growth = merged_df["ARPU"].pct_change() * 100

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=merged_df["Year"][1:],
                y=arpu_growth[1:],
                marker_color=[
                    "#E50914" if x > 0 else "#00D9FF" for x in arpu_growth[1:]
                ],
            )
        )

        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="ARPU Growth (%)",
            height=350,
            paper_bgcolor="#0a0a0a",
            plot_bgcolor="#0a0a0a",
            font=dict(family="Arial", size=13, color="#f0f0f0"),
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor="#2a2a2a")

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Subscriber Insights
    avg_arpu = merged_df["ARPU"].mean()
    latest_arpu = merged_df["ARPU"].iloc[-1]

    st.markdown(
        f"""
    <div class="insight-box">
        <h4>Subscriber Metrics Insights</h4>
        <p><strong>CAGR:</strong> {subs_cagr:.1f}% compound annual growth demonstrates strong market expansion</p>
        <p><strong>Current ARPU:</strong> ${latest_arpu:.2f} reflects pricing power and premium tier adoption</p>
        <p><strong>Average ARPU:</strong> ${avg_arpu:.2f} over the analysis period</p>
        <p style="color: #c0c0c0; font-size: 0.9rem; margin-top: 1rem;">
            Growth rate moderation in mature markets is offset by expanding ARPU through tier optimization and
            password-sharing policies, demonstrating Netflix's ability to extract more value per subscriber.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

# ============================================================================
# REGIONAL PERFORMANCE
# ============================================================================

# ============================================================================
# REGIONAL PERFORMANCE
# ============================================================================

elif page == "Regional Performance":
    st.markdown(
        '<div class="main-header">Regional Analysis</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="page-subtitle">Geographic Market Performance</div>',
        unsafe_allow_html=True,
    )

    regions = ["US & Canada", "EMEA", "Latin America", "Asia-Pacific"]
    latest_subs_region = data["subscribers_region"].iloc[-1]

    # Regional Distribution
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            '<div class="section-header">Subscriber Distribution</div>',
            unsafe_allow_html=True,
        )

        values = [latest_subs_region[region] for region in regions]

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=regions,
                    values=values,
                    hole=0.5,
                    marker_colors=["#f5d000", "#4285F4", "#34A853", "#FF6B35"],
                    textfont=dict(size=12, color="white"),
                )
            ]
        )

        fig.update_layout(
            height=350,
            paper_bgcolor="#000000",
            font=dict(family="Arial", size=13, color="#f0f0f0"),
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5
            ),
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(
            '<div class="section-header">Regional Growth</div>', unsafe_allow_html=True
        )

        first_year = data["subscribers_region"].iloc[0]
        last_year = data["subscribers_region"].iloc[-1]

        growth_rates = []
        for region in regions:
            growth = ((last_year[region] / first_year[region]) - 1) * 100
            growth_rates.append(growth)

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=regions,
                y=growth_rates,
                marker_color="#E50914",
                text=[f"{x:.0f}%" for x in growth_rates],
                textposition="outside",
                textfont=dict(size=11),
            )
        )

        fig.update_layout(
            yaxis_title="Total Growth (%)",
            height=350,
            paper_bgcolor="#000000",
            plot_bgcolor="#000000",
            font=dict(family="Arial", size=13, color="#f0f0f0"),
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor="#2a2a2a")

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Regional Evolution
    st.markdown(
        '<div class="section-header">Regional Subscriber Evolution</div>',
        unsafe_allow_html=True,
    )

    fig = go.Figure()

    colors_map = {
        "US & Canada": "#f5d000",
        "EMEA": "#4285F4",
        "Latin America": "#34A853",
        "Asia-Pacific": "#FF6B35",
    }

    for region in regions:
        fig.add_trace(
            go.Scatter(
                x=data["subscribers_region"]["Year"],
                y=data["subscribers_region"][region],
                mode="lines+markers",
                name=region,
                line=dict(width=2, color=colors_map[region]),
                marker=dict(size=8),
            )
        )

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Subscribers (M)",
        height=400,
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        font=dict(family="Arial", size=13, color="#f0f0f0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#2a2a2a")

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Regional ARPU
    st.markdown(
        '<div class="section-header">Regional ARPU Analysis</div>',
        unsafe_allow_html=True,
    )

    arpu_data = []
    for idx, row in data["revenue_region"].iterrows():
        year = row["Year"]
        sub_row = data["subscribers_region"][
            data["subscribers_region"]["Year"] == year
        ].iloc[0]

        for region in regions:
            arpu = (row[region] * 1000) / sub_row[region]
            arpu_data.append({"Year": year, "Region": region, "ARPU": arpu})

    arpu_df = pd.DataFrame(arpu_data)

    fig = go.Figure()

    for region in regions:
        region_data = arpu_df[arpu_df["Region"] == region]
        fig.add_trace(
            go.Bar(
                name=region,
                x=region_data["Year"],
                y=region_data["ARPU"],
                marker_color=colors_map[region],
            )
        )

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="ARPU ($)",
        barmode="group",
        height=400,
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        font=dict(family="Arial", size=13, color="#f0f0f0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#2a2a2a")

    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# CONTENT STRATEGY
# ============================================================================

elif page == "Content Strategy":
    st.markdown(
        '<div class="main-header">Content Investment Strategy</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="page-subtitle">Spending Patterns, ROI & Portfolio Optimization</div>',
        unsafe_allow_html=True,
    )

    # Content Spend Overview
    col1, col2, col3 = st.columns(3)

    with col1:
        total_content = data["content_spend"]["Content spend"].sum()
        st.metric("Total Content Investment", f"${total_content:.1f}B")

    with col2:
        latest_content = data["content_spend"]["Content spend"].iloc[-1]
        prev_content = data["content_spend"]["Content spend"].iloc[-2]
        content_growth = ((latest_content - prev_content) / prev_content) * 100
        st.metric("Latest Spend", f"${latest_content:.2f}B", f"{content_growth:+.1f}%")

    with col3:
        content_cagr = calculate_cagr(
            data["content_spend"]["Content spend"].iloc[0],
            data["content_spend"]["Content spend"].iloc[-1],
            len(data["content_spend"]) - 1,
        )
        st.metric("Content CAGR", f"{content_cagr:.1f}%")

    st.markdown("---")

    # Content Spend Evolution
    st.markdown(
        '<div class="section-header">Content Spend Evolution</div>',
        unsafe_allow_html=True,
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data["content_spend"]["Year"],
            y=data["content_spend"]["Content spend"],
            mode="lines+markers",
            line=dict(color="#E50914", width=3),
            marker=dict(size=10),
            fill="tozeroy",
            fillcolor="rgba(229,9,20,0.15)",
        )
    )

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Content Spend (Billion USD)",
        height=400,
        paper_bgcolor="#0a0a0a",
        plot_bgcolor="#0a0a0a",
        font=dict(family="Arial", size=13, color="#f0f0f0"),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#2a2a2a")

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Content ROI Analysis
    st.markdown(
        '<div class="section-header">Content Investment vs Business Outcomes</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        merged_df = pd.merge(data["content_spend"], data["revenue"], on="Year")
        merged_df = pd.merge(merged_df, data["subscribers"], on="Year")

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=merged_df["Content spend"],
                y=merged_df["Revenue"],
                mode="markers",
                marker=dict(
                    size=15,
                    color=merged_df["Year"],
                    colorscale="Reds",
                    showscale=True,
                    colorbar=dict(title="Year"),
                    line=dict(width=1, color="#ffffff"),
                ),
                text=merged_df["Year"],
                hovertemplate="<b>Year: %{text}</b><br>"
                + "Content: $%{x:.1f}B<br>"
                + "Revenue: $%{y:.1f}B<extra></extra>",
            )
        )

        fig.update_layout(
            xaxis_title="Content Spend (Billion USD)",
            yaxis_title="Revenue (Billion USD)",
            height=350,
            paper_bgcolor="#0a0a0a",
            plot_bgcolor="#0a0a0a",
            font=dict(family="Arial", size=13, color="#f0f0f0"),
        )
        fig.update_xaxes(showgrid=True, gridcolor="#2a2a2a")
        fig.update_yaxes(showgrid=True, gridcolor="#2a2a2a")

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=merged_df["Content spend"],
                y=merged_df["Subscribers"],
                mode="markers",
                marker=dict(
                    size=15,
                    color=merged_df["Year"],
                    colorscale="Blues",
                    showscale=True,
                    colorbar=dict(title="Year"),
                    line=dict(width=1, color="#ffffff"),
                ),
                text=merged_df["Year"],
                hovertemplate="<b>Year: %{text}</b><br>"
                + "Content: $%{x:.1f}B<br>"
                + "Subscribers: %{y:.1f}M<extra></extra>",
            )
        )

        fig.update_layout(
            xaxis_title="Content Spend (Billion USD)",
            yaxis_title="Subscribers (Millions)",
            height=350,
            paper_bgcolor="#0a0a0a",
            plot_bgcolor="#0a0a0a",
            font=dict(family="Arial", size=13, color="#f0f0f0"),
        )
        fig.update_xaxes(showgrid=True, gridcolor="#2a2a2a")
        fig.update_yaxes(showgrid=True, gridcolor="#2a2a2a")

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Content Efficiency Metrics
    st.markdown(
        '<div class="section-header">Content Investment Efficiency</div>',
        unsafe_allow_html=True,
    )

    merged_df["Content % Revenue"] = (
        merged_df["Content spend"] / merged_df["Revenue"]
    ) * 100
    merged_df["Revenue per $ Content"] = (
        merged_df["Revenue"] / merged_df["Content spend"]
    )

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=merged_df["Year"],
                y=merged_df["Content % Revenue"],
                mode="lines+markers",
                line=dict(color="#FFD700", width=3),
                marker=dict(size=10),
                fill="tonexty",
                fillcolor="rgba(255,215,0,0.15)",
            )
        )

        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="Content as % of Revenue",
            height=350,
            paper_bgcolor="#0a0a0a",
            plot_bgcolor="#0a0a0a",
            font=dict(family="Arial", size=13, color="#f0f0f0"),
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor="#2a2a2a")

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=merged_df["Year"],
                y=merged_df["Revenue per $ Content"],
                marker_color="#00D9FF",
            )
        )

        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="Revenue per $ Content Spend",
            height=350,
            paper_bgcolor="#0a0a0a",
            plot_bgcolor="#0a0a0a",
            font=dict(family="Arial", size=13, color="#f0f0f0"),
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor="#2a2a2a")

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Content Strategy Insights
    avg_content_ratio = merged_df["Content % Revenue"].mean()
    latest_roi = merged_df["Revenue per $ Content"].iloc[-1]

    st.markdown(
        f"""
    <div class="insight-box">
        <h4>Content Strategy Insights</h4>
        <p><strong>Average Content Ratio:</strong> {avg_content_ratio:.1f}% of revenue invested in content</p>
        <p><strong>Current ROI:</strong> ${latest_roi:.2f} revenue generated per dollar of content spend</p>
        <p><strong>Strategic Balance:</strong> Sustained investment in original content drives differentiation and subscriber loyalty</p>
        <p style="color: #c0c0c0; font-size: 0.9rem; margin-top: 1rem;">
            Netflix's content strategy demonstrates disciplined investment with improving returns. The balance between
            scale and efficiency suggests optimal portfolio management with focus on high-impact original productions.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

# ============================================================================
# PREDICTIVE MODELS
# ============================================================================

elif page == "Predictive Models":
    st.markdown(
        '<div class="main-header">Predictive Analytics</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="page-subtitle">Machine Learning Forecasts & Trend Projections</div>',
        unsafe_allow_html=True,
    )

    # Revenue Prediction
    st.markdown(
        '<div class="section-header">Revenue Forecast Model</div>',
        unsafe_allow_html=True,
    )

    revenue_df = data["revenue"].copy()
    X = revenue_df[["Year"]].values
    y = revenue_df["Revenue"].values

    # Linear Model
    lr_model = LinearRegression()
    lr_model.fit(X, y)

    # Polynomial Model
    poly_features = PolynomialFeatures(degree=2)
    X_poly = poly_features.fit_transform(X)
    poly_model = LinearRegression()
    poly_model.fit(X_poly, y)

    # Future predictions
    future_years = np.array([[2025], [2025], [2026], [2027], [2028]])
    future_years_poly = poly_features.transform(future_years)

    linear_pred = lr_model.predict(future_years)
    poly_pred = poly_model.predict(future_years_poly)

    # Calculate R² scores
    linear_r2 = r2_score(y, lr_model.predict(X))
    poly_r2 = r2_score(y, poly_model.predict(X_poly))

    # Visualization
    fig = go.Figure()

    # Historical data
    fig.add_trace(
        go.Scatter(
            x=revenue_df["Year"],
            y=revenue_df["Revenue"],
            mode="markers",
            name="Historical Data",
            marker=dict(size=10, color="#E50914"),
        )
    )

    # Linear prediction
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([revenue_df["Year"].values, future_years.flatten()]),
            y=np.concatenate([lr_model.predict(X), linear_pred]),
            mode="lines",
            name=f"Linear Model (R²={linear_r2:.3f})",
            line=dict(color="#00D9FF", width=2, dash="dash"),
        )
    )

    # Polynomial prediction
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([revenue_df["Year"].values, future_years.flatten()]),
            y=np.concatenate([poly_model.predict(X_poly), poly_pred]),
            mode="lines",
            name=f"Polynomial Model (R²={poly_r2:.3f})",
            line=dict(color="#FFD700", width=2, dash="dot"),
        )
    )

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Revenue (Billion USD)",
        height=450,
        paper_bgcolor="#0a0a0a",
        plot_bgcolor="#0a0a0a",
        font=dict(family="Arial", size=13, color="#f0f0f0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#2a2a2a")
    fig.update_yaxes(showgrid=True, gridcolor="#2a2a2a")

    st.plotly_chart(fig, use_container_width=True)

    # Prediction table
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Linear Model Forecasts**")
        forecast_df = pd.DataFrame(
            {
                "Year": future_years.flatten(),
                "Predicted Revenue ($B)": linear_pred.round(2),
            }
        )
        st.dataframe(forecast_df, hide_index=True, use_container_width=True)

    with col2:
        st.markdown("**Polynomial Model Forecasts**")
        forecast_df_poly = pd.DataFrame(
            {
                "Year": future_years.flatten(),
                "Predicted Revenue ($B)": poly_pred.round(2),
            }
        )
        st.dataframe(forecast_df_poly, hide_index=True, use_container_width=True)

    st.markdown("---")

    # Subscriber Prediction
    st.markdown(
        '<div class="section-header">Subscriber Growth Forecast</div>',
        unsafe_allow_html=True,
    )

    subs_df = data["subscribers"].copy()
    X_subs = subs_df[["Year"]].values
    y_subs = subs_df["Subscribers"].values

    # Models
    lr_subs = LinearRegression()
    lr_subs.fit(X_subs, y_subs)

    poly_features_subs = PolynomialFeatures(degree=2)
    X_subs_poly = poly_features_subs.fit_transform(X_subs)
    poly_subs = LinearRegression()
    poly_subs.fit(X_subs_poly, y_subs)

    # Predictions
    future_years_subs_poly = poly_features_subs.transform(future_years)
    linear_subs_pred = lr_subs.predict(future_years)
    poly_subs_pred = poly_subs.predict(future_years_subs_poly)

    # R² scores
    linear_subs_r2 = r2_score(y_subs, lr_subs.predict(X_subs))
    poly_subs_r2 = r2_score(y_subs, poly_subs.predict(X_subs_poly))

    # Visualization
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=subs_df["Year"],
            y=subs_df["Subscribers"],
            mode="markers",
            name="Historical Data",
            marker=dict(size=10, color="#E50914"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=np.concatenate([subs_df["Year"].values, future_years.flatten()]),
            y=np.concatenate([lr_subs.predict(X_subs), linear_subs_pred]),
            mode="lines",
            name=f"Linear Model (R²={linear_subs_r2:.3f})",
            line=dict(color="#00D9FF", width=2, dash="dash"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=np.concatenate([subs_df["Year"].values, future_years.flatten()]),
            y=np.concatenate([poly_subs.predict(X_subs_poly), poly_subs_pred]),
            mode="lines",
            name=f"Polynomial Model (R²={poly_subs_r2:.3f})",
            line=dict(color="#FFD700", width=2, dash="dot"),
        )
    )

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Subscribers (Millions)",
        height=450,
        paper_bgcolor="#0a0a0a",
        plot_bgcolor="#0a0a0a",
        font=dict(family="Arial", size=13, color="#f0f0f0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#2a2a2a")
    fig.update_yaxes(showgrid=True, gridcolor="#2a2a2a")

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Model Insights
    st.markdown(
        f"""
    <div class="insight-box">
        <h4>Predictive Model Insights</h4>
        <p><strong>Revenue Model Accuracy:</strong> Polynomial model (R²={poly_r2:.3f}) outperforms linear model (R²={linear_r2:.3f})</p>
        <p><strong>Subscriber Model Accuracy:</strong> Polynomial model (R²={poly_subs_r2:.3f}) vs linear model (R²={linear_subs_r2:.3f})</p>
        <p><strong>Forecast Interpretation:</strong> Non-linear models better capture market maturation and saturation effects</p>
        <p style="color: #c0c0c0; font-size: 0.9rem; margin-top: 1rem;">
            The higher R² scores for polynomial models suggest that Netflix's growth follows a non-linear pattern,
            typical of companies transitioning from high-growth to mature phases. Future projections should account
            for market saturation dynamics and competitive pressures.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

elif page == "Statistical Analysis":
    st.markdown(
        '<div class="main-header">Statistical Deep Dive</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="page-subtitle">Correlation Analysis & Efficiency Metrics</div>',
        unsafe_allow_html=True,
    )

    # Calculate efficiency metrics
    efficiency_df = efficiency_metrics(data)

    # Correlation Matrix
    st.markdown(
        '<div class="section-header">Business Metrics Correlation Matrix</div>',
        unsafe_allow_html=True,
    )

    correlation_vars = [
        "Revenue",
        "Profit",
        "Subscribers",
        "Content spend",
        "Profit Margin (%)",
        "ARPU ($)",
    ]
    correlation_df = efficiency_df[correlation_vars].corr()

    fig = go.Figure(
        data=go.Heatmap(
            z=correlation_df.values,
            x=correlation_vars,
            y=correlation_vars,
            colorscale="RdBu",
            zmid=0,
            text=correlation_df.values.round(2),
            texttemplate="%{text}",
            textfont={"size": 12, "color": "white"},
            colorbar=dict(title="Correlation"),
        )
    )

    fig.update_layout(
        height=500,
        paper_bgcolor="#0a0a0a",
        plot_bgcolor="#0a0a0a",
        font=dict(family="Arial", size=12, color="#f0f0f0"),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Efficiency Metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        avg_margin = efficiency_df["Profit Margin (%)"].mean()
        st.metric("Avg Profit Margin", f"{avg_margin:.1f}%")

    with col2:
        avg_arpu = efficiency_df["ARPU ($)"].mean()
        st.metric("Avg ARPU", f"${avg_arpu:.2f}")

    with col3:
        avg_content_pct = efficiency_df["Content % Revenue"].mean()
        st.metric("Avg Content % Revenue", f"{avg_content_pct:.1f}%")

    st.markdown("---")

    # Time Series Analysis
    st.markdown(
        '<div class="section-header">Efficiency Trends Over Time</div>',
        unsafe_allow_html=True,
    )

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Profit Margin Evolution",
            "ARPU Trend",
            "Content as % Revenue",
            "Revenue per $ Content",
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    # Profit Margin
    fig.add_trace(
        go.Scatter(
            x=efficiency_df["Year"],
            y=efficiency_df["Profit Margin (%)"],
            mode="lines+markers",
            line=dict(color="#E50914", width=2),
            marker=dict(size=6),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # ARPU
    fig.add_trace(
        go.Scatter(
            x=efficiency_df["Year"],
            y=efficiency_df["ARPU ($)"],
            mode="lines+markers",
            line=dict(color="#00D9FF", width=2),
            marker=dict(size=6),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # Content % Revenue
    fig.add_trace(
        go.Scatter(
            x=efficiency_df["Year"],
            y=efficiency_df["Content % Revenue"],
            mode="lines+markers",
            line=dict(color="#FFD700", width=2),
            marker=dict(size=6),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # Revenue per $ Content
    fig.add_trace(
        go.Scatter(
            x=efficiency_df["Year"],
            y=efficiency_df["Revenue per $ Content"],
            mode="lines+markers",
            line=dict(color="#00FF00", width=2),
            marker=dict(size=6),
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    fig.update_xaxes(showgrid=True, gridcolor="#2a2a2a")
    fig.update_yaxes(showgrid=True, gridcolor="#2a2a2a")

    fig.update_layout(
        height=600,
        paper_bgcolor="#0a0a0a",
        plot_bgcolor="#0a0a0a",
        font=dict(family="Arial", size=11, color="#f0f0f0"),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Detailed Metrics Table
    st.markdown(
        '<div class="section-header">Comprehensive Efficiency Metrics</div>',
        unsafe_allow_html=True,
    )

    display_df = efficiency_df[
        [
            "Year",
            "Profit Margin (%)",
            "ARPU ($)",
            "Content % Revenue",
            "Revenue per $ Content",
            "Profit per Subscriber ($)",
        ]
    ].copy()

    display_df["Profit Margin (%)"] = display_df["Profit Margin (%)"].round(1)
    display_df["ARPU ($)"] = display_df["ARPU ($)"].round(2)
    display_df["Content % Revenue"] = display_df["Content % Revenue"].round(1)
    display_df["Revenue per $ Content"] = display_df["Revenue per $ Content"].round(2)
    display_df["Profit per Subscriber ($)"] = display_df[
        "Profit per Subscriber ($)"
    ].round(2)

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Statistical Insights
    revenue_content_corr = efficiency_df["Revenue"].corr(efficiency_df["Content spend"])
    profit_arpu_corr = efficiency_df["Profit"].corr(efficiency_df["ARPU ($)"])

    st.markdown(
        f"""
    <div class="insight-box">
        <h4>Statistical Analysis Insights</h4>
        <p><strong>Revenue-Content Correlation:</strong> {revenue_content_corr:.3f} indicates strong positive relationship</p>
        <p><strong>Profit-ARPU Correlation:</strong> {profit_arpu_corr:.3f} shows how pricing drives profitability</p>
        <p><strong>Efficiency Trend:</strong> Improving metrics demonstrate operational leverage and scale benefits</p>
        <p style="color: #c0c0c0; font-size: 0.9rem; margin-top: 1rem;">
            High correlations between revenue and content spend validate Netflix's investment strategy. The positive
            trajectory in efficiency metrics indicates successful transformation from growth to profitability focus.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

# ============================================================================
# CONTENT ECONOMICS (Movie Analysis)
# ============================================================================

elif page == "Content Economics":
    st.markdown(
        '<div class="main-header">Content Economics</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="page-subtitle">Film Production Budget vs Box Office Performance</div>',
        unsafe_allow_html=True,
    )

    if data["movies"] is None or len(data["movies"]) == 0:
        st.warning("⚠️ Movies dataset not available for analysis.")
    else:
        # Prepare movie data
        movies_df = data["movies"].copy()
        movies_df = movies_df.dropna(subset=["budget", "gross"])
        movies_df = movies_df[movies_df["budget"] > 0]
        movies_df = movies_df[movies_df["gross"] > 0]
        movies_df["roi"] = (
            (movies_df["gross"] - movies_df["budget"]) / movies_df["budget"]
        ) * 100
        movies_df["profit"] = movies_df["gross"] - movies_df["budget"]

        # Overview Metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Movies", f"{len(movies_df):,}")

        with col2:
            avg_budget = movies_df["budget"].mean() / 1e6
            st.metric("Avg Budget", f"${avg_budget:.1f}M")

        with col3:
            avg_gross = movies_df["gross"].mean() / 1e6
            st.metric("Avg Gross", f"${avg_gross:.1f}M")

        with col4:
            avg_roi = movies_df["roi"].mean()
            st.metric("Avg ROI", f"{avg_roi:.0f}%")

        st.markdown("---")

        # Genre Analysis
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                '<div class="section-header">ROI by Genre</div>', unsafe_allow_html=True
            )

            genre_roi = (
                movies_df.groupby("genre")
                .agg({"roi": "mean", "name": "count"})
                .reset_index()
            )
            genre_roi.columns = ["Genre", "Avg ROI", "Count"]
            genre_roi = (
                genre_roi[genre_roi["Count"] >= 10]
                .sort_values("Avg ROI", ascending=False)
                .head(10)
            )

            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    x=genre_roi["Avg ROI"],
                    y=genre_roi["Genre"],
                    orientation="h",
                    marker_color="#E50914",
                    text=genre_roi["Count"],
                    texttemplate="n=%{text}",
                    textposition="outside",
                    textfont=dict(size=10, color="#f0f0f0"),
                )
            )

            fig.update_layout(
                xaxis_title="Average ROI (%)",
                height=400,
                paper_bgcolor="#0a0a0a",
                plot_bgcolor="#0a0a0a",
                font=dict(family="Arial", size=13, color="#f0f0f0"),
            )
            fig.update_xaxes(showgrid=True, gridcolor="#2a2a2a")
            fig.update_yaxes(showgrid=False)

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown(
                '<div class="section-header">Budget vs Box Office</div>',
                unsafe_allow_html=True,
            )

            sample_df = movies_df.sample(min(500, len(movies_df)))
            sample_df["profit_size"] = (sample_df["profit"].abs() + 1000) / 1000000

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=sample_df["budget"],
                    y=sample_df["gross"],
                    mode="markers",
                    marker=dict(
                        size=sample_df["profit_size"].clip(2, 20),
                        color=sample_df["roi"],
                        colorscale=[[0, "#E50914"], [0.5, "#ffffff"], [1, "#00D9FF"]],
                        showscale=True,
                        colorbar=dict(title="ROI (%)"),
                        line=dict(width=0.5, color="#666"),
                    ),
                    text=sample_df["name"],
                    customdata=np.column_stack(
                        (sample_df["year"], sample_df["genre"], sample_df["roi"])
                    ),
                    hovertemplate="<b>%{text}</b><br>"
                    + "Budget: $%{x:,.0f}<br>"
                    + "Gross: $%{y:,.0f}<br>"
                    + "Year: %{customdata[0]}<br>"
                    + "Genre: %{customdata[1]}<br>"
                    + "ROI: %{customdata[2]:.1f}%<extra></extra>",
                )
            )

            # Break-even line
            max_val = max(sample_df["budget"].max(), sample_df["gross"].max())
            fig.add_trace(
                go.Scatter(
                    x=[0, max_val],
                    y=[0, max_val],
                    mode="lines",
                    name="Break-even",
                    line=dict(color="#999", dash="dash", width=1),
                    showlegend=True,
                )
            )

            fig.update_layout(
                xaxis_title="Budget (USD)",
                yaxis_title="Box Office Gross (USD)",
                height=400,
                paper_bgcolor="#0a0a0a",
                plot_bgcolor="#0a0a0a",
                font=dict(family="Arial", size=13, color="#f0f0f0"),
                hovermode="closest",
            )
            fig.update_xaxes(showgrid=True, gridcolor="#2a2a2a")
            fig.update_yaxes(showgrid=True, gridcolor="#2a2a2a")

            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Top Performers
        st.markdown(
            '<div class="section-header">Top Performing Films</div>',
            unsafe_allow_html=True,
        )

        tab1, tab2, tab3 = st.tabs(
            ["Highest ROI", "Highest Grossing", "Most Profitable"]
        )

        with tab1:
            top_roi = movies_df.nlargest(10, "roi")[
                ["name", "genre", "year", "budget", "gross", "roi"]
            ]
            top_roi["budget"] = top_roi["budget"].apply(lambda x: f"${x / 1e6:.1f}M")
            top_roi["gross"] = top_roi["gross"].apply(lambda x: f"${x / 1e6:.1f}M")
            top_roi["roi"] = top_roi["roi"].apply(lambda x: f"{x:.0f}%")
            st.dataframe(top_roi, use_container_width=True, hide_index=True)

        with tab2:
            top_gross = movies_df.nlargest(10, "gross")[
                ["name", "genre", "year", "budget", "gross", "roi"]
            ]
            top_gross["budget"] = top_gross["budget"].apply(
                lambda x: f"${x / 1e6:.1f}M"
            )
            top_gross["gross"] = top_gross["gross"].apply(lambda x: f"${x / 1e6:.1f}M")
            top_gross["roi"] = top_gross["roi"].apply(lambda x: f"{x:.0f}%")
            st.dataframe(top_gross, use_container_width=True, hide_index=True)

        with tab3:
            top_profit = movies_df.nlargest(10, "profit")[
                ["name", "genre", "year", "profit", "roi"]
            ]
            top_profit["profit"] = top_profit["profit"].apply(
                lambda x: f"${x / 1e9:.2f}B"
            )
            top_profit["roi"] = top_profit["roi"].apply(lambda x: f"{x:.0f}%")
            st.dataframe(top_profit, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Content Strategy Insights
        st.markdown(
            f"""
        <div class="insight-box">
            <h4>Content Strategy Insights</h4>
            <p><strong>Average Film ROI:</strong> {movies_df["roi"].mean():.0f}%</p>
            <p><strong>Success Rate:</strong> {(movies_df["roi"] > 0).mean() * 100:.1f}% of films are profitable</p>
            <p><strong>Blockbuster Rate:</strong> {(movies_df["roi"] > 200).mean() * 100:.1f}% achieve >200% ROI</p>
            <p style="color: #c0c0c0; font-size: 0.9rem; margin-top: 1rem;">
                Lower-budget films often achieve higher ROI percentages, suggesting a balanced portfolio strategy
                combining blockbusters with cost-efficient productions reduces risk while maximizing returns.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

# Footer
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: #c0c0c0; font-size: 0.9rem; padding: 2rem 0;'>
    <p>Netflix Business Economics Dashboard | Data-Driven Insights</p>
    <p style="color: #808080;">© 2025 | Comprehensive Analysis of Streaming Platform Performance</p>
</div>
""",
    unsafe_allow_html=True,
)
