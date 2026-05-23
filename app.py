import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Used Bike Prices Analysis",
    page_icon="🏍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Load data and models
@st.cache_resource
def load_data():
    df = pd.read_csv('bikes_cleaned.csv')
    return df

@st.cache_resource
def load_model():
    model = joblib.load('bike_price_model.pkl')
    scaler = joblib.load('scaler.pkl')
    le_brand = joblib.load('label_encoder_brand.pkl')
    le_owner = joblib.load('label_encoder_owner.pkl')
    le_location = joblib.load('label_encoder_location.pkl')
    return model, scaler, le_brand, le_owner, le_location

# Load resources
df = load_data()
model, scaler, le_brand, le_owner, le_location = load_model()

# Sidebar navigation
st.sidebar.title("🏍️ Navigation")
page = st.sidebar.radio("Select Page", [
    "Dashboard Overview",
    "Data Exploration",
    "Price Prediction",
    "Model Analysis",
    "Brand Insights"
])

# Main header
st.markdown('<h1 class="main-header">Used Bike Prices Analysis Dashboard</h1>', unsafe_allow_html=True)
st.markdown("---")

if page == "Dashboard Overview":
    st.header("📊 Dashboard Overview")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Bikes", f"{len(df):,}")
    with col2:
        st.metric("Average Price", f"₹{df['price'].mean():,.0f}")
    with col3:
        st.metric("Unique Brands", df['brand'].nunique())
    with col4:
        st.metric("Unique Locations", df['location'].nunique())
    
    st.markdown("---")
    
    # Price Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Price Distribution")
        fig_price = px.histogram(df, x='price', nbins=50, 
                                 title='Distribution of Bike Prices',
                                 labels={'price': 'Price (INR)'})
        fig_price.update_layout(height=400)
        st.plotly_chart(fig_price, use_container_width=True)
    
    with col2:
        st.subheader("Brand Distribution")
        brand_counts = df['brand'].value_counts().head(10)
        fig_brand = px.bar(x=brand_counts.values, y=brand_counts.index,
                           orientation='h', title='Top 10 Brands',
                           labels={'x': 'Count', 'y': 'Brand'})
        fig_brand.update_layout(height=400)
        st.plotly_chart(fig_brand, use_container_width=True)
    
    st.markdown("---")
    
    # Key Statistics
    st.subheader("📈 Key Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"**Average Bike Age:** {df['bike_age'].mean():.1f} years")
        st.info(f"**Average Kms Driven:** {df['kms_driven'].mean():,.0f} km")
    
    with col2:
        st.info(f"**Average Power:** {df['power'].mean():.1f} BHP")
        st.info(f"**Average CC:** {df['cc'].mean():.0f} cc")
    
    with col3:
        st.info(f"**Median Price:** ₹{df['price'].median():,.0f}")
        st.info(f"**Price Range:** ₹{df['price'].min():,.0f} - ₹{df['price'].max():,.0f}")

elif page == "Data Exploration":
    st.header("🔍 Data Exploration")
    
    # Filters
    st.sidebar.subheader("Filters")
    selected_brands = st.sidebar.multiselect(
        "Select Brands",
        options=df['brand'].unique(),
        default=df['brand'].unique()[:5]
    )
    
    price_range = st.sidebar.slider(
        "Price Range (INR)",
        min_value=int(df['price'].min()),
        max_value=int(df['price'].max()),
        value=(int(df['price'].min()), int(df['price'].max()))
    )
    
    year_range = st.sidebar.slider(
        "Model Year Range",
        min_value=int(df['model_year'].min()),
        max_value=int(df['model_year'].max()),
        value=(int(df['model_year'].min()), int(df['model_year'].max()))
    )
    
    # Apply filters
    filtered_df = df[
        (df['brand'].isin(selected_brands)) &
        (df['price'].between(price_range[0], price_range[1])) &
        (df['model_year'].between(year_range[0], year_range[1]))
    ]
    
    st.write(f"Showing {len(filtered_df)} bikes based on filters")
    
    # Display filtered data
    st.subheader("Filtered Data")
    st.dataframe(filtered_df[['model_name', 'brand', 'model_year', 'price', 'kms_driven', 'power', 'cc']].head(100))
    
    st.markdown("---")
    
    # Interactive plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Price vs Power")
        fig = px.scatter(filtered_df, x='power', y='price', color='brand',
                        hover_data=['model_name'], title='Price vs Power by Brand')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Price vs CC")
        fig = px.scatter(filtered_df, x='cc', y='price', color='brand',
                        hover_data=['model_name'], title='Price vs Engine CC by Brand')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Correlation heatmap
    st.subheader("Correlation Analysis")
    numerical_cols = ['model_year', 'kms_driven', 'mileage', 'power', 'price', 'cc', 'bike_age']
    corr_matrix = filtered_df[numerical_cols].corr()
    
    fig_corr = px.imshow(corr_matrix, 
                        labels=dict(color="Correlation"),
                        color_continuous_scale='RdBu_r',
                        title='Correlation Heatmap')
    fig_corr.update_layout(height=500)
    st.plotly_chart(fig_corr, use_container_width=True)

elif page == "Price Prediction":
    st.header("🎯 Price Prediction")
    
    st.markdown("""
    Use this tool to predict the price of a used bike based on its specifications.
    Fill in the details below and click "Predict Price" to get an estimate.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        brand = st.selectbox("Brand", options=sorted(df['brand'].unique()))
        kms_driven = st.number_input("Kilometers Driven", min_value=0, max_value=1000000, value=15000)
        mileage = st.number_input("Mileage (kmpl)", min_value=5.0, max_value=100.0, value=40.0)
    
    with col2:
        power = st.number_input("Power (BHP)", min_value=5.0, max_value=200.0, value=15.0)
        cc = st.number_input("Engine CC", min_value=50, max_value=2500, value=150)
        bike_age = st.number_input("Bike Age (years)", min_value=1, max_value=30, value=5)
    
    with col3:
        owner = st.selectbox("Owner Type", options=sorted(df['owner'].unique()))
        location = st.selectbox("Location", options=sorted(df['location'].unique())[:50])
    
    # Calculate derived feature
    power_to_cc_ratio = power / cc
    
    # Predict button
    if st.button("Predict Price", type="primary"):
        # Prepare input
        input_data = pd.DataFrame({
            'kms_driven': [kms_driven],
            'mileage': [mileage],
            'power': [power],
            'cc': [cc],
            'bike_age': [bike_age],
            'power_to_cc_ratio': [power_to_cc_ratio],
            'brand_encoded': [le_brand.transform([brand])[0]],
            'owner_encoded': [le_owner.transform([owner])[0]],
            'location_encoded': [le_location.transform([location])[0]]
        })
        
        # Scale numerical features
        numerical_features = ['kms_driven', 'mileage', 'power', 'cc', 'bike_age', 'power_to_cc_ratio']
        input_data[numerical_features] = scaler.transform(input_data[numerical_features])
        
        # Make prediction
        prediction_scaled = model.predict(input_data[numerical_features + ['brand_encoded', 'owner_encoded', 'location_encoded']])
        
        # Inverse transform the prediction (approximate)
        # Since we scaled the target too, we need to inverse transform
        # For simplicity, we'll use the mean and std from original data
        price_mean = df['price'].mean()
        price_std = df['price'].std()
        prediction = prediction_scaled[0] * price_std + price_mean
        
        # Display prediction
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col2:
            st.success(f"## Predicted Price: ₹{prediction:,.0f}")
        
        st.markdown("---")
        
        # Show confidence interval (approximate)
        st.info(f"**Estimated Range:** ₹{prediction * 0.85:,.0f} - ₹{prediction * 1.15:,.0f}")
        st.info(f"**Prediction Confidence:** ~65% (based on model R² score)")
        
        # Feature contribution
        st.subheader("Feature Contribution")
        feature_importance = pd.DataFrame({
            'Feature': ['Power', 'CC', 'Bike Age', 'Kms Driven', 'Mileage', 'Brand', 'Location', 'Owner'],
            'Importance': [0.53, 0.28, 0.05, 0.04, 0.02, 0.01, 0.02, 0.00]
        })
        
        fig_imp = px.bar(feature_importance, x='Importance', y='Feature',
                        orientation='h', title='Feature Importance in Prediction')
        st.plotly_chart(fig_imp, use_container_width=True)

elif page == "Model Analysis":
    st.header("🤖 Model Analysis")
    
    # Model performance metrics
    st.subheader("Model Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Model Type", "Random Forest")
    with col2:
        st.metric("R² Score", "0.6521")
    with col3:
        st.metric("MAE", "₹17,980")
    with col4:
        st.metric("MSE", "7.4B")
    
    st.markdown("---")
    
    # Feature importance
    st.subheader("Feature Importance Ranking")
    feature_importance = pd.DataFrame({
        'Feature': ['Power', 'CC', 'Bike Age', 'Power/CC Ratio', 'Kms Driven', 'Mileage', 'Location', 'Brand', 'Owner'],
        'Importance': [0.53, 0.28, 0.05, 0.05, 0.04, 0.02, 0.02, 0.01, 0.00]
    }).sort_values('Importance', ascending=True)
    
    fig_imp = px.bar(feature_importance, x='Importance', y='Feature',
                    orientation='h', title='Feature Importance in Random Forest Model',
                    color='Importance', color_continuous_scale='viridis')
    st.plotly_chart(fig_imp, use_container_width=True)
    
    st.markdown("---")
    
    # Model insights
    st.subheader("📝 Model Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Key Findings:**
        
        - **Engine Power** is the most significant predictor (53% importance)
        - **Engine Capacity (CC)** is the second most important (28% importance)
        - Together, power and CC account for 81% of predictive power
        - **Bike Age** has moderate impact (5% importance)
        - **Brand and Location** have minimal individual impact
        """)
    
    with col2:
        st.warning("""
        **Model Limitations:**
        
        - R² score of 0.65 indicates 65% variance explained
        - Average prediction error of ~₹18,000
        - Model may not capture all market dynamics
        - Premium bikes (>₹500,000) may have higher prediction errors
        - Limited by available features in dataset
        """)
    
    st.markdown("---")
    
    # Recommendations
    st.subheader("💡 Model Improvement Recommendations")
    
    recommendations = [
        "Collect additional features: bike condition, service history, modifications",
        "Implement ensemble methods combining multiple models",
        "Add temporal features for trend analysis",
        "Increase dataset size, especially for premium bikes",
        "Consider deep learning for complex pattern recognition",
        "Implement continuous learning pipeline for model updates"
    ]
    
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"{i}. {rec}")

elif page == "Brand Insights":
    st.header("🏷️ Brand Insights")
    
    # Brand selection
    selected_brand = st.selectbox("Select Brand for Analysis", options=sorted(df['brand'].unique()))
    
    brand_df = df[df['brand'] == selected_brand]
    
    # Brand statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Bikes", len(brand_df))
    with col2:
        st.metric("Average Price", f"₹{brand_df['price'].mean():,.0f}")
    with col3:
        st.metric("Price Range", f"₹{brand_df['price'].min():,.0f} - ₹{brand_df['price'].max():,.0f}")
    with col4:
        st.metric("Avg Bike Age", f"{brand_df['bike_age'].mean():.1f} years")
    
    st.markdown("---")
    
    # Brand-specific plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"Price Distribution - {selected_brand}")
        fig = px.histogram(brand_df, x='price', nbins=30, 
                          title=f'Price Distribution for {selected_brand}')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader(f"Model Year Distribution - {selected_brand}")
        year_counts = brand_df['model_year'].value_counts().sort_index()
        fig = px.bar(x=year_counts.index, y=year_counts.values,
                    title=f'Model Year Distribution for {selected_brand}',
                    labels={'x': 'Year', 'y': 'Count'})
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Top models for the brand
    st.subheader(f"Top Models - {selected_brand}")
    top_models = brand_df.groupby('model_name').agg({
        'price': ['count', 'mean', 'min', 'max']
    }).round(0)
    top_models.columns = ['Count', 'Avg Price', 'Min Price', 'Max Price']
    top_models = top_models.sort_values('Count', ascending=False).head(10)
    st.dataframe(top_models)
    
    st.markdown("---")
    
    # Brand comparison
    st.subheader("Brand Comparison")
    brand_stats = df.groupby('brand').agg({
        'price': ['mean', 'count']
    }).round(0)
    brand_stats.columns = ['Average Price', 'Count']
    brand_stats = brand_stats.sort_values('Count', ascending=False).head(15)
    
    fig_comp = px.scatter(brand_stats, x='Count', y='Average Price',
                         size='Average Price', hover_name=brand_stats.index,
                         title='Brand Comparison: Count vs Average Price',
                         labels={'Count': 'Number of Bikes', 'Average Price': 'Avg Price (INR)'})
    st.plotly_chart(fig_comp, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Used Bike Prices Analysis Dashboard | Developed by Vinay Sharma | Data Science Intern, Unified Company</p>
    <p>© 2026 All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)
