import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="Geo-Spatial Hazard Predictor",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 Geo-Spatial Air Hazard & Risk Predictor")
st.markdown("Machine Learning model predicting environmental hazard categories with geospatial mapping.")

# ----------------- LOAD DATA & TRAIN MODELS -----------------
@st.cache_data
def load_data():
    df = pd.read_csv("hazard_data.csv")
    df = df.dropna(subset=[
        'lat', 'lng', 'AQI Category', 
        'CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value'
    ])
    return df

@st.cache_resource
def train_models(df):
    feature_cols = ['CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value']
    X = df[feature_cols]
    y = df['AQI Category']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Primary: Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf_model.predict(X_test))
    
    # Baseline: Decision Tree Classifier
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    dt_acc = accuracy_score(y_test, dt_model.predict(X_test))
    
    # Feature Importance DataFrame
    importance_df = pd.DataFrame({
        'Pollutant Feature': feature_cols,
        'Importance Weight': rf_model.feature_importances_
    }).sort_values(by='Importance Weight', ascending=False)
    
    return rf_model, rf_acc, dt_acc, importance_df

df = load_data()
model, rf_accuracy, dt_accuracy, feature_importance = train_models(df)

# ----------------- TOP KPI METRICS -----------------
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Monitored Stations / Cities", f"{len(df):,}")
kpi2.metric("Random Forest Accuracy", f"{rf_accuracy * 100:.2f}%")
kpi3.metric("Decision Tree Baseline", f"{dt_accuracy * 100:.2f}%")
kpi4.metric("Leading Risk Driver", feature_importance.iloc[0]['Pollutant Feature'].replace(' AQI Value', ''))

st.markdown("---")

# ----------------- SIDEBAR: RISK PREDICTOR -----------------
st.sidebar.header("🧪 Predict Hazard Level")
st.sidebar.markdown(f"**Random Forest Accuracy:** `{rf_accuracy * 100:.2f}%`")

co_val = st.sidebar.slider("CO AQI Value", min_value=0, max_value=int(df['CO AQI Value'].max()), value=10)
ozone_val = st.sidebar.slider("Ozone AQI Value", min_value=0, max_value=int(df['Ozone AQI Value'].max()), value=35)
no2_val = st.sidebar.slider("NO2 AQI Value", min_value=0, max_value=int(df['NO2 AQI Value'].max()), value=15)
pm25_val = st.sidebar.slider("PM2.5 AQI Value", min_value=0, max_value=int(df['PM2.5 AQI Value'].max()), value=50)

input_sample = pd.DataFrame(
    [[co_val, ozone_val, no2_val, pm25_val]], 
    columns=['CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value']
)

prediction = model.predict(input_sample)[0]

# Display Prediction
st.sidebar.markdown("---")
st.sidebar.subheader("Predicted Hazard Category:")
if "Good" in prediction:
    st.sidebar.success(f"✅ {prediction}")
elif "Moderate" in prediction:
    st.sidebar.warning(f"⚠️ {prediction}")
else:
    st.sidebar.error(f"🚨 {prediction}")

# Health Advisory Card
st.sidebar.markdown("### 🛡️ Recommended Safety Actions")
if "Good" in prediction:
    st.sidebar.info(
        "**Air Quality Safe**\n\n"
        "- Air quality is satisfactory; poses little or no risk.\n"
        "- Normal outdoor exertion and ventilation permitted.\n"
        "- No personal protective equipment (PPE) required."
    )
elif "Moderate" in prediction:
    st.sidebar.warning(
        "**Mild Atmospheric Concern**\n\n"
        "- Unusually sensitive individuals may experience respiratory irritation.\n"
        "- Consider reducing prolonged outdoor heavy exertion.\n"
        "- Keep basic ventilation filtration active."
    )
elif "Sensitive" in prediction:
    st.sidebar.warning(
        "**Unhealthy for Sensitive Groups**\n\n"
        "- Children, elderly, and individuals with respiratory conditions at elevated risk.\n"
        "- Limit strenuous outdoor activities.\n"
        "- Wear basic particulate masks if commuting along major arterial roads."
    )
elif "Very Unhealthy" in prediction:
    st.sidebar.error(
        "**Severe Hazard Alert**\n\n"
        "- Significant health risk for entire exposed population.\n"
        "- Mandate N95/FFP2 respirators during outdoor occupational tasks.\n"
        "- Keep windows sealed; engage indoor HEPA air filtration."
    )
elif "Hazardous" in prediction:
    st.sidebar.error(
        "**Emergency Health Condition**\n\n"
        "- Toxic exposure threshold exceeded. Serious risk of adverse health effects.\n"
        "- Evacuate or strictly avoid outdoor activities.\n"
        "- Industrial workers require multi-gas particulate containment gear."
    )
else:
    st.sidebar.error(
        "**Unhealthy Environment**\n\n"
        "- Everyone may begin to experience noticeable respiratory symptoms.\n"
        "- Restrict outdoor shifts and vocational field operations.\n"
        "- Wear certified particulate-filtering masks."
    )

# ----------------- MAP SECTION -----------------
st.subheader("🗺️ Geospatial Risk & Regional Exploration")

col1, col2, col3 = st.columns([1.5, 1.5, 1.5])

with col1:
    country_list = ["Global (All Countries)"] + sorted(df['Country'].dropna().unique().tolist())
    selected_country = st.selectbox("Select Country / Region:", country_list)

with col2:
    categories = ["All Categories"] + sorted(df['AQI Category'].dropna().unique().tolist())
    selected_category = st.selectbox("Filter Hazard Level:", categories)

with col3:
    map_view_type = st.radio(
        "Visualization Style:",
        ["Individual Pin Markers", "Thermal Density Heatmap"],
        horizontal=True
    )

# Filter Data
filtered_df = df.copy()

if selected_country != "Global (All Countries)":
    filtered_df = filtered_df[filtered_df['Country'] == selected_country]

if selected_category != "All Categories":
    filtered_df = filtered_df[filtered_df['AQI Category'] == selected_category]

# Dynamic Camera Positioning & Sampling
if selected_country == "Global (All Countries)":
    sample_size = min(400, len(filtered_df))
    sample_df = filtered_df.sample(sample_size, random_state=42) if sample_size > 0 else filtered_df
    avg_lat = sample_df['lat'].mean() if not sample_df.empty else 20.0
    avg_lng = sample_df['lng'].mean() if not sample_df.empty else 0.0
    zoom_level = 3
else:
    sample_df = filtered_df
    avg_lat = sample_df['lat'].mean() if not sample_df.empty else 20.0
    avg_lng = sample_df['lng'].mean() if not sample_df.empty else 0.0
    zoom_level = 5

st.caption(f"Displaying **{len(sample_df)}** data points for `{selected_country}`.")

def get_color(cat):
    if "Good" in str(cat):
        return "green"
    elif "Moderate" in str(cat):
        return "orange"
    elif "Unhealthy" in str(cat):
        return "red"
    elif "Hazardous" in str(cat):
        return "darkred"
    return "blue"

m = folium.Map(location=[avg_lat, avg_lng], zoom_start=zoom_level, tiles="OpenStreetMap")

if map_view_type == "Thermal Density Heatmap":
    if not sample_df.empty:
        heat_data = sample_df[['lat', 'lng', 'AQI Value']].values.tolist()
        HeatMap(heat_data, radius=16, blur=18, min_opacity=0.35, max_zoom=10).add_to(m)
else:
    for _, row in sample_df.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lng']],
            radius=5,
            popup=f"<b>{row['City']}, {row['Country']}</b><br>AQI: {row['AQI Value']}<br>Status: {row['AQI Category']}",
            color=get_color(row['AQI Category']),
            fill=True,
            fill_opacity=0.7
        ).add_to(m)

st_folium(m, width="100%", height=530, key=f"map_{selected_country}_{map_view_type}_{selected_category}")

# ----------------- PHASE 3: ML EXPLAINABILITY & BENCHMARKING -----------------
st.markdown("---")
st.subheader("📊 Model Diagnostics & Explainability")

m_col1, m_col2 = st.columns([1.5, 1])

with m_col1:
    st.markdown("##### Relative Feature Importance (Random Forest)")
    st.caption("Quantifies the contribution of each chemical metric in determining atmospheric hazard category.")
    chart_data = feature_importance.set_index('Pollutant Feature')
    st.bar_chart(chart_data)

with m_col2:
    st.markdown("##### Algorithmic Benchmark Breakdown")
    benchmark_df = pd.DataFrame({
        "Model Architecture": ["Random Forest Classifier (Ensemble)", "Decision Tree Classifier (Baseline)"],
        "Test Split Accuracy": [f"{rf_accuracy * 100:.2f}%", f"{dt_accuracy * 100:.2f}%"],
        "Primary Hyperparameters": ["n_estimators=100, random_state=42", "criterion='gini', random_state=42"]
    })
    st.table(benchmark_df)

# ----------------- DATA PREVIEW -----------------
with st.expander("🔍 View Raw Filtered Records"):
    st.dataframe(sample_df[[
        'Country', 'City', 'lat', 'lng', 'AQI Category', 
        'PM2.5 AQI Value', 'CO AQI Value', 'NO2 AQI Value', 'Ozone AQI Value'
    ]].head(25))