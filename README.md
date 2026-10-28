# 🌍 Geo-Spatial Air Hazard & Risk Predictor

An end-to-end Machine Learning web application that predicts atmospheric hazard categories, visualizes global pollutant dispersion, and provides automated civic safety advisories.

🔗 **Live Deployment:** [Launch Web App](https://geospatial-hazard-predictor-zapufnd6rlesa6at3orxwa.streamlit.app/)

---

## 📌 Key Features

* **Multi-Pollutant Classification:** Predicts real-time air hazard categories using an ensemble **Random Forest Classifier** trained on global environmental monitoring data.
* **Algorithmic Benchmarking:** Evaluates performance against a **Decision Tree baseline** to justify model architecture choices.
* **Explainable AI (XAI):** Visualizes feature importance weights across key pollutants (PM2.5, NO2, CO, Ozone) to make model inference transparent.
* **Interactive GIS Mapping:** Built with **Folium**, supporting:
  * Dynamic national drill-downs with automated camera re-centering.
  * Dual-mode visualization (Thermal Density Heatmaps vs. Categorical Pin Markers).
* **Automated Civic Advisories:** Generates actionable safety recommendations (PPE/N95 masks, outdoor shift limitations, filtration requirements) based on real-time input levels.

---

## 🛠️ Tech Stack

* **Language:** Python
* **Machine Learning:** Scikit-Learn (Random Forest, Decision Tree, Metrics)
* **Geospatial & Visualization:** Folium, Streamlit-Folium
* **Data Processing:** Pandas, NumPy
* **Application Framework:** Streamlit Community Cloud

---

## 🚀 Local Installation & Run

1. Clone the repository:
   ```bash
   git clone [https://github.com/mansi04-codes/geospatial-hazard-predictor.git](https://github.com/mansi04-codes/geospatial-hazard-predictor.git)
   cd geospatial-hazard-predictor