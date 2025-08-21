# Healthcare Analytics – Patient Encounters & Forecasts

A **dynamic app** for interactive exploration of patient encounters and treatment trends. Users can filter by providers and treatments, visualize demographic insights, and track time series trends and forecasts for key metrics, all in an intuitive dashboard interface.

---

## 🚀 Features

- **Upload or Use Default Data:** Upload your Excel file or use the default dataset provided.
- **Interactive Filtering:** Filter encounters by providers and treatments in the sidebar.
- **Patient Demographics:** Explore gender, age group, and other key demographic insights.
- **Treatment Analysis:** View treatment-wise unique patient counts across age groups.
- **Time Series Forecasting:** Forecast patient encounters using methods like ETS, Holt’s Linear, Holt-Winters, ARIMA, and optionally XGBoost.
- **Visualizations:** Histograms, boxplots, line charts, and metric tiles for easy interpretation.

---

## 🌐 Try it Out

Check out the live app here: [HP Healthcare Analytics](https://hp-healthcare-analytics.streamlit.app/)  

---

## 🗂 Overview of Uploaded DataFrames

### Dim.Treatment
- **Shape:** 10 rows × 2 columns  
- **Columns:**
  1. `Treatment_ID`
  2. `Treatment_Desc`

### Dim.Physician
- **Shape:** 5 rows × 2 columns  
- **Columns:**
  1. `Physician_ID`
  2. `Specialty`

### Dim.Patient
- **Shape:** 172 rows × 3 columns  
- **Columns:**
  1. `Patient_ID`
  2. `Patient_Age`
  3. `Gender_at_Birth`

### EncounterFact
- **Shape:** 5451 rows × 5 columns  
- **Columns:**
  1. `Encounter_Number`
  2. `Date_of_Service`
  3. `Patient_ID`
  4. `Treatment_ID`
  5. `Provider_ID`

---
## 🖼 Screenshots

### 1️⃣ Patient Encounters
![Encounters Breakdown](screenshots/1-encounters.png)

### 2️⃣ Time Series Forecasting
![Time Series Forecasting](screenshots/2-forecast.png)

### 3️⃣ Demographics & Key Insights
![Demographics Insights](screenshots/3-demographics.png)

---

## ⚙️ Dependencies

- Python 3.9+
- [Streamlit](https://streamlit.io/)
- Pandas, NumPy
- Plotly
- Statsmodels (for forecasting)
- Scikit-learn
