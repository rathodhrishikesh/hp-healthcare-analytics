import warnings
warnings.filterwarnings("ignore")

import io
from pathlib import Path
from urllib.parse import urlparse
from typing import Tuple, Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import requests

# Forecasting libs
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing, Holt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from io import BytesIO

# XGBoost (optional)
try:
    from xgboost import XGBRegressor
    XGB_OK = True
except Exception:
    XGB_OK = False
    
# ------------------------------- Session state for default data -------------------------------
if "use_default" not in st.session_state:
    st.session_state.use_default = False

# ------------------------------- Page config / Theme -------------------------------
st.set_page_config(
    page_title="Healthcare Analytics – Encounters & Forecasts",
    page_icon="🏥",
    layout="wide"
)

# ------------------------------- Sidebar: Uploader / Defaults -------------------------------
st.sidebar.title("📦 Data Version 12:55 AM")
st.sidebar.caption("Upload your Excel or use the default placed in `public/`")

uploaded = st.sidebar.file_uploader("Upload Excel file", type=["xlsx"])
    
col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("📂 Use Default Data"):
        st.session_state.use_default = True
        st.rerun()

with col2:
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

PUBLIC_DIR = Path("public")
DEFAULT_PATH = PUBLIC_DIR / "Hospital-Encounters-Test-Data.xlsx"

# ------------------------------- Data loading helpers -------------------------------
@st.cache_data(show_spinner=True)
def parse_dataframes_from_path(path: str):
    excel_file = pd.ExcelFile(path)
    dim_treatment_df = excel_file.parse(sheet_name='Dimensions', skiprows=8, nrows=10, usecols='B:C')
    dim_physician_df = excel_file.parse(sheet_name='Dimensions', skiprows=8, nrows=5, usecols='E:F')
    dim_patient_df   = excel_file.parse(sheet_name='Dimensions', skiprows=8, usecols='H:J')
    encounter_fact_df= excel_file.parse(sheet_name='Fact Table', skiprows=2, usecols='B:F')
    return dim_treatment_df, dim_physician_df, dim_patient_df, encounter_fact_df

@st.cache_data(show_spinner=True)
def parse_dataframes_from_bytes(file_bytes: bytes):
    excel_file = pd.ExcelFile(io.BytesIO(file_bytes))
    dim_treatment_df = excel_file.parse(sheet_name='Dimensions', skiprows=8, nrows=10, usecols='B:C')
    dim_physician_df = excel_file.parse(sheet_name='Dimensions', skiprows=8, nrows=5, usecols='E:F')
    dim_patient_df   = excel_file.parse(sheet_name='Dimensions', skiprows=8, usecols='H:J')
    encounter_fact_df= excel_file.parse(sheet_name='Fact Table', skiprows=2, usecols='B:F')
    return dim_treatment_df, dim_physician_df, dim_patient_df, encounter_fact_df

@st.cache_data(show_spinner=False)
def get_filtered_encounters(encounter_fact_df, dim_patient_df, dim_treatment_df, sel_providers, sel_treatments, provider_col, treat_col):
    df = (
        encounter_fact_df
        .merge(dim_patient_df, on="Patient_ID", how="left")
        .merge(dim_treatment_df, on="Treatment_ID", how="left")
    )
    if provider_col and sel_providers:
        df = df[df[provider_col].isin(sel_providers)]
    if treat_col and sel_treatments:
        df = df[df[treat_col].isin(sel_treatments)]
    return df

def preprocess(
    dim_treatment_df: pd.DataFrame,
    dim_physician_df: pd.DataFrame,
    dim_patient_df: pd.DataFrame,
    encounter_fact_df: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    # Ensure expected column names exist (light validation)
    # Convert dates
    if 'Date_of_Service' in encounter_fact_df.columns:
        encounter_fact_df['Date_of_Service'] = pd.to_datetime(encounter_fact_df['Date_of_Service'])
    # Merge helpers
    # Provider specialty join expects Physician_ID in physician dim; align names if necessary
    # (Assuming your columns exactly match the baseline workbook)
    return {
        "dim_treatment_df": dim_treatment_df,
        "dim_physician_df": dim_physician_df,
        "dim_patient_df": dim_patient_df,
        "encounter_fact_df": encounter_fact_df
    }

def display_patient_encounter_metrics(encounters_with_details):
    """
    Displays patient encounter metrics and key insights in Streamlit.
    
    Args:
        encounters_with_details (pd.DataFrame): Merged and optionally filtered DataFrame
            containing encounters with patient and treatment details.
    """
    # Apply provider filter if selected
    if provider_col and sel_providers:
        encounters_with_details = encounters_with_details[encounters_with_details[provider_col].isin(sel_providers)]

    # Apply treatment filter if selected
    if treat_col and sel_treatments:
        encounters_with_details = encounters_with_details[encounters_with_details[treat_col].isin(sel_treatments)]

    # Gender-specific subsets
    female_patients = encounters_with_details[encounters_with_details['Gender_at_Birth'] == 'F']
    male_patients = encounters_with_details[encounters_with_details['Gender_at_Birth'] == 'M']
    other_gender_encounters = encounters_with_details[
        (~encounters_with_details['Gender_at_Birth'].isin(['F', 'M']))
    ]
    
    total_encounters = len(encounters_with_details)
    
    # Women 23–45
    women_23_45 = female_patients[(female_patients['Patient_Age'] >= 23) & (female_patients['Patient_Age'] <= 45)].shape[0]
    total_women_encounters = len(female_patients)
    percent_women_23_45 = (women_23_45 / total_women_encounters * 100) if total_women_encounters > 0 else 0

    # Men 23–65
    men_23_65 = male_patients[(male_patients['Patient_Age'] >= 23) & (male_patients['Patient_Age'] <= 65)].shape[0]
    total_men_encounters = len(male_patients)
    percent_men_23_65 = (men_23_65 / total_men_encounters * 100) if total_men_encounters > 0 else 0

    # Very young + elderly
    very_young_count = encounters_with_details[(encounters_with_details['Patient_Age'] <= 22)].shape[0]
    elderly_count = encounters_with_details[(encounters_with_details['Patient_Age'] >= 66)].shape[0]
    very_young_and_elderly_count = very_young_count + elderly_count
    percent_very_young_and_elderly = (very_young_and_elderly_count / total_encounters * 100) if total_encounters > 0 else 0

    # Male 23–65 IV Vitamin B12 (Treatment_ID = 99131)
    male_23_65_df = male_patients[(male_patients['Patient_Age'] >= 23) & (male_patients['Patient_Age'] <= 65)]
    b12_encounters_male_23_65 = male_23_65_df[male_23_65_df['Treatment_ID'] == 99131].shape[0]
    total_encounters_male_23_65 = len(male_23_65_df)
    percent_b12_male_23_65 = (b12_encounters_male_23_65 / total_encounters_male_23_65 * 100) if total_encounters_male_23_65 > 0 else 0

    # Top Treatment
    if not encounters_with_details.empty:
        top_treatment = encounters_with_details['Treatment_Desc'].value_counts().idxmax()
        top_treatment_count = encounters_with_details['Treatment_Desc'].value_counts().max()
        percent_top_treatment = (top_treatment_count / total_encounters * 100) if total_encounters > 0 else 0
    else:
        top_treatment = "N/A"
        top_treatment_count = 0
        percent_top_treatment = 0

    # -----------------------
    # Display as 4 tiles
    # -----------------------
    st.markdown("## Patient Encounters Breakdown")
    
    metrics_tiles = {
        "Total Encounters": (total_encounters, "#006600"),
        "Female": (total_women_encounters, "#cc0066"),
        "Male": (total_men_encounters, "#004c99"),
        "Other Genders": (len(other_gender_encounters), "#ff9900"),
    }

    cols = st.columns(len(metrics_tiles))

    for col, (label, (value, color)) in zip(cols, metrics_tiles.items()):
        col.markdown(f"""
            <div style="
                background-color:{color};
                padding:10px;
                border-radius:10px;
                text-align:center;
                color:white;
                box-shadow: 0px 3px 6px rgba(0,0,0,0.15);
            ">
                <div style="font-size:14px; font-weight:600;">{label}</div>
                <div style="font-size:20px; font-weight:700;">{value}</div>
            </div>
        """, unsafe_allow_html=True)

    # -----------------------
    # Display key insights metrics
    # -----------------------
    st.markdown("## Key Insights")
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Women 23–45", f"{percent_women_23_45:.1f}%", f"{women_23_45}/{total_women_encounters}", delta_color="off")
    col2.metric("Men 23–65", f"{percent_men_23_65:.1f}%", f"{men_23_65}/{total_men_encounters}", delta_color="off")
    col3.metric("Young + Elderly", f"{percent_very_young_and_elderly:.1f}%", f"{very_young_and_elderly_count}/{total_encounters}", delta_color="off")
    col4.metric("B12 Male 23–65", f"{percent_b12_male_23_65:.1f}%", f"{b12_encounters_male_23_65}/{total_encounters_male_23_65}", delta_color="off")
    col5.metric(f"Top Treatment: {top_treatment}", f"{percent_top_treatment:.1f}%", f"{top_treatment_count}/{total_encounters}", delta_color="off")

# ------------------------------- Obtain data -------------------------------
excel: Optional[pd.ExcelFile] = None
data_source_label = None

if uploaded is not None:
    file_bytes = uploaded.read()
    dim_treatment_df, dim_physician_df, dim_patient_df, encounter_fact_df = parse_dataframes_from_bytes(file_bytes)
    data_source_label = f"Uploaded file: **{uploaded.name}**"
elif st.session_state.use_default:
    if not DEFAULT_PATH.exists():
        st.error(f"❌ Default file not found at {DEFAULT_PATH}")
        st.stop()
    dim_treatment_df, dim_physician_df, dim_patient_df, encounter_fact_df = parse_dataframes_from_path(str(DEFAULT_PATH))
    data_source_label = f"Default file: **{DEFAULT_PATH.name}**"
else:
    st.info("👋 Upload an Excel file on the left, or click the button to use the default file.")
    st.stop()

data = preprocess(dim_treatment_df, dim_physician_df, dim_patient_df, encounter_fact_df)

# ------------------------------- Derived data / shared widgets -------------------------------
st.sidebar.markdown("---")
st.sidebar.success(data_source_label)

# Provider filter (if present)
provider_col = 'Provider_ID' if 'Provider_ID' in data["encounter_fact_df"].columns else None
providers = sorted(data["encounter_fact_df"][provider_col].dropna().unique()) if provider_col else []
sel_providers = st.sidebar.multiselect("Filter providers", providers, default=providers[:5] if len(providers) > 0 else [])

# Treatment filter (if present)
treat_col = 'Treatment_ID' if 'Treatment_ID' in data["encounter_fact_df"].columns else None
treatments = sorted(data["encounter_fact_df"][treat_col].dropna().unique()) if treat_col else []
sel_treatments = st.sidebar.multiselect("Filter treatments", treatments, default=treatments[:10] if len(treatments) > 0 else [])

encounters_joined = get_filtered_encounters(
    data["encounter_fact_df"], data["dim_patient_df"], data["dim_treatment_df"],
    sel_providers, sel_treatments, provider_col, treat_col
)

# ------------------------------- Tabs / Slides -------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣ Hospital Encounters Data Analysis",
    #"2️⃣ Time Series Analysis & Forecasting",
    "3️⃣ Demographics",
    "4️⃣ Treatment-wise Unique Patient Counts",
    "5️⃣ Uploaded Data Overview"
])

# ===================================== Slide 1 =====================================
with tab1:
    st.subheader("Q1. Distribution of Encounters per Day & Centricity Measure")
    df_f = encounters_joined.copy()
    if provider_col and len(sel_providers) > 0:
        df_f = df_f[df_f[provider_col].isin(sel_providers)]

    # Group by day
    if "Date_of_Service" in df_f.columns and "Encounter_Number" in df_f.columns:
        enc_day = df_f.groupby("Date_of_Service")["Encounter_Number"].count().rename("Encounters")

        mean_val = float(enc_day.mean()) if len(enc_day) else np.nan
        median_val = float(enc_day.median()) if len(enc_day) else np.nan
        skew_val = float(enc_day.skew()) if len(enc_day) else np.nan
        q1 = enc_day.quantile(0.25)
        q3 = enc_day.quantile(0.75)

        # Histogram & Boxplot
        c1, c2 = st.columns(2)
        with c1:
            # fig_hist = px.histogram(enc_day, nbins=20, title="Distribution: Encounters per Day")
            fig_hist = px.histogram(x=enc_day.values, nbins=20, title="Distribution: Encounters per Day")
            # fig_hist.update_layout(bargap=0.05)
            fig_hist.update_layout(bargap=0.05, xaxis_title="Number of Encounters", yaxis_title="Frequency (days)")
            
            fig_hist.add_vline(x=mean_val, line_dash="dash", line_color="green", annotation_text=f"Mean: {mean_val:.2f}", annotation_position="top right")
            fig_hist.add_vline(x=median_val, line_dash="dot", line_color="red", annotation_text=f"Median: {median_val:.2f}", annotation_position="bottom right")
            fig_hist.add_vrect(x0=q1, x1=q3, fillcolor="yellow", opacity=0.2, layer="above")
            st.plotly_chart(fig_hist, use_container_width=True)

        with c2:
            fig_box = px.box(enc_day, points="suspectedoutliers", title="Boxplot: Encounters per Day")
            st.plotly_chart(fig_box, use_container_width=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Mean", f"{mean_val:.2f}" if not np.isnan(mean_val) else "—")
        m2.metric("Median", f"{median_val:.2f}" if not np.isnan(median_val) else "—")
        m3.metric("Skewness", f"{skew_val:.2f}" if not np.isnan(skew_val) else "—")

        st.caption(
            "Rule of thumb: if |skewness| > 0.5, the distribution is meaningfully skewed → prefer **median** as the centricity measure; "
            "otherwise **mean** is acceptable."
        )

    st.markdown("---")
    st.subheader("Q1a. Monthly Encounter Trend")
    if "Date_of_Service" in df_f.columns:
        enc_month = df_f.groupby(df_f["Date_of_Service"].dt.to_period("M"))["Encounter_Number"].count()
        enc_month.index = enc_month.index.to_timestamp()
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(x=enc_month.index, y=enc_month.values, mode="lines+markers", name="Encounters"))
        fig_trend.update_layout(title="Monthly Encounters trend", xaxis_title="Month", yaxis_title="Encounters")
        st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("---")
    st.subheader("Q1b. Is the trend the same for all providers?")
    if provider_col and "Date_of_Service" in data["encounter_fact_df"].columns:
        enc_prov_month = (
            data["encounter_fact_df"]
            .groupby([data["encounter_fact_df"]["Date_of_Service"].dt.to_period("M"), provider_col])["Encounter_Number"]
            .count().unstack(provider_col).fillna(0)
        )
        enc_prov_month.index = enc_prov_month.index.to_timestamp()
        fig_multi = go.Figure()
        for col in enc_prov_month.columns:
            fig_multi.add_trace(go.Scatter(x=enc_prov_month.index, y=enc_prov_month[col], mode="lines+markers", name=f"Provider {col}"))
        fig_multi.update_layout(title="Monthly Encounters by Provider", xaxis_title="Month", yaxis_title="Encounters")
        st.plotly_chart(fig_multi, use_container_width=True)

# ===================================== Slide 3 =====================================
with tab2:

    display_patient_encounter_metrics(encounters_joined)
    
    st.subheader("Q2. Are treatment types distributed differently across provider specialties?")
    # Join encounters with dimensions for Specialty and Treatment_Desc
    # Align physician dimension names: expect columns Physician_ID and Specialty
    if "Physician_ID" in data["dim_physician_df"].columns:
        merged = (
            data["encounter_fact_df"]
            .merge(data["dim_treatment_df"], on="Treatment_ID", how="left")
            .merge(data["dim_physician_df"], left_on="Provider_ID", right_on="Physician_ID", how="left")
        )
        # Provider/treatment filters
        if provider_col and len(sel_providers) > 0:
            merged = merged[merged["Provider_ID"].isin(sel_providers)]
        if treat_col and len(sel_treatments) > 0:
            merged = merged[merged["Treatment_ID"].isin(sel_treatments)]

        grp = (
            merged.groupby(["Specialty", "Treatment_Desc"])
            .size().reset_index(name="Encounter_Count")
            .sort_values(["Specialty", "Encounter_Count"], ascending=[True, False])
        )

        if len(grp):
            fig_bar = px.bar(grp, x="Specialty", y="Encounter_Count", color="Treatment_Desc",
                             barmode="group", title="Treatment counts by Specialty",
                             hover_data=["Treatment_Desc", "Encounter_Count"])
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No rows after filters.")
    else:
        st.warning("Physician dimension missing expected columns (Physician_ID / Specialty).")

    st.markdown("---")
    st.subheader("Q2a. Patterns by Age Group or Gender at Birth")
    # Build age groups
    if "Patient_Age" in encounters_joined.columns:
        bins = [0, 5, 13, 22, 45, 65, 120]
        labels = ['0–5', '6–13', '14–22', '23–45', '46–65', '66–120']
        tmp = encounters_joined.copy()
        tmp["Age_Group"] = pd.cut(tmp["Patient_Age"], bins=bins, labels=labels)
        # optional provider/treatment filters
        if provider_col and len(sel_providers) > 0:
            tmp = tmp[tmp[provider_col].isin(sel_providers)]
        if treat_col and len(sel_treatments) > 0:
            tmp = tmp[tmp[treat_col].isin(sel_treatments)]

        # Heatmap: Age group vs Treatment (counts), faceted by Gender
        crosstab = (
            tmp.groupby(["Gender_at_Birth", "Age_Group", "Treatment_Desc"])
            .size().reset_index(name="Encounter_Count")
        )
        st.caption("Heatmap per gender: Age group × Treatment (Encounter counts)")
        genders = list(crosstab["Gender_at_Birth"].dropna().unique())
        for g in genders:
            sub = crosstab[crosstab["Gender_at_Birth"] == g]
            if len(sub) == 0:
                continue

            heat = sub.pivot_table(
                index="Age_Group",
                columns="Treatment_Desc",
                values="Encounter_Count",
                fill_value=0
            )

            # Choose color scale based on gender
            if g == "F":
                color_scale = ["#ffe6f0", "#ffb3cc", "#ff6699", "#cc0066"]  # light pink → deep pink
            elif g == "M":
                color_scale = ["#e6f0ff", "#99ccff", "#3399ff", "#004c99"]  # light blue → deep blue
            else:
                color_scale = "Blues"  # fallback, in case other genders appear

            fig_heat = px.imshow(
                heat,
                aspect="auto",
                title=f"Gender at Birth: {g}",
                labels=dict(x="Treatment", y="Age Group", color="Encounters"),
                color_continuous_scale=color_scale
            )

            st.plotly_chart(fig_heat, use_container_width=True)

    else:
        st.warning("Patient_Age not available in joined encounters table.")

# ===================================== Slide 4 =====================================
with tab3:
    st.subheader("Treatment-wise Unique Patient Counts by Age Group")
    # Build matrix: rows Treatment_ID, columns Age Groups, values distinct patients
    enc = encounters_joined.copy()
    bins = [0, 5, 13, 22, 45, 65, 120]
    labels = ['0–5', '6–13', '14–22', '23–45', '46–65', '66–120']
    enc["Age_Group"] = pd.cut(enc["Patient_Age"], bins=bins, labels=labels)
    # optional filters
    if provider_col and len(sel_providers) > 0:
        enc = enc[enc[provider_col].isin(sel_providers)]
    if treat_col and len(sel_treatments) > 0:
        enc = enc[enc[treat_col].isin(sel_treatments)]

    # Distinct patients per Treatment_ID × Age_Group
    def nunique_patients(df):
        return df["Patient_ID"].nunique()

    mat = (
        enc.groupby(["Treatment_ID", "Age_Group"])
        .apply(nunique_patients)
        .reset_index(name="Distinct_Patients")
        .pivot(index="Treatment_ID", columns="Age_Group", values="Distinct_Patients")
        .fillna(0).astype(int)
        .sort_index()
    )

    # Append Treatment_Desc for readability
    if "Treatment_Desc" in enc.columns:
        desc_map = enc.drop_duplicates(subset=["Treatment_ID"]).set_index("Treatment_ID")["Treatment_Desc"].to_dict()
        mat.insert(0, "Treatment_Desc", [desc_map.get(tid, "") for tid in mat.index])

    st.dataframe(mat, use_container_width=True)

# ------------------------------- Footer / Notes -------------------------------
st.caption("Tip: use the left sidebar to filter providers/treatments. Hover any chart for tooltips; legends are clickable to isolate series.")
if not XGB_OK:
    st.caption("⚠️ XGBoost not installed – the XGBoost option is hidden. Install `xgboost` to enable it.")

# ===================================== Slide 5 =====================================
with tab4:
    st.subheader("Overview of Uploaded DataFrames")
    
    # st.write("HR Testing Area")
    
    dfs = {
        "Dim.Treatment": dim_treatment_df,
        "Dim.Physician": dim_physician_df,
        "Dim.Patient": dim_patient_df,
        "EncounterFact": encounter_fact_df
    }
    
    for name, df in dfs.items():
        st.markdown(f"### {name}")
        st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        st.write("**Columns:**", list(df.columns))
        st.dataframe(df.head(), use_container_width=True)
        st.markdown("---")