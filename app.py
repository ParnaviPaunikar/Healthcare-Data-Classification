import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Healthcare Risk Assessment",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# LOAD MODEL, SCALER AND DATA
# =========================================================

model = joblib.load("model/disease_risk_model.pkl")
scaler = joblib.load("model/scaler.pkl")

data = pd.read_csv("data/diabetes.csv")

# =========================================================
# DATA PREPROCESSING
# =========================================================

columns_with_invalid_zero = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

clean_data = data.copy()

for column in columns_with_invalid_zero:
    clean_data[column] = pd.to_numeric(
        clean_data[column], errors="coerce"
    )
    clean_data[column] = clean_data[column].replace(0, pd.NA)
    clean_data[column] = clean_data[column].fillna(
        clean_data[column].median()
    )
    clean_data[column] = clean_data[column].astype(float)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f4f8f7;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .main-header {
        background: linear-gradient(135deg, #0b5d4b, #14866d);
        padding: 30px;
        border-radius: 18px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 8px 25px rgba(11, 93, 75, 0.18);
    }

    .main-header h1 {
        font-size: 38px;
        margin-bottom: 5px;
    }

    .main-header p {
        font-size: 17px;
        margin: 0;
        opacity: 0.92;
    }

    .section-title {
        color: #0b5d4b;
        font-size: 25px;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    .risk-high {
        background: linear-gradient(135deg, #fff0f0, #ffe1e1);
        border-left: 6px solid #d9534f;
        padding: 22px;
        border-radius: 15px;
        margin-top: 15px;
    }

    .risk-low {
        background: linear-gradient(135deg, #effaf5, #dcf5ea);
        border-left: 6px solid #198754;
        padding: 22px;
        border-radius: 15px;
        margin-top: 15px;
    }

    .footer-box {
        background-color: #eaf4f1;
        padding: 18px;
        border-radius: 12px;
        margin-top: 30px;
        font-size: 13px;
        color: #355c53;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 🩺 Healthcare AI")

    st.markdown(
        """
        **Disease Risk Classification**

        Machine learning based healthcare
        analytics system.
        """
    )

    st.divider()

    st.markdown("### 📌 Project Information")

    st.write("**Model:** Logistic Regression")
    st.write("**Dataset:** Diabetes Healthcare Dataset")
    st.write("**Features:** 8")
    st.write("**Target:** Outcome")

    st.divider()

    st.markdown("### 🛡️ Academic Use")

    st.info(
        "This application is developed for academic "
        "and educational purposes."
    )

# =========================================================
# MAIN HEADER
# =========================================================

st.markdown(
    """
    <div class="main-header">
        <h1>🩺 Healthcare Risk Assessment</h1>
        <p>
        Machine Learning Based Diabetes Risk Classification
        & Healthcare Analytics Dashboard
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# DASHBOARD METRICS
# =========================================================

total_records = len(data)
lower_risk = int((data["Outcome"] == 0).sum())
higher_risk = int((data["Outcome"] == 1).sum())

higher_risk_percentage = (
    higher_risk / total_records
) * 100

metric1, metric2, metric3, metric4 = st.columns(4)

with metric1:
    st.metric(
        "👥 Patient Records",
        f"{total_records:,}"
    )

with metric2:
    st.metric(
        "🟢 Lower Risk",
        f"{lower_risk:,}"
    )

with metric3:
    st.metric(
        "🟠 Higher Risk",
        f"{higher_risk:,}"
    )

with metric4:
    st.metric(
        "📊 Higher Risk %",
        f"{higher_risk_percentage:.1f}%"
    )

st.divider()

# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🏠 Dashboard",
        "🔍 Risk Assessment",
        "📈 Health Analytics",
        "📂 Dataset Management",
        "📋 Patient Data"
    ]
)

# =========================================================
# TAB 1 - DASHBOARD
# =========================================================

with tab1:

    st.markdown(
        '<div class="section-title">'
        '📊 Healthcare Dataset Overview'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    # -----------------------------------------------------
    # RISK DISTRIBUTION
    # -----------------------------------------------------

    with col1:

        risk_data = pd.DataFrame({
            "Risk Classification": [
                "Lower Risk",
                "Higher Risk"
            ],
            "Patients": [
                lower_risk,
                higher_risk
            ]
        })

        fig_risk = px.pie(
            risk_data,
            names="Risk Classification",
            values="Patients",
            hole=0.55,
            title="Patient Risk Classification Distribution"
        )

        fig_risk.update_layout(
            height=420
        )

        st.plotly_chart(
            fig_risk,
            width="stretch"
        )

    # -----------------------------------------------------
    # AGE GROUP ANALYSIS
    # -----------------------------------------------------

    with col2:

        age_bins = [0, 20, 30, 40, 50, 60, 100]

        age_labels = [
            "≤20",
            "21–30",
            "31–40",
            "41–50",
            "51–60",
            "60+"
        ]

        age_data = data.copy()

        age_data["Age Group"] = pd.cut(
            age_data["Age"],
            bins=age_bins,
            labels=age_labels,
            include_lowest=True
        )

        age_risk = (
            age_data.groupby(
                ["Age Group", "Outcome"],
                observed=False
            )
            .size()
            .reset_index(name="Patients")
        )

        age_risk["Risk"] = age_risk["Outcome"].map({
            0: "Lower Risk",
            1: "Higher Risk"
        })

        fig_age = px.bar(
            age_risk,
            x="Age Group",
            y="Patients",
            color="Risk",
            barmode="group",
            title="Risk Distribution by Age Group"
        )

        fig_age.update_layout(
            height=420,
            xaxis_title="Age Group",
            yaxis_title="Number of Patients"
        )

        st.plotly_chart(
            fig_age,
            width="stretch"
        )

    # -----------------------------------------------------
    # DATASET SUMMARY
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        '📌 Dataset Summary'
        '</div>',
        unsafe_allow_html=True
    )

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    with summary_col1:
        st.info(
            f"""
            **Average Age**

            {data["Age"].mean():.1f} years
            """
        )

    with summary_col2:
        st.info(
            f"""
            **Average Glucose**

            {data["Glucose"].mean():.1f}
            """
        )

    with summary_col3:
        st.info(
            f"""
            **Average BMI**

            {data["BMI"].mean():.1f}
            """
        )

# =========================================================
# TAB 2 - RISK ASSESSMENT
# =========================================================

with tab2:

    st.markdown(
        '<div class="section-title">'
        '🔍 Patient Risk Assessment'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Enter patient health parameters to generate a "
        "model-based diabetes risk classification."
    )

    input_col1, input_col2 = st.columns(2)

    with input_col1:

        pregnancies = st.number_input(
            "Pregnancies",
            min_value=0,
            max_value=20,
            value=1
        )

        glucose = st.number_input(
            "Glucose Level",
            min_value=0,
            max_value=250,
            value=120
        )

        blood_pressure = st.number_input(
            "Blood Pressure",
            min_value=0,
            max_value=200,
            value=70
        )

        skin_thickness = st.number_input(
            "Skin Thickness",
            min_value=0,
            max_value=100,
            value=20
        )

    with input_col2:

        insulin = st.number_input(
            "Insulin",
            min_value=0,
            max_value=900,
            value=80
        )

        bmi = st.number_input(
            "BMI",
            min_value=0.0,
            max_value=70.0,
            value=25.0
        )

        diabetes_pedigree = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=3.0,
            value=0.5
        )

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=30
        )

    st.markdown("")

    predict_button = st.button(
        "🔍 Assess Diabetes Risk",
        type="primary",
        width="stretch"
    )

    if predict_button:

        patient_data = pd.DataFrame(
            [[
                pregnancies,
                glucose,
                blood_pressure,
                skin_thickness,
                insulin,
                bmi,
                diabetes_pedigree,
                age
            ]],
            columns=[
                "Pregnancies",
                "Glucose",
                "BloodPressure",
                "SkinThickness",
                "Insulin",
                "BMI",
                "DiabetesPedigreeFunction",
                "Age"
            ]
        )

        # Handle zero values like training data
        for column in columns_with_invalid_zero:

            if patient_data.loc[0, column] == 0:

                patient_data.loc[0, column] = clean_data[
                    column
                ].median()

        patient_scaled = scaler.transform(
            patient_data
        )

        prediction = model.predict(
            patient_scaled
        )[0]

        probability = model.predict_proba(
            patient_scaled
        )[0][1]

        probability_percentage = probability * 100

        st.divider()

        st.markdown(
            '<div class="section-title">'
            '📊 Model Assessment Result'
            '</div>',
            unsafe_allow_html=True
        )

        result_col1, result_col2 = st.columns([1, 1])

        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        with result_col1:

            if prediction == 1:

                st.markdown(
                    """
                    <div class="risk-high">
                        <h2>⚠️ Higher Risk Classification</h2>
                        <p>
                        The machine learning model classified
                        this patient record as higher risk.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    """
                    <div class="risk-low">
                        <h2>✅ Lower Risk Classification</h2>
                        <p>
                        The machine learning model classified
                        this patient record as lower risk.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.metric(
                "Estimated Model Risk Probability",
                f"{probability_percentage:.2f}%"
            )

        # -------------------------------------------------
        # GAUGE
        # -------------------------------------------------

        with result_col2:

            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=probability_percentage,
                    title={
                        "text": "Estimated Risk Probability"
                    },
                    gauge={
                        "axis": {
                            "range": [0, 100]
                        },
                        "bar": {
                            "color": "#0b5d4b"
                        },
                        "steps": [
                            {
                                "range": [0, 50],
                                "color": "#dff3e9"
                            },
                            {
                                "range": [50, 100],
                                "color": "#ffe1e1"
                            }
                        ]
                    }
                )
            )

            gauge.update_layout(
                height=330,
                margin=dict(
                    l=20,
                    r=20,
                    t=60,
                    b=20
                )
            )

            st.plotly_chart(
                gauge,
                width="stretch"
            )

        # -------------------------------------------------
        # PATIENT SUMMARY
        # -------------------------------------------------

        st.markdown("### 👤 Patient Summary")

        summary_table = pd.DataFrame({
            "Parameter": [
                "Pregnancies",
                "Glucose",
                "Blood Pressure",
                "BMI",
                "Age"
            ],
            "Value": [
                pregnancies,
                glucose,
                blood_pressure,
                bmi,
                age
            ]
        })

        st.dataframe(
            summary_table,
            width="stretch",
            hide_index=True
        )

# =========================================================
# TAB 3 - HEALTH ANALYTICS
# =========================================================

with tab3:

    st.markdown(
        '<div class="section-title">'
        '📈 Health Data Analytics'
        '</div>',
        unsafe_allow_html=True
    )

    analytics_col1, analytics_col2 = st.columns(2)

    # -----------------------------------------------------
    # GLUCOSE DISTRIBUTION
    # -----------------------------------------------------

    with analytics_col1:

        fig_glucose = px.histogram(
            clean_data,
            x="Glucose",
            color="Outcome",
            nbins=25,
            title="Glucose Level Distribution",
            labels={
                "Glucose": "Glucose Level",
                "Outcome": "Outcome"
            }
        )

        fig_glucose.update_layout(
            height=420
        )

        st.plotly_chart(
            fig_glucose,
            width="stretch"
        )

    # -----------------------------------------------------
    # BMI VS GLUCOSE
    # -----------------------------------------------------

    with analytics_col2:

        scatter_data = clean_data.copy()

        scatter_data["Risk"] = scatter_data[
            "Outcome"
        ].map({
            0: "Lower Risk",
            1: "Higher Risk"
        })

        fig_scatter = px.scatter(
            scatter_data,
            x="BMI",
            y="Glucose",
            color="Risk",
            hover_data=[
                "Age",
                "BloodPressure"
            ],
            title="BMI vs Glucose Level",
            labels={
                "BMI": "Body Mass Index",
                "Glucose": "Glucose Level"
            }
        )

        fig_scatter.update_layout(
            height=420
        )

        st.plotly_chart(
            fig_scatter,
            width="stretch"
        )

    # -----------------------------------------------------
    # AGE VS GLUCOSE
    # -----------------------------------------------------

    fig_age_glucose = px.scatter(
        clean_data,
        x="Age",
        y="Glucose",
        color="Outcome",
        size="BMI",
        hover_data=[
            "BloodPressure",
            "Insulin"
        ],
        title="Age, Glucose and BMI Relationship",
        labels={
            "Age": "Age",
            "Glucose": "Glucose Level",
            "BMI": "BMI",
            "Outcome": "Outcome"
        }
    )

    fig_age_glucose.update_layout(
        height=450
    )

    st.plotly_chart(
        fig_age_glucose,
        width="stretch"
    )

    # -----------------------------------------------------
    # STATISTICAL SUMMARY
    # -----------------------------------------------------

    st.markdown("### 📊 Statistical Summary")

    statistics = clean_data[
        [
            "Glucose",
            "BloodPressure",
            "BMI",
            "Age"
        ]
    ].describe().round(2)

    st.dataframe(
        statistics,
        width="stretch"
    )

# =========================================================
# TAB 5 - PATIENT DATA
# =========================================================

with tab5:

    st.markdown(
        '<div class="section-title">'
        '📋 Healthcare Dataset'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Interactive view of the healthcare records used "
        "for the classification project."
    )

    display_data = data.copy()

    display_data["Risk Classification"] = display_data[
        "Outcome"
    ].map({
        0: "Lower Risk",
        1: "Higher Risk"
    })

    display_data = display_data[
        [
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age",
            "Risk Classification"
        ]
    ]

    st.dataframe(
        display_data,
        width="stretch",
        height=500,
        hide_index=True
    )

# =========================================================
# TAB 4 - DATASET MANAGEMENT
# =========================================================

with tab4:

    st.markdown(
        '<div class="section-title">'
        '📂 Dataset Management'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Upload a healthcare CSV dataset to inspect its "
        "structure, statistics, missing values, and basic "
        "data distribution."
    )

    uploaded_file = st.file_uploader(
        "📤 Upload Healthcare Dataset (CSV)",
        type=["csv"]
    )

    if uploaded_file is not None:

        uploaded_data = pd.read_csv(uploaded_file)

        st.success(
            f"Dataset uploaded successfully: {uploaded_file.name}"
        )

        st.markdown("### 📊 Dataset Overview")

        overview_col1, overview_col2, overview_col3 = st.columns(3)

        with overview_col1:
            st.metric(
                "Total Records",
                f"{uploaded_data.shape[0]:,}"
            )

        with overview_col2:
            st.metric(
                "Total Columns",
                uploaded_data.shape[1]
            )

        with overview_col3:
            st.metric(
                "Missing Values",
                int(uploaded_data.isnull().sum().sum())
            )

        st.markdown("### 📋 Dataset Preview")

        st.dataframe(
            uploaded_data.head(20),
            width="stretch",
            hide_index=True
        )

        st.markdown("### 🔎 Missing Values")

        missing_data = pd.DataFrame({
            "Column": uploaded_data.columns,
            "Missing Values": uploaded_data.isnull().sum().values
        })

        st.dataframe(
            missing_data,
            width="stretch",
            hide_index=True
        )

        st.markdown("### 📈 Statistical Summary")

        numeric_data = uploaded_data.select_dtypes(
            include="number"
        )

        if not numeric_data.empty:

            st.dataframe(
                numeric_data.describe().round(2),
                width="stretch"
            )

        else:

            st.info(
                "No numeric columns were found in this dataset."
            )

        st.markdown("### 📊 Dataset Visualization")

        if not numeric_data.empty and len(numeric_data.columns) >= 2:

            selected_x = st.selectbox(
                "Select X-axis",
                numeric_data.columns
            )

            selected_y = st.selectbox(
                "Select Y-axis",
                numeric_data.columns
            )

            fig_uploaded = px.scatter(
                uploaded_data,
                x=selected_x,
                y=selected_y,
                title=f"{selected_x} vs {selected_y}"
            )

            fig_uploaded.update_layout(height=450)

            st.plotly_chart(
                fig_uploaded,
                width="stretch"
            )

        else:

            st.info(
                "At least two numeric columns are required "
                "to create a visualization."
            )
# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer-box">
        <b>⚠️ Academic Disclaimer:</b><br>
        This application is developed for educational and
        academic purposes as part of a Fundamentals of
        Machine Learning project. The model output is a
        machine-learning classification and should not be
        considered a medical diagnosis or a substitute for
        professional medical advice.
    </div>
    """,
    unsafe_allow_html=True
)