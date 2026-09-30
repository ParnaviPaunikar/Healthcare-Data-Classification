import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Healthcare Risk Assessment",
    page_icon="🩺",
    layout="wide"
)


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

model = joblib.load("model/disease_risk_model.pkl")
scaler = joblib.load("model/scaler.pkl")


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 40px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .risk-box {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
    }

    .disclaimer {
        padding: 15px;
        border-radius: 10px;
        margin-top: 30px;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🩺 Healthcare Risk Assessment</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning Based Diabetes Risk Classification Framework</div>',
    unsafe_allow_html=True
)


st.divider()


# =========================================================
# INTRODUCTION
# =========================================================

st.markdown("## 📋 Patient Information")

st.write(
    "Enter the patient's health information below. "
    "The trained machine learning model will classify the "
    "record into a diabetes risk category."
)


# =========================================================
# INPUT FORM
# =========================================================

col1, col2 = st.columns(2)


with col1:

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


with col2:

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


st.divider()


# =========================================================
# PREDICTION BUTTON
# =========================================================

predict_button = st.button(
    "🔍 Assess Diabetes Risk",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

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


    # Scale patient data
    patient_scaled = scaler.transform(patient_data)


    # Prediction
    prediction = model.predict(patient_scaled)[0]

    probability = model.predict_proba(patient_scaled)[0][1]

    probability_percentage = probability * 100


    # =====================================================
    # RESULT
    # =====================================================

    st.markdown("## 📊 Risk Assessment Result")


    if prediction == 1:

        st.error(
            "⚠️ Higher Diabetes Risk Classification"
        )

        risk_label = "Higher Risk"

    else:

        st.success(
            "✅ Lower Diabetes Risk Classification"
        )

        risk_label = "Lower Risk"


    st.metric(
        "Estimated Risk Probability",
        f"{probability_percentage:.2f}%"
    )


    # =====================================================
    # GAUGE CHART
    # =====================================================

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
                }
            }
        )
    )


    gauge.update_layout(
        height=350
    )


    st.plotly_chart(
        gauge,
        use_container_width=True
    )


    # =====================================================
    # PATIENT SUMMARY
    # =====================================================

    st.markdown("### Patient Summary")

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    with summary_col1:

        st.metric(
            "Age",
            age
        )

    with summary_col2:

        st.metric(
            "Glucose",
            glucose
        )

    with summary_col3:

        st.metric(
            "BMI",
            bmi
        )


# =========================================================
# DISCLAIMER
# =========================================================

st.divider()

st.markdown(
    """
    <div class="disclaimer">

    ⚠️ <b>Academic Disclaimer:</b><br>
    This application is developed for educational and academic
    purposes as part of a Fundamental of Machine Learning project.
    The prediction is generated by a machine learning model and
    should not be considered a medical diagnosis or a substitute
    for professional medical advice.

    </div>
    """,
    unsafe_allow_html=True
)