import os
import base64
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from typing import Optional, List
from pydantic import BaseModel, Field, EmailStr
from langchain_openai import ChatOpenAI 

# Load .env
load_dotenv()

# ---------- Pydantic Models ----------
class PatientInfo(BaseModel):
    name: Optional[str] = Field(None, description="Full name of the patient")
    age: Optional[int] = Field(None, description="Age of the patient")
    gender: Optional[str] = Field(None, description="Gender of the patient")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    report_date: Optional[str] = Field(None, description="Date of the report")

class MedicalMetric(BaseModel):
    name: Optional[str] = Field(..., description="Name of the medical metric")
    value: Optional[str] = Field(..., description="Value with units")
    status: Optional[str] = Field(None, description="Status - High, Low, or Normal")

class MedicalReport(BaseModel):
    patient: PatientInfo
    metrics: List[MedicalMetric]
    summary : Optional[str] = Field(None, description="Summary of report, include some final review")

# ---------- UI Components ----------
def card_box(title, value):
    st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #f9f9f9, #ffffff);
            border-left: 5px solid #4CAF50;
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 12px;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.05);
        ">
            <div style="font-size: 14px; color: #888; font-weight: 600;">{title}</div>
            <div style="font-size: 18px; font-weight: 500; color: #333; margin-top: 4px;">{value}</div>
        </div>
    """, unsafe_allow_html=True)

def show_patient_info(info: PatientInfo):
    st.markdown("## 👤 Patient Info")
    col1, col2 = st.columns(2)
    with col1:
        card_box("Name", info.name or "N/A")
        card_box("Age", info.age or "N/A")
        card_box("Gender", info.gender or "N/A")
    with col2:
        card_box("Email", info.email or "N/A")
        card_box("Phone", info.phone or "N/A")
        card_box("Report Date", info.report_date or "N/A")

def show_medical_metrics(metrics: List[MedicalMetric]):
    st.markdown("## 🧪 Medical Metrics")
    for metric in metrics:
        # Fallbacks for missing data
        name = metric.name or "N/A"
        value = metric.value or "N/A"
        status = metric.status or "N/A"

        # Status color logic
        color = "#2196F3"  # Default blue
        if status.lower() == "high":
            color = "#f44336"  # Red
        elif status.lower() == "low":
            color = "#ff9800"  # Orange
        elif status.lower() == "normal":
            color = "#4CAF50"  # Green

        # Display each metric as a styled card
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #ffffff, #f0f0f0);
            border-left: 5px solid {color};
            border-radius: 10px;
            padding: 12px 16px;
            margin-bottom: 10px;
            box-shadow: 1px 1px 6px rgba(0,0,0,0.05);
        ">
            <div style='color: black; font-size: 16px; font-weight: 600;'>
                <strong>{name}</strong>: {value}
            </div>
            <div style='color:{color}; font-weight: bold; font-size: 14px; margin-top: 4px;'>
                Status: {status}
            </div>
        </div>
        """, unsafe_allow_html=True)


# ---------- Groq Vision API ----------
#llm = ChatGroq(model_name="meta-llama/llama-4-scout-17b-16e-instruct")
llm = ChatOpenAI (model_name="gpt-4o-mini")
def getStructuredMedicalData(base64_image: str) -> MedicalReport:
    message = HumanMessage(
        content=[{
            "type": "text", 
            "text": """You are an advanced medical information extraction and summarization system. Your primary task is to analyze any type of medical report image and extract all structured information clearly presented in the document. This includes, but is not limited to:​

Identifiable entities such as patient or user details, contact information, and dates.​

All kinds of medical measurements, metrics, observations, results, or findings.​

Every medical term with its associated value, unit, and, if available, any indicators like High/Low/Normal.​

Any identifiable fields like names, contact info, or timestamps—whether they belong to a patient, user, doctor, or organization.​

Do not infer or hallucinate data. If any field is not available in the report, leave it null. Ensure the final output strictly follows the predefined schema and includes everything that could be relevant in a medical context.​

After extracting all structured data from the medical report, provide a concise summary that encapsulates the key findings, notable abnormalities, and overall assessment. This summary should reflect the information presented in the report without introducing any assumptions or external knowledge.​

If any values are significantly outside the normal range or indicate potential health risks, advise the patient to consult a healthcare professional promptly. For minor deviations or common issues, suggest appropriate lifestyle changes, dietary adjustments, or over-the-counter remedies, ensuring that all recommendations are based solely on the information provided in the report.​

Additionally, when appropriate and based strictly on the data extracted from the medical report, provide a suggested medical prescription. This should include:​

Medication Name: The recommended drug based on the findings.​

Dosage: The amount and frequency of the medication.​

Route of Administration: How the medication should be taken (e.g., orally, intravenously).​

Duration: The length of time the medication should be taken.​

Precautions: Any important warnings or considerations.​

Ensure that all prescription suggestions are grounded solely in the information provided within the medical report and do not incorporate external data or assumptions. Always emphasize that any prescription provided is a suggestion and must be reviewed and authorized by a licensed healthcare professional before implementation.
"""
        }, {
            "type": "image_url", 
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        }]
    )
    system_message = SystemMessage(
    content="""
You are an advanced medical information extraction and summarization system. Your primary task is to analyze any type of medical report image and extract all structured information clearly presented in the document. This includes, but is not limited to:​

Identifiable entities such as patient or user details, contact information, and dates.​

All kinds of medical measurements, metrics, observations, results, or findings.​

Every medical term with its associated value, unit, and, if available, any indicators like High/Low/Normal.​

Any identifiable fields like names, contact info, or timestamps—whether they belong to a patient, user, doctor, or organization.​

Do not infer or hallucinate data. If any field is not available in the report, leave it null. Ensure the final output strictly follows the predefined schema and includes everything that could be relevant in a medical context.​

After extracting all structured data from the medical report, provide a concise summary that encapsulates the key findings, notable abnormalities, and overall assessment. This summary should reflect the information presented in the report without introducing any assumptions or external knowledge.​

If any values are significantly outside the normal range or indicate potential health risks, advise the patient to consult a healthcare professional promptly. For minor deviations or common issues, suggest appropriate lifestyle changes, dietary adjustments, or over-the-counter remedies, ensuring that all recommendations are based solely on the information provided in the report.​

Additionally, when appropriate and based strictly on the data extracted from the medical report, provide a suggested medical prescription. This should include:​

Medication Name: The recommended drug based on the findings.​

Dosage: The amount and frequency of the medication.​

Route of Administration: How the medication should be taken (e.g., orally, intravenously).​

Duration: The length of time the medication should be taken.​

Precautions: Any important warnings or considerations.​

Ensure that all prescription suggestions are grounded solely in the information provided within the medical report and do not incorporate external data or assumptions. Always emphasize that any prescription provided is a suggestion and must be reviewed and authorized by a licensed healthcare professional before implementation.
"""
)
    
    structured_llm = llm.with_structured_output(MedicalReport)
    return structured_llm.invoke([system_message, message])

# ---------- Streamlit UI ----------
st.set_page_config(page_title="🧾 Medical Report Analyzer", layout="centered")
st.title("🧾 Medical Report Analyzer")

with st.sidebar:
    st.title("🧾 Upload Report")
    st.info("Upload a medical report image to extract data.")
    st.markdown("""## Supported Examples""")
    st.markdown("""<li>Blood Report</li>
                    <li>Pathology Reports</li>
                    <li>Lab Tests (WBC, RBC, Sugar, etc.)</li>
    """, unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a medical report image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    resized_image = image.resize((600, 800))
    st.image(resized_image, caption="Uploaded Report", use_column_width=True)

    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    base64_image = base64.b64encode(buffered.getvalue()).decode()

    if st.button("Analyze Report"):
        with st.spinner("Analyzing with LLM..."):
            report = getStructuredMedicalData(base64_image)
            st.markdown(report.model_dump())
            show_patient_info(report.patient)
            show_medical_metrics(report.metrics)
            if report.summary:
                st.markdown("## 📝 Summary")
                st.markdown(f"""
    <div style='
        background: linear-gradient(135deg, #f0fff0, #e0f7e0);
        padding: 20px;
        border-radius: 12px;
        border-left: 6px solid #4CAF50;
        font-size: 17px;
        color: #333;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
        margin-top: 20px;
    '>
        {report.summary}
    </div>
""", unsafe_allow_html=True)
