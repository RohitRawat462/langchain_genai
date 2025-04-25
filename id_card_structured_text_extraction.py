import os
import base64
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from typing import Optional
from pydantic import BaseModel, Field, EmailStr
# Load .env
load_dotenv()

# ---------- Pydantic Model ----------
class IDCardInfo(BaseModel):
    name: str = Field(..., description="Full name as on the ID card")
    age: Optional[int] = Field(None, description="Age of the person")
    gender: Optional[str] = Field(None, description="Gender of the cardholder")
    dob: Optional[str] = Field(None, description="Date of Birth in YYYY-MM-DD format")
    email: Optional[EmailStr] = Field(None, description="Email ID")
    phone: Optional[str] = Field(None, description="Phone number")
    id_type: Optional[str] = Field(None, description="Type of ID Card (e.g., Aadhaar, PAN, Passport, etc.)")
    id_number: Optional[str] = Field(None, description="ID Card number")
    address: Optional[str] = Field(None, description="Address as printed on the ID card")
    issue_date: Optional[str] = Field(None, description="Date of issue of the ID card")
    expiry_date: Optional[str] = Field(None, description="Expiry date of the ID card if applicable")
    nationality: Optional[str] = Field(None, description="Nationality or citizenship status")
    image: Optional[str] = Field(None, description="Image of the ID card holder")

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


def display_id_card_info(output_text: dict):
    st.markdown("## 🪪 Extracted ID Card Information")
    st.markdown("---")

    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            card_box("Name", output_text.get("name", "N/A"))
            card_box("Age", output_text.get("age", "N/A"))
            card_box("Gender", output_text.get("gender", "N/A"))
            card_box("Date of Birth", output_text.get("dob", "N/A"))
            card_box("ID Type", output_text.get("id_type", "N/A"))
            card_box("ID Number", output_text.get("id_number", "N/A"))

        with col2:
            card_box("Email", output_text.get("email", "N/A"))
            card_box("Phone", output_text.get("phone", "N/A"))
            card_box("Address", output_text.get("address", "N/A"))
            card_box("Issue Date", output_text.get("issue_date", "N/A"))
            card_box("Expiry Date", output_text.get("expiry_date", "N/A"))
            card_box("Nationality", output_text.get("nationality", "N/A"))

# ---------- Groq Vision API ----------
llm = ChatGroq(model_name="llama-3.2-90b-vision-preview")
def getStructuredDataFromVisionModel(base64_image: str) -> str:
    message = HumanMessage(
             content=[
             {"type": "text", "text": 'Extract strurued text from this image, please extract image pic of card holder if present'},
             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
             ])
    structured_llm = llm.with_structured_output(IDCardInfo)
    response = structured_llm.invoke([message])
    return response
    
# ---------- Streamlit UI ----------
st.set_page_config(page_title="🪪 ID Card Extractor", layout="centered")
st.title("🪪 ID Card Data Extractor with Groq Vision")


with st.sidebar:
    st.title("🪪 ID Card Data Extractor")
    st.info("Upload an image of an ID card to extract structured data.")

    st.markdown("""## List of Supported doc""")
    st.markdown("""
     <li>Aadhaar Card</li>
            <li>PAN Card</li>
            <li>Driving License</li>
            <li>Voter ID</li>
            <li>Passport</li>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload an ID card image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)

    # Preview
    st.image(image, caption="Uploaded ID Card", use_column_width=True)

    # Convert to base64
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    base64_image = base64.b64encode(buffered.getvalue()).decode()

    if st.button("Extract Info"):
        with st.spinner("Processing image..."):
            output_text = getStructuredDataFromVisionModel(base64_image)
            output_json  = output_text.dict()
            st.markdown(output_json)
            display_id_card_info(output_json)
            
