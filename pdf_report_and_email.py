import os
import tempfile
import base64
import streamlit as st
from dotenv import load_dotenv
from fpdf import FPDF
from docx import Document
import smtplib
from email.message import EmailMessage

# LangChain imports (make sure you have the latest versions)
from langchain_groq import ChatGroq
from langchain_core.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")  # Preferably an app password

# Initialize the Groq LLM
llm = ChatGroq(api_key=groq_api_key, model_name="meta-llama/llama-4-scout-17b-16e-instruct")
parser = StrOutputParser()

# -------------------------------
# Define Chat Prompt Templates with roles
# -------------------------------

# 1. Summarization prompt – given the extracted PDF text, produce a report summary.
summary_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("You are a concise and detail-oriented report generator."),
    HumanMessagePromptTemplate.from_template("Summarize the following PDF content into a clear and professional report:\n\n{text}\n")
])

# Create summarization chain using the pipeline operator
summarize_chain = summary_prompt | llm | parser

# -------------------------------
# PDF Extraction Function
# -------------------------------
# (For demonstration, using PyMuPDF is a common choice. You can substitute any PDF text extraction library.)
def extract_text_from_pdf(file_bytes) -> str:
    import fitz  # PyMuPDF; install via pip install pymupdf
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# -------------------------------
# File Generation Functions
# -------------------------------
def create_txt(blog_text: str):
    return blog_text

def create_pdf(blog_text: str, output_path: str):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    # Attempt to encode text to avoid encoding errors using replace
    pdf.set_font("Arial", size=12)
    for line in blog_text.split("\n"):
        safe_line = line.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 10, safe_line)
    pdf.output(output_path)

def create_docx(blog_text: str, output_path: str):
    doc = Document()
    for para in blog_text.split("\n\n"):
        doc.add_paragraph(para)
    doc.save(output_path)
    return output_path

# Wrap file generation in a RunnableLambda so it fits in the chain pipeline if desired.
#file_generator = RunnableLambda(lambda content: create_txt(content))

# -------------------------------
# Email Sending Function
# -------------------------------
def send_email(subject: str, body: str, to_email: str, attachment_path: str = None, attachment_filename: str = None):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = to_email
    msg.set_content(body)
    
    if attachment_path and attachment_filename:
        with open(attachment_path, "rb") as f:
            file_data = f.read()
            msg.add_attachment(file_data, maintype="application", subtype="octet-stream", filename=attachment_filename)
    
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        smtp.send_message(msg)

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="📄 PDF Report & Emailer", layout="centered")
st.title("📄 PDF Summarizer and Email Report Generator")
st.write("Upload a PDF, generate a report, and email it to a client!")

# PDF upload
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf","docx"])
recipient_email = st.text_input("Enter recipient email", placeholder="client@example.com")
send_btn = st.button("Send Report via Email")

if uploaded_file:
    st.success(f"PDF uploaded: {uploaded_file.name}")
    # Extract text from PDF
    with st.spinner("Extracting text from PDF..."): 
        pdf_text = extract_text_from_pdf(uploaded_file.read())
    st.text_area("Extracted Text Preview", value=pdf_text, height=200)

    # Generate Report using the summarization chain
    with st.spinner("Generating report summary..."):
        report_summary = summarize_chain.invoke({"text": pdf_text})
    st.subheader("Generated Report")
    st.write(report_summary)
    
    # Create downloadable files
    blog_content = report_summary  # In a real use-case, you might add more formatting
    
    # TXT file path
    txt_path = os.path.join(tempfile.gettempdir(), "report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(blog_content)
    
    # PDF file path
    pdf_path = os.path.join(tempfile.gettempdir(), "report.pdf")
    create_pdf(blog_content, pdf_path)
    
    # DOCX file path
    docx_path = os.path.join(tempfile.gettempdir(), "report.docx")
    create_docx(blog_content, docx_path)
    
    st.markdown("---")
    st.subheader("Download Report")
    st.download_button("⬇️ Download as TXT", data=blog_content, file_name="report.txt")
    
    with open(pdf_path, "rb") as f:
        st.download_button("⬇️ Download as PDF", data=f.read(), file_name="report.pdf")
    
    with open(docx_path, "rb") as f:
        st.download_button("⬇️ Download as DOCX", data=f.read(), file_name="report.docx")
    
    # Sending Email
    if send_btn and recipient_email:
        with st.spinner("Sending email..."):
            try:
                email_subject = f"Report Summary for {uploaded_file.name}"
                email_body = f"Hello,\n\nPlease find attached the report summary for the PDF {uploaded_file.name}.\n\nReport:\n{blog_content}\n\nRegards,\nYour AI Assistant"
                # Here, you can attach one of the file types; let's attach the PDF.
                send_email(email_subject, email_body, recipient_email, attachment_path=pdf_path, attachment_filename="report.pdf")
                st.success("Email sent successfully!")
            except Exception as e:
                st.error(f"Error sending email: {e}")
