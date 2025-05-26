import streamlit as st
import spacy
import joblib
import pandas as pd
import docx
import PyPDF2
import easyocr
import numpy as np
from PIL import Image
import zipfile
import os
import io

# Load models
model = joblib.load("document_classifier_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
nlp = spacy.load("custom_ner_model")

# Initialize EasyOCR
reader = easyocr.Reader(['en'], gpu=False)

# Page configuration
st.set_page_config(page_title="Customs Document Classifier", layout="wide")

# UI Header
st.markdown("""
    <h1 style='text-align: center; color: #2C3E50;'>📦 Customs Document Classifier & Extractor</h1>
""", unsafe_allow_html=True)

# Upload Method Selection
upload_mode = st.radio("Select how to upload files:", ["Upload individual file", "Upload ZIP file"], horizontal=True)

# File uploader
uploaded_file = st.file_uploader("Upload file", type=["txt", "docx", "pdf", "jpg", "jpeg", "png", "xlsx", "csv", "zip"])

# Optional: Manual document type selector
doc_types = ["Bill of Lading", "Commercial Invoice", "Packing List", "Certificate of Origin"]
manual_type_option = st.selectbox("Manually select the document type:", options=doc_types + ["Other"], key="manual_doc_type")
manual_type = st.text_input("Or type your own document type:", "") if manual_type_option == "Other" else manual_type_option

# Text extraction function with OCR error handling
def extract_text(file, ext):
    if ext == "txt":
        return file.read().decode("utf-8")
    elif ext == "docx":
        return "\n".join([para.text for para in docx.Document(file).paragraphs])
    elif ext == "pdf":
        reader_pdf = PyPDF2.PdfReader(file)
        return "\n".join([page.extract_text() for page in reader_pdf.pages if page.extract_text()])
    elif ext == "xlsx":
        df = pd.read_excel(file)
        return df.to_string()
    elif ext == "csv":
        df = pd.read_csv(file)
        return df.to_string()
    elif ext in ["jpg", "jpeg", "png"]:
        try:
            image = Image.open(file)
            image_np = np.array(image)
            results = reader.readtext(image_np)
            return "\n".join([res[1] for res in results])
        except Exception as e:
            return f"⚠️ OCR error: {str(e)}"
    return None

# Processing function
def process_file(file, filename):
    ext = filename.split(".")[-1].lower()
    text = extract_text(file, ext)
    if not text:
        return None, None, None
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]
    doc = nlp(text)
    entities = {ent.label_: ent.text for ent in doc.ents}
    return text, prediction, entities

# ZIP Upload Handling
if uploaded_file and upload_mode == "Upload ZIP file" and uploaded_file.name.endswith(".zip"):
    with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
        st.subheader("📂 Processing Files Inside ZIP")
        for filename in zip_ref.namelist():
            with zip_ref.open(filename) as file:
                text, doc_type, entity_dict = process_file(file, filename)
                if text:
                    st.markdown(f"### 📄 File: `{filename}`")
                    st.text_area("Extracted Text", text, height=200)
                    st.success(f"Predicted Type: {doc_type}")
                    if entity_dict:
                        keys = list(entity_dict.keys()) + ["Other"]
                        selected = st.selectbox(f"View entities for `{filename}`:", keys, key=filename)
                        if selected == "Other":
                            custom_key = st.text_input(f"Type your custom entity label for `{filename}`:", key=f"custom_{filename}")
                            st.info(f"**{custom_key}**: {entity_dict.get(custom_key, 'Not Found')}")
                        else:
                            st.info(f"**{selected}**: {entity_dict[selected]}")
                    else:
                        st.warning("No entities found.")
                else:
                    st.error(f"⚠️ Could not extract text from `{filename}`")

# Single File Upload Handling
elif uploaded_file and upload_mode == "Upload individual file":
    text, doc_type, entity_dict = process_file(uploaded_file, uploaded_file.name)
    if text:
        st.subheader("📄 Document Content")
        st.text_area("Extracted Text", text, height=200)
        st.success(f"Predicted Type: {doc_type}")
        if entity_dict:
            keys = list(entity_dict.keys()) + ["Other"]
            selected = st.selectbox("Select entity to view:", keys)
            if selected == "Other":
                custom_key = st.text_input("Type your custom entity label:")
                st.info(f"**{custom_key}**: {entity_dict.get(custom_key, 'Not Found')}")
            else:
                st.info(f"**{selected}**: {entity_dict[selected]}")
        else:
            st.warning("No entities found.")
    else:
        st.error("❌ Could not extract text from the uploaded file.")
else:
    st.info("📁 Upload a file or ZIP to get started.")
