import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
from huggingface_hub import hf_hub_download
from PIL import Image
import requests
from io import BytesIO
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
import nltk

# === Fungsi Praproses Teks ===
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    return text
    
# Inisialisasi resource
@st.cache_resource
def load_resources():
    model = AutoModelForSequenceClassification.from_pretrained("Adkurrr/ikd_ft_woss")
    tokenizer = AutoTokenizer.from_pretrained("Adkurrr/ikd_ft_woss")
    return tokenizer, model

@st.cache_resource
def load_bert_pretrained():
    model = AutoModelForSequenceClassification.from_pretrained("Adkurrr/ikd_pretrained_woss")
    tokenizer = AutoTokenizer.from_pretrained("Adkurrr/ikd_pretrained_woss")
    return model, tokenizer

@st.cache_resource
def load_lr_model():
    file_path = hf_hub_download(repo_id="Adkurrr/lr-SVM-woss", filename="lr_model.pkl")
    return joblib.load(file_path)

@st.cache_resource
def load_svm_model():
    file_path = hf_hub_download(repo_id="Adkurrr/Lr-SVM-woss", filename="svm_model.pkl")
    return joblib.load(file_path)

# === Prediction Functions ===
def predict_with_bert(text, model, tokenizer):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).squeeze()
        pred = torch.argmax(probs).item()
    return pred, probs.numpy()

def predict_with_model(text, model):
    return model.predict([text])[0]
    
st.title("Prediksi Sentimen Ulasan IKD")
st.write("Skenario : Tanpa Stopword Removal dan Stemming")

text_input = st.text_area("Masukkan ulasan:", "")
model_choice = st.selectbox("Pilih Model", [
        "BERT Finetuned", "BERT Pretrained", "Logistic Regression", "SVM"
    ])

if st.button("🔍 Prediksi Sentimen"):
        if not text_input.strip():
            st.warning("⚠️ Ulasan Tidak Boleh Kosong")
        else:
            # Praproses semua input sebelum diprediksi
            processed_text = preprocess_text(text_input)

            if model_choice == "BERT Finetuned":
                model, tokenizer = load_bert_finetuned()
                label, probs = predict_with_bert(processed_text, model, tokenizer)
            elif model_choice == "BERT Pretrained":
                model, tokenizer = load_bert_pretrained()
                label, probs = predict_with_bert(processed_text, model, tokenizer)
            elif model_choice == "Logistic Regression":
                model = load_lr_model()
                label = predict_with_model(processed_text, model)
            elif model_choice == "SVM":
                model = load_svm_model()
                label = predict_with_model(processed_text, model)
            else:
                label = "?"

            sentimen_label = "Positif" if str(label) in ["1", "positif", "positive"] else "Negatif"
            st.success(f"Prediksi Sentimen: {sentimen_label}")
