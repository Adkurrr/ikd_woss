import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import re

# Inisialisasi resource
@st.cache_resource
def load_resources():
    tokenizer = AutoTokenizer.from_pretrained("Adkurrr/ikd_ft_withoutStemStopwords")
    model = AutoModelForSequenceClassification.from_pretrained("Adkurrr/ikd_ft_withoutStemStopwords")

    return tokenizer, model
tokenizer, model = load_resources()

# --- Preprocessing sesuai training ---
def cleansing_text(review):
    if not isinstance(review, str):
        return ""
    review = review.lower()
    review = re.sub(r'[^a-zA-Z\s]', '', review)
    return review

def preprocess_input(text):
    clean = cleansing_text(text)
    return clean

# --- Streamlit UI ---
st.title("Analisis Sentimen Ulasan IKD 🇮🇩")
st.write("Masukkan ulasan aplikasi Identitas Kependudukan Digital untuk dianalisis sentimennya.")

text_input = st.text_area("Tulis ulasan di sini:")

if st.button("Prediksi Sentimen"):
    if text_input.strip() == "":
        st.warning("Masukkan teks terlebih dahulu.")
    else:
        # Preprocess dulu
        preprocessed_text = preprocess_input(text_input)

        # Tokenisasi dan prediksi
        inputs = tokenizer(preprocessed_text, return_tensors="pt", truncation=True, padding=True)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()

        # Mapping label: pastikan sesuai config.json
        label_map = {
            0: "Negatif",
            1: "Positif"
        }

        sentiment = label_map.get(pred, "Tidak diketahui")
        
        st.subheader("Hasil Analisis")
        st.write(f"**Sentimen:** {sentiment}")
        st.write(f"**Kepercayaan:** {confidence * 100:.2f}%")
