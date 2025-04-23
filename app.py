import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import re
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

#Download fungsi yg dibutuhkan
import nltk
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Inisialisasi resource
@st.cache_resource
def load_resources():
    tokenizer = AutoTokenizer.from_pretrained("Adkurrr/ikd_ft_fullpreprocessing")
    model = AutoModelForSequenceClassification.from_pretrained("Adkurrr/ikd_ft_fullpreprocessing")

    stop_words = set(stopwords.words('indonesian'))
    custom_stopwords = {'nya', 'yg', 'kali', 'bgt', 'mls'}
    stop_words.update(custom_stopwords)

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    return tokenizer, model, stop_words, stemmer

tokenizer, model, stop_words, stemmer = load_resources()

# --- Preprocessing sesuai training ---
def cleansing_text(review):
    if not isinstance(review, str):
        return ""
    review = review.lower()
    review = re.sub(r'[^a-zA-Z\s]', '', review)
    return review

def tokenize_text(text):
    # Gunakan tokenisasi sederhana untuk bahasa Indonesia
    return text.split()

def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]

def stemming_tokens(tokens):
    return [stemmer.stem(token) for token in tokens]

def preprocess_input(text):
    clean = cleansing_text(text)
    tokens = tokenize_text(clean)
    tokens_no_stopword = remove_stopwords(tokens)
    stemmed_tokens = stemming_tokens(tokens_no_stopword)
    return " ".join(stemmed_tokens)

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
