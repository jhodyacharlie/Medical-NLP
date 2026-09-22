import re
import random

import numpy as np
import pandas as pd
import streamlit as st

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from datasets import load_dataset

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="MedBot - Medical Chatbot",
    page_icon="🏥",
    layout="centered",
)

# ============================================================
# SETUP NLTK & SENTENCE-BERT (dijalankan sekali, lalu di-cache)
# ============================================================
@st.cache_resource(show_spinner="Menyiapkan resource NLP...")
def setup_nlp():
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)

    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        return model, True
    except Exception:
        return None, False


SBERT_MODEL, USE_SBERT = setup_nlp()

# ============================================================
# PREPROCESSING TEKS
# ============================================================
INDONESIAN_STOPWORDS = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "dengan", "untuk",
    "pada", "adalah", "atau", "juga", "dalam", "tidak", "akan", "ada",
    "saya", "kamu", "anda", "ia", "mereka", "kami", "kita", "bisa",
    "sudah", "bila", "jika", "maka", "oleh", "karena", "apa",
    "bagaimana", "berapa", "kapan", "dimana", "siapa", "apakah", "cara",
    "lebih", "sangat", "dapat", "nya", "pun", "lagi", "belum",
    "telah", "namun", "tapi", "serta", "meski", "agar", "supaya", "hal",
    "the", "is", "are", "was", "what", "how", "why", "when", "where",
}
ENGLISH_STOPWORDS = set(stopwords.words("english"))
ALL_STOPWORDS = INDONESIAN_STOPWORDS | ENGLISH_STOPWORDS
LEMMATIZER = WordNetLemmatizer()


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in ALL_STOPWORDS and len(t) > 2]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)


# ============================================================
# KATEGORI SEDERHANA (dataset tidak menyediakan label kategori)
# ============================================================
CATEGORY_KEYWORDS = {
    "diabetes": ["diabetes", "insulin", "blood sugar", "glucose", "hyperglycemia"],
    "hipertensi": ["hypertension", "blood pressure"],
    "jantung": ["heart", "cardiac", "cardiovascular", "myocardial"],
    "asma": ["asthma", "wheezing", "inhaler", "bronchitis"],
    "covid": ["covid", "coronavirus", "sars-cov"],
    "kanker": ["cancer", "tumor", "tumour", "carcinoma", "malignant"],
    "kesehatan mental": ["depression", "anxiety", "stress", "mental", "psychiatric"],
    "infeksi": ["infection", "virus", "bacteria", "bacterial", "fungal"],
    "pencernaan": ["stomach", "gastritis", "digestive", "intestine", "bowel"],
}


def assign_category(text: str) -> str:
    text_lower = str(text).lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in text_lower for keyword in keywords):
            return category
    return "umum"


# ============================================================
# LOAD DATASET (di-cache supaya tidak diunduh ulang tiap interaksi)
# ============================================================
@st.cache_data(show_spinner="Memuat dataset medis dari HuggingFace...")
def load_data(dataset_config: str = "medical_meadow_medical_flashcards",
              n_samples: int = 3000,
              random_state: int = 42) -> pd.DataFrame:
    hf_dataset = load_dataset(
        "Malikeh1375/medical-question-answering-datasets",
        dataset_config,
        split="train",
    )
    df = hf_dataset.to_pandas()

    df["input"] = df["input"].astype(str).str.strip()
    df["output"] = df["output"].astype(str).str.strip()
    df = df[(df["input"] != "") & (df["output"] != "")]
    df = df.drop_duplicates(subset="input").reset_index(drop=True)

    if n_samples is not None and len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=random_state).reset_index(drop=True)

    df = df.rename(columns={"input": "question", "output": "answer"})
    df["category"] = (df["question"] + " " + df["answer"]).apply(assign_category)
    df["processed_question"] = df["question"].apply(preprocess_text)
    return df


# ============================================================
# ENGINE CHATBOT
# ============================================================
class MedicalChatbotEngine:
    def __init__(self, dataframe: pd.DataFrame, threshold: float = 0.35, top_k: int = 3):
        self.df = dataframe
        self.threshold = threshold if USE_SBERT else 0.15
        self.top_k = top_k
        self.conversation_history = []
        self._build_index()
        self._define_rules()

    def _build_index(self):
        if USE_SBERT:
            self.sbert_embeddings = SBERT_MODEL.encode(
                self.df["question"].tolist(),
                convert_to_tensor=True,
                show_progress_bar=False,
                batch_size=64,
            )

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["processed_question"])

    def _define_rules(self):
        self.rules = {
            "emergency": {
                "patterns": [
                    r"(sesak.*berat|nyeri dada.*berat|tidak.*bernapas|pingsan)",
                    r"(can'?t breathe|chest pain|unconscious|heart attack|severe bleeding)",
                ],
                "responses": ["🚨 DARURAT! Hubungi 119 atau segera ke IGD terdekat!"],
            },
            "greeting": {
                "patterns": [r"\b(halo|hai|hi|hello|hey|good morning|good afternoon)\b"],
                "responses": ["👋 Halo! Ada yang bisa saya bantu terkait kesehatan hari ini?"],
            },
        }

    def _check_rules(self, text: str):
        for _, data in self.rules.items():
            for pattern in data["patterns"]:
                if re.search(pattern, text.lower()):
                    return random.choice(data["responses"])
        return None

    def _search_sbert(self, query: str):
        from sentence_transformers import util

        emb = SBERT_MODEL.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(emb, self.sbert_embeddings)[0].cpu().numpy()
        top_results = np.argsort(-scores)[: self.top_k]
        return [(idx, float(scores[idx])) for idx in top_results]

    def _search_tfidf(self, query: str):
        processed = preprocess_text(query)
        vec = self.vectorizer.transform([processed])
        scores = cosine_similarity(vec, self.tfidf_matrix).flatten()
        top_results = np.argsort(scores)[::-1][: self.top_k]
        return [(idx, scores[idx]) for idx in top_results]

    def _find_best_match(self, query: str):
        results = self._search_sbert(query) if USE_SBERT else self._search_tfidf(query)
        return results[0]

    def _build_context_query(self, user_input: str) -> str:
        if self.conversation_history:
            return self.conversation_history[-1] + " " + user_input
        return user_input

    def get_response(self, user_input: str) -> str:
        if not user_input.strip():
            return "Silakan ketik pertanyaan."

        rule = self._check_rules(user_input)
        if rule:
            return rule

        query = self._build_context_query(user_input)
        results = self._search_sbert(query) if USE_SBERT else self._search_tfidf(query)
        method = "SBERT" if USE_SBERT else "TF-IDF"

        best_idx, best_score = results[0]
        if best_score < self.threshold:
            return "🤔 Tidak menemukan jawaban yang cukup relevan. Coba ubah kata kunci pertanyaan Anda."

        boosted = []
        for idx, score in results:
            text = self.df.iloc[idx]["question"]
            bonus = sum(1 for word in user_input.lower().split() if word in text.lower())
            boosted.append((idx, score + 0.05 * bonus))

        best_idx = sorted(boosted, key=lambda x: x[1], reverse=True)[0][0]
        row = self.df.iloc[best_idx]
        self.conversation_history.append(user_input)

        return (
            f"**[Kategori: {row['category']} | {method}]**\n\n"
            f"{row['answer']}\n\n"
            "─────────────────\n"
            "⚠️ Jawaban bersumber dari dataset medis (Malikeh1375/medical-question-answering-datasets). "
            "Untuk kondisi serius, konsultasikan ke dokter."
        )


@st.cache_resource(show_spinner="Menyiapkan mesin pencarian jawaban (bisa memakan waktu di run pertama)...")
def build_bot(dataset_config: str, n_samples: int) -> MedicalChatbotEngine:
    df = load_data(dataset_config, n_samples)
    return MedicalChatbotEngine(df)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("🏥 Tentang MedBot")
    st.write(
        "Chatbot edukasi kesehatan berbasis pencarian semantik (Sentence-BERT) "
        "di atas dataset **Malikeh1375/medical-question-answering-datasets** dari HuggingFace."
    )
    st.caption(f"Metode aktif: **{'Sentence-BERT' if USE_SBERT else 'TF-IDF (fallback)'}**")

    dataset_config = st.selectbox(
        "Subset dataset",
        [
            "medical_meadow_medical_flashcards",
            "medical_meadow_wikidoc_patient_information",
            "medical_meadow_medqa",
        ],
        index=0,
    )
    n_samples = st.slider("Jumlah sampel data", 500, 5000, 3000, step=500)

    st.divider()
    st.warning("⚠️ Bukan pengganti dokter. Darurat medis: hubungi **119**.")

    if st.button("🗑️ Reset percakapan"):
        st.session_state.messages = []
        st.rerun()

# ============================================================
# MAIN CHAT UI
# ============================================================
st.title("🏥 MedBot — Asisten Kesehatan")
st.caption("Tanya seputar kesehatan (Bahasa Indonesia/Inggris) — jawaban ditampilkan dalam Bahasa Inggris karena sumber datanya berbahasa Inggris.")

bot = build_bot(dataset_config, n_samples)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ketik pertanyaan kesehatan Anda... (contoh: diabetes symptoms)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("MedBot sedang mengetik..."):
            response = bot.get_response(prompt)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
