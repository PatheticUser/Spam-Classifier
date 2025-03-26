import streamlit as st
import pickle
import string
import nltk
import pandas as pd
import numpy as np
import base64
from sklearn.metrics import confusion_matrix, classification_report
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from typing import Tuple, Optional

# NLTK Resource Management
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)


# Preprocessing Utility
class TextPreprocessor:
    def __init__(self):
        self.ps = PorterStemmer()
        self.stop_words = set(stopwords.words("english"))
        self.punctuation = set(string.punctuation)

    def transform_text(self, text: str) -> str:
        """
        Comprehensive text preprocessing:
        1. Lowercase conversion
        2. Tokenization
        3. Remove stopwords and punctuation
        4. Stemming
        """
        # Lowercase and tokenize
        tokens = nltk.word_tokenize(text.lower())

        # Advanced cleaning
        cleaned_tokens = [
            self.ps.stem(token)
            for token in tokens
            if token.isalnum()
            and token not in self.stop_words
            and token not in self.punctuation
        ]

        return " ".join(cleaned_tokens)


# Model Manager
class SpamClassificationModel:
    def __init__(self, vectorizer_path: str, model_path: str):
        self.vectorizer_path = vectorizer_path
        self.model_path = model_path
        self.tfidf = None
        self.model = None
        self.preprocessor = TextPreprocessor()

    def load_model(self) -> Tuple[bool, Optional[str]]:
        """
        Load TF-IDF vectorizer and spam classification model
        Returns success status and optional error message
        """
        try:
            with open(self.vectorizer_path, "rb") as f:
                self.tfidf = pickle.load(f)
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            return True, None
        except FileNotFoundError as e:
            return False, f"Model files not found: {str(e)}"
        except Exception as e:
            return False, f"Error loading model: {str(e)}"

    def predict(self, message: str) -> Tuple[int, float]:
        """
        Predict spam probability and classification
        Returns (prediction, probability)
        """
        if not self.model or not self.tfidf:
            raise ValueError("Model not loaded")

        # Preprocess and vectorize
        processed_text = self.preprocessor.transform_text(message)
        vectorized_text = self.tfidf.transform([processed_text])

        # Predict with probability
        prediction = self.model.predict(vectorized_text)[0]
        proba = self.model.predict_proba(vectorized_text)[0]
        spam_proba = proba[1] if prediction == 1 else proba[0]

        return prediction, spam_proba


# Helper function to create download link
def get_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'


# Streamlit App Configuration
def configure_app():
    """Sets up the Streamlit app with a custom favicon."""
    st.set_page_config(
        page_title="Spam Shield",
        page_icon="https://png.pngtree.com/png-vector/20220609/ourmid/pngtree-green-shield-icon-for-web-design-isolated-on-white-background-png-image_4839869.png",
        layout="wide",
    )

    # Custom CSS
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #121212;
            color: #E0E0E0;
        }
        .main-container {
            background-color: #1E1E1E;
            border-radius: 12px;
            padding: 30px;
            max-width: 800px;
            margin: 20px auto;
            box-shadow: 0 8px 24px rgba(0,0,0,0.3);
            border: 1px solid #333;
        }
        .title {
            color: #4CAF50;
            text-align: center;
            font-weight: 700;
            margin-bottom: 20px;
        }
        .stTextArea textarea {
            background-color: #2C2C2C;
            color: #E0E0E0;
            border: 2px solid #4CAF50;
            border-radius: 8px;
            font-size: 16px;
            padding: 15px;
            min-height: 250px;
        }
        .stButton > button {
            background-color: #4CAF50 !important;
            color: white !important;
            border-radius: 8px;
            font-weight: 600;
            border: none;
            padding: 12px 24px;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #45a049 !important;
            transform: scale(1.05);
            box-shadow: 0 4px 12px rgba(76, 175, 80, 0.3);
        }
        .result-container {
            margin-top: 20px;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            font-size: 18px;
            font-weight: 600;
        }
        .spam-result {
            background-color: rgba(244, 67, 54, 0.2);
            color: #F44336;
            border: 2px solid #F44336;
        }
        .not-spam-result {
            background-color: rgba(76, 175, 80, 0.2);
            color: #4CAF50;
            border: 2px solid #4CAF50;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )


def main():
    # App Configuration
    configure_app()

    # Model Initialization
    model_manager = SpamClassificationModel(
        vectorizer_path="vectorizer1.pkl", model_path="model1.pkl"
    )

    # Model Loading
    model_loaded, error_msg = model_manager.load_model()

    # Title
    st.markdown("<h1 class='title'>Spam Shield</h1>", unsafe_allow_html=True)

    # Error Handling for Model Loading
    if not model_loaded:
        st.error(f"Model Loading Error: {error_msg}")
        st.stop()

    # Tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Spam Detection", "Bulk Analysis", "Model Performance"])

    with tab1:
        # Single Message Detection
        input_sms = st.text_area(
            "Enter your message:",
            placeholder="Type or paste your SMS/Email here...",
            key="single_message",
        )

        col1, col2 = st.columns([3, 1])

        with col1:
            predict_clicked = st.button("Analyze Message", type="primary")

        # Prediction Logic
        if predict_clicked and input_sms.strip():
            try:
                prediction, probability = model_manager.predict(input_sms)

                result_class = "spam-result" if prediction == 1 else "not-spam-result"
                result_text = (
                    f"SPAM DETECTED! (Probability: {probability:.2%})"
                    if prediction == 1
                    else f"NOT SPAM (Confidence: {1-probability:.2%})"
                )

                st.markdown(
                    f"""
                    <div class="result-container {result_class}">
                        {result_text}
                    </div>
                """,
                    unsafe_allow_html=True,
                )

            except Exception as e:
                st.error(f"Prediction Error: {str(e)}")

    with tab2:
        # Bulk Analysis
        uploaded_file = st.file_uploader("Upload CSV with messages", type=["csv"])

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)

                # Validate DataFrame
                if "message" not in df.columns:
                    st.error("CSV must contain a 'message' column")
                    st.stop()

                # Bulk Prediction
                with st.spinner("Analyzing messages..."):
                    df["prediction"] = df["message"].apply(
                        lambda x: model_manager.predict(x)[0]
                    )
                    df["spam_probability"] = df["message"].apply(
                        lambda x: model_manager.predict(x)[1]
                    )

                # Display Results
                st.dataframe(df)

                # Download Link
                st.markdown(
                    get_download_link(
                        df, "spam_analysis_results.csv", "Download Analysis Results"
                    ),
                    unsafe_allow_html=True,
                )

            except Exception as e:
                st.error(f"Bulk Analysis Error: {str(e)}")

    with tab3:
        # Model Performance Metrics (Placeholder - you'd typically use test data)
        st.subheader("Model Performance Metrics")
        st.write("Note: These are placeholder metrics. Update with actual test data.")

        performance_data = {
            "Accuracy": 0.98,
            "Precision": 0.99,
            "Recall": 0.94,
            "F1 Score": 0.93,
        }

        for metric, value in performance_data.items():
            st.metric(metric, f"{value:.2%}")

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

# README.md
"""
# Spam Shield

## Overview
Spam Shield is an intelligent message classification application using machine learning.

## Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download NLTK resources: `python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"`
4. Run the app: `streamlit run app.py`

## Features
- Single message spam detection
- Bulk message analysis
- Model performance insights

"""
