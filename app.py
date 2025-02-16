import streamlit as st
import pickle
import string
import nltk
import base64
import os
from nltk.stem.porter import PorterStemmer


# Set up NLTK data directory
NLTK_DIR = os.path.expanduser("~/.nltk_data")  
nltk.data.path.append(NLTK_DIR)  

# Download required resources if not available
nltk.download('punkt', download_dir=NLTK_DIR)
nltk.download('stopwords', download_dir=NLTK_DIR)
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
# Initialize Porter Stemmer
ps = PorterStemmer()

# Load stopwords once to optimize processing
stop_words = set(stopwords.words('english'))

# Load model & vectorizer
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    st.error("🚨 Model or vectorizer file not found! Please check your directory.")
    st.stop()

# Function to encode background image
def get_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        return ""

# Encode background image (Ensure "bg.jpg" exists)
base64_image = get_base64("bg.jpg")

# Custom Styling
st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{base64_image}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    .container {{
        max-width: 700px;
        margin: auto;
        padding: 20px;
    }}
    .title {{
        background: linear-gradient(to right, #ff416c, #ff4b2b);
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        color: #ffffff;
        padding: 10px;
        text-shadow: 3px 3px 6px rgba(0, 0, 0, 1);
    }}
    .stTextArea textarea, .stTextInput input {{
        border-radius: 10px;
        border: 2px solid #ff6f61;
        caret-color: black !important;
        padding: 12px;
        font-size: 16px;
        color: #000;
        font-weight: bold;
        background: rgba(255, 255, 255, 0.95);
    }}
    .stButton button {{
        background: linear-gradient(to right, #ff416c, #ff4b2b);
        color: white;
        border-radius: 20px;
        padding: 12px 25px;
        font-size: 18px;
        font-weight: bold;
        transition: 0.3s;
        box-shadow: 2px 2px 10px rgba(255, 75, 43, 0.6);
    }}
    .stButton button:hover {{
        transform: scale(1.1);
        box-shadow: 2px 2px 15px rgba(255, 75, 43, 0.9);
    }}
    .safe-container {{
        background-color: green;
        color: yellow;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }}
    </style>
""", unsafe_allow_html=True)

# Function to preprocess text
def transform_text(text):
    text = text.lower()
    words = word_tokenize(text)

    # Remove stopwords, punctuations, and apply stemming
    filtered_words = [
        ps.stem(word) for word in words if word.isalnum() and word not in stop_words
    ]

    return " ".join(filtered_words)

# App Layout
st.markdown("<h1 class='title'>📩 Pankaj Singh SMS/Email Spam Predictor 🚀</h1>", unsafe_allow_html=True)

with st.container():
    st.markdown("<div class='container'>", unsafe_allow_html=True)
    
    st.markdown("<h4 style='color: red; text-align: center; background-color:black'>📩 Enter your message:</h4>", unsafe_allow_html=True)

    # Input field
    input_sms = st.text_area("", height=120, key="input_sms")

    # Predict Button
    if st.button("🔍 Click to Predict", help="Analyze the SMS for spam detection"):
        if not input_sms.strip():
            st.warning("⚠️ Please enter a message before predicting!")
        else:
            with st.spinner("🕵️‍♂️ Analyzing your message..."):
                transformed_sms = transform_text(input_sms)
                vector_input = tfidf.transform([transformed_sms])
                result = model.predict(vector_input)[0]
                spam_prob = model.predict_proba(vector_input)[0][1]

                # Display Result
                result_class = "spam-container" if result == 1 else "safe-container"
                result_text = "🚨 This is SPAM! 🚨" if result == 1 else "✅ This is NOT Spam ✅"
                probability = f"{spam_prob * 100:.2f}%"

                st.markdown(f"""
                    <div class='{result_class}'>
                        <h2>{result_text}</h2>
                        <p>Spam Probability: {probability}</p>
                    </div>
                """, unsafe_allow_html=True)

                # Show Confidence Level with Progress Bar
                st.progress(float(spam_prob))
