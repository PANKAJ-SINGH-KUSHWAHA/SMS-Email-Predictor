import streamlit as st
import pickle
import string
import nltk
import base64
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
# Download necessary NLTK data
import nltk
import os

NLTK_DIR = os.path.expanduser("~/.nltk_data")  # Set custom path
nltk.data.path.append(NLTK_DIR)  # Ensure nltk looks here

# Download 'punkt' to the custom directory
nltk.download("punkt", download_dir=NLTK_DIR)
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize


# Initialize Stemmer
ps = PorterStemmer()

import nltk
import os

# Set up NLTK data directory
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

# Set the NLTK data path
nltk.data.path.append(nltk_data_path)

# Download required NLTK resources
nltk.download('punkt', download_dir=nltk_data_path)

# Now your tokenizer will work without lookup errors


# Load vectorizer and model with error handling
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    st.error("🚨 Model or vectorizer file not found! Please check your directory.")
    st.stop()

# Preload stopwords once for efficiency
stop_words = set(stopwords.words('english'))

# Function to encode background image
def get_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        st.warning("⚠️ Background image not found. Using default styling.")
        return ""

# Encode background image (Ensure "bg.jpg" exists in your directory)
base64_image = get_base64("bg.jpg")

# Custom CSS for Background & Styling
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

    label {{
        font-size: 30px !important;
        font-weight: bold !important;
        color: white !important;
        text-shadow: 3px 3px 6px rgba(0, 0, 0, 1);
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

    .spam-container {{
        background-color: #e74c3c;
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }}

    .safe-container {{
        background-color: #2ecc71;
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }}

    .stProgress > div > div {{
        border-radius: 10px;
        height: 12px;
    }}
    </style>
    """, unsafe_allow_html=True)


# Function to preprocess text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# App Layout
st.markdown("<h1 class='title'>📩 Pankaj Singh SMS/Email Spam Predictor Service 🚀</h1>", unsafe_allow_html=True)

with st.container():
    st.markdown("<div class='container'>", unsafe_allow_html=True)
    
    st.markdown("<h4 style='color: red; text-align: center; background-color:black'>📩 Enter your message:</h2>", unsafe_allow_html=True)

    # Input field without default label (to prevent overlap)
    input_sms = st.text_area("", height=120, key="input_sms")
    
    # Predict Button
    if st.button("🔍 Click to Predict", help="Analyze the SMS for spam detection"):
        if not input_sms.strip():  # Check if input is empty
            st.warning("⚠️ Please enter a message before predicting!")
        else:
            if "last_input" not in st.session_state or st.session_state.last_input != input_sms:
                with st.spinner("🕵️‍♂️ Analyzing your message..."):
                    transformed_sms = transform_text(input_sms)
                    vector_input = tfidf.transform([transformed_sms])
                    result = model.predict(vector_input)[0]
                    spam_prob = model.predict_proba(vector_input)[0][1]  

                    # Store result in session state
                    st.session_state.last_input = input_sms
                    st.session_state.result = result
                    st.session_state.spam_prob = spam_prob

            # Display stored result inside a styled div
            if "result" in st.session_state:
                result_class = "spam-container" if st.session_state.result == 1 else "safe-container"
                result_text = "🚨 This is SPAM! 🚨" if st.session_state.result == 1 else "✅ This is NOT Spam ✅"
                probability = f"{st.session_state.spam_prob * 100:.2f}%"
                
                st.markdown(f"""
                    <div class='{result_class}'>
                        <h2>{result_text}</h2>
                        <p>Spam Probability: {probability}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show Confidence Level with Progress Bar
                st.progress(float(st.session_state.spam_prob))