import streamlit as st
import joblib
import re, string
import nltk

# -------------------
# Load Model & Vectorizer
# -------------------
model = joblib.load("best_model.pkl")   # Logistic Regression
tfidf = joblib.load("tfidf_vectorizer.pkl")

MODEL_NAME = "Logistic Regression"

# -------------------
# Drug Suggestions
# -------------------
drug_suggestions = {
    "Depression": ["Fluoxetine", "Sertraline", "Escitalopram", "Venlafaxine", "Desvenlafaxine"],
    "Diabetes, Type 2": ["Metformin", "Gliclazide", "Sitagliptin", "Empagliflozin", "Jardiance"],
    "High Blood Pressure": ["Amlodipine", "Lisinopril", "Losartan", "Prazosin", "Hydrochlorothiazide"]
}

# -------------------
# Preprocessing (MATCHES NOTEBOOK)
# -------------------
def preprocess(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'[‘’“”…]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -------------------
# Page Config
# -------------------
st.set_page_config(
    page_title="Drug Review Condition Predictor",
    page_icon="💊",
    layout="wide"
)

# -------------------
# Sidebar Navigation
# -------------------
st.sidebar.title("Navigation")

if "page" not in st.session_state:
    st.session_state.page = "Overview"

page = st.sidebar.radio(
    "Go to",
    ["Overview", "Review Input", "Prediction & Drugs"],
    index=["Overview", "Review Input", "Prediction & Drugs"].index(st.session_state.page)
)

st.session_state.page = page

# =====================================================
# PAGE 1: OVERVIEW
# =====================================================
if page == "Overview":
    st.markdown(
        "<h1 style='text-align:center; color:#4B8BBE;'>💊 Drug Review Condition Predictor</h1>",
        unsafe_allow_html=True
    )

    st.markdown(f"""
    **Welcome!**

    This app predicts the **medical condition** from a drug review using  
    **NLP (TF-IDF + {MODEL_NAME})**.

    **Supported conditions:**
    - Depression  
    - Diabetes (Type 2)  
    - High Blood Pressure  

    👉 Go to **Review Input** to get started.
    """)

    st.image(
        "https://images.unsplash.com/photo-1588776814546-0f1e2f4ee801",
        use_column_width=True
    )

# =====================================================
# PAGE 2: REVIEW INPUT + PREDICTION
# =====================================================
elif page == "Review Input":
    st.markdown(
        "<h2 style='color:#FF4B4B;'>Step 1: Enter Your Review</h2>",
        unsafe_allow_html=True
    )

    review_input = st.text_area(
        "Enter your drug review:",
        height=200
    )

    st.info("Tip: Natural, real-world reviews give best predictions.")

    if st.button("Predict Condition & Go"):
        if review_input.strip() == "":
            st.warning("Please enter a review before predicting.")
        else:
            cleaned_review = preprocess(review_input)
            review_vec = tfidf.transform([cleaned_review])
            prediction = model.predict(review_vec)[0]

            st.session_state.review_input = review_input
            st.session_state.prediction = prediction
            st.session_state.page = "Prediction & Drugs"

            st.rerun()

# =====================================================
# PAGE 3: RESULTS (AUTO DISPLAY)
# =====================================================
elif page == "Prediction & Drugs":
    st.markdown(
        "<h2 style='color:#4CAF50;'>Prediction Result</h2>",
        unsafe_allow_html=True
    )

    if "prediction" not in st.session_state:
        st.warning("Please enter a review first.")
    else:
        prediction = st.session_state.prediction
        review_input = st.session_state.review_input

        st.success(f"🩺 **Predicted Condition:** {prediction}")

        drugs = drug_suggestions.get(prediction, [])
        if drugs:
            st.info(f"💊 **Suggested Drugs:** {', '.join(drugs)}")
        else:
            st.info("No drug suggestions available.")

        st.markdown("### 📝 Entered Review")
        st.write(review_input)

        if st.button("🔄 Try Another Review"):
            for key in ["prediction", "review_input"]:
                st.session_state.pop(key, None)
            st.session_state.page = "Review Input"
            st.rerun()
