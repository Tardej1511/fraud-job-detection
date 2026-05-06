import streamlit as st
import pickle
import re
import numpy as np
import os

# 🔷 Page config
st.set_page_config(page_title="Fraud Detection", page_icon="🔍", layout="centered")

# 🔷 Safe file loading
BASE_DIR = os.path.dirname(__file__)

model_path = os.path.join(BASE_DIR, "model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")

# 🔷 Load with error handling
try:
    model = pickle.load(open(model_path, "rb"))
    vectorizer = pickle.load(open(vectorizer_path, "rb"))
except FileNotFoundError:
    st.error("❌ Model files not found! Please upload model.pkl and vectorizer.pkl to GitHub.")
    st.stop()

# 🔷 Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    return text

# 🔷 Explain prediction
def explain_prediction(text, vectorizer, model, top_n=10):
    cleaned = text.lower()
    X = vectorizer.transform([cleaned])
    feature_names = np.array(vectorizer.get_feature_names_out())

    if hasattr(model, "coef_"):
        coefs = model.coef_[0]
        row = X.toarray()[0]
        contrib = row * coefs
        idx = np.argsort(np.abs(contrib))[-top_n:]
        return list(zip(feature_names[idx], contrib[idx]))

    row = X.toarray()[0]
    idx = np.argsort(row)[-top_n:]
    return list(zip(feature_names[idx], row[idx]))

# 🔷 UI
st.markdown("<h1 style='text-align:center;color:#00ADB5;'>💼 Fraud Job Detection System</h1>", unsafe_allow_html=True)
st.markdown("### 🔍 AI-based Fake Job Detection")

text = st.text_area("📄 Enter Job Description", height=200)

# 🔷 Prediction
if st.button("🚀 Analyze Job"):

    if text.strip() == "":
        st.warning("⚠️ Please enter job description")
    else:
        cleaned = clean_text(text)
        vec = vectorizer.transform([cleaned]).toarray()

        pred = model.predict(vec)
        prob = model.predict_proba(vec)[0][1]

        st.markdown("---")

        # RESULT
        if pred[0] == 1:
            st.error("🚨 Fraud Job")
        else:
            st.success("✅ Genuine Job")

        # 📊 Score (NO matplotlib)
        st.markdown("### 📊 Prediction Score")
        st.metric("Fraud Probability", f"{prob*100:.2f}%")
        st.progress(float(prob))

        # Risk
        if prob > 0.75:
            st.error("🔴 High Risk")
        elif prob > 0.4:
            st.warning("🟡 Medium Risk")
        else:
            st.success("🟢 Low Risk")

        # 🔍 Explanation
        st.markdown("### 🔍 Why this prediction?")
        explanations = explain_prediction(text, vectorizer, model)

        for word, val in explanations:
            if val > 0:
                st.write(f"🔺 {word} → increases fraud likelihood")
            else:
                st.write(f"🔻 {word} → indicates genuine")

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center;'>Made with ❤️ | ML Project</p>", unsafe_allow_html=True)
