import streamlit as st
import pickle
import re
import numpy as np
import os

# 🔷 Page config
st.set_page_config(page_title="Fraud Detection", page_icon="🔍", layout="centered")

# 🔥 🎨 CUSTOM UI (NEXT LEVEL)
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}

/* Title */
.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #00ADB5;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #cbd5e1;
}

/* Card */
.card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
    margin-top: 20px;
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg, #00ADB5, #22c55e);
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    border: none;
}

/* Text area */
textarea {
    border-radius: 10px !important;
}

</style>
""", unsafe_allow_html=True)

# 🔷 Safe file loading
BASE_DIR = os.path.dirname(__file__)

model_path = os.path.join(BASE_DIR, "model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")

try:
    model = pickle.load(open(model_path, "rb"))
    vectorizer = pickle.load(open(vectorizer_path, "rb"))
except:
    st.error("❌ Model files not found! Check GitHub upload.")
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

# 🔷 HEADER
st.markdown("<div class='title'>💼 Fraud Job Detection System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>🔍 AI-powered Fake Job Detection</div>", unsafe_allow_html=True)

# 🔷 INPUT CARD
st.markdown("<div class='card'>", unsafe_allow_html=True)
text = st.text_area("📄 Enter Job Description", height=200)
st.markdown("</div>", unsafe_allow_html=True)

# 🔷 BUTTON
if st.button("🚀 Analyze Job"):

    if text.strip() == "":
        st.warning("⚠️ Please enter job description")
    else:
        cleaned = clean_text(text)
        vec = vectorizer.transform([cleaned]).toarray()

        pred = model.predict(vec)
        prob = model.predict_proba(vec)[0][1]

        # 🔷 RESULT CARD
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            if pred[0] == 1:
                st.error("🚨 Fraud Job")
            else:
                st.success("✅ Genuine Job")

        with col2:
            st.metric("Confidence", f"{prob*100:.2f}%")

        st.progress(float(prob))

        st.markdown("</div>", unsafe_allow_html=True)

        # 🔷 RISK CARD
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        if prob > 0.75:
            st.error("🔴 High Risk - Avoid this job")
        elif prob > 0.4:
            st.warning("🟡 Medium Risk - Be careful")
        else:
            st.success("🟢 Low Risk - Looks safe")

        st.markdown("</div>", unsafe_allow_html=True)

        # 🔷 EXPLANATION CARD
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        st.markdown("### 🔍 Why this prediction?")
        explanations = explain_prediction(text, vectorizer, model)

        for word, val in explanations:
            if val > 0:
                st.write(f"🔺 **{word}** increases fraud risk")
            else:
                st.write(f"🔻 **{word}** indicates genuine")

        st.markdown("</div>", unsafe_allow_html=True)

# 🔷 FOOTER
st.markdown("""
<hr>
<p style='text-align:center;color:gray;'>
Made with ❤️ | Machine Learning Project | Streamlit App
</p>
""", unsafe_allow_html=True)
