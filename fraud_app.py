import streamlit as st
import pickle
import re
import matplotlib.pyplot as plt
import numpy as np

# 🔷 Page config
st.set_page_config(page_title="Fraud Detection", page_icon="🔍", layout="centered")

# 🔷 Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# 🔷 Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    return text

# 🔷 Explain prediction (Why Fraud)
def explain_prediction(text, vectorizer, model, top_n=10):
    cleaned = text.lower()
    X = vectorizer.transform([cleaned])
    feature_names = np.array(vectorizer.get_feature_names_out())

    # For linear models
    if hasattr(model, "coef_"):
        coefs = model.coef_[0]
        row = X.toarray()[0]
        contrib = row * coefs
        idx = np.argsort(np.abs(contrib))[-top_n:]
        return list(zip(feature_names[idx], contrib[idx]))

    # For tree models
    row = X.toarray()[0]
    idx = np.argsort(row)[-top_n:]
    return list(zip(feature_names[idx], row[idx]))

# 🔥 Custom CSS (Premium UI)
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
h1 {
    color: #00ADB5;
}
.stButton>button {
    background-color: #00ADB5;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# 🔷 Header
st.markdown("<h1 style='text-align:center;'>💼 Fraud Job Detection System</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center;'>🔍 AI-based Fake Job Detection</h3>", unsafe_allow_html=True)

# 🔷 Input
text = st.text_area("📄 Enter Job Description", height=200)

# 🔷 Prediction Button
if st.button("🚀 Analyze Job"):

    if text.strip() == "":
        st.warning("⚠️ Please enter job description")
    else:
        cleaned = clean_text(text)
        vec = vectorizer.transform([cleaned]).toarray()

        pred = model.predict(vec)
        prob = model.predict_proba(vec)[0][1]

        st.markdown("---")

        # 🔥 Result + Confidence (2 columns)
        col1, col2 = st.columns(2)

        with col1:
            if pred[0] == 1:
                st.error("🚨 Fraud Job")
            else:
                st.success("✅ Genuine Job")

        with col2:
            st.info(f"Confidence: {prob*100:.2f}%")

        # 🔥 Progress bar
        st.progress(float(prob))

        # 🔥 Risk Level
        if prob > 0.75:
            st.error("🔴 High Risk")
        elif prob > 0.4:
            st.warning("🟡 Medium Risk")
        else:
            st.success("🟢 Low Risk")

        # 🔥 Graph
        st.markdown("### 📊 Prediction Visualization")
        fig, ax = plt.subplots()
        ax.bar(["Genuine", "Fraud"], [1-prob, prob])
        ax.set_ylabel("Probability")
        st.pyplot(fig)

        # 🔥 Explanation (Why Fraud)
        st.markdown("### 🔍 Why this prediction?")
        explanations = explain_prediction(text, vectorizer, model)

        for word, val in explanations:
            if val > 0:
                st.write(f"🔺 {word} → increases fraud likelihood")
            else:
                st.write(f"🔻 {word} → indicates genuine")

        # 🔥 Smart Insight
        st.markdown("### 🧠 Smart Insight")
        if prob > 0.75:
            st.write("⚠️ High risk – Avoid this job")
        elif prob > 0.4:
            st.write("🤔 Suspicious – Verify carefully")
        else:
            st.write("✅ Likely safe")

# 🔷 Footer
st.markdown("---")
st.markdown("<p style='text-align:center;'>Made with ❤️ | Machine Learning Project</p>", unsafe_allow_html=True)