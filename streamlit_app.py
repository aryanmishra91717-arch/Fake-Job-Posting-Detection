
import streamlit as st
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).parent

@st.cache_resource
def load_model():
    model = joblib.load(BASE_DIR / "fake_job_model.pkl")
    vectorizer = joblib.load(BASE_DIR / "tfidf_vectorizer.pkl")
    return model, vectorizer

st.set_page_config(
    page_title="Fake Job Detection",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 Fake Job Posting Detection")
st.subheader("AI-Powered Job Verification System")

st.write(
    "Enter a job title and description to check "
    "whether the job posting is potentially fraudulent."
)

try:
    model, vectorizer = load_model()

    title = st.text_input(
        "Job Title",
        placeholder="e.g. Software Engineer"
    )

    description = st.text_area(
        "Job Description",
        placeholder="Paste the complete job description here...",
        height=200
    )

    if st.button("CHECK JOB", use_container_width=True):
        if not title.strip() or not description.strip():
            st.warning("Please enter both job title and description.")
        else:
            job_text = title.strip() + " " + description.strip()
            features = vectorizer.transform([job_text])
            prediction = model.predict(features)[0]

            if str(prediction) == "1":
                st.error("⚠️ POTENTIALLY FRAUDULENT JOB")
            else:
                st.success("✅ JOB APPEARS REAL")

            st.caption(
                "This is an ML prediction, not a guarantee. "
                "Always verify employers independently."
            )

except Exception as error:
    st.error(f"Unable to load the model: {error}")
