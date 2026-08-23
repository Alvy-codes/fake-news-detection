"""
Fake News Detection - Streamlit App (Enhanced UI)
Silicon Delta Innovation Hub DS Final Project
Model: Logistic Regression (TF-IDF + Chi-square feature selection)
"""

import streamlit as st
import joblib
import re
import html
import matplotlib.pyplot as plt
import numpy as np

# ---------------- Page config (must be first st. call) ----------------
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

# ---------------- Custom styling ----------------
st.markdown("""
<style>
    .main { background-color: #F7F9FB; }
    .hero {
        background: linear-gradient(135deg, #1B263B 0%, #2E4057 100%);
        padding: 2.2rem 2rem;
        border-radius: 14px;
        margin-bottom: 1.8rem;
    }
    .hero h1 { color: white; font-size: 2.1rem; margin-bottom: 0.3rem; }
    .hero p { color: #A9B4C0; font-size: 1.05rem; margin: 0; }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.2rem;
    }
    .example-card {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem;
    }
    div[data-testid="stForm"] {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- Text cleaning (must match training pipeline) ----------------
def strip_dateline(text):
    return re.sub(r'^[A-Z][A-Za-z\.\s]{0,40}\(Reuters\)\s*-\s*', '', text)

def clean_text(text):
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = strip_dateline(text)
    text = re.sub(r'[^A-Za-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

# ---------------- Load model artifacts ----------------
@st.cache_resource
def load_artifacts():
    tfidf = joblib.load('tfidf_vectorizer.joblib')
    selector = joblib.load('chi2_selector.joblib')
    model = joblib.load('best_model.joblib')
    return tfidf, selector, model

tfidf, selector, model = load_artifacts()

# ---------------- Gauge chart ----------------
def draw_gauge(proba_fake):
    fig, ax = plt.subplots(figsize=(4, 2.3), subplot_kw={'aspect': 'equal'})
    fig.patch.set_alpha(0)

    theta = np.linspace(np.pi, 0, 100)
    colors_bg = plt.cm.RdYlGn_r(np.linspace(0.15, 0.9, 100))
    for i in range(len(theta) - 1):
        ax.plot(np.cos(theta[i:i+2]), np.sin(theta[i:i+2]), color=colors_bg[i], linewidth=14, solid_capstyle='butt')

    needle_angle = np.pi * (1 - proba_fake)
    ax.plot([0, 0.75*np.cos(needle_angle)], [0, 0.75*np.sin(needle_angle)], color='#1B263B', linewidth=3)
    ax.scatter([0], [0], color='#1B263B', s=60, zorder=5)

    ax.text(0, -0.35, f"{proba_fake*100:.1f}%", ha='center', va='center', fontsize=26, fontweight='bold', color='#1B263B')
    ax.text(0, -0.62, "probability FAKE", ha='center', va='center', fontsize=10, color='#6C7A89')

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-0.75, 1.15)
    ax.axis('off')
    return fig

# ---------------- Session state for example pre-fill ----------------
if 'title_input' not in st.session_state:
    st.session_state.title_input = ""
if 'body_input' not in st.session_state:
    st.session_state.body_input = ""

def load_fake_example():
    st.session_state.title_input = "WATCH: Hillary Caught on Camera Doing Something SHOCKING"
    st.session_state.body_input = "You wont believe what happened next featured image via getty. Share this if you agree!"

def load_real_example():
    st.session_state.title_input = "Senate passes budget bill in bipartisan vote"
    st.session_state.body_input = ("WASHINGTON (Reuters) - The U.S. Senate voted on Wednesday to approve a new "
                                    "budget bill, according to congressional aides.")

def clear_inputs():
    st.session_state.title_input = ""
    st.session_state.body_input = ""

# ---------------- Hero header ----------------
st.markdown("""
<div class="hero">
    <h1>📰 Fake News Detector</h1>
    <p>NAME: ALVERNIA OJO &nbsp;·&nbsp; DATA SCIENCE PROJECT &nbsp;·&nbsp; MODEL: LOGISTICS REGRESSION (TF-IDF)</p>
</div>
""", unsafe_allow_html=True)

col_main, col_side = st.columns([2, 1])

with col_main:
    st.markdown("#### Try it out")
    ex1, ex2, ex3 = st.columns(3)
    with ex1:
        st.button("📰 Load real-news example", on_click=load_real_example, use_container_width=True)
    with ex2:
        st.button("🚨 Load fake-news example", on_click=load_fake_example, use_container_width=True)
    with ex3:
        st.button("🗑️ Clear", on_click=clear_inputs, use_container_width=True)

    with st.form("prediction_form"):
        title = st.text_input("Article Title", key="title_input", placeholder="e.g. Senate passes new budget bill")
        body = st.text_area("Article Text", key="body_input", height=200, placeholder="Paste the article body here...")
        submitted = st.form_submit_button("🔍 Analyze Article", use_container_width=True, type="primary")

    if submitted:
        if not title.strip() and not body.strip():
            st.warning("Please enter a title or article text to analyze.")
        else:
            full_text = clean_text(title) + " " + clean_text(body)
            X_vec = tfidf.transform([full_text])
            X_sel = selector.transform(X_vec)

            proba_fake = model.predict_proba(X_sel)[0][1]
            pred = model.predict(X_sel)[0]

            st.divider()
            res_col1, res_col2 = st.columns([1, 1.3])

            with res_col1:
                fig = draw_gauge(proba_fake)
                st.pyplot(fig, use_container_width=True)

            with res_col2:
                st.markdown("<br>", unsafe_allow_html=True)
                if pred == 1:
                    st.error(f"⚠️ **Likely FAKE NEWS**\n\nThe model predicts a **{proba_fake*100:.2f}% probability** "
                             f"that this article is fake news.")
                else:
                    st.success(f"✅ **Likely REAL NEWS**\n\nThe model predicts a **{proba_fake*100:.2f}% probability** "
                               f"that this article is fake news (i.e. {(1-proba_fake)*100:.2f}% likely real).")

                with st.expander("How to read this"):
                    st.write(
                        "This model was trained on articles from politics / world news categories "
                        "(2015-2018). It flags linguistic patterns common in tabloid-style, "
                        "sensationalized content (e.g. heavy use of 'watch', 'featured image', "
                        "vote-baiting phrasing) versus wire-service style reporting (e.g. attributive "
                        "'said', dateline structure). It is a decision-support tool, not a ground-truth "
                        "fact-checker — always verify claims against primary sources."
                    )

with col_side:
    st.markdown("#### About this model")
    st.markdown("""
    <div class="example-card">
    <b>Dataset:</b> ~38,600 labeled political news articles (2015–2018)<br><br>
    <b>Pipeline:</b> TF-IDF (unigrams + bigrams) → Chi-square feature selection → Logistic Regression<br><br>
    <b>Test performance:</b><br>
    • F1-score: <b>0.981</b><br>
    • AUC-ROC: <b>0.998</b><br>
    • Accuracy: <b>98.3%</b><br><br>
    <b>Key signals learned:</b><br>
    Fake → "featured image", "video", "watch", "pic twitter"<br>
    Real → "said", weekday names, "reuters"
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Built as part of the Data Science capstone at Silicon Delta Innovation Hub. "
               "Supervisor: Constance Okere.")
