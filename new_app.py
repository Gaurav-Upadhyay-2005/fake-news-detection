import streamlit as st
import joblib
import numpy as np
import pickle
import re
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# NLTK DATA DOWNLOAD
# ============================================================================
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Fake News Detection System v3.0",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM STYLING
# ============================================================================
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0a0e27 0%, #16213e 100%);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a2f4a 0%, #0f1d2d 100%);
    }
    h1, h2, h3 {
        color: #00d4ff;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0, 212, 255, 0.2);
    }
    body { color: #e0e0e0; }
    .stButton > button {
        background: linear-gradient(90deg, #00d4ff 0%, #0099cc 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0, 212, 255, 0.4);
    }
    [data-testid="metric-container"] {
        background: rgba(0, 212, 255, 0.05);
        border-left: 4px solid #00d4ff;
        padding: 15px;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab-list"] button { color: #00d4ff; }
    /* PDF download buttons */
    .stDownloadButton > button {
        background: linear-gradient(90deg, #00d4ff 0%, #0099cc 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    .stDownloadButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 16px rgba(0, 212, 255, 0.4) !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LABEL ENCODING - EXACT FROM TRAINING
# ============================================================================
"""
⚠️ IMPORTANT - LABEL ENCODING (From Training):
    0 = REAL NEWS ✅
    1 = FAKE NEWS 🚨
"""

# ============================================================================
# LOAD ML MODELS
# ============================================================================
@st.cache_resource
def load_ml_models():
    try:
        current_dir     = os.path.dirname(os.path.abspath(__file__))
        model_dir       = os.path.join(current_dir, "V2_Saved_Models_ML")

        if not os.path.exists(model_dir):
            return {"status": "error", "message": f"Model directory not found at: {model_dir}"}

        lr_path         = os.path.join(model_dir, "logistic_regression.pkl")
        svc_path        = os.path.join(model_dir, "svc.pkl")
        voting_path     = os.path.join(model_dir, "voting_classifier_86acc.pkl")
        vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")

        for path, name in [
            (lr_path,         "Logistic Regression"),
            (svc_path,        "SVC"),
            (voting_path,     "Voting Classifier"),
            (vectorizer_path, "TF-IDF Vectorizer"),
        ]:
            if not os.path.exists(path):
                return {"status": "error", "message": f"{name} not found at: {path}"}

        return {
            "lr":         joblib.load(lr_path),
            "svc":        joblib.load(svc_path),
            "voting":     joblib.load(voting_path),
            "vectorizer": joblib.load(vectorizer_path),
            "status":     "success",
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ============================================================================
# LOAD ALL MODELS
# ============================================================================
ml_dict = load_ml_models()

# ML must succeed
if ml_dict.get("status") == "error":
    st.error(f"❌ Model Loading Error: {ml_dict.get('message')}")
    st.info("""
    **Make sure your folder structure is correct:**
    ```
    FAKE NEWS PROJECT FINAL/
    ├── app.py
    └── V2_Saved_Models_ML/
        ├── logistic_regression.pkl
        ├── svc.pkl
        ├── tfidf_vectorizer.pkl
        └── voting_classifier_86acc.pkl
    ```
    """)
    st.stop()

lr_model     = ml_dict["lr"]
svc_model    = ml_dict["svc"]
voting_model = ml_dict["voting"]
vectorizer   = ml_dict["vectorizer"]

st.sidebar.success("✅ All ML models loaded successfully!")

# ============================================================================
# MODEL LIST
# ============================================================================
ALL_MODELS = ["Logistic Regression", "SVC", "Voting Classifier"]

# ============================================================================
# TEXT PREPROCESSING
# ============================================================================
class MLPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def preprocess_text(self, text):
        cleaned          = self.clean_text(text)
        tokens           = word_tokenize(cleaned)
        orig_count       = len(tokens)
        processed        = [self.lemmatizer.lemmatize(t)
                            for t in tokens
                            if t not in self.stop_words and len(t) > 2]
        return {
            "text":              ' '.join(processed),
            "final_tokens":      len(processed),
            "original_tokens":   orig_count,
            "stopwords_removed": orig_count - len(processed),
        }


ml_preprocessor = MLPreprocessor()

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def predict_logistic_regression(text):
    prep   = ml_preprocessor.preprocess_text(text)
    vector = vectorizer.transform([prep["text"]])
    proba  = lr_model.predict_proba(vector)[0]
    pred   = int(lr_model.predict(vector)[0])
    return {"class": pred, "prob_real": float(proba[0]), "prob_fake": float(proba[1]),
            "confidence": float(max(proba) * 100), "prep_info": prep}


def predict_svc(text):
    prep   = ml_preprocessor.preprocess_text(text)
    vector = vectorizer.transform([prep["text"]])
    proba  = svc_model.predict_proba(vector)[0]          # CalibratedClassifierCV
    pred   = int(svc_model.predict(vector)[0])
    return {"class": pred, "prob_real": float(proba[0]), "prob_fake": float(proba[1]),
            "confidence": float(max(proba) * 100), "prep_info": prep}


def predict_voting_classifier(text):
    prep   = ml_preprocessor.preprocess_text(text)
    vector = vectorizer.transform([prep["text"]])
    try:
        proba = voting_model.predict_proba(vector)[0]
    except Exception:
        p     = int(voting_model.predict(vector)[0])
        proba = np.array([1 - p, p])
    pred = int(voting_model.predict(vector)[0])
    return {"class": pred, "prob_real": float(proba[0]), "prob_fake": float(proba[1]),
            "confidence": float(max(proba) * 100), "prep_info": prep}


# ============================================================================
# PDF HELPER — load sample PDFs for download buttons
# ============================================================================
def _load_pdf(filename):
    """Load a PDF file from the same directory as app.py"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path        = os.path.join(current_dir, filename)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    return None

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================
st.sidebar.markdown("# ⚙️ Configuration")
st.sidebar.markdown("---")

selected_model = st.sidebar.selectbox(
    "🤖 Select Model",
    ALL_MODELS,
    help="Choose which trained model to use for prediction"
)

st.sidebar.markdown("---")

with st.sidebar.expander("ℹ️ Model Information"):
    st.markdown("""
    ### Trained Models Overview
    
    **Label Encoding** (Training Setup):
    - `0` = REAL NEWS ✅
    - `1` = FAKE NEWS 🚨
    
    **ML Models**:
    1. **Logistic Regression** — Test Acc: 85.50%
    2. **SVC** — Test Acc: 85.60% (Best generalization)
    3. **Voting Classifier** — Test Acc: 86.09% (Highest ML)
    
    **ML Vectorization**:
    - TF-IDF with max 5000 features
    - Trained on 166,418 articles
    - Train/Test split: 80-20
    
    **Threshold Logic**:
    - ≥ 60% Real → ✅ REAL NEWS
    - ≥ 60% Fake → 🚨 FAKE NEWS
    - Both < 60% → ⚠️ Verify from other sources
    """)

st.sidebar.markdown("---")

with st.sidebar.expander("📊 Preprocessing Pipeline"):
    st.markdown("""
    ### ML Models (LR, SVC, Voting):
    1. **Lowercase** text
    2. **Remove URLs, Emails**
    3. **Remove Special Chars & Numbers**
    4. **Tokenize** (NLTK)
    5. **Remove Stopwords**
    6. **Lemmatize**
    7. **TF-IDF Vectorize** (5000 features)
    """)

st.sidebar.markdown("---")
st.sidebar.text("📜 Version: 3.0")
st.sidebar.text("👨‍💻 Developed by: Gaurav Upadhyay")

# ============================================================================
# MAIN CONTENT
# ============================================================================
st.title("📰 Fake News Detection System v3.0")
st.markdown("**Comprehensive ML News Classification**")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Prediction",
    "📊 Analytics",
    "📚 Documentation",
    "⚙️ Advanced"
])

# ============================================================================
# TAB 1: PREDICTION
# ============================================================================
with tab1:
    st.subheader("📝 Analyze News Content")

    col_input1, col_input2 = st.columns([1.5, 1])

    with col_input1:
        title = st.text_input(
            "📌 News Title",
            placeholder="Enter the news headline or title...",
            help="The headline of the news article"
        )
        article = st.text_area(
            "📄 News Article",
            placeholder="Enter the full news content or body...",
            height=200,
            help="The main body of the news article"
        )

    with col_input2:
        st.markdown("### 💡 Quick Actions")

        # ── FAKE NEWS PDF download ──────────────────────────────────────────
        fake_pdf = _load_pdf("fake_news_samples.pdf")
        if fake_pdf:
            st.download_button(
                label="📌 Load Fake Example",
                data=fake_pdf,
                file_name="fake_news_samples.pdf",
                mime="application/pdf",
                use_container_width=True,
                help="Download 10 fake news examples. Copy any title+text into the fields and predict."
            )
        else:
            st.warning("⚠️ fake_news_samples.pdf not found in project folder")

        # ── REAL NEWS PDF download ──────────────────────────────────────────
        real_pdf = _load_pdf("real_news_samples.pdf")
        if real_pdf:
            st.download_button(
                label="✓ Load Real Example",
                data=real_pdf,
                file_name="real_news_samples.pdf",
                mime="application/pdf",
                use_container_width=True,
                help="Download 10 real news examples. Copy any title+text into the fields and predict."
            )
        else:
            st.warning("⚠️ real_news_samples.pdf not found in project folder")

        if st.button("🗑️ Clear All", use_container_width=True):
            for k in ["ex_title", "ex_article"]:
                st.session_state.pop(k, None)
            st.rerun()

        st.markdown("---")
        st.info("📥 Download a PDF, pick any news, paste title & text above, then click Analyze")

    # Load session examples if set
    if "ex_title" in st.session_state:
        title = st.session_state["ex_title"]
    if "ex_article" in st.session_state:
        article = st.session_state["ex_article"]

    st.markdown("---")

    if st.button("🚀 Analyze News", use_container_width=True, type="primary"):

        if not title or not article:
            st.error("⚠️ Error: Please enter both title and article content")
            st.stop()

        full_text = f"{title} {article}"

        with st.spinner("🔄 Analyzing news content..."):
            try:
                if selected_model == "Logistic Regression":
                    result = predict_logistic_regression(full_text)
                elif selected_model == "SVC":
                    result = predict_svc(full_text)
                else:
                    result = predict_voting_classifier(full_text)
            except Exception as e:
                st.error(f"❌ Prediction error: {e}")
                st.stop()

        prob_real  = result["prob_real"]
        prob_fake  = result["prob_fake"]
        confidence = result["confidence"]
        prep_info  = result["prep_info"]

        # ── THRESHOLD LOGIC (60%) ───────────────────────────────────────────
        THRESHOLD = 0.60
        if prob_real >= THRESHOLD:
            final_verdict   = "REAL"
            final_label     = 0
        elif prob_fake >= THRESHOLD:
            final_verdict   = "FAKE"
            final_label     = 1
        else:
            final_verdict   = "UNCERTAIN"
            final_label     = -1

        st.markdown("---")
        st.markdown("## 📊 Prediction Results")

        result_col1, result_col2, result_col3 = st.columns([2, 1, 1])
        with result_col1:
            if final_verdict == "REAL":
                st.success("### ✅ REAL NEWS")
                st.info("This news appears to be authentic and credible based on the trained model")
            elif final_verdict == "FAKE":
                st.error("### 🚨 FAKE NEWS")
                st.warning("This news appears to contain misinformation or unverified claims")
            else:
                st.warning("### ⚠️ UNCERTAIN — Cannot Determine")
                st.info("""
                **The model confidence is below 60% for both categories.**

                🔍 Please verify this news from multiple credible sources before sharing or trusting it.

                Suggested sources: Reuters, AP News, BBC, The Hindu, NDTV, PTI, or official press releases.
                """)
        with result_col2:
            st.metric("Confidence", f"{confidence:.2f}%")
        with result_col3:
            st.metric("Threshold", "≥ 60% to decide")

        st.markdown("---")
        st.markdown("### 📈 Probability Distribution")

        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Real News Probability", f"{prob_real*100:.2f}%", delta_color="normal")
        with metric_col2:
            st.metric("Fake News Probability", f"{prob_fake*100:.2f}%", delta_color="off")

        st.markdown("---")
        st.markdown("### 📊 Probability Visualization")

        bar_colors = ['#00d400', '#ff4141']
        if final_verdict == "UNCERTAIN":
            bar_colors = ['#f0a500', '#f0a500']

        fig = go.Figure(data=[
            go.Bar(
                x=['REAL NEWS (0)', 'FAKE NEWS (1)'],
                y=[prob_real, prob_fake],
                marker=dict(
                    color=bar_colors,
                    line=dict(color='rgba(255,255,255,0.3)', width=2)
                ),
                text=[f'{prob_real*100:.1f}%', f'{prob_fake*100:.1f}%'],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Probability: %{y:.4f}<br>Percentage: %{text}<extra></extra>'
            )
        ])
        fig.add_hline(
            y=THRESHOLD,
            line_dash="dash",
            line_color="white",
            opacity=0.5,
            annotation_text="60% Threshold",
            annotation_position="top right"
        )
        fig.update_layout(
            title=f"Prediction Probability Distribution ({selected_model})",
            xaxis_title="News Category (Label)",
            yaxis_title="Probability",
            yaxis=dict(range=[0, 1]),
            height=400,
            template="plotly_dark",
            showlegend=False,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("### 🔧 Text Processing Details")

        d1, d2, d3, d4 = st.columns(4)
        with d1: st.metric("Original Tokens", prep_info["original_tokens"])
        with d2: st.metric("Final Tokens",    prep_info["final_tokens"])
        with d3: st.metric("Removed (Stopwords)", prep_info["stopwords_removed"])
        with d4: st.metric("Text Length", f"{len(full_text)} chars")

        st.markdown("---")
        st.markdown("### ℹ️ Model Information")

        i1, i2, i3 = st.columns(3)
        with i1: st.write(f"**Model**: {selected_model}")
        with i2: st.write("**Vectorizer**: TF-IDF (5000 features)")
        with i3: st.write("**Label Encoding**: 0=Real, 1=Fake")

        if 'predictions' not in st.session_state:
            st.session_state.predictions = []
        st.session_state.predictions.append({
            "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "title":      title[:40] + "..." if len(title) > 40 else title,
            "model":      selected_model,
            "prediction": final_verdict,
            "confidence": f"{confidence:.2f}%",
            "prob_real":  f"{prob_real*100:.2f}%",
            "prob_fake":  f"{prob_fake*100:.2f}%",
        })

# ============================================================================
# TAB 2: ANALYTICS
# ============================================================================
with tab2:
    st.subheader("📊 Prediction History & Statistics")

    if 'predictions' in st.session_state and st.session_state.predictions:
        predictions_df = pd.DataFrame(st.session_state.predictions)
        st.dataframe(predictions_df, use_container_width=True, height=300)

        st.markdown("---")
        st.markdown("### 📈 Statistics")

        s1, s2, s3, s4 = st.columns(4)
        with s1: st.metric("Total Predictions", len(predictions_df))
        with s2: st.metric("Real News",      len(predictions_df[predictions_df['prediction'] == 'REAL']))
        with s3: st.metric("Fake News",      len(predictions_df[predictions_df['prediction'] == 'FAKE']))
        with s4:
            avg_conf = predictions_df['confidence'].str.rstrip('%').astype(float).mean()
            st.metric("Avg Confidence", f"{avg_conf:.2f}%")

        if st.button("🗑️ Clear Prediction History"):
            st.session_state.predictions = []
            st.rerun()
    else:
        st.info("📌 No predictions yet. Go to Prediction tab to analyze news!")

# ============================================================================
# TAB 3: DOCUMENTATION
# ============================================================================
with tab3:
    st.subheader("📚 Complete Documentation")

    doc_tab1, doc_tab2, doc_tab3 = st.tabs(["How to Use", "Preprocessing Pipeline", "Model Details"])

    with doc_tab1:
        st.markdown("""
        ### 📖 How to Use This Application
        
        **Step 1: Enter News Content**
        - Paste the news title in "News Title" field
        - Paste the full article in "News Article" field
        
        **Step 2: Select Model**
        - Choose model from sidebar
        - Each model has different accuracy/speed tradeoff
        
        **Step 3: Analyze**
        - Click "🚀 Analyze News" button
        - Wait for processing (ML: <1 sec)
        
        **Step 4: Review Results**
        - Check prediction (REAL, FAKE, or UNCERTAIN)
        - Review confidence percentage
        - Check probability distribution
        
        ### 🎯 Label Encoding
        - **0** = REAL NEWS ✅
        - **1** = FAKE NEWS 🚨
        
        ### 📊 Threshold Logic
        - **≥ 60% Real** → ✅ REAL NEWS
        - **≥ 60% Fake** → 🚨 FAKE NEWS
        - **Both < 60%** → ⚠️ UNCERTAIN — verify from other sources
        
        ### 📥 Sample PDFs
        - Click **"📌 Load Fake Example"** to download 10 fake news samples
        - Click **"✓ Load Real Example"** to download 10 real news samples
        - Open the PDF, copy any title + text into the input fields, and click Analyze
        
        ### 💡 Pro Tips
        - Longer articles give better predictions
        - Include both title and content
        - Try multiple models for comparison
        """)

    with doc_tab2:
        st.markdown("""
        ### 🔄 Text Preprocessing Pipeline
        
        ---
        #### ML Models (Logistic Regression, SVC, Voting Classifier):
        1. **Lowercase** → "Breaking" → "breaking"
        2. **Remove URLs** → http://, https://, www.
        3. **Remove Emails** → user@domain.com
        4. **Remove Special Chars & Numbers** → !, @, #, 0-9
        5. **Clean Spaces** → Remove extra whitespace
        6. **Tokenize** → NLTK word_tokenize
        7. **Remove Stopwords** → the, and, is, a, an, etc.
        8. **Lemmatize** → breaking→break, controls→control
        9. **TF-IDF Vectorize** → 5000 numerical features
        """)

    with doc_tab3:
        st.markdown("""
        ### 🤖 Model Information
        
        **Training Dataset:** 166,418 articles | 50% Real / 50% Fake | 80-20 split
        
        **ML Models:**
        | Model | Train Acc | Test Acc | Gap | Status |
        |-------|-----------|----------|-----|--------|
        | Logistic Regression | 88.86% | 85.50% | 3.36% | Good |
        | SVC | 85.69% | 85.60% | 0.09% | Excellent |
        | Voting Classifier | 88.98% | 86.09% | 2.89% | Good |
        
        ### ✅ Model Selection Tips
        - **Logistic Regression**: Fastest, real-time
        - **SVC**: Best generalization (0.09% overfit gap)
        - **Voting Classifier**: Highest ML accuracy
        """)

# ============================================================================
# TAB 4: ADVANCED
# ============================================================================
with tab4:
    st.subheader("⚙️ Advanced Settings & Information")

    adv_col1, adv_col2 = st.columns(2)
    with adv_col1:
        st.markdown("### 🔧 System Information")
        st.write("**Framework**: Streamlit")
        st.write("**ML Library**: Scikit-learn")
        st.write("**NLP**: NLTK")
        st.write("**Vectorizer**: TF-IDF")
        st.write("**Version**: 3.0")

    with adv_col2:
        st.markdown("### 📊 Model Statistics")
        st.write("**Total Training Samples**: 166,418")
        st.write("**TF-IDF Features**: 5,000")
        st.write("**Preprocessing Steps (ML)**: 9")
        st.write("**Best ML Accuracy**: 86.09%")
        st.write("**Total Models**: 3")
        st.write("**Decision Threshold**: 60%")

    st.markdown("---")
    st.markdown("### 📋 Preprocessing Configuration")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **ML Models:**
        - Tokenization: NLTK word_tokenize()
        - Stopwords: English (NLTK)
        - Lemmatization: WordNetLemmatizer
        - Min Token Length: > 2 chars
        """)
    with c2:
        st.markdown("""
        **ML Cleaning Rules:**
        - Remove URLs: ✅ Yes
        - Remove Emails: ✅ Yes
        - Remove Symbols: ✅ Yes
        - Remove Numbers: ✅ Yes
        - TF-IDF Max Features: 5,000
        - Min DF: 2
        """)

    st.markdown("---")
    st.markdown("### ⚠️ Important Notes")
    st.warning("""
    1. **Label Encoding**: 0=Real, 1=Fake (Exact from training)
    2. **Preprocessing**: Exact same pipeline as training (Critical!)
    3. **Accuracy**: ~86% (Not 100% — always verify from multiple sources)
    4. **Threshold**: Results below 60% confidence are marked UNCERTAIN
    5. **Real-world usage**: This is for educational/demonstration purposes
    6. **Always verify**: Important news should be verified from credible sources
    """)

    st.markdown("---")
    st.markdown("### 🔍 Debug Information")

    db1, db2 = st.columns(2)
    with db1:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        ml_dir      = os.path.join(current_dir, "V2_Saved_Models_ML")
        st.write(f"**Current Directory**: {current_dir}")
        st.write(f"**ML Dir Exists**: {os.path.exists(ml_dir)}")
    with db2:
        if os.path.exists(ml_dir):
            st.write("**Files in V2_Saved_Models_ML:**")
            for f in os.listdir(ml_dir):
                st.write(f"  - {f}")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.8em; padding: 20px;'>
    <p>🎓 Fake News Detection System v3.0</p>
    <p>⚠️ Disclaimer: This tool is for educational purposes. 
    Always verify important news from multiple credible sources.</p>
    <p>📧 Developed with ❤️ by Gaurav Upadhyay</p>
</div>
""", unsafe_allow_html=True)