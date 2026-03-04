import streamlit as st
import joblib
import re
import os
import io
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from datetime import datetime

# ── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResumeAI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0f;
    color: #e8e6f0;
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse 80% 50% at 50% -10%, #1a0a3a 0%, #0a0a0f 60%);
    min-height: 100vh;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { display: none; }
.block-container { padding: 2rem 3rem 4rem; max-width: 1100px; margin: auto; }

.hero { text-align: center; padding: 3.5rem 0 2rem; }
.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, rgba(139,92,246,0.2), rgba(59,130,246,0.2));
    border: 1px solid rgba(139,92,246,0.4);
    border-radius: 50px;
    padding: 0.35rem 1.1rem;
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #a78bfa;
    margin-bottom: 1.5rem;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.4rem, 5vw, 3.8rem);
    font-weight: 800;
    line-height: 1.1;
    background: linear-gradient(135deg, #f0eeff 0%, #a78bfa 50%, #60a5fa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 1rem;
}
.hero p {
    font-size: 1.05rem;
    color: #9990b8;
    font-weight: 300;
    max-width: 480px;
    margin: 0 auto 2.5rem;
    line-height: 1.7;
}
.stats-bar { display: flex; justify-content: center; gap: 2.5rem; margin-bottom: 3rem; flex-wrap: wrap; }
.stat-item { text-align: center; }
.stat-number { font-family: 'Syne', sans-serif; font-size: 1.6rem; font-weight: 700; color: #a78bfa; }
.stat-label { font-size: 0.72rem; color: #6b6480; text-transform: uppercase; letter-spacing: 0.08em; }

.glass-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 2rem;
    backdrop-filter: blur(10px);
    transition: border-color 0.3s ease;
}
.glass-card:hover { border-color: rgba(139,92,246,0.3); }
.section-label { font-family: 'Syne', sans-serif; font-size: 0.7rem; font-weight: 600; letter-spacing: 0.15em; text-transform: uppercase; color: #7c6fa0; margin-bottom: 0.6rem; }
.section-title { font-family: 'Syne', sans-serif; font-size: 1.25rem; font-weight: 700; color: #e8e6f0; margin-bottom: 1.2rem; }

.result-card {
    background: linear-gradient(135deg, rgba(139,92,246,0.12), rgba(59,130,246,0.08));
    border: 1px solid rgba(139,92,246,0.35);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    text-align: center;
    animation: fadeInUp 0.5s ease;
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-label { font-size: 0.72rem; font-weight: 500; letter-spacing: 0.15em; text-transform: uppercase; color: #7c6fa0; margin-bottom: 0.5rem; }
.result-category {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.result-confidence { margin-top: 0.4rem; font-size: 0.85rem; color: #6b6480; }

/* ── Resume Details Card ── */
.details-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px;
    padding: 1.8rem 2rem;
    margin-top: 1.2rem;
    animation: fadeInUp 0.6s ease;
}
.details-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #7c6fa0;
    margin-bottom: 1.2rem;
}
.detail-row {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    padding: 0.75rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.detail-row:last-child { border-bottom: none; }
.detail-icon {
    font-size: 1rem;
    width: 2rem;
    height: 2rem;
    background: rgba(139,92,246,0.1);
    border: 1px solid rgba(139,92,246,0.2);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    text-align: center;
    line-height: 2rem;
}
.detail-key { font-size: 0.72rem; color: #6b6480; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.2rem; }
.detail-value { font-size: 0.92rem; color: #c4b5fd; font-weight: 500; }
.detail-value.muted { color: #4a4060; font-style: italic; }

.skills-wrap { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.3rem; }
.skill-tag {
    background: rgba(59,130,246,0.1);
    border: 1px solid rgba(59,130,246,0.25);
    border-radius: 50px;
    padding: 0.2rem 0.7rem;
    font-size: 0.75rem;
    color: #93c5fd;
    font-weight: 500;
}

.categories-grid { display: flex; flex-wrap: wrap; gap: 0.6rem; margin-top: 0.8rem; }
.cat-pill {
    background: rgba(139,92,246,0.1);
    border: 1px solid rgba(139,92,246,0.25);
    border-radius: 50px;
    padding: 0.3rem 0.85rem;
    font-size: 0.78rem;
    color: #a78bfa;
    font-weight: 500;
}

.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    padding: 4px;
    border: 1px solid rgba(255,255,255,0.07);
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 9px;
    color: #7c6fa0;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    font-size: 0.9rem;
    padding: 0.5rem 1.2rem;
    border: none;
}
.stTabs [aria-selected="true"] { background: rgba(139,92,246,0.2) !important; color: #c4b5fd !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.5rem; }

div[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.02);
    border: 2px dashed rgba(139,92,246,0.3);
    border-radius: 16px;
    padding: 1rem;
}
div[data-testid="stFileUploader"]:hover { border-color: rgba(139,92,246,0.6); }

textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    color: #e8e6f0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
}

.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.65rem 2rem;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 0.95rem;
    width: 100%;
    transition: all 0.25s ease;
    box-shadow: 0 4px 20px rgba(124,58,237,0.35);
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 28px rgba(124,58,237,0.5); }

.divider { height: 1px; background: linear-gradient(90deg, transparent, rgba(139,92,246,0.3), transparent); margin: 2rem 0; }
.preview-box {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    font-size: 0.82rem;
    color: #7c6fa0;
    line-height: 1.7;
    max-height: 160px;
    overflow-y: auto;
}
.steps-row { display: flex; gap: 1rem; margin-top: 0.5rem; }
.step-box { flex: 1; background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.07); border-radius: 14px; padding: 1.2rem; text-align: center; }
.step-num { font-family: 'Syne', sans-serif; font-size: 1.4rem; font-weight: 800; color: rgba(139,92,246,0.4); margin-bottom: 0.4rem; }
.step-text { font-size: 0.8rem; color: #7c6fa0; line-height: 1.5; }
</style>
""", unsafe_allow_html=True)

# ── Load Model ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    base  = os.path.dirname(os.path.abspath(__file__))
    svm   = joblib.load(os.path.join(base, 'models', 'svm_model.pkl'))
    tfidf = joblib.load(os.path.join(base, 'models', 'tfidf_vectorizer.pkl'))
    le    = joblib.load(os.path.join(base, 'models', 'label_encoder.pkl'))
    return svm, tfidf, le

svm, tfidf, le = load_models()
stop_words = set(stopwords.words('english'))

# ── Helpers ────────────────────────────────────────────────────────────────
def preprocess(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    words = [w for w in text.split() if w not in stop_words]
    return ' '.join(words)

def extract_text(uploaded_file):
    name = uploaded_file.name
    if name.endswith('.pdf'):
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
            return ' '.join(p.extract_text() or '' for p in reader.pages)
        except Exception as e:
            st.error(f"Could not read PDF: {e}. Try uploading a .docx instead.")
            return ''
    elif name.endswith('.docx'):
        try:
            from docx import Document
            doc = Document(io.BytesIO(uploaded_file.read()))
            return ' '.join(p.text for p in doc.paragraphs)
        except Exception as e:
            st.error(f"Could not read DOCX: {e}")
            return ''
    return ''

def predict(text):
    cleaned = preprocess(text)
    vec     = tfidf.transform([cleaned])
    pred    = svm.predict(vec)
    return le.inverse_transform(pred)[0]

# ── Resume Detail Extraction ───────────────────────────────────────────────
def extract_details(text):
    details = {}

    # Name — check for "Name: John Doe" label first, then fallback to first title-case line
    name_labeled = re.search(r'(?i)(?:name|full\s*name)\s*[:\-]\s*([A-Za-z]+(?: [A-Za-z]+){1,4})', text)
    if name_labeled:
        details['name'] = name_labeled.group(1).strip()
    else:
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        found_name = None
        for line in lines[:10]:
            if any(skip in line.lower() for skip in ['resume','curriculum','cv','objective','summary','profile','@','http','linkedin','phone','email','address']):
                continue
            if re.search(r'\d{5,}', line):
                continue
            if re.match(r'^[A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+){1,3}$', line):
                found_name = line
                break
        details['name'] = found_name

    # Email — handle "Email: x@y.com", bare emails, and multi-part domains
    email = re.findall(r'[a-zA-Z0-9_.+\-]+@[a-zA-Z0-9\-]+(?:\.[a-zA-Z0-9\-]+)*\.[a-zA-Z]{2,}', text)
    details['email'] = email[0] if email else None

    # Phone
    phone = re.findall(r'(?:\+91[\s\-]?)?[6-9]\d{9}|(?:\+\d{1,3}[\s\-]?)?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}', text)
    details['phone'] = phone[0] if phone else None

    # Graduation year
    years = re.findall(r'\b(19[89]\d|20[0-3]\d)\b', text)
    grad_years = [int(y) for y in years if int(y) <= datetime.now().year]
    details['graduation_year'] = str(max(grad_years)) if grad_years else None

    # College
    uni_pattern = r'(?i)((?:university|college|institute|school|iit|nit|bits|vit|srm|anna|jntu|osmania|manipal|amity|symbiosis)[^\n,]{0,60})'
    uni = re.findall(uni_pattern, text)
    details['college'] = uni[0].strip() if uni else None

    # Experience
    exp_pattern = r'(\d+\.?\d*)\+?\s*(?:years?|yrs?)[\s\w]{0,15}(?:experience|exp|work)'
    exp = re.findall(exp_pattern, text, re.IGNORECASE)
    if exp:
        details['experience'] = f"{exp[0]} years"
    else:
        date_ranges = re.findall(r'(20\d{2})\s*[-–]\s*(20\d{2}|present|current)', text, re.IGNORECASE)
        if date_ranges:
            cur = datetime.now().year
            total = sum(
                (cur if 'present' in end.lower() or 'current' in end.lower() else int(end)) - int(start)
                for start, end in date_ranges
            )
            details['experience'] = f"~{total} years (estimated)" if total > 0 else None
        else:
            details['experience'] = None

    # Skills
    skill_keywords = [
        'python','java','sql','javascript','react','node','angular','vue','django','flask',
        'machine learning','deep learning','nlp','tensorflow','pytorch','keras','scikit',
        'aws','azure','gcp','docker','kubernetes','git','linux','excel','power bi','tableau',
        'workday','peoplesoft','sap','oracle','hadoop','spark','mongodb','postgresql','mysql',
        'c++','c#','.net','spring','html','css','typescript','r','matlab','selenium','rest api',
        'xgboost','pandas','numpy','opencv','hcm','eib','integration','bi publisher'
    ]
    text_lower = text.lower()
    found = [s.title() for s in skill_keywords if s in text_lower]
    details['skills'] = found[:18] if found else []

    return details

def render_details(details):
    rows = [
        ("👤", "Full Name",           details.get('name'),            "Not detected"),
        ("📧", "Email",               details.get('email'),           "Not detected"),
        ("📱", "Phone",               details.get('phone'),           "Not detected"),
        ("🎓", "Graduation Year",     details.get('graduation_year'), "Not detected"),
        ("🏫", "College / University",details.get('college'),         "Not detected"),
        ("💼", "Experience",          details.get('experience'),      "Not detected"),
    ]

    html = '<div class="details-card"><div class="details-title">✦ &nbsp;Extracted Resume Details</div>'

    for icon, key, value, fallback in rows:
        muted   = 'muted' if not value else ''
        display = value if value else fallback
        html += f'''
        <div class="detail-row">
            <div class="detail-icon">{icon}</div>
            <div>
                <div class="detail-key">{key}</div>
                <div class="detail-value {muted}">{display}</div>
            </div>
        </div>'''

    skills = details.get('skills', [])
    tags   = ''.join([f'<span class="skill-tag">{s}</span>' for s in skills]) if skills else '<span class="detail-value muted">Not detected</span>'
    html  += f'''
        <div class="detail-row">
            <div class="detail-icon">🛠</div>
            <div>
                <div class="detail-key">Skills &amp; Technologies</div>
                <div class="skills-wrap">{tags}</div>
            </div>
        </div>'''

    html += '</div>'
    return html

# ── Hero ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">✦ AI-Powered NLP Classifier</div>
    <h1>Resume Intelligence<br>Engine</h1>
    <p>Upload a resume for instant category classification and automatic detail extraction.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="stats-bar">
    <div class="stat-item"><div class="stat-number">4</div><div class="stat-label">Categories</div></div>
    <div class="stat-item"><div class="stat-number">500</div><div class="stat-label">TF-IDF Features</div></div>
    <div class="stat-item"><div class="stat-number">SVM</div><div class="stat-label">Model</div></div>
    <div class="stat-item"><div class="stat-number">&lt;1s</div><div class="stat-label">Inference</div></div>
</div>
""", unsafe_allow_html=True)

# ── Main Layout ────────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2], gap="large")
result_text = None

with col_left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Input</div><div class="section-title">Classify a Resume</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📁  Upload File", "✏️  Paste Text"])

    with tab1:
        uploaded = st.file_uploader("Drag & drop or browse", type=["pdf","docx"], label_visibility="collapsed")
        if uploaded:
            with st.spinner("Extracting text..."):
                raw = extract_text(uploaded)
            if raw.strip():
                st.markdown(f'<div class="preview-box">{raw[:400]}…</div>', unsafe_allow_html=True)
                result_text = raw

    with tab2:
        pasted = st.text_area("Resume text", placeholder="Paste resume content here — skills, experience, tools...", height=200, label_visibility="collapsed")
        if st.button("Classify Resume →"):
            if pasted.strip():
                result_text = pasted
            else:
                st.warning("Please paste some text first.")

    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="glass-card" style="height:100%">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Result</div><div class="section-title">Prediction</div>', unsafe_allow_html=True)

    if result_text:
        category         = predict(result_text)
        category_display = category.replace("_", " ")
        word_count       = len(result_text.split())
        char_count       = len(result_text)

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Detected Category</div>
            <div class="result-category">{category_display}</div>
            <div class="result-confidence">LinearSVC · TF-IDF 500 features</div>
        </div>
        <div style="display:flex; gap:1rem; margin-top:1rem;">
            <div style="flex:1; background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.07);
                        border-radius:12px; padding:1rem; text-align:center;">
                <div style="font-family:'Syne',sans-serif; font-size:1.4rem; font-weight:700; color:#a78bfa;">{word_count:,}</div>
                <div style="font-size:0.72rem; color:#6b6480; text-transform:uppercase; letter-spacing:0.08em;">Words</div>
            </div>
            <div style="flex:1; background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.07);
                        border-radius:12px; padding:1rem; text-align:center;">
                <div style="font-family:'Syne',sans-serif; font-size:1.4rem; font-weight:700; color:#60a5fa;">{char_count:,}</div>
                <div style="font-size:0.72rem; color:#6b6480; text-transform:uppercase; letter-spacing:0.08em;">Chars</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center; padding:3rem 1rem; color:#3d3555;">
            <div style="font-size:3rem; margin-bottom:1rem;">⬡</div>
            <div style="font-family:'Syne',sans-serif; font-size:0.9rem; font-weight:600;">Awaiting input</div>
            <div style="font-size:0.78rem; margin-top:0.4rem;">Upload or paste a resume to classify</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ── Resume Details Panel (below prediction) ────────────────────────────────
if result_text:
    with st.spinner("Extracting resume details..."):
        details = extract_details(result_text)
    st.markdown(render_details(details), unsafe_allow_html=True)

# ── Divider ────────────────────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── Bottom Info ────────────────────────────────────────────────────────────
col_a, col_b = st.columns([1, 1], gap="large")

with col_a:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Supported</div><div class="section-title">Categories</div>', unsafe_allow_html=True)
    cats  = [c.replace("_", " ") for c in le.classes_]
    pills = "".join([f'<span class="cat-pill">{c}</span>' for c in cats])
    st.markdown(f'<div class="categories-grid">{pills}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_b:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Pipeline</div><div class="section-title">How It Works</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="steps-row">
        <div class="step-box"><div class="step-num">01</div><div class="step-text">Text extracted from PDF or DOCX</div></div>
        <div class="step-box"><div class="step-num">02</div><div class="step-text">Cleaned & stopwords removed</div></div>
        <div class="step-box"><div class="step-num">03</div><div class="step-text">TF-IDF vectorized → SVM predicts</div></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; margin-top:3rem; padding-top:1.5rem;
     border-top:1px solid rgba(255,255,255,0.05);
     font-size:0.75rem; color:#3d3555; letter-spacing:0.05em;">
    RESUME INTELLIGENCE ENGINE &nbsp;·&nbsp; NLP CLASSIFICATION SYSTEM
</div>
""", unsafe_allow_html=True)
