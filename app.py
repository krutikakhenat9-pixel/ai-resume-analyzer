import streamlit as st
import pdfplumber
import numpy as np
import pandas as pd
import re
import time
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background: #0a0a0f;
    color: #e8e6f0;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 4rem;
    max-width: 1200px;
}
.hero-header {
    text-align: center;
    padding: 3rem 2rem 2rem;
    background: linear-gradient(135deg, #12121e 0%, #1a1a2e 50%, #16213e 100%);
    border-radius: 24px;
    border: 1px solid #2a2a4a;
    margin-bottom: 2rem;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #818cf8 0%, #c084fc 50%, #fb7185 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-subtitle {
    font-size: 1.1rem;
    color: #6b6b8a;
}
.hero-badge {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.4);
    color: #818cf8;
    font-size: 0.7rem;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 1rem;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.glass-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0.01) 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.section-label {
    font-size: 0.65rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #4a4a6a;
    margin-bottom: 0.5rem;
}
.section-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #e8e6f0;
    margin-bottom: 1rem;
}
.score-container {
    text-align: center;
    padding: 2rem 1rem;
}
.score-ring {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 160px;
    height: 160px;
    border-radius: 50%;
    font-size: 2.8rem;
    font-weight: 700;
    margin: 1rem auto;
}
.score-excellent {
    background: conic-gradient(#22c55e var(--pct), #1a2a1a var(--pct));
    color: #22c55e;
}
.score-good {
    background: conic-gradient(#818cf8 var(--pct), #1a1a2e var(--pct));
    color: #818cf8;
}
.score-average {
    background: conic-gradient(#f59e0b var(--pct), #2a1a0a var(--pct));
    color: #f59e0b;
}
.score-poor {
    background: conic-gradient(#fb7185 var(--pct), #2a1a1a var(--pct));
    color: #fb7185;
}
.score-inner {
    width: 120px;
    height: 120px;
    border-radius: 50%;
    background: #0a0a0f;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
}
.score-label {
    font-size: 0.6rem;
    letter-spacing: 2px;
    color: #4a4a6a;
}
.pill-container {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 0.75rem;
}
.pill-missing {
    background: rgba(251,113,133,0.1);
    border: 1px solid rgba(251,113,133,0.3);
    color: #fb7185;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
}
.pill-present {
    background: rgba(34,197,94,0.1);
    border: 1px solid rgba(34,197,94,0.3);
    color: #22c55e;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
}
.metric-row {
    display: flex;
    justify-content: space-around;
    gap: 1rem;
    margin: 1rem 0;
}
.metric-box {
    text-align: center;
    flex: 1;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 1rem;
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #818cf8;
}
.metric-name {
    font-size: 0.75rem;
    color: #4a4a6a;
    margin-top: 0.25rem;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.suggestion-item {
    border-left: 3px solid #818cf8;
    padding: 0.75rem 1rem;
    margin-bottom: 0.75rem;
    background: rgba(99,102,241,0.05);
    border-radius: 0 8px 8px 0;
    font-size: 0.9rem;
    color: #c4c4d8;
    line-height: 1.6;
}
.question-item {
    border-left: 3px solid #c084fc;
    padding: 0.75rem 1rem;
    margin-bottom: 0.75rem;
    background: rgba(192,132,252,0.05);
    border-radius: 0 8px 8px 0;
    font-size: 0.9rem;
    color: #c4c4d8;
    line-height: 1.6;
}
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
}
.styled-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(129,140,248,0.4), transparent);
    margin: 1.5rem 0;
}
.status-tag {
    display: inline-block;
    font-size: 0.65rem;
    padding: 2px 10px;
    border-radius: 12px;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.tag-green { background: rgba(34,197,94,0.15); color: #22c55e; border: 1px solid rgba(34,197,94,0.3); }
.tag-red { background: rgba(251,113,133,0.15); color: #fb7185; border: 1px solid rgba(251,113,133,0.3); }
.tag-blue { background: rgba(129,140,248,0.15); color: #818cf8; border: 1px solid rgba(129,140,248,0.3); }
</style>
""", unsafe_allow_html=True)


def extract_text_from_pdf(uploaded_file) -> str:
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()


def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\.\,\-\+\#]', ' ', text)
    return text.lower().strip()


def compute_ats_score(resume_text: str, jd_text: str) -> dict:
    c_resume = clean_text(resume_text)
    c_jd = clean_text(jd_text)

    vect = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    try:
        tfidf = vect.fit_transform([c_resume, c_jd])
        cos_sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    except Exception:
        cos_sim = 0.0

    jd_words = re.findall(r'\b[a-zA-Z][a-zA-Z\+\#\.]{2,}\b', jd_text)

    stop = {
        'the','and','for','are','this','that','with','have','from','your',
        'will','you','our','all','has','can','been','was','not','any','its',
        'their','they','them','who','but','also','such','more','both','each',
        'about','into','than','what','when','where','which','while','would',
        'should','could','must','shall','may','might','only','some','other',
        'over','after','before','between','through','during','following',
        'required','experience','role','position','team','work','skills',
        'ability','knowledge','strong','excellent','good','well','high',
        'include','including','responsible','responsibilities','requirements',
        'preferred','plus','using','use','used','ensure','provide','support',
        'develop','maintain','manage','create','build','design','implement',
        'collaborate','communicate','working','works'
    }

    jd_keywords = list({
        w.lower() for w in jd_words
        if w.lower() not in stop and len(w) > 2
    })

    resume_lower = resume_text.lower()
    present = [k for k in jd_keywords if k in resume_lower]
    missing = [k for k in jd_keywords if k not in resume_lower]

    kw_score = len(present) / max(len(jd_keywords), 1)
    ats_score = round((cos_sim * 0.5 + kw_score * 0.5) * 100, 1)
    ats_score = min(ats_score, 100)

    return {
        "ats_score": ats_score,
        "cos_sim": round(cos_sim * 100, 1),
        "kw_coverage": round(kw_score * 100, 1),
        "total_kw": len(jd_keywords),
        "present_kw": present[:40],
        "missing_kw": missing[:40],
    }


def call_openrouter(prompt: str, system: str = "") -> str:
    api_key = st.secrets.get("OPENROUTER_API_KEY", "")

    if not api_key:
        return "Missing OPENROUTER_API_KEY. Add it inside .streamlit/secrets.toml"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "AI Resume Analyzer"
    }

    payload = {
        "model": "mistralai/mistral-7b-instruct:free",
        "messages": [
            {
                "role": "system",
                "content": system or "You are an expert resume coach and HR specialist."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 1200,
        "temperature": 0.7
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )

        if response.status_code != 200:
            return f"OpenRouter Error {response.status_code}: {response.text}"

        data = response.json()
        return data["choices"][0]["message"]["content"]

    except Exception as e:
        return f"Request failed: {str(e)}"


def get_suggestions(resume_text: str, jd_text: str, missing_kw: list) -> str:
    missing_str = ", ".join(missing_kw[:20]) if missing_kw else "none identified"

    prompt = f"""
Analyze this resume against the job description and provide exactly 6 specific, actionable improvement suggestions.

RESUME:
{resume_text[:3000]}

JOB DESCRIPTION:
{jd_text[:2000]}

MISSING KEYWORDS:
{missing_str}

Format your response as a numbered list.
Focus on:
- Missing skills to add
- How to reframe existing experience
- Quantifiable achievements
- Formatting improvements
- ATS optimization tips
"""
    return call_openrouter(prompt)


def get_interview_questions(jd_text: str, resume_text: str) -> str:
    prompt = f"""
Based on this job description and candidate resume, generate 8 targeted interview questions.

JOB DESCRIPTION:
{jd_text[:2000]}

RESUME HIGHLIGHTS:
{resume_text[:1500]}

Provide:
- 3 technical questions
- 2 behavioral STAR questions
- 2 role-specific scenario questions
- 1 cultural fit question

Format as numbered list.
"""
    return call_openrouter(prompt)


def score_color_class(score: float) -> str:
    if score >= 80:
        return "score-excellent"
    if score >= 60:
        return "score-good"
    if score >= 40:
        return "score-average"
    return "score-poor"


def render_score_ring(score: float):
    pct = f"{score}%"
    cls = score_color_class(score)
    label = "EXCELLENT" if score >= 80 else "GOOD" if score >= 60 else "AVERAGE" if score >= 40 else "NEEDS WORK"

    st.markdown(f"""
    <div class="score-container">
        <div class="score-ring {cls}" style="--pct:{pct}">
            <div class="score-inner">
                <span>{int(score)}</span>
                <span class="score-label">/ 100</span>
            </div>
        </div>
        <div style="margin-top:0.5rem;">
            <span class="status-tag {'tag-green' if score >= 80 else 'tag-blue' if score >= 60 else 'tag-red'}">{label}</span>
        </div>
        <p style="color:#4a4a6a;font-size:0.8rem;margin-top:0.5rem;">ATS COMPATIBILITY SCORE</p>
    </div>
    """, unsafe_allow_html=True)


st.markdown("""
<div class="hero-header">
    <div class="hero-badge">✦ AI-Powered ✦</div>
    <h1 class="hero-title">Resume Analyzer<br>+ Job Matcher</h1>
    <p class="hero-subtitle">Upload your resume · Paste the JD · Get your ATS score instantly</p>
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="section-label">Step 01</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📄 Upload Resume</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop your PDF resume here",
        type=["pdf"],
        help="Only PDF files are supported",
        label_visibility="collapsed"
    )

    if uploaded_file:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;margin-top:0.5rem;">
            <span class="status-tag tag-green">✓ UPLOADED</span>
            <span style="color:#4a4a6a;font-size:0.8rem;">{uploaded_file.name}</span>
        </div>
        """, unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="section-label">Step 02</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📋 Job Description</div>', unsafe_allow_html=True)

    jd_text = st.text_area(
        "Paste the full job description",
        height=200,
        placeholder="Paste the complete job description here...",
        label_visibility="collapsed"
    )

    if jd_text:
        word_count = len(jd_text.split())
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;margin-top:0.5rem;">
            <span class="status-tag tag-green">✓ READY</span>
            <span style="color:#4a4a6a;font-size:0.8rem;">{word_count} words</span>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

btn_col = st.columns([1, 2, 1])[1]
with btn_col:
    analyze_btn = st.button("⚡ ANALYZE RESUME", use_container_width=True)

if analyze_btn:
    if not uploaded_file:
        st.error("Please upload a PDF resume first.")
    elif not jd_text.strip():
        st.error("Please paste a job description.")
    else:
        progress_bar = st.progress(0, text="Extracting resume text...")
        time.sleep(0.3)

        resume_text = extract_text_from_pdf(uploaded_file)

        if not resume_text:
            st.error("Could not extract text from the PDF. Make sure it is not scanned.")
            st.stop()

        progress_bar.progress(35, text="Computing ATS score...")
        results = compute_ats_score(resume_text, jd_text)

        progress_bar.progress(60, text="Generating improvement suggestions...")
        suggestions_text = get_suggestions(resume_text, jd_text, results["missing_kw"])

        progress_bar.progress(80, text="Generating interview questions...")
        questions_text = get_interview_questions(jd_text, resume_text)

        progress_bar.progress(100, text="Done!")
        time.sleep(0.4)
        progress_bar.empty()

        st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)
        st.markdown("## 📊 Analysis Results")

        r1_left, r1_right = st.columns([1, 2], gap="large")

        with r1_left:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            render_score_ring(results["ats_score"])
            st.markdown('</div>', unsafe_allow_html=True)

        with r1_right:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Breakdown</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Score Components</div>', unsafe_allow_html=True)

            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-box">
                    <div class="metric-value">{results['cos_sim']}%</div>
                    <div class="metric-name">Semantic Match</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">{results['kw_coverage']}%</div>
                    <div class="metric-name">Keyword Coverage</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">{len(results['present_kw'])}/{results['total_kw']}</div>
                    <div class="metric-name">Keywords Found</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**Semantic Similarity**")
            st.progress(results["cos_sim"] / 100)

            st.markdown("**Keyword Coverage**")
            st.progress(results["kw_coverage"] / 100)

            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["✗ Missing Keywords", "✓ Matched Keywords"])

        with tab1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="section-title">🔴 {len(results["missing_kw"])} Keywords Missing</div>', unsafe_allow_html=True)

            if results["missing_kw"]:
                pills = "".join(f'<span class="pill-missing">{k}</span>' for k in results["missing_kw"][:30])
                st.markdown(f'<div class="pill-container">{pills}</div>', unsafe_allow_html=True)
            else:
                st.success("Great — no critical keywords missing!")

            st.markdown('</div>', unsafe_allow_html=True)

        with tab2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="section-title">🟢 {len(results["present_kw"])} Keywords Found</div>', unsafe_allow_html=True)

            if results["present_kw"]:
                pills = "".join(f'<span class="pill-present">{k}</span>' for k in results["present_kw"][:30])
                st.markdown(f'<div class="pill-container">{pills}</div>', unsafe_allow_html=True)
            else:
                st.warning("No matching keywords found.")

            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

        s_col, q_col = st.columns([1, 1], gap="large")

        with s_col:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">💡 Resume Improvements</div>', unsafe_allow_html=True)

            lines = [l.strip() for l in suggestions_text.split('\n') if l.strip()]
            for line in lines:
                if line[0].isdigit() or line.startswith("-") or line.startswith("•"):
                    clean = re.sub(r'^[\d\.\-\•\*]\s*', '', line).strip()
                    if clean:
                        st.markdown(f'<div class="suggestion-item">✦ {clean}</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        with q_col:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">🎤 Likely Interview Questions</div>', unsafe_allow_html=True)

            q_lines = [l.strip() for l in questions_text.split('\n') if l.strip()]
            for line in q_lines:
                if line[0].isdigit() or line.startswith("-") or line.startswith("•"):
                    clean = re.sub(r'^[\d\.\-\•\*]\s*', '', line).strip()
                    if clean:
                        st.markdown(f'<div class="question-item">❓ {clean}</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("📄 View Extracted Resume Text"):
            st.text_area("", resume_text, height=300, label_visibility="collapsed")

        ats = results["ats_score"]

        if ats >= 80:
            msg = "🎉 Outstanding match! Your resume is highly optimized for this role."
            color = "#22c55e"
        elif ats >= 60:
            msg = "👍 Good match! A few tweaks can improve your score further."
            color = "#818cf8"
        elif ats >= 40:
            msg = "⚠️ Moderate match. Add missing skills and stronger keywords."
            color = "#f59e0b"
        else:
            msg = "🔴 Low match. Tailor your resume strongly for this role."
            color = "#fb7185"

        st.markdown(f"""
        <div style="text-align:center;padding:1.5rem;background:rgba(255,255,255,0.02);
                    border:1px solid rgba(255,255,255,0.06);border-radius:16px;margin-top:1rem;">
            <p style="color:{color};font-size:1rem;font-weight:500;margin:0;">{msg}</p>
            <p style="color:#4a4a6a;font-size:0.75rem;margin-top:0.5rem;">
                Powered by OpenRouter AI · Analysis complete
            </p>
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center;padding:3rem 2rem;color:#2a2a4a;">
        <div style="font-size:3rem;margin-bottom:1rem;">🎯</div>
        <p style="font-size:0.8rem;letter-spacing:2px;text-transform:uppercase;color:#2a2a4a;">
            Upload your resume & paste a job description to begin
        </p>
    </div>
    """, unsafe_allow_html=True)