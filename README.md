# 🎯 AI Resume Analyzer + Job Matcher

A sleek, production-grade Streamlit app that analyzes your resume against any job description using TF-IDF scoring and Claude AI — giving you an ATS compatibility score, missing keywords, improvement suggestions, and tailored interview questions.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📄 PDF Resume Upload | Extracts text from any PDF resume using `pdfplumber` |
| 📋 Job Description Input | Paste any JD for instant comparison |
| 🎯 ATS Score (0–100) | TF-IDF cosine similarity + keyword coverage scoring |
| 🔑 Keyword Gap Analysis | See exactly which JD keywords are missing vs. matched |
| 💡 AI Improvement Suggestions | 6 specific, actionable resume fixes via Claude AI |
| 🎤 Interview Question Generator | 8 targeted interview questions based on JD + resume |
| 📊 Visual Score Ring | Animated ring with colour-coded feedback |
| 🌑 Dark Mode UI | Polished dark theme with glassmorphism cards |

---

## 🚀 Quick Start

### 1. Clone / download the project

```bash
git clone https://github.com/your-username/ai-resume-analyzer.git
cd ai-resume-analyzer
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your Anthropic API key

The app uses Claude (via the `anthropic` SDK). Export your key before running:

```bash
# macOS / Linux
export ANTHROPIC_API_KEY="sk-ant-..."

# Windows (PowerShell)
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

Or create a `.streamlit/secrets.toml` file:

```toml
OPENROUTER_API_KEY="your_openrouter_api_key_here"```

### 4. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 📁 Folder Structure

```
ai_resume_analyzer/
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── assets/              # (Optional) logos, sample resumes
```

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `streamlit` | UI framework |
| `pdfplumber` | PDF text extraction |
| `scikit-learn` | TF-IDF vectorization & cosine similarity |
| `numpy` / `pandas` | Data processing |
| `anthropic` | Claude AI — suggestions & interview Qs |

---

## 📐 Scoring Methodology

The **ATS Score** is a blended metric:

```
ATS Score = (Cosine Similarity × 0.5) + (Keyword Coverage × 0.5)
```

- **Cosine Similarity** — TF-IDF semantic match between resume and JD text (bigrams included)
- **Keyword Coverage** — % of meaningful JD keywords found in the resume

| Score | Band |
|---|---|
| 80–100 | 🟢 Excellent |
| 60–79 | 🔵 Good |
| 40–59 | 🟡 Average |
| 0–39 | 🔴 Needs Work |

---


## 💡 Tips for Best Results

1. **Use a text-based PDF** — scanned image PDFs won't extract text. Use Adobe Acrobat, Google Docs, or similar to export a proper PDF.
2. **Paste the full JD** — include responsibilities, requirements, and nice-to-haves for the most accurate keyword analysis.
3. **Iterate** — after seeing the missing keywords, update your resume and re-analyze to watch your score climb.

---

## 📄 License

MIT License — free to use and modify.

---

> Built with ❤️ using Streamlit + Claude AI
