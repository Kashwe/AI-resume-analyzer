# 🤖 AI Resume Analyzer

An intelligent resume analysis tool powered by LangChain and LLM APIs. Upload a resume PDF, extract skills, match against job descriptions, and get an ATS score with actionable improvement suggestions.

## ✨ Features

- **PDF Resume Upload** — Upload any resume in PDF format
- **Skill Extraction** — Automatically identify technical and soft skills using LLM
- **Job Description Matching** — Semantic similarity matching between resume and JD
- **ATS Score Generation** — Score your resume on ATS (Applicant Tracking System) criteria
- **Improvement Suggestions** — Get specific, actionable feedback to improve your resume
- **Keyword Gap Analysis** — See missing keywords from the job description
- **Experience Level Detection** — Auto-detect seniority from resume content

## 🗂️ Project Structure

```
ai-resume-analyzer/
├── backend/
│   ├── main.py              # FastAPI app — REST endpoints
│   ├── pdf_extractor.py     # US-03: PDF text extraction
│   ├── skill_extractor.py   # US-04: LLM-powered skill extraction
│   ├── job_matcher.py       # US-05: Job description matching
│   ├── ats_scorer.py        # ATS score + improvement suggestions
│   └── models.py            # Pydantic data models
├── frontend/
│   └── index.html           # UI (can be replaced with React)
├── tests/
│   ├── test_pdf_extractor.py
│   ├── test_skill_extractor.py
│   └── test_job_matcher.py
├── sample_data/
│   └── sample_job_description.txt
├── requirements.txt
├── .env.example
└── README.md
```

## 🚀 Setup & Installation (US-01)

### 1. Clone the repository

```bash
git clone <repo-url>
cd ai-resume-analyzer
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
# Edit .env and add your API keys
```

### 5. Run the backend server

```bash
cd backend
uvicorn main:app --reload --port 8000
```

### 6. Open the UI

Open `frontend/index.html` in your browser, or serve it:

```bash
python -m http.server 3000 --directory frontend
```

Then visit `http://localhost:3000`

## 🔑 API Keys Required

- **Groq API Key** — Get from [console.groq.com/keys](https://console.groq.com/keys)
  - Set as `GROQ_API_KEY` in `.env`

> 💡 The project uses Groq's ultra-fast inference with the `llama-3.3-70b-versatile` model by default. You can change the model in `.env`.

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/upload-resume` | Upload PDF, extract text |
| `POST` | `/analyze` | Full analysis (skills + ATS score) |
| `POST` | `/match-job` | Match resume against a job description |
| `GET`  | `/health` | Health check |

### Example: Full Analysis

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@resume.pdf" \
  -F "job_description=We are looking for a Python developer..."
```

## 🧪 Running Tests

```bash
pytest tests/ -v
```

## 🏗️ Architecture

```
PDF Upload → Text Extraction → LLM Skill Extraction
                                        ↓
Job Description ────────────→ Semantic Matching → ATS Score
                                        ↓
                               Improvement Report
```

### Key LangChain Components Used
- `ChatGroq` — LLM backbone (Groq API for ultra-fast inference)
- `PydanticOutputParser` — Structured output parsing
- `PromptTemplate` — Engineered prompts for skill extraction
- LCEL (LangChain Expression Language) — Composing extraction and matching pipelines

## 👥 Team Handoff Notes

The following user stories are **complete** and ready for your team:

| Story | Status | Module |
|-------|--------|--------|
| US-01 Project Setup | ✅ Done | `requirements.txt`, `.env.example` |
| US-02 PDF Upload | ✅ Done | `backend/main.py` → `/upload-resume` |
| US-03 PDF Text Extraction | ✅ Done | `backend/pdf_extractor.py` |
| US-04 Skill Extraction | ✅ Done | `backend/skill_extractor.py` |
| US-05 Job Description Matching | ✅ Done | `backend/job_matcher.py` |

**Remaining stories** (ATS scoring, UI polish, deployment) can build directly on top of these modules.

## 📄 License

MIT
