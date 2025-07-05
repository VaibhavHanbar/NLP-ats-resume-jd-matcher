# NLP-ats-resume-jd-matcher
This project uses Natural Language Processing (NLP) and cosine similarity to compare a candidate’s resume with a job description and generate an ATS (Applicant Tracking System) score. It includes a Flask web application for user interaction and an API for integration.

## 🔍 Features

- Upload resume (PDF or DOCX)
- Input job description
- Get a comprehensive analysis:
  - Overall match score
  - Section similarity
  - Skills match (matched, missing, and additional)
  - Experience and education match
  - Requirement extraction and match
- API endpoint for integration

## 🛠️ Tech Stack

- Python
- Flask
- Scikit-learn
- NLTK
- pdfplumber, docx2txt
- HTML, Jinja2 (for templates)

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/your-username/ats-resume-jd-matcher.git
cd ats-resume-jd-matcher
