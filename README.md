<div align="center">

# 🎯 CV Analyzer — Intelligent Resume Screening Dashboard

**ML-powered resume screening with skill gap analysis across 50+ job roles**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://futureml03-i5a4nzvzqgaaawhb3y5bpv.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)


<br/>

![](https://img.shields.io/badge/50%2B%20Job%20Roles-Tech%20%26%20Non--Tech-a78bfa?style=for-the-badge)
![](https://img.shields.io/badge/100%2B%20Skills%20Detected-NLP%20Powered-667eea?style=for-the-badge)
![](https://img.shields.io/badge/Batch%20Screening-Leaderboard%20%26%20CSV-43e97b?style=for-the-badge)

</div>

---

## 📌 Overview

**CV Analyzer** is a beautifully designed, ML-powered resume screening web app built with Streamlit. It helps recruiters, HR teams, and hiring managers instantly evaluate resumes against job descriptions — scoring candidates on semantic similarity and skill match, detecting skill gaps, and ranking multiple candidates on a leaderboard.

No paid APIs. No external databases. Everything runs fresh on every submission.

### 🚀 Live Demo
👉 **[https://futureml03-i5a4nzvzqgaaawhb3y5bpv.streamlit.app/](https://futureml03-i5a4nzvzqgaaawhb3y5bpv.streamlit.app/)**

---

## ✨ Features

### 📄 Screen a Resume
- Paste resume text **or** upload a PDF directly
- Choose from **50+ preset job descriptions** across Tech & Non-Tech roles
- Or type a fully custom job description
- Instant scoring with two complementary ML methods:
  - **TF-IDF Cosine Similarity** — semantic vocabulary match
  - **Skill Match Score** — direct keyword skill comparison
- Composite final score with Grade (A / B / C / D) and hiring recommendation
- Visual **radar chart** showing skill coverage across 8 categories
- Matched skills ✅, missing skills ❌, and bonus extra skills 🌟

### 📊 Batch Compare — Candidate Leaderboard
- Add **multiple candidates** via paste text or PDF upload
- Screen all candidates against a single job description in one click
- Ranked leaderboard with scores, grades, skill tags, and verdict
- Horizontal bar chart comparison across all candidates
- **Download results as CSV** for offline review

### 💡 How It Works
- Step-by-step methodology explanation for non-technical users
- Grade thresholds table, important caveats, and full role directory
- Skill taxonomy browser covering 100+ skills across 8 categories

---

## 🧠 Scoring Methodology

```
Final Score = 50% × TF-IDF Cosine Similarity + 50% × Skill Match Score
```

| Component | What it measures |
|---|---|
| **TF-IDF Cosine Similarity** | Overall vocabulary & semantic overlap between CV and JD |
| **Skill Match Score** | `(Matched skills ÷ Required skills) × 100` |

### Grade Thresholds

| Grade | Verdict | Score Range |
|---|---|---|
| **A** | ✅ Shortlist | 70% and above |
| **B** | 📋 Consider | 50% – 69% |
| **C** | ⚠️ Hold | 32% – 49% |
| **D** | ❌ Reject | Below 32% |

---

## 🗂️ Skill Categories Detected

| Category | Examples |
|---|---|
| **Programming** | Python, Java, JavaScript, C++, SQL, Rust, Go… |
| **Web & APIs** | React, Node.js, Django, FastAPI, GraphQL, Next.js… |
| **Data & ML** | TensorFlow, PyTorch, scikit-learn, LangChain, LLM, RAG… |
| **Cloud/DevOps** | AWS, Docker, Kubernetes, Terraform, CI/CD, GitHub… |
| **Databases** | PostgreSQL, MongoDB, Snowflake, Redis, BigQuery… |
| **Security** | OWASP, Penetration Testing, SIEM, Zero Trust, DevSecOps… |
| **Design** | Figma, Adobe XD, Wireframing, UX Research, Prototyping… |
| **Soft Skills** | Leadership, Agile, Scrum, Communication, Stakeholder Mgmt… |

---

## 🏢 Supported Job Roles (50+)

<details>
<summary><b>🖥️ Tech Roles (33)</b> — click to expand</summary>

**Data & AI:** Data Science Intern, Data Analyst Intern, Machine Learning Engineer, AI / Generative AI Engineer, NLP Engineer, Computer Vision Engineer, Data Engineer, BI / Analytics Engineer

**Web & Software:** Full-Stack Intern, Frontend Developer, Backend Developer, React Native Developer, Mobile App Developer, Software Engineer (SDE-1), WordPress / CMS Developer, Game Developer

**Cloud & DevOps:** Cloud / DevOps Intern, DevOps Engineer, Site Reliability Engineer (SRE), Cloud Architect, Platform Engineer

**Security:** Cybersecurity Analyst, Penetration Tester (Ethical Hacker), Security Engineer

**Database & Systems:** Database Engineer, Database Administrator (DBA)

**Specialized:** Embedded Systems Engineer, Blockchain Developer, IoT Engineer, Robotics Engineer, AR / VR Developer, QA Engineer, Technical Writer, Solutions Architect

</details>

<details>
<summary><b>💼 Non-Tech Roles (42)</b> — click to expand</summary>

**Business & Strategy:** Operations Executive, Business Analyst, Product Manager, Project Manager, Strategy Consultant, Entrepreneurship Associate, Management Trainee

**Finance:** Finance Analyst, Investment Banking Analyst, Chartered Accountant (CA), Risk Analyst, FP&A Analyst

**Marketing & Content:** Social Media Manager, Digital Marketing Executive, Content Writer / Copywriter, SEO Specialist, Performance Marketing Manager, Brand Manager, PR Executive

**Design:** Graphic Designer, UI/UX Designer, Motion Graphics Designer, Product Designer

**HR & People:** HR Executive, Talent Acquisition Specialist, L&D Executive

**Sales & Customer:** Customer Success Executive, Sales Executive (B2B), Account Manager, E-commerce Manager

**Research:** Research Analyst, Market Research Analyst, Policy Analyst

**Healthcare & Science:** Clinical Research Associate (CRA), Pharmaceutical / Biotech Analyst

**Education:** EdTech Content Creator, Academic Counsellor

**Supply Chain:** Supply Chain Analyst, Logistics Coordinator

</details>

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/richa-sinha28/FUTURE_ML_03.git
cd FUTURE_ML_03
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run resume.py
```

The app will open at `http://localhost:8501`

---

## 📦 Requirements

```
streamlit
pandas
scikit-learn
nltk
plotly
matplotlib
PyPDF2
```


---

## 📁 Project Structure

```
FUTURE_ML_03/
 ┣ 📄 resume.py          # Main Streamlit application
 ┣ 📄 requirements.txt   # Python dependencies
 ┗ 📄 README.md          # Project documentation
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| [Streamlit](https://streamlit.io) | Web app framework |
| [scikit-learn](https://scikit-learn.org) | TF-IDF vectorisation & cosine similarity |
| [NLTK](https://www.nltk.org) | Stopword removal & text preprocessing |
| [Plotly](https://plotly.com) | Interactive radar chart |
| [Matplotlib](https://matplotlib.org) | Batch comparison bar chart |
| [PyPDF2](https://pypdf2.readthedocs.io) | PDF text extraction |
| [Pandas](https://pandas.pydata.org) | Data handling & CSV export |

---

## ⚠️ Limitations & Notes

- Scores are based on **text present in the CV** — a candidate may possess a skill that is simply not written in the resume
- TF-IDF captures **vocabulary overlap**, not years of experience or depth of knowledge
- Scores are intended as **decision support**, not a replacement for human judgement
- PDF extraction works best on **text-based PDFs** (not scanned image PDFs)
- Analysis runs **fresh on every submission** — no model files or database required

---

## 🙌 Contributing

Contributions are welcome! Feel free to open issues or pull requests for:
- New job role presets
- Additional skill taxonomy entries
- UI improvements
- Bug fixes

---

<div align="center">



⭐ **Star this repo if you found it useful!**

</div>
