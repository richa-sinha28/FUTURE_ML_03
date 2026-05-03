"""
CV Analyzer — Beautiful Gradient Resume Screening Dashboard
=============================================================
HOW TO RUN:
    pip install streamlit pandas scikit-learn nltk plotly matplotlib PyPDF2
    streamlit run cv_analyzer_app.py
"""

import streamlit as st
import pandas as pd
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import plotly.graph_objects as go
import matplotlib.pyplot as plt

try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

nltk.download('stopwords', quiet=True)
nltk.download('punkt',     quiet=True)
STOP_WORDS = set(stopwords.words('english'))

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CV Analyzer — Resume Screening",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# SKILL TAXONOMY
# ─────────────────────────────────────────────────────────────────────────────
SKILL_TAXONOMY = {
    "Programming":  ['python','java','javascript','typescript','c++','c#','r','scala',
                     'go','rust','kotlin','swift','php','ruby','sql','bash','matlab',
                     'perl','dart','julia','haskell','erlang','elixir','groovy','solidity'],
    "Web & APIs":   ['react','angular','vue','html','css','nodejs','express','django',
                     'flask','fastapi','spring','graphql','rest','tailwind','bootstrap',
                     'nextjs','nuxtjs','svelte','gatsby','wordpress','webflow','jquery',
                     'sass','webpack','vite','storybook','three.js'],
    "Data & ML":    ['machine learning','deep learning','nlp','tensorflow','pytorch',
                     'scikit-learn','pandas','numpy','keras','xgboost','spark','hadoop',
                     'tableau','powerbi','statistics','data analysis','data science',
                     'computer vision','reinforcement learning','feature engineering',
                     'data pipeline','mlops','hugging face','langchain','llm','rag',
                     'generative ai','prompt engineering','time series','a/b testing',
                     'data visualization','etl','dbt','airflow','looker'],
    "Cloud/DevOps": ['aws','azure','gcp','docker','kubernetes','terraform','jenkins',
                     'github','gitlab','ci/cd','linux','devops','ansible','helm',
                     'prometheus','grafana','nginx','apache','cloudflare','vercel',
                     'netlify','heroku','digital ocean','pulumi','datadog','splunk'],
    "Databases":    ['mysql','postgresql','mongodb','redis','cassandra','oracle',
                     'sql server','elasticsearch','dynamodb','sqlite','neo4j',
                     'influxdb','cockroachdb','bigquery','snowflake','databricks',
                     'supabase','firebase','prisma','sequelize'],
    "Soft Skills":  ['communication','teamwork','leadership','problem solving',
                     'critical thinking','agile','scrum','project management',
                     'collaboration','time management','adaptability','analytical',
                     'mentoring','stakeholder management','negotiation','creativity',
                     'decision making','conflict resolution','presentation skills',
                     'strategic thinking','emotional intelligence'],
    "Security":     ['cybersecurity','penetration testing','ethical hacking','soc',
                     'siem','vulnerability assessment','network security','owasp',
                     'cryptography','firewall','ids','ips','zero trust','devsecops',
                     'incident response','threat modeling','burp suite','metasploit'],
    "Design":       ['figma','sketch','adobe xd','photoshop','illustrator','indesign',
                     'ui design','ux design','wireframing','prototyping','user research',
                     'design thinking','motion graphics','after effects','premiere pro',
                     'canva','blender','cinema 4d','accessibility'],
}
ALL_SKILLS = [s for grp in SKILL_TAXONOMY.values() for s in grp]

# ─────────────────────────────────────────────────────────────────────────────
# JD PRESETS
# ─────────────────────────────────────────────────────────────────────────────
JD_PRESETS = {
    # ── always first ──
    "Custom (type your own)": "",

    # ══════════════════════════════════════
    # ── TECH ROLES ──
    # ══════════════════════════════════════

    # --- Data & AI ---
    "Data Science Intern": (
        "We are looking for a Data Science Intern with strong Python skills. "
        "Must know machine learning, deep learning, NLP, TensorFlow, PyTorch, "
        "scikit-learn, pandas, numpy, and statistics. SQL and data analysis required. "
        "AWS or GCP experience is a plus. Strong communication, teamwork, and agile working style expected."
    ),
    "Data Analyst Intern": (
        "Data Analyst Intern with Python and SQL skills. Experience with Tableau or Power BI. "
        "Statistical analysis and data analysis skills needed. Pandas, numpy knowledge helpful. "
        "Strong communication and leadership for presenting insights."
    ),
    "Machine Learning Engineer": (
        "Seeking an ML Engineer with expertise in designing and deploying ML models. "
        "Experience with TensorFlow, PyTorch, scikit-learn, XGBoost. "
        "MLOps knowledge using Airflow, Docker, Kubernetes. "
        "Python, SQL, feature engineering, A/B testing, and data pipeline skills required. "
        "AWS or GCP deployment experience preferred. Communication and teamwork essential."
    ),
    "AI / Generative AI Engineer": (
        "Hiring an AI Engineer with knowledge of generative models and LLMs. "
        "Experience with Python, APIs, prompt engineering, LangChain, Hugging Face, RAG. "
        "Understanding of NLP concepts and model fine-tuning. "
        "Skills: Python, machine learning, deep learning, NLP, TensorFlow, PyTorch, generative AI."
    ),
    "NLP Engineer": (
        "NLP Engineer with expertise in natural language processing and text analytics. "
        "Experience with Hugging Face, LLM, RAG, LangChain, TensorFlow, PyTorch. "
        "Python, pandas, numpy, scikit-learn required. "
        "Understanding of transformer architectures and generative AI. "
        "Strong communication and analytical skills needed."
    ),
    "Computer Vision Engineer": (
        "Computer Vision Engineer with deep learning and image processing expertise. "
        "Experience with TensorFlow, PyTorch, OpenCV, and computer vision. "
        "Knowledge of object detection, segmentation, and classification models. "
        "Python, numpy, pandas required. AWS or GCP experience is a plus. "
        "Problem solving and analytical skills expected."
    ),
    "Data Engineer": (
        "Data Engineer with expertise in building and maintaining data pipelines. "
        "Experience with Spark, Hadoop, Airflow, dbt, ETL processes. "
        "Python, SQL, PostgreSQL, BigQuery, Snowflake, Databricks required. "
        "AWS, GCP, or Azure cloud platform knowledge. Docker and CI/CD helpful. "
        "Analytical mindset and strong teamwork skills."
    ),
    "BI / Analytics Engineer": (
        "Business Intelligence Engineer with expertise in data visualization. "
        "Experience with Tableau, Power BI, Looker, and data analysis. "
        "SQL, PostgreSQL, BigQuery, Snowflake knowledge required. "
        "Python and dbt skills helpful. Strong communication and presentation skills."
    ),

    # --- Web & Software ---
    "Full-Stack Intern": (
        "Full-Stack Intern needed. Must have JavaScript, React, Node.js, HTML, CSS experience. "
        "Backend: Django or Flask or FastAPI. REST API and GraphQL knowledge. "
        "Databases: PostgreSQL, MongoDB. Docker and CI/CD experience preferred. "
        "Good communication and problem solving skills."
    ),
    "Frontend Developer": (
        "Frontend Developer with strong React, TypeScript, HTML, CSS, and JavaScript skills. "
        "Experience with Next.js, Tailwind, Webpack, Vite, and Storybook. "
        "REST API integration and GraphQL knowledge required. "
        "Performance optimization, accessibility, and UI design experience. "
        "Agile team environment; communication and collaboration skills essential."
    ),
    "Backend Developer": (
        "Backend Developer with strong Python or Java or Node.js skills. "
        "Experience with Django, FastAPI, Spring, or Express. "
        "REST APIs, GraphQL, PostgreSQL, MongoDB, Redis required. "
        "Docker, Kubernetes, CI/CD, AWS or GCP experience. "
        "Problem solving and teamwork expected."
    ),
    "React Native Developer": (
        "React Native Developer for cross-platform mobile apps. "
        "JavaScript, TypeScript, React Native, REST APIs required. "
        "Understanding of iOS and Android app lifecycle. "
        "Redux, navigation, and third-party SDK integration experience. "
        "Problem solving, teamwork, communication skills."
    ),
    "Mobile App Developer": (
        "Looking for a Mobile App Developer with experience in Android (Kotlin/Java) or iOS (Swift). "
        "Understanding of mobile app lifecycle, UI design, and API integration required. "
        "Experience with app deployment and debugging is a plus. "
        "Skills: Kotlin, Java, Swift, REST, Git, problem solving."
    ),
    "Software Engineer (SDE-1)": (
        "Software Development Engineer with strong coding fundamentals in Python or Java or C++. "
        "Data structures, algorithms, object-oriented design required. "
        "Experience with REST APIs, SQL, PostgreSQL, Git, CI/CD. "
        "Agile, scrum, collaboration, and communication skills. "
        "Docker and cloud exposure preferred."
    ),
    "WordPress / CMS Developer": (
        "WordPress Developer with experience in custom theme and plugin development. "
        "PHP, HTML, CSS, JavaScript, and MySQL skills required. "
        "Experience with Elementor, WooCommerce, SEO basics. "
        "REST API integration and performance optimization. "
        "Communication and time management skills."
    ),
    "Game Developer": (
        "Game Developer with Unity or Unreal Engine experience. "
        "C# or C++ programming skills required. "
        "Understanding of game physics, shaders, and 3D rendering. "
        "Experience with game optimization, debugging, and deployment. "
        "Problem solving, creativity, and teamwork skills."
    ),

    # --- Cloud, DevOps & Infra ---
    "Cloud / DevOps Intern": (
        "DevOps Intern with AWS, Azure or GCP experience. "
        "Must know Docker, Kubernetes, Terraform, Jenkins, Linux. "
        "Strong CI/CD pipeline knowledge. Git, Ansible experience useful. "
        "Agile team environment. Good communication required."
    ),
    "DevOps Engineer": (
        "DevOps Engineer with expertise in CI/CD pipelines and infrastructure automation. "
        "AWS, Azure, or GCP required. Docker, Kubernetes, Terraform, Ansible, Helm. "
        "Jenkins, GitHub, GitLab CI experience. Linux, Bash, Python scripting. "
        "Monitoring with Prometheus, Grafana, Datadog. "
        "Problem solving, collaboration, agile mindset."
    ),
    "Site Reliability Engineer (SRE)": (
        "Looking for an SRE with knowledge of system monitoring and reliability. "
        "Experience with Linux, scripting (Python/Bash), and cloud platforms. "
        "Skills: Linux, Python, bash, AWS, GCP, CI/CD, Docker, Kubernetes, DevOps, "
        "Prometheus, Grafana, incident response, analytical thinking."
    ),
    "Cloud Architect": (
        "Cloud Architect with deep expertise in designing scalable cloud solutions. "
        "AWS, Azure, or GCP certification preferred. "
        "Terraform, Kubernetes, Docker, CI/CD, microservices architecture experience. "
        "Security best practices, cost optimization, and networking knowledge. "
        "Leadership, stakeholder management, strategic thinking required."
    ),
    "Platform Engineer": (
        "Platform Engineer to build and maintain internal developer platforms. "
        "Kubernetes, Docker, Terraform, Helm, CI/CD, GitHub required. "
        "Python or Go scripting skills. AWS or GCP exposure. "
        "Prometheus, Grafana monitoring. Collaboration and communication skills."
    ),

    # --- Security ---
    "Cybersecurity Analyst": (
        "Cybersecurity Analyst with knowledge of network security and threat detection. "
        "Experience with SIEM, SOC, IDS, IPS, vulnerability assessment, OWASP. "
        "Incident response, firewall management, cryptography basics. "
        "Penetration testing, Burp Suite, Metasploit exposure helpful. "
        "Analytical, problem solving, communication, and teamwork skills."
    ),
    "Penetration Tester (Ethical Hacker)": (
        "Penetration Tester with hands-on ethical hacking and offensive security experience. "
        "OWASP top 10, Burp Suite, Metasploit, network security, vulnerability assessment. "
        "Linux, Python, bash scripting. Experience with CTF challenges. "
        "Zero trust, DevSecOps, incident response knowledge. "
        "Analytical mindset and strong communication."
    ),
    "Security Engineer": (
        "Security Engineer to secure infrastructure and applications. "
        "DevSecOps, vulnerability assessment, threat modeling, OWASP, cryptography. "
        "AWS or Azure security services knowledge. "
        "Python scripting, CI/CD integration, Linux required. "
        "Problem solving, critical thinking, and collaboration skills."
    ),

    # --- Database & Systems ---
    "Database Engineer": (
        "Seeking a Database Engineer with strong SQL skills. "
        "Experience with relational databases like MySQL/PostgreSQL. "
        "Indexing, query optimization, and normalization expertise. "
        "Skills: SQL, MySQL, PostgreSQL, MongoDB, Redis, data analysis."
    ),
    "Database Administrator (DBA)": (
        "DBA with expertise in managing and optimizing production databases. "
        "MySQL, PostgreSQL, Oracle, SQL Server required. "
        "Backup, replication, performance tuning, indexing experience. "
        "Snowflake or BigQuery cloud database knowledge preferred. "
        "Analytical, problem solving, and time management skills."
    ),

    # --- Specialized Engineering ---
    "Embedded Systems Engineer": (
        "Seeking an Embedded Systems Engineer with knowledge of C/C++ and microcontrollers. "
        "Experience with Arduino or Raspberry Pi. "
        "Understanding of hardware-software integration and real-time systems. "
        "Skills: C++, bash, linux, problem solving, analytical."
    ),
    "Blockchain Developer": (
        "Seeking a Blockchain Developer with knowledge of Ethereum, smart contracts, and Solidity. "
        "Understanding of Web3.js or Ethers.js. Familiarity with decentralized applications (dApps). "
        "Skills: Solidity, Ethereum, JavaScript, Node.js, cryptography basics."
    ),
    "IoT Engineer": (
        "IoT Engineer with experience in connected device development. "
        "C/C++, Python, Raspberry Pi, Arduino, embedded Linux experience. "
        "MQTT, REST API, cloud connectivity (AWS IoT or Azure IoT) skills. "
        "Linux, bash scripting, networking fundamentals. "
        "Problem solving, analytical, and teamwork skills."
    ),
    "Robotics Engineer": (
        "Robotics Engineer with hands-on experience in robot programming and control systems. "
        "C++, Python, ROS (Robot Operating System) required. "
        "Computer vision, sensor fusion, and motion planning experience. "
        "Linux, bash, embedded systems knowledge. "
        "Analytical, problem solving, and collaboration skills."
    ),
    "AR / VR Developer": (
        "AR/VR Developer with experience building immersive experiences. "
        "Unity, C#, Three.js or Unreal Engine required. "
        "Experience with ARKit, ARCore, WebXR, or OpenXR. "
        "3D modeling, blender, and performance optimization skills. "
        "Creativity, problem solving, and teamwork."
    ),
    "Quality Assurance (QA) Engineer": (
        "QA Engineer with software testing and automation expertise. "
        "Experience with Selenium, Cypress, Jest, or Pytest. "
        "Manual and automated testing, bug tracking, CI/CD integration. "
        "Python or JavaScript scripting skills. "
        "Agile, attention to detail, analytical, and communication skills."
    ),
    "Technical Writer": (
        "Technical Writer with ability to produce clear technical documentation. "
        "Experience documenting APIs, SDKs, and developer tools. "
        "Familiarity with Markdown, Git, GitHub, and static site generators. "
        "Understanding of REST APIs and basic programming concepts (Python or JavaScript). "
        "Communication, collaboration, and time management skills."
    ),
    "Solutions Architect": (
        "Solutions Architect to design and present technical solutions to clients. "
        "AWS, Azure or GCP expertise required. Microservices, Docker, Kubernetes knowledge. "
        "REST APIs, SQL, NoSQL databases experience. "
        "Strong communication, presentation skills, leadership, and stakeholder management."
    ),

    # ══════════════════════════════════════
    # ── NON-TECH / BUSINESS ROLES ──
    # ══════════════════════════════════════

    # --- Business & Strategy ---
    "Operations Executive": (
        "Looking for an Operations Executive to support daily business activities. "
        "Process optimization, coordination, and reporting responsibilities. "
        "Skills: communication, teamwork, leadership, time management, problem solving, analytical."
    ),
    "Business Analyst": (
        "Business Analyst to bridge technical and business teams. "
        "Requirements gathering, process mapping, and documentation. "
        "Data analysis, SQL basics, and reporting skills. "
        "Strong communication, stakeholder management, presentation skills, "
        "critical thinking, and agile methodology knowledge."
    ),
    "Product Manager": (
        "Product Manager to define and drive product roadmaps. "
        "Experience with agile, scrum, backlog management, and user research. "
        "Data analysis and SQL basics helpful. A/B testing knowledge. "
        "Strong communication, leadership, stakeholder management, "
        "strategic thinking, decision making, and collaboration skills."
    ),
    "Project Manager": (
        "Project Manager to lead cross-functional teams and deliver projects on time. "
        "PMP or agile/scrum certification preferred. "
        "Risk management, budgeting, project planning experience. "
        "Strong communication, leadership, time management, "
        "stakeholder management, and conflict resolution skills."
    ),
    "Strategy Consultant": (
        "Strategy Consultant for market research and business strategy development. "
        "Data analysis, financial modeling, and reporting skills. "
        "Strong communication, analytical, critical thinking, "
        "presentation skills, problem solving, and leadership."
    ),
    "Entrepreneurship Associate": (
        "Entrepreneurship Associate to assist in startup operations. "
        "Market research, business strategy, and idea validation. "
        "Skills: communication, leadership, problem solving, teamwork, analytical, collaboration."
    ),
    "Management Trainee": (
        "Management Trainee for cross-departmental rotational program. "
        "Strong academic background with leadership experience. "
        "Communication, analytical, critical thinking, teamwork, "
        "time management, adaptability, and problem solving skills."
    ),

    # --- Finance ---
    "Finance Analyst": (
        "Hiring a Finance Analyst with basic knowledge of accounting and financial analysis. "
        "Understanding of Excel, budgeting, and financial reporting. "
        "Skills: data analysis, statistics, communication, analytical, leadership, time management."
    ),
    "Investment Banking Analyst": (
        "Investment Banking Analyst with financial modeling and valuation expertise. "
        "Excel, PowerPoint, financial analysis, data analysis, statistics skills. "
        "Understanding of capital markets, M&A concepts. "
        "Strong analytical, communication, critical thinking, time management, and leadership skills."
    ),
    "Chartered Accountant (CA)": (
        "CA with expertise in audit, taxation, and financial reporting. "
        "GST, TDS, income tax, financial analysis, accounting standards knowledge. "
        "Tally, Excel, data analysis skills. "
        "Analytical, communication, time management, and critical thinking skills."
    ),
    "Risk Analyst": (
        "Risk Analyst to identify, assess, and mitigate financial and operational risks. "
        "Data analysis, statistics, financial modeling, and reporting skills. "
        "SQL or Python basics helpful. "
        "Analytical, critical thinking, communication, and problem solving skills."
    ),
    "Financial Planning & Analysis (FP&A)": (
        "FP&A Analyst for budgeting, forecasting, and financial reporting. "
        "Excel, data analysis, statistics, PowerBI or Tableau knowledge. "
        "SQL basics helpful. Strong analytical, communication, "
        "presentation skills, and strategic thinking."
    ),

    # --- Marketing & Content ---
    "Social Media Manager": (
        "Social Media Manager to manage online presence and content creation. "
        "Content scheduling, analytics tracking, and marketing trends knowledge. "
        "Skills: communication, leadership, teamwork, analytical, time management, collaboration."
    ),
    "Digital Marketing Executive": (
        "Digital Marketing Executive with SEO, SEM, Google Ads, Meta Ads expertise. "
        "Email marketing, content marketing, and social media strategy. "
        "Google Analytics, data analysis, and A/B testing knowledge. "
        "Communication, creativity, analytical, time management, and teamwork skills."
    ),
    "Content Writer / Copywriter": (
        "Content Writer with excellent writing and editing skills. "
        "SEO writing, blog content, and social media copy experience. "
        "Research and storytelling skills. "
        "Communication, creativity, time management, adaptability, and teamwork."
    ),
    "SEO Specialist": (
        "SEO Specialist to improve organic search rankings and website traffic. "
        "On-page, off-page, and technical SEO knowledge. "
        "Google Search Console, Analytics, Ahrefs or SEMrush experience. "
        "Data analysis, analytical, communication, critical thinking, and problem solving."
    ),
    "Performance Marketing Manager": (
        "Performance Marketing Manager for paid media campaigns. "
        "Google Ads, Meta Ads, LinkedIn Ads, programmatic advertising experience. "
        "Data analysis, A/B testing, Google Analytics required. "
        "Strong analytical, communication, leadership, and strategic thinking skills."
    ),
    "Brand Manager": (
        "Brand Manager to develop and maintain brand identity. "
        "Market research, campaign management, and content strategy. "
        "Communication, creativity, leadership, stakeholder management, "
        "strategic thinking, and presentation skills."
    ),
    "Public Relations (PR) Executive": (
        "PR Executive to manage media relations and communications. "
        "Press release writing, crisis communication, and event management. "
        "Communication, creativity, leadership, collaboration, "
        "time management, and negotiation skills."
    ),

    # --- Design ---
    "Graphic Designer": (
        "Seeking a Graphic Designer with experience in design tools and visual content creation. "
        "Ability to create content for social media and branding. "
        "Skills: Figma, Photoshop, Illustrator, Canva, communication, teamwork, creativity, "
        "collaboration, time management, ui design."
    ),
    "UI/UX Designer": (
        "UI/UX Designer to create intuitive digital experiences. "
        "Figma, Sketch, Adobe XD required. Wireframing, prototyping, user research. "
        "Design thinking, accessibility standards, and usability testing. "
        "Communication, collaboration, creativity, analytical, and presentation skills."
    ),
    "Motion Graphics Designer": (
        "Motion Graphics Designer to create engaging animations and video content. "
        "After Effects, Premiere Pro, Illustrator required. "
        "Motion graphics, animation principles, and storytelling skills. "
        "Cinema 4D or Blender experience preferred. "
        "Creativity, communication, time management, and teamwork."
    ),
    "Product Designer": (
        "Product Designer to craft end-to-end user experiences. "
        "Figma, Sketch, prototyping, wireframing, user research required. "
        "Design thinking, accessibility, design systems, and usability testing. "
        "Communication, collaboration, creativity, and strategic thinking."
    ),

    # --- HR & People ---
    "Human Resources (HR) Executive": (
        "HR Executive for talent acquisition and employee relations. "
        "Recruitment, onboarding, payroll, and compliance experience. "
        "Communication, leadership, time management, "
        "conflict resolution, teamwork, and adaptability skills."
    ),
    "Talent Acquisition Specialist": (
        "Talent Acquisition Specialist to source and hire top candidates. "
        "LinkedIn Recruiter, job portals, and sourcing strategy experience. "
        "ATS tools, screening, and interview coordination. "
        "Communication, collaboration, negotiation, "
        "time management, and stakeholder management skills."
    ),
    "Learning & Development (L&D) Executive": (
        "L&D Executive to design and deliver training programs. "
        "Training needs analysis, curriculum design, and facilitation skills. "
        "Communication, leadership, creativity, collaboration, "
        "presentation skills, and time management."
    ),

    # --- Sales & Customer ---
    "Customer Success Executive": (
        "Customer Success Executive to manage client relationships and onboarding. "
        "Resolve issues and ensure satisfaction. "
        "Skills: communication, leadership, problem solving, teamwork, collaboration, time management."
    ),
    "Sales Executive (B2B)": (
        "B2B Sales Executive to generate leads and close enterprise deals. "
        "CRM tools, outbound calling, email outreach, and pipeline management. "
        "Communication, negotiation, leadership, problem solving, "
        "collaboration, and time management skills."
    ),
    "Account Manager": (
        "Account Manager to manage key client accounts and drive retention. "
        "Client relationship management, upselling, and reporting. "
        "Communication, negotiation, stakeholder management, "
        "leadership, collaboration, and strategic thinking skills."
    ),
    "E-commerce Manager": (
        "E-commerce Manager to oversee online store operations. "
        "Marketplace management (Amazon, Flipkart), product listings, "
        "inventory, and performance tracking. "
        "Data analysis, analytical, communication, leadership, "
        "time management, and problem solving skills."
    ),

    # --- Research & Analysis ---
    "Research Analyst": (
        "Research Analyst with strong analytical and writing skills. "
        "Gather data, analyze information, and prepare reports. "
        "Skills: data analysis, statistics, analytical, communication, critical thinking, python, sql."
    ),
    "Market Research Analyst": (
        "Market Research Analyst to gather and interpret market data. "
        "Qualitative and quantitative research methods. "
        "Data analysis, statistics, PowerBI or Tableau, and reporting. "
        "Communication, analytical, critical thinking, and presentation skills."
    ),
    "Policy Analyst": (
        "Policy Analyst to research, evaluate, and recommend policy solutions. "
        "Qualitative and quantitative data analysis, report writing, and literature review. "
        "Statistics, communication, critical thinking, analytical, "
        "research skills, and stakeholder management."
    ),

    # --- Healthcare & Science ---
    "Clinical Research Associate (CRA)": (
        "Clinical Research Associate to monitor clinical trials. "
        "ICH-GCP guidelines, protocol knowledge, and site management. "
        "Data analysis, reporting, and documentation skills. "
        "Communication, analytical, critical thinking, teamwork, and time management."
    ),
    "Pharmaceutical / Biotech Analyst": (
        "Pharmaceutical Analyst with knowledge of drug development and regulatory processes. "
        "Data analysis, statistics, and scientific report writing. "
        "Communication, analytical, critical thinking, teamwork, and time management."
    ),

    # --- Education & Training ---
    "Education Technology (EdTech) Content Creator": (
        "EdTech Content Creator to develop engaging learning materials. "
        "Curriculum design, e-learning authoring tools, and video production. "
        "Communication, creativity, teamwork, time management, and collaboration."
    ),
    "Academic Counsellor": (
        "Academic Counsellor to guide students in course selection and career planning. "
        "Communication, leadership, empathy, problem solving, "
        "presentation skills, and time management."
    ),

    # --- Supply Chain & Logistics ---
    "Supply Chain Analyst": (
        "Supply Chain Analyst to optimize logistics and inventory management. "
        "Data analysis, SQL, Excel, and ERP system knowledge. "
        "Analytical, problem solving, communication, "
        "time management, and critical thinking skills."
    ),
    "Logistics Coordinator": (
        "Logistics Coordinator to manage shipments and vendor coordination. "
        "Inventory management, order tracking, and supply chain coordination. "
        "Communication, teamwork, time management, "
        "analytical, and problem solving skills."
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=DM+Serif+Display:ital@0;1&display=swap');

:root {
    --grad-1: linear-gradient(135deg,#667eea 0%,#764ba2 100%);
    --grad-3: linear-gradient(135deg,#4facfe 0%,#00f2fe 100%);
    --grad-4: linear-gradient(135deg,#43e97b 0%,#38f9d7 100%);
    --grad-5: linear-gradient(135deg,#fa709a 0%,#fee140 100%);
    --text:   #f0eeff;
    --muted:  #8884a8;
    --accent: #a78bfa;
}

*,*::before,*::after{box-sizing:border-box;}

html,body,[class*="css"],.stApp{
    font-family:'Outfit',sans-serif !important;
    background:#0a0818 !important;
    color:var(--text) !important;
}
.stApp{
    background:
        radial-gradient(ellipse at 20% 50%,rgba(102,126,234,.15) 0%,transparent 50%),
        radial-gradient(ellipse at 80% 20%,rgba(118,75,162,.12) 0%,transparent 50%),
        radial-gradient(ellipse at 60% 80%,rgba(79,172,254,.10) 0%,transparent 50%),
        #0a0818 !important;
}

/* Sidebar */
[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#0f0c29 0%,#1a1040 100%) !important;
    border-right:1px solid rgba(167,139,250,.15) !important;
    box-shadow:4px 0 30px rgba(0,0,0,.4) !important;
}
[data-testid="stSidebar"] *{color:#c4b5fd !important;}

/* Layout */
.block-container{padding:1.8rem 2.2rem !important;max-width:1400px;}
#MainMenu,footer,header{visibility:hidden;}

/* Buttons */
.stButton>button{
    background:linear-gradient(135deg,#667eea 0%,#764ba2 100%) !important;
    color:#fff !important; border:none !important;
    border-radius:12px !important; padding:.7rem 1.6rem !important;
    font-family:'Outfit',sans-serif !important; font-weight:700 !important;
    font-size:.9rem !important; width:100% !important;
    letter-spacing:.03em !important;
    transition:all .25s cubic-bezier(.4,0,.2,1) !important;
    box-shadow:0 4px 20px rgba(102,126,234,.4) !important;
}
.stButton>button:hover{
    transform:translateY(-2px) !important;
    box-shadow:0 8px 30px rgba(102,126,234,.6) !important;
    filter:brightness(1.1) !important;
}

/* Inputs */
.stTextArea textarea,.stTextInput input{
    background:rgba(255,255,255,.04) !important;
    border:1.5px solid rgba(167,139,250,.25) !important;
    border-radius:12px !important;
    font-family:'Outfit',sans-serif !important;
    font-size:.875rem !important; color:var(--text) !important;
}
.stTextArea textarea:focus,.stTextInput input:focus{
    border-color:rgba(167,139,250,.7) !important;
    box-shadow:0 0 0 3px rgba(167,139,250,.12) !important;
}
.stTextArea textarea::placeholder,.stTextInput input::placeholder{color:var(--muted) !important;}

/* File uploader */
div[data-testid="stFileUploader"]{
    border:2px dashed rgba(167,139,250,.3) !important;
    border-radius:14px !important;
    background:rgba(167,139,250,.04) !important;
}
div[data-testid="stFileUploader"]:hover{
    border-color:rgba(167,139,250,.6) !important;
}

/* Selectbox */
div[data-baseweb="select"]>div{
    background:rgba(255,255,255,.05) !important;
    border:1.5px solid rgba(167,139,250,.25) !important;
    border-radius:12px !important; color:var(--text) !important;
}

/* Expander */
details summary{
    background:rgba(167,139,250,.08) !important;
    border-radius:10px !important; color:#c4b5fd !important;
}

/* Page header */
.page-hd{
    background:linear-gradient(135deg,#1a1040 0%,#0f1a3d 60%,#1a0a2e 100%);
    border:1px solid rgba(167,139,250,.2); border-radius:20px;
    padding:2rem 2.5rem; margin-bottom:1.75rem;
    position:relative; overflow:hidden;
}
.page-hd::before{
    content:''; position:absolute; top:-80px; right:-80px;
    width:280px; height:280px;
    background:radial-gradient(circle,rgba(102,126,234,.25) 0%,transparent 65%);
    pointer-events:none;
}
.page-hd::after{
    content:''; position:absolute; bottom:-60px; left:30%;
    width:200px; height:200px;
    background:radial-gradient(circle,rgba(240,147,251,.12) 0%,transparent 65%);
    pointer-events:none;
}
.hd-wrap{display:flex;align-items:flex-end;gap:1.25rem;position:relative;z-index:1;}
.hd-icon{
    width:56px;height:56px;border-radius:16px;
    background:linear-gradient(135deg,#667eea,#764ba2);
    display:flex;align-items:center;justify-content:center;
    font-size:1.5rem;flex-shrink:0;
    box-shadow:0 8px 24px rgba(102,126,234,.5);
}
.hd-text h1{
    font-family:'DM Serif Display',serif; font-size:2.3rem; font-weight:400;
    background:linear-gradient(135deg,#e0d7ff 0%,#a78bfa 50%,#f093fb 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    background-clip:text; letter-spacing:-.01em; line-height:1.1; margin:0;
}
.hd-text p{color:#8884a8;font-size:.9rem;margin-top:.4rem;font-weight:400;}

/* KPI grid */
.kpi-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-bottom:1.75rem;}
.kpi{
    border-radius:16px;padding:1.3rem 1.5rem;
    border:1px solid rgba(255,255,255,.07);
    position:relative;overflow:hidden;
}
.kpi::before{
    content:'';position:absolute;top:0;left:0;right:0;
    height:3px;border-radius:16px 16px 0 0;
}
.kpi-total{background:linear-gradient(145deg,#1a1040,#221550);}
.kpi-total::before{background:linear-gradient(90deg,#667eea,#764ba2);}
.kpi-green{background:linear-gradient(145deg,#0a1f16,#0d2a1e);}
.kpi-green::before{background:linear-gradient(90deg,#43e97b,#38f9d7);}
.kpi-amber{background:linear-gradient(145deg,#1f1408,#2a1c0a);}
.kpi-amber::before{background:linear-gradient(90deg,#fa709a,#fee140);}
.kpi-blue{background:linear-gradient(145deg,#071832,#0a2040);}
.kpi-blue::before{background:linear-gradient(90deg,#4facfe,#00f2fe);}
.kpi .v{font-family:'DM Serif Display',serif;font-size:2rem;font-weight:400;line-height:1;}
.kpi-total .v{background:var(--grad-1);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.kpi-green .v{background:var(--grad-4);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.kpi-amber .v{background:var(--grad-5);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.kpi-blue  .v{background:var(--grad-3);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.kpi .l{font-size:.68rem;color:#6a6890;text-transform:uppercase;letter-spacing:.12em;margin-top:.4rem;}

/* Cards */
.card{
    background:linear-gradient(145deg,rgba(26,16,64,.9) 0%,rgba(22,33,62,.9) 100%);
    border:1px solid rgba(167,139,250,.15); border-radius:18px; padding:1.6rem;
    box-shadow:0 4px 24px rgba(0,0,0,.3); margin-bottom:1.1rem;
    backdrop-filter:blur(10px); position:relative; overflow:hidden;
}
.card::before{
    content:'';position:absolute;top:0;left:0;right:0;height:1px;
    background:linear-gradient(90deg,transparent,rgba(167,139,250,.4),transparent);
}
.card-t{font-family:'Outfit',sans-serif;font-size:1rem;font-weight:700;color:#e0d7ff;margin-bottom:1rem;}

/* Progress bars */
.pr-lbl{display:flex;justify-content:space-between;font-size:.8rem;color:#8884a8;margin-bottom:.3rem;}
.pr-lbl span:last-child{font-weight:700;color:#e0d7ff;}
.pr-track{height:8px;background:rgba(255,255,255,.06);border-radius:99px;overflow:hidden;margin-bottom:1rem;}
.pr-fill{height:100%;border-radius:99px;}
.pr-violet{background:linear-gradient(90deg,#667eea,#764ba2);}
.pr-teal  {background:linear-gradient(90deg,#43e97b,#38f9d7);}
.pr-orange{background:linear-gradient(90deg,#fa709a,#fee140);}
.pr-red   {background:linear-gradient(90deg,#f5576c,#f093fb);}
.pr-blue  {background:linear-gradient(90deg,#4facfe,#00f2fe);}

/* Skill tags */
.skill-tag{
    display:inline-block;padding:.26rem .75rem;border-radius:99px;
    font-size:.72rem;font-weight:600;margin:.2rem .2rem .2rem 0;letter-spacing:.02em;
}
.sk-m{background:linear-gradient(135deg,rgba(67,233,123,.15),rgba(56,249,215,.1));color:#6ee7b7;border:1px solid rgba(67,233,123,.3);}
.sk-x{background:linear-gradient(135deg,rgba(250,112,154,.15),rgba(254,225,64,.08));color:#fca5a5;border:1px solid rgba(250,112,154,.3);}
.sk-e{background:linear-gradient(135deg,rgba(102,126,234,.15),rgba(118,75,162,.1));color:#c4b5fd;border:1px solid rgba(102,126,234,.3);}

/* Verdict */
.verdict{
    border-radius:14px;padding:1.1rem 1.4rem;font-weight:600;font-size:.92rem;
    display:flex;align-items:center;gap:.8rem;margin-top:.6rem;
}
.v-sl{background:linear-gradient(135deg,rgba(67,233,123,.12),rgba(56,249,215,.06));border:1px solid rgba(67,233,123,.35);color:#6ee7b7;}
.v-co{background:linear-gradient(135deg,rgba(79,172,254,.12),rgba(0,242,254,.06));border:1px solid rgba(79,172,254,.35);color:#93c5fd;}
.v-ho{background:linear-gradient(135deg,rgba(250,112,154,.12),rgba(254,225,64,.06));border:1px solid rgba(250,112,154,.35);color:#fcd34d;}
.v-re{background:linear-gradient(135deg,rgba(245,87,108,.12),rgba(240,147,251,.06));border:1px solid rgba(245,87,108,.35);color:#f87171;}

/* Section label */
.slbl{
    font-size:.67rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;
    background:linear-gradient(90deg,#a78bfa,#f093fb);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
    margin-bottom:.55rem;margin-top:1.1rem;display:block;
}

/* Glow divider */
.glow-div{
    height:1px;
    background:linear-gradient(90deg,transparent,rgba(167,139,250,.4),rgba(240,147,251,.3),transparent);
    margin:1.2rem 0;
}

/* Placeholder */
.placeholder{
    background:linear-gradient(145deg,rgba(26,16,64,.6),rgba(22,33,62,.6));
    border:2px dashed rgba(167,139,250,.2); border-radius:18px;
    padding:4rem 2rem; text-align:center; margin-top:.25rem;
}

/* Step cards */
.step-card{
    background:linear-gradient(145deg,rgba(26,16,64,.8),rgba(22,33,62,.8));
    border:1px solid rgba(167,139,250,.12); border-radius:16px;
    padding:1.3rem 1.5rem; margin-bottom:.9rem;
    display:flex; gap:1.1rem; align-items:flex-start;
    transition:all .2s; backdrop-filter:blur(8px);
}
.step-card:hover{border-color:rgba(167,139,250,.3);transform:translateX(4px);box-shadow:0 4px 20px rgba(102,126,234,.15);}
.step-n{
    width:36px;height:36px;border-radius:10px;flex-shrink:0;
    display:flex;align-items:center;justify-content:center;
    font-weight:800;font-size:.85rem;color:#fff;
    background:linear-gradient(135deg,#667eea,#764ba2);
    box-shadow:0 4px 12px rgba(102,126,234,.4);
}
.step-title{font-weight:700;color:#e0d7ff;margin-bottom:.28rem;font-size:.95rem;}
.step-desc{font-size:.83rem;color:#8884a8;line-height:1.7;}

/* Leaderboard card */
.lb-card{
    border-radius:18px;padding:1.4rem 1.6rem;margin-bottom:.9rem;
    border:1px solid rgba(167,139,250,.12);
    position:relative;overflow:hidden;transition:all .2s;backdrop-filter:blur(8px);
}
.lb-card:hover{border-color:rgba(167,139,250,.3);transform:translateY(-2px);box-shadow:0 8px 30px rgba(102,126,234,.15);}
.lb-card.rank-1{background:linear-gradient(145deg,rgba(102,126,234,.15),rgba(118,75,162,.1));border-color:rgba(167,139,250,.4);box-shadow:0 0 30px rgba(102,126,234,.2);}
.lb-card.rank-other{background:linear-gradient(145deg,rgba(26,16,64,.85),rgba(22,33,62,.85));}
.lb-card::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,rgba(167,139,250,.3),transparent);}

/* Table */
table{width:100%;border-collapse:collapse;font-size:.85rem;}
th{padding:.5rem 0;color:#6a6890;font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;border-bottom:1px solid rgba(167,139,250,.15);}
td{padding:.5rem 0;color:#c4b5fd;border-bottom:1px solid rgba(167,139,250,.07);}

/* Sidebar */
.sb-logo{
    font-family:'DM Serif Display',serif;font-size:1.5rem;font-weight:400;
    background:linear-gradient(135deg,#e0d7ff,#a78bfa,#f093fb);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
    padding:1.3rem 0 .2rem;
}
.sb-tag{font-size:.68rem;color:#4a4870;margin-bottom:1.1rem;letter-spacing:.06em;text-transform:uppercase;}
.sb-note{font-size:.73rem;color:#3a3860;line-height:1.8;padding-bottom:.75rem;border-top:1px solid rgba(167,139,250,.1);padding-top:1rem;margin-top:.5rem;}

/* Role category badge */
.role-badge{
    display:inline-block;padding:.15rem .6rem;border-radius:6px;font-size:.65rem;
    font-weight:700;text-transform:uppercase;letter-spacing:.08em;margin-right:.4rem;
    background:linear-gradient(135deg,rgba(167,139,250,.15),rgba(240,147,251,.1));
    color:#c4b5fd;border:1px solid rgba(167,139,250,.25);
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def pdf_to_text(file):
    if not PDF_SUPPORT:
        return ""
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    except Exception:
        return ""


def clean_text(text):
    if not text or pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'https?://\S+', ' ', text)
    text = re.sub(r'[^a-z0-9\s\+\#]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [w for w in text.split() if w not in STOP_WORDS and len(w) > 1]
    return ' '.join(words)


def extract_skills(text):
    tl = str(text).lower()
    return [s for s in ALL_SKILLS
            if re.search(r'\b' + re.escape(s) + r'\b', tl)]


def cat_scores(skills_found):
    sf = set(skills_found)
    return {c: round(len(sf & set(s)) / len(s) * 100)
            for c, s in SKILL_TAXONOMY.items()}


def score_resume(resume_text, jd_text):
    jd_skills  = extract_skills(jd_text)
    res_skills = extract_skills(resume_text)

    matched   = [s for s in jd_skills if s in res_skills]
    missing   = [s for s in jd_skills if s not in res_skills]
    extra     = [s for s in res_skills if s not in jd_skills]
    skill_pct = round(len(matched) / len(jd_skills) * 100, 1) if jd_skills else 0.0

    try:
        vec = TfidfVectorizer(ngram_range=(1, 2), max_features=8000, sublinear_tf=True)
        mat = vec.fit_transform([clean_text(jd_text), clean_text(resume_text)])
        tfidf_pct = round(float(cosine_similarity(mat[0], mat[1])[0][0]) * 100, 1)
    except Exception:
        tfidf_pct = 0.0

    final = round(0.5 * tfidf_pct + 0.5 * skill_pct, 1)

    if   final >= 70: grade, verdict, vcls = "A", "Shortlist", "v-sl"
    elif final >= 50: grade, verdict, vcls = "B", "Consider",  "v-co"
    elif final >= 32: grade, verdict, vcls = "C", "Hold",      "v-ho"
    else:             grade, verdict, vcls = "D", "Reject",    "v-re"

    icons = {"Shortlist": "✅", "Consider": "📋", "Hold": "⚠️", "Reject": "❌"}

    return dict(
        tfidf=tfidf_pct, skill=skill_pct, final=final,
        grade=grade, verdict=verdict, vcls=vcls, vicon=icons[verdict],
        matched=matched, missing=missing, extra=extra,
        res_skills=res_skills, jd_skills=jd_skills,
        cat=cat_scores(res_skills)
    )


def grade_color_css(grade):
    return {
        "A": "background:linear-gradient(135deg,#43e97b,#38f9d7);color:#0a1f16;",
        "B": "background:linear-gradient(135deg,#4facfe,#00f2fe);color:#071832;",
        "C": "background:linear-gradient(135deg,#fa709a,#fee140);color:#1a0a06;",
        "D": "background:linear-gradient(135deg,#f5576c,#f093fb);color:#1a0612;",
    }.get(grade, "background:#333;color:#fff;")


def bar_class(score):
    if score >= 70: return "pr-teal"
    if score >= 50: return "pr-violet"
    if score >= 32: return "pr-orange"
    return "pr-red"


def score_grad(grade):
    return {
        "A": "#43e97b,#38f9d7",
        "B": "#4facfe,#00f2fe",
        "C": "#fa709a,#fee140",
        "D": "#f5576c,#f093fb",
    }.get(grade, "#667eea,#764ba2")


def glow_color(grade):
    return {
        "A": "rgba(67,233,123",
        "B": "rgba(79,172,254",
        "C": "rgba(250,112,154",
        "D": "rgba(245,87,108",
    }.get(grade, "rgba(102,126,234")


# Group presets by category for display
TECH_ROLES = [
    "Custom (type your own)",
    # Data & AI
    "Data Science Intern","Data Analyst Intern","Machine Learning Engineer",
    "AI / Generative AI Engineer","NLP Engineer","Computer Vision Engineer",
    "Data Engineer","BI / Analytics Engineer",
    # Web & Software
    "Full-Stack Intern","Frontend Developer","Backend Developer",
    "React Native Developer","Mobile App Developer","Software Engineer (SDE-1)",
    "WordPress / CMS Developer","Game Developer",
    # Cloud, DevOps
    "Cloud / DevOps Intern","DevOps Engineer","Site Reliability Engineer (SRE)",
    "Cloud Architect","Platform Engineer",
    # Security
    "Cybersecurity Analyst","Penetration Tester (Ethical Hacker)","Security Engineer",
    # Database & Systems
    "Database Engineer","Database Administrator (DBA)",
    # Specialized
    "Embedded Systems Engineer","Blockchain Developer","IoT Engineer",
    "Robotics Engineer","AR / VR Developer","Quality Assurance (QA) Engineer",
    "Technical Writer","Solutions Architect",
]

NON_TECH_ROLES = [
    # Business & Strategy
    "Operations Executive","Business Analyst","Product Manager","Project Manager",
    "Strategy Consultant","Entrepreneurship Associate","Management Trainee",
    # Finance
    "Finance Analyst","Investment Banking Analyst","Chartered Accountant (CA)",
    "Risk Analyst","Financial Planning & Analysis (FP&A)",
    # Marketing & Content
    "Social Media Manager","Digital Marketing Executive","Content Writer / Copywriter",
    "SEO Specialist","Performance Marketing Manager","Brand Manager",
    "Public Relations (PR) Executive",
    # Design
    "Graphic Designer","UI/UX Designer","Motion Graphics Designer","Product Designer",
    # HR & People
    "Human Resources (HR) Executive","Talent Acquisition Specialist",
    "Learning & Development (L&D) Executive",
    # Sales & Customer
    "Customer Success Executive","Sales Executive (B2B)","Account Manager","E-commerce Manager",
    # Research
    "Research Analyst","Market Research Analyst","Policy Analyst",
    # Healthcare & Science
    "Clinical Research Associate (CRA)","Pharmaceutical / Biotech Analyst",
    # Education
    "Education Technology (EdTech) Content Creator","Academic Counsellor",
    # Supply Chain
    "Supply Chain Analyst","Logistics Coordinator",
]


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-logo">🎯 CV Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-tag">Intelligent Resume Screening Platform</div>', unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "nav",
        ["Screen Resume", "Batch Compare", "How It Works"],
        label_visibility="collapsed",
        format_func=lambda x: {
            "Screen Resume": "📄  Screen a Resume",
            "Batch Compare": "📊  Compare Candidates",
            "How It Works":  "💡  How It Works",
        }[x]
    )

    st.markdown("---")

    # Role stats in sidebar
    total_roles = len(TECH_ROLES) - 1 + len(NON_TECH_ROLES)  # minus Custom
    st.markdown(f"""
    <div style="background:rgba(102,126,234,.08);border:1px solid rgba(102,126,234,.2);
                border-radius:12px;padding:.9rem 1rem;margin-bottom:.8rem;">
        <div style="font-size:.65rem;text-transform:uppercase;letter-spacing:.12em;
                    color:#6a6890;margin-bottom:.5rem;">Role Library</div>
        <div style="display:flex;gap:.6rem;flex-wrap:wrap;">
            <div style="text-align:center;">
                <div style="font-family:'DM Serif Display',serif;font-size:1.4rem;
                            background:linear-gradient(135deg,#4facfe,#00f2fe);
                            -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                            background-clip:text;">{len(TECH_ROLES)-1}</div>
                <div style="font-size:.62rem;color:#4a4870;">Tech Roles</div>
            </div>
            <div style="color:#3a3860;font-size:1.2rem;align-self:center;">|</div>
            <div style="text-align:center;">
                <div style="font-family:'DM Serif Display',serif;font-size:1.4rem;
                            background:linear-gradient(135deg,#43e97b,#38f9d7);
                            -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                            background-clip:text;">{len(NON_TECH_ROLES)}</div>
                <div style="font-size:.62rem;color:#4a4870;">Non-Tech Roles</div>
            </div>
            <div style="color:#3a3860;font-size:1.2rem;align-self:center;">|</div>
            <div style="text-align:center;">
                <div style="font-family:'DM Serif Display',serif;font-size:1.4rem;
                            background:linear-gradient(135deg,#a78bfa,#f093fb);
                            -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                            background-clip:text;">{total_roles}</div>
                <div style="font-size:.62rem;color:#4a4870;">Total</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sb-note">
        <b style="color:#7a70a8;">What this does</b><br>
        🔹 Cleans &amp; preprocesses text<br>
        🔹 Extracts 100+ skills via NLP<br>
        🔹 Scores semantic similarity<br>
        🔹 Ranks candidates by role fit<br>
        🔹 Shows exact skill gaps<br>
        🔹 Covers 50+ job roles
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — SCREEN A RESUME
# ─────────────────────────────────────────────────────────────────────────────
if page == "Screen Resume":

    st.markdown("""
    <div class="page-hd">
      <div class="hd-wrap">
        <div class="hd-icon">🎯</div>
        <div class="hd-text">
          <h1>CV Analyzer</h1>
          <p>Paste or upload a resume, select the job role — get instant ML-powered scoring &amp; skill gap analysis across 50+ roles.</p>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap="large")

    with left:
        cand_name = st.text_input("👤  Candidate Name", placeholder="e.g. Priya Sharma")

        st.markdown('<span class="slbl">Resume Input</span>', unsafe_allow_html=True)
        mode = st.radio("mode", ["Paste Text", "Upload PDF"],
                        horizontal=True, label_visibility="collapsed")

        resume_text = ""
        if mode == "Paste Text":
            resume_text = st.text_area(
                "Resume", height=210, label_visibility="collapsed",
                placeholder=(
                    "Paste full resume text here...\n\n"
                    "Example:\n"
                    "Python developer, 6 months internship at a startup.\n"
                    "Skills: Python, scikit-learn, pandas, numpy, TensorFlow, SQL, Git.\n"
                    "Built an NLP-based sentiment classifier. Agile team, good communication."
                )
            )
        else:
            if not PDF_SUPPORT:
                st.warning("Install PyPDF2:  pip install PyPDF2")
            else:
                uploaded = st.file_uploader("Upload PDF", type=["pdf"],
                                             label_visibility="collapsed")
                if uploaded:
                    resume_text = pdf_to_text(uploaded)
                    if resume_text.strip():
                        with st.expander("📋 Extracted Text Preview"):
                            st.text(resume_text[:900] +
                                    ("…" if len(resume_text) > 900 else ""))
                    else:
                        st.error("Could not extract text. Please try paste mode.")

        st.markdown('<span class="slbl">Job Description</span>', unsafe_allow_html=True)

        # Role category filter
        role_cat = st.radio(
            "Role Category", ["All Roles", "🖥️ Tech Roles", "💼 Non-Tech Roles"],
            horizontal=True, label_visibility="collapsed"
        )

        if role_cat == "🖥️ Tech Roles":
            preset_keys = TECH_ROLES
        elif role_cat == "💼 Non-Tech Roles":
            preset_keys = ["Custom (type your own)"] + NON_TECH_ROLES
        else:
            preset_keys = list(JD_PRESETS.keys())

        preset = st.selectbox("Preset", preset_keys, label_visibility="collapsed")

        jd_text = st.text_area(
            "JD", value=JD_PRESETS.get(preset, ""), height=160,
            label_visibility="collapsed",
            placeholder="Paste job description here..."
        )

        st.markdown("<br>", unsafe_allow_html=True)
        go_btn = st.button("🔍  Analyse CV")

    with right:
        if go_btn:
            if not resume_text.strip():
                st.warning("Please provide a resume.")
            elif not jd_text.strip():
                st.warning("Please add a job description.")
            else:
                name = cand_name.strip() or "Candidate"
                with st.spinner("Running ML analysis…"):
                    R = score_resume(resume_text, jd_text)

                gcss   = grade_color_css(R["grade"])
                glow   = glow_color(R["grade"])
                barcls = bar_class(R["final"])
                sgrad  = score_grad(R["grade"])

                role_label = preset if preset != "Custom (type your own)" else "Custom Role"

                st.markdown(f"""
                <div class="card" style="box-shadow:0 0 40px {glow},0.18);">
                  <div style="display:flex;justify-content:space-between;
                              align-items:flex-start;margin-bottom:1.3rem;">
                    <div>
                      <div style="font-size:.67rem;font-weight:700;letter-spacing:.14em;
                                  text-transform:uppercase;
                                  background:linear-gradient(90deg,#a78bfa,#f093fb);
                                  -webkit-background-clip:text;
                                  -webkit-text-fill-color:transparent;
                                  background-clip:text;margin-bottom:.35rem;">
                        CV Analysis Result
                      </div>
                      <div style="font-family:'DM Serif Display',serif;font-size:1.4rem;
                                  color:#e0d7ff;">{name}</div>
                      <div style="font-size:.75rem;color:#6a6890;margin-top:.2rem;">
                        Role: <span style="color:#a78bfa;">{role_label}</span>
                      </div>
                    </div>
                    <div style="text-align:right;">
                      <div style="font-family:'DM Serif Display',serif;font-size:3.2rem;
                                  line-height:1;
                                  background:linear-gradient(135deg,{sgrad});
                                  -webkit-background-clip:text;
                                  -webkit-text-fill-color:transparent;
                                  background-clip:text;">{R['final']}%</div>
                      <div style="display:inline-block;padding:.18rem .75rem;
                                  border-radius:8px;font-size:.7rem;font-weight:800;
                                  letter-spacing:.1em;text-transform:uppercase;
                                  margin-top:.2rem;{gcss}">
                        Grade {R['grade']}
                      </div>
                    </div>
                  </div>

                  <div class="pr-lbl">
                    <span>Semantic Similarity (TF-IDF)</span>
                    <span>{R['tfidf']}%</span>
                  </div>
                  <div class="pr-track">
                    <div class="pr-fill {bar_class(R['tfidf'])}"
                         style="width:{R['tfidf']}%;"></div>
                  </div>

                  <div class="pr-lbl">
                    <span>Skill Match Score</span>
                    <span>{R['skill']}%</span>
                  </div>
                  <div class="pr-track">
                    <div class="pr-fill {barcls}" style="width:{R['skill']}%;"></div>
                  </div>

                  <div class="glow-div"></div>

                  <div class="verdict {R['vcls']}">
                    <span style="font-size:1.2rem;">{R['vicon']}</span>
                    <span>Recommendation: <strong>{R['verdict']}</strong>
                    {'— Strong match. Schedule interview.' if R['verdict'] == 'Shortlist'
                     else '— Good fit. Review skill gaps.' if R['verdict'] == 'Consider'
                     else '— Average fit. Notable gaps present.' if R['verdict'] == 'Hold'
                     else '— Poor fit for this role.'}</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                c1, c2 = st.columns(2)
                with c1:
                    mhtml = ''.join(
                        [f'<span class="skill-tag sk-m">✓ {s}</span>'
                         for s in R["matched"]]
                    ) or '<span style="color:#6a6890;font-size:.83rem;">None matched</span>'
                    st.markdown(f"""
                    <div class="card" style="min-height:148px;">
                      <div class="card-t">✅ Matched Skills
                        <span style="font-size:.75rem;font-weight:600;
                          background:linear-gradient(90deg,#43e97b,#38f9d7);
                          -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                          background-clip:text;">
                          ({len(R['matched'])} / {len(R['jd_skills'])})
                        </span>
                      </div>
                      {mhtml}
                    </div>
                    """, unsafe_allow_html=True)

                with c2:
                    xhtml = ''.join(
                        [f'<span class="skill-tag sk-x">✗ {s}</span>'
                         for s in R["missing"]]
                    ) or '<span style="color:#6a6890;font-size:.83rem;">No critical gaps!</span>'
                    st.markdown(f"""
                    <div class="card" style="min-height:148px;">
                      <div class="card-t">❌ Missing Skills
                        <span style="font-size:.75rem;font-weight:600;
                          background:linear-gradient(90deg,#fa709a,#fee140);
                          -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                          background-clip:text;">
                          ({len(R['missing'])})
                        </span>
                      </div>
                      {xhtml}
                    </div>
                    """, unsafe_allow_html=True)

                cats = list(R["cat"].keys())
                vals = list(R["cat"].values())
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=vals + [vals[0]], theta=cats + [cats[0]],
                    fill='toself',
                    fillcolor='rgba(102,126,234,0.18)',
                    line=dict(color='#a78bfa', width=2.5),
                    marker=dict(color='#c4b5fd', size=7,
                                line=dict(color='#667eea', width=2)),
                ))
                fig.update_layout(
                    polar=dict(
                        bgcolor='rgba(15,12,41,0.8)',
                        radialaxis=dict(
                            visible=True, range=[0, 100],
                            tickfont=dict(color='#4a4870', size=8),
                            gridcolor='rgba(167,139,250,0.1)',
                            linecolor='rgba(167,139,250,0.1)'
                        ),
                        angularaxis=dict(
                            tickfont=dict(color='#8884a8', size=10),
                            gridcolor='rgba(167,139,250,0.08)',
                            linecolor='rgba(167,139,250,0.08)'
                        ),
                    ),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    showlegend=False,
                    margin=dict(t=25, b=25, l=40, r=40),
                    height=265,
                )
                st.markdown('<span class="slbl">Skill Coverage by Category</span>',
                            unsafe_allow_html=True)
                st.plotly_chart(fig, use_container_width=True,
                                config={"displayModeBar": False})

                if R["extra"]:
                    ehtml = ''.join(
                        [f'<span class="skill-tag sk-e">{s}</span>'
                         for s in R["extra"][:14]]
                    )
                    st.markdown(f"""
                    <div class="card">
                      <div class="card-t">🌟 Additional Skills
                        <span style="font-size:.78rem;color:#8884a8;font-weight:400;">
                          (not in JD — bonus)
                        </span>
                      </div>
                      {ehtml}
                    </div>
                    """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="placeholder">
              <div style="font-size:2.5rem;margin-bottom:1rem;">🎯</div>
              <div style="font-family:'DM Serif Display',serif;font-size:1.1rem;
                          background:linear-gradient(135deg,#e0d7ff,#a78bfa);
                          -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                          background-clip:text;">
                Results will appear here
              </div>
              <div style="font-size:.83rem;color:#4a4870;margin-top:.5rem;">
                Fill in the form and click Analyse CV
              </div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — BATCH COMPARE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Batch Compare":

    st.markdown("""
    <div class="page-hd">
      <div class="hd-wrap">
        <div class="hd-icon">📊</div>
        <div class="hd-text">
          <h1>Compare Candidates</h1>
          <p>Add multiple CVs against one job description — get a ranked leaderboard instantly across 50+ roles.</p>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<span class="slbl">Job Description</span>', unsafe_allow_html=True)

    b_cat = st.radio(
        "Batch Role Category", ["All Roles", "🖥️ Tech Roles", "💼 Non-Tech Roles"],
        horizontal=True, label_visibility="collapsed"
    )
    if b_cat == "🖥️ Tech Roles":
        p2_keys = TECH_ROLES
    elif b_cat == "💼 Non-Tech Roles":
        p2_keys = ["Custom (type your own)"] + NON_TECH_ROLES
    else:
        p2_keys = list(JD_PRESETS.keys())

    p2 = st.selectbox("p2", p2_keys, label_visibility="collapsed")
    batch_jd = st.text_area(
        "JD2", value=JD_PRESETS.get(p2, ""), height=90,
        label_visibility="collapsed",
        placeholder="Paste job description here…"
    )

    st.markdown('<div class="glow-div"></div>', unsafe_allow_html=True)

    if "candidates" not in st.session_state:
        st.session_state.candidates = []

    # ── ADD CANDIDATE EXPANDER ──────────────────────────────────────────────
    with st.expander("➕  Add a Candidate",
                     expanded=len(st.session_state.candidates) == 0):

        nn = st.text_input("Candidate Name", placeholder="e.g. Rahul Verma", key="nn")

        # Input mode toggle
        batch_input_mode = st.radio(
            "Resume Input",
            ["✏️ Paste Text", "📄 Upload PDF"],
            horizontal=True,
            key="batch_input_mode"
        )

        batch_resume_text = ""

        if batch_input_mode == "✏️ Paste Text":
            batch_resume_text = st.text_area(
                "Resume Text",
                height=100,
                key="nt",
                placeholder="Paste resume text here…"
            )

        else:  # Upload PDF
            if not PDF_SUPPORT:
                st.warning("PDF support requires PyPDF2:  pip install PyPDF2")
            else:
                batch_pdf = st.file_uploader(
                    "Upload Resume PDF",
                    type=["pdf"],
                    key="batch_pdf",
                    label_visibility="collapsed"
                )
                if batch_pdf is not None:
                    batch_resume_text = pdf_to_text(batch_pdf)
                    if batch_resume_text.strip():
                        st.success(
                            f"✅ PDF extracted — {len(batch_resume_text.split())} words detected"
                        )
                        with st.expander("📋 Preview extracted text"):
                            st.text(
                                batch_resume_text[:600] +
                                ("…" if len(batch_resume_text) > 600 else "")
                            )
                    else:
                        st.error(
                            "Could not extract text from this PDF. "
                            "Try paste mode or use a text-based PDF."
                        )

        if st.button("Add Candidate"):
            name_ok = nn.strip()
            text_ok = batch_resume_text.strip()
            if name_ok and text_ok:
                st.session_state.candidates.append(
                    {"name": name_ok, "text": text_ok}
                )
                st.success(f"✅ {name_ok} added!")
                st.rerun()
            elif not name_ok:
                st.warning("Please enter the candidate's name.")
            else:
                st.warning(
                    "Please paste resume text or upload a PDF before adding."
                )

    # ── CANDIDATE LIST ──────────────────────────────────────────────────────
    if st.session_state.candidates:
        st.markdown(
            f'<span class="slbl">{len(st.session_state.candidates)} Candidate(s) Loaded</span>',
            unsafe_allow_html=True
        )
        for i, c in enumerate(st.session_state.candidates):
            col_a, col_b = st.columns([6, 1])
            with col_a:
                ns = len(extract_skills(c["text"]))
                st.markdown(f"""
                <div style="background:rgba(167,139,250,.06);
                            border:1px solid rgba(167,139,250,.15);
                            border-radius:11px;padding:.65rem 1rem;
                            margin-bottom:.35rem;font-size:.86rem;color:#e0d7ff;">
                    <b>{c['name']}</b>
                    <span style="color:#6a6890;margin-left:.6rem;">
                        — {len(c['text'].split())} words · {ns} skills detected
                    </span>
                </div>
                """, unsafe_allow_html=True)
            with col_b:
                if st.button("✕", key=f"del{i}"):
                    st.session_state.candidates.pop(i)
                    st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("⚡  Run Batch Screening")

        if run_btn:
            if not batch_jd.strip():
                st.warning("Please enter a job description.")
            else:
                with st.spinner("Screening all candidates…"):
                    rows = []
                    for c in st.session_state.candidates:
                        R = score_resume(c["text"], batch_jd)
                        rows.append({
                            "Candidate":  c["name"],
                            "Final Score": R["final"],
                            "TF-IDF":     R["tfidf"],
                            "Skill Score": R["skill"],
                            "Grade":      R["grade"],
                            "Verdict":    R["verdict"],
                            "vcls":       R["vcls"],
                            "vicon":      R["vicon"],
                            "_matched":   R["matched"],
                            "_missing":   R["missing"],
                            "Matched #":  len(R["matched"]),
                            "Missing #":  len(R["missing"]),
                        })

                df = (pd.DataFrame(rows)
                      .sort_values("Final Score", ascending=False)
                      .reset_index(drop=True))
                df.index += 1

                total  = len(df)
                shortl = len(df[df["Final Score"] >= 50])
                avg_s  = round(df["Final Score"].mean(), 1)
                top_s  = round(df["Final Score"].max(), 1)

                st.markdown(f"""
                <div class="kpi-grid">
                  <div class="kpi kpi-total"><div class="v">{total}</div><div class="l">Candidates</div></div>
                  <div class="kpi kpi-green"><div class="v">{shortl}</div><div class="l">Shortlisted</div></div>
                  <div class="kpi kpi-amber"><div class="v">{avg_s}%</div><div class="l">Avg Score</div></div>
                  <div class="kpi kpi-blue"><div class="v">{top_s}%</div><div class="l">Top Score</div></div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown('<span class="slbl">Leaderboard</span>',
                            unsafe_allow_html=True)

                vmsg = {
                    "Shortlist": "Strong match — schedule interview.",
                    "Consider":  "Good fit — review skill gaps.",
                    "Hold":      "Average fit — notable gaps.",
                    "Reject":    "Poor fit for this role.",
                }

                for rank, row in df.iterrows():
                    gcss      = grade_color_css(row["Grade"])
                    barcl     = bar_class(row["Final Score"])
                    glow      = glow_color(row["Grade"])
                    sgrad     = score_grad(row["Grade"])
                    rank_cls  = "rank-1" if rank == 1 else "rank-other"
                    crown     = "👑 " if rank == 1 else ""
                    glow_sh   = f"box-shadow:0 0 30px {glow},0.25);" if rank == 1 else ""

                    mt = ''.join([f'<span class="skill-tag sk-m">{s}</span>'
                                   for s in row["_matched"][:7]])
                    mx = ''.join([f'<span class="skill-tag sk-x">{s}</span>'
                                   for s in row["_missing"][:7]])

                    st.markdown(f"""
                    <div class="lb-card {rank_cls}" style="{glow_sh}">
                      <div style="display:flex;align-items:center;gap:1rem;">
                        <div style="font-family:'DM Serif Display',serif;font-size:2rem;
                                    line-height:1;min-width:2.8rem;
                                    background:linear-gradient(135deg,#667eea,#764ba2);
                                    -webkit-background-clip:text;
                                    -webkit-text-fill-color:transparent;
                                    background-clip:text;">{crown}#{rank}</div>
                        <div style="flex:1;">
                          <div style="display:flex;justify-content:space-between;
                                      align-items:center;margin-bottom:.45rem;">
                            <div>
                              <span style="font-family:'Outfit',sans-serif;font-size:1rem;
                                           font-weight:700;color:#e0d7ff;">{row['Candidate']}</span>
                              <span style="font-size:.7rem;font-weight:800;margin-left:.55rem;
                                           padding:.15rem .6rem;border-radius:6px;{gcss}">
                                {row['vicon']} {row['Verdict']}
                              </span>
                            </div>
                            <div style="font-family:'DM Serif Display',serif;font-size:1.65rem;
                                        background:linear-gradient(135deg,{sgrad});
                                        -webkit-background-clip:text;
                                        -webkit-text-fill-color:transparent;
                                        background-clip:text;">{row['Final Score']}%</div>
                          </div>
                          <div style="height:6px;background:rgba(255,255,255,.05);
                                      border-radius:99px;overflow:hidden;margin-bottom:.55rem;">
                            <div class="pr-fill {barcl}"
                                 style="width:{row['Final Score']}%;"></div>
                          </div>
                          <div style="font-size:.75rem;color:#6a6890;margin-bottom:.45rem;">
                            TF-IDF: <b style="color:#a78bfa;">{row['TF-IDF']}%</b>
                            &nbsp;·&nbsp;
                            Skills: <b style="color:#a78bfa;">{row['Skill Score']}%</b>
                            &nbsp;·&nbsp;
                            ✓ <b style="color:#6ee7b7;">{row['Matched #']}</b>
                            &nbsp;·&nbsp;
                            ✗ <b style="color:#fca5a5;">{row['Missing #']}</b>
                            &nbsp;·&nbsp;
                            <i style="color:#4a4870;">{vmsg.get(row['Verdict'], '')}</i>
                          </div>
                          {f'<div>{mt}</div>' if mt else ''}
                          {f'<div style="margin-top:.3rem;">{mx}</div>' if mx else ''}
                        </div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown(
                    '<span class="slbl" style="margin-top:1.5rem;display:block;">Score Comparison</span>',
                    unsafe_allow_html=True
                )

                fig2, ax = plt.subplots(figsize=(10, max(3, len(df) * 0.65)))
                fig2.patch.set_facecolor('#0f0c29')
                ax.set_facecolor('#0f0c29')

                names_  = df["Candidate"].tolist()[::-1]
                scores_ = df["Final Score"].tolist()[::-1]
                bar_colors = [
                    '#43e97b' if s >= 70 else
                    '#667eea' if s >= 50 else
                    '#fa709a' if s >= 32 else '#f5576c'
                    for s in scores_
                ]

                ax.barh(names_, scores_, color=bar_colors, height=0.5, edgecolor='none')
                ax.axvline(x=50, color='#a78bfa', linestyle='--',
                           linewidth=1.2, alpha=0.6, label='Shortlist threshold (50%)')
                ax.set_xlabel('Final Score (%)', color='#6a6890', fontsize=10)
                ax.set_xlim(0, 115)
                ax.tick_params(colors='#8884a8', labelsize=10)
                for sp in ax.spines.values():
                    sp.set_visible(False)
                ax.xaxis.grid(True, color='#1a1040', linewidth=0.8)
                ax.set_axisbelow(True)
                ax.legend(facecolor='#1a1040', edgecolor='#2e2060', labelcolor='#8884a8', fontsize=9)
         
                for bar_, s_ in zip(ax.patches, scores_):
                    ax.text(
                        bar_.get_width() + 0.8,
                        bar_.get_y() + bar_.get_height() / 2,
                        f'{s_}%', va='center',
                        color='#e0d7ff', fontsize=9, fontweight='700'
                    )

                plt.tight_layout()
                st.pyplot(fig2)
                plt.close()

                exp = df.drop(columns=["vcls", "vicon", "_matched", "_missing"]).copy()
                st.download_button(
                    "⬇️  Download Results CSV",
                    data=exp.to_csv(index=True, index_label="Rank").encode("utf-8"),
                    file_name="cv_analyzer_results.csv",
                    mime="text/csv"
                )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — HOW IT WORKS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "How It Works":

    st.markdown("""
    <div class="page-hd">
      <div class="hd-wrap">
        <div class="hd-icon">💡</div>
        <div class="hd-text">
          <h1>How It Works</h1>
          <p>Plain-English explanation of the CV Analyzer scoring methodology for recruiters, HR managers, and hiring teams.</p>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    steps = [
        ("1", "Text Cleaning & Preprocessing",
         "The CV and job description are converted to lowercase, special characters removed, "
         "and common filler words ('the', 'is', 'and') are stripped. "
         "This focuses the ML model on meaningful words only."),
        ("2", "Skill Extraction via NLP",
         "We scan the cleaned text for 100+ skills across 8 categories using pattern matching. "
         "Categories include Programming, Web & APIs, Data & ML, Cloud/DevOps, Databases, "
         "Security, Design, and Soft Skills."),
        ("3", "Semantic Similarity — TF-IDF + Cosine",
         "TF-IDF converts text into weighted numbers by word importance. "
         "Cosine Similarity measures how similar the CV is to the job description. "
         "This catches strong candidates even when they use different wording."),
        ("4", "Skill Match Score",
         "We directly compare the required skills listed in the JD against what the candidate mentions. "
         "Formula: (Skills Matched ÷ Total Required) × 100. Transparent and auditable."),
        ("5", "Composite Score & Grading",
         "Final Score = 50% TF-IDF + 50% Skill Match. "
         "Grade A (≥70%) → Shortlist · B (50–69%) → Consider · C (32–49%) → Hold · D (<32%) → Reject."),
        ("6", "Skill Gap Report",
         "Missing skills are shown as tags — exactly what the JD requires that the candidate did NOT mention. "
         "Use these to ask targeted interview questions or plan onboarding."),
    ]

    for num, title, desc in steps:
        st.markdown(f"""
        <div class="step-card">
          <div class="step-n">{num}</div>
          <div>
            <div class="step-title">{title}</div>
            <div class="step-desc">{desc}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class="card">
          <div class="card-t">📊 Grade Thresholds</div>
          <table>
            <tr><th>Grade</th><th>Verdict</th><th>Score Range</th></tr>
            <tr>
              <td><span style="background:linear-gradient(135deg,#43e97b,#38f9d7);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                background-clip:text;font-weight:800;">A</span></td>
              <td>Shortlist ✅</td><td>70% and above</td>
            </tr>
            <tr>
              <td><span style="background:linear-gradient(135deg,#4facfe,#00f2fe);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                background-clip:text;font-weight:800;">B</span></td>
              <td>Consider 📋</td><td>50% – 69%</td>
            </tr>
            <tr>
              <td><span style="background:linear-gradient(135deg,#fa709a,#fee140);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                background-clip:text;font-weight:800;">C</span></td>
              <td>Hold ⚠️</td><td>32% – 49%</td>
            </tr>
            <tr>
              <td><span style="background:linear-gradient(135deg,#f5576c,#f093fb);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                background-clip:text;font-weight:800;">D</span></td>
              <td>Reject ❌</td><td>Below 32%</td>
            </tr>
          </table>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="card">
          <div class="card-t">⚠️ Important Notes</div>
          <div style="font-size:.83rem;color:#8884a8;line-height:1.88;">
            🔹 A candidate may <b style="color:#c4b5fd;">have</b> a skill but not
               <b style="color:#c4b5fd;">write</b> it — tags reflect CV text only.<br>
            🔹 TF-IDF captures vocabulary match, not years of experience.<br>
            🔹 Treat scores as <b style="color:#c4b5fd;">decision support</b>,
               not a final hiring decision.<br>
            🔹 Scores improve when the CV uses keywords matching the JD language.<br>
            🔹 No pkl dependency — analysis runs fresh on every submission.<br>
            🔹 Supports both Tech and Non-Tech roles across 50+ categories.
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Role directory
    st.markdown('<span class="slbl" style="margin-top:1.5rem;display:block;">Role Directory</span>',
                unsafe_allow_html=True)

    r1, r2 = st.columns(2)

    with r1:
        tech_tags = ''.join([
            f'<span class="skill-tag sk-e">🖥️ {r}</span>' for r in TECH_ROLES[1:]
        ])
        st.markdown(f"""
        <div class="card">
          <div class="card-t">🖥️ Tech Roles <span style="font-size:.75rem;color:#6a6890;font-weight:400;">({len(TECH_ROLES)-1} roles)</span></div>
          {tech_tags}
        </div>
        """, unsafe_allow_html=True)

    with r2:
        nontech_tags = ''.join([
            f'<span class="skill-tag sk-m">💼 {r}</span>' for r in NON_TECH_ROLES
        ])
        st.markdown(f"""
        <div class="card">
          <div class="card-t">💼 Non-Tech Roles <span style="font-size:.75rem;color:#6a6890;font-weight:400;">({len(NON_TECH_ROLES)} roles)</span></div>
          {nontech_tags}
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<span class="slbl" style="margin-top:1rem;display:block;">Skills We Detect (100+)</span>',
                unsafe_allow_html=True)

    cat_items  = list(SKILL_TAXONOMY.items())
    grad_pairs = [
        ("rgba(102,126,234,0.15)", "rgba(102,126,234,0.3)"),
        ("rgba(240,147,251,0.15)", "rgba(240,147,251,0.3)"),
        ("rgba(79,172,254,0.15)",  "rgba(79,172,254,0.3)"),
        ("rgba(67,233,123,0.15)",  "rgba(67,233,123,0.3)"),
    ]

    for row_i in range(0, len(cat_items), 4):
        chunk = cat_items[row_i:row_i + 4]
        cols  = st.columns(len(chunk))
        for col, (cat, skills), (bg, bd) in zip(cols, chunk, grad_pairs):
            tags = ''.join(
                [f'<span class="skill-tag sk-e">{s}</span>' for s in skills]
            )
            col.markdown(f"""
            <div class="card"
                 style="background:linear-gradient(145deg,{bg},{bg.replace('0.15','0.06')});
                        border-color:{bd};">
              <div class="card-t" style="font-size:.88rem;">{cat}</div>
              {tags}
            </div>
            """, unsafe_allow_html=True)