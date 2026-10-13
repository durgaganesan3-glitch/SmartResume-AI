import re
import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="SmartResume AI",
    page_icon="📄",
    layout="wide"
)

SKILLS = [
    "python", "sql", "machine learning", "deep learning",
    "artificial intelligence", "data science", "nlp",
    "natural language processing", "generative ai", "llm",
    "rag", "tensorflow", "pytorch", "scikit-learn",
    "pandas", "numpy", "opencv", "computer vision",
    "data analysis", "statistics", "django", "fastapi",
    "flask", "streamlit", "react", "javascript",
    "html", "css", "git", "github", "mysql",
    "postgresql", "rest api", "aws", "docker",
    "transformers", "hugging face", "prompt engineering",
    "langchain", "vector database", "power bi"
]

def extract_resume_text(uploaded_file):
    reader = PdfReader(uploaded_file)
    pages = [
        page.extract_text() or ""
        for page in reader.pages
    ]
    return "\n".join(pages).strip()

def clean_text(text):
    return re.sub(r"\s+", " ", text.lower()).strip()

def find_skills(text):
    text = clean_text(text)
    found = []
    for skill in SKILLS:
        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"
        if re.search(pattern, text):
            found.append(skill)
    return sorted(set(found))

def calculate_similarity(resume_text, job_text):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )
    matrix = vectorizer.fit_transform(
        [resume_text, job_text]
    )
    score = cosine_similarity(
        matrix[0:1], matrix[1:2]
    )[0][0]
    return round(float(score) * 100, 1)

st.title("📄 SmartResume AI")
st.subheader("Resume Analyzer and Job Matcher")
st.write(
    "Compare your resume with a job description, "
    "find relevant skills, and identify areas to improve."
)

with st.sidebar:
    st.header("About this project")
    st.write("Built with Python, NLP and machine learning.")
    st.write("No paid AI API required.")
    st.caption(
        "The match score measures text similarity. "
        "It is not a hiring probability."
    )

uploaded_file = st.file_uploader(
    "1. Upload your resume (PDF)",
    type=["pdf"]
)

job_description = st.text_area(
    "2. Paste the job description",
    height=220,
    placeholder=(
        "Paste an AI Engineer or AI/ML job description here..."
    )
)

if st.button("Analyze Resume", type="primary"):
    if uploaded_file is None:
        st.error("Please upload your resume PDF.")
    elif not job_description.strip():
        st.error("Please paste a job description.")
    else:
        try:
            resume_text = extract_resume_text(uploaded_file)

            if len(resume_text) < 30:
                st.error(
                    "Not enough readable text found. "
                    "Try a text-based PDF instead of a scanned image."
                )
            else:
                resume_clean = clean_text(resume_text)
                job_clean = clean_text(job_description)

                score = calculate_similarity(
                    resume_clean, job_clean
                )
                resume_skills = find_skills(resume_text)
                job_skills = find_skills(job_description)

                matching = sorted(
                    set(resume_skills) & set(job_skills)
                )
                missing = sorted(
                    set(job_skills) - set(resume_skills)
                )

                st.divider()
                st.header("Your Analysis")

                col1, col2, col3 = st.columns(3)
                col1.metric("Text Similarity", f"{score}%")
                col2.metric("Matching Skills", len(matching))
                col3.metric("Potential Skill Gaps", len(missing))

                st.progress(int(score))

                left, right = st.columns(2)

                with left:
                    st.subheader("✅ Matching Skills")
                    if matching:
                        st.write(", ".join(matching))
                    else:
                        st.info(
                            "No matching skills from the built-in "
                            "skill list were detected."
                        )

                with right:
                    st.subheader("📌 Skills to Review")
                    if missing:
                        st.write(", ".join(missing))
                    else:
                        st.success(
                            "All detected job skills also appear "
                            "in your resume."
                        )

                st.subheader("💡 Improvement Suggestions")
                suggestions = []

                if missing:
                    suggestions.append(
                        "Review the missing skills and learn those "
                        "relevant to your experience. Add them only "
                        "if you genuinely have the skills."
                    )

                if score < 30:
                    suggestions.append(
                        "Compare your project descriptions and "
                        "technical skills with the job requirements."
                    )

                if len(resume_skills) < 5:
                    suggestions.append(
                        "Consider adding a clear technical skills "
                        "section with accurate skills you can explain."
                    )

                if not suggestions:
                    suggestions.append(
                        "Check that your projects clearly explain "
                        "your contribution, tools, and results."
                    )

                for suggestion in suggestions:
                    st.write(f"• {suggestion}")

                st.subheader("📋 Detected Resume Skills")
                st.write(
                    ", ".join(resume_skills)
                    if resume_skills
                    else "No listed skills detected."
                )

                st.download_button(
                    "Download Analysis Report",
                    data=(
                        "SMARTRESUME AI - ANALYSIS REPORT\n\n"
                        f"Text similarity: {score}%\n"
                        f"Matching skills: {', '.join(matching)}\n"
                        f"Skills to review: {', '.join(missing)}\n"
                        f"Detected resume skills: "
                        f"{', '.join(resume_skills)}\n"
                    ),
                    file_name="smartresume_report.txt",
                    mime="text/plain"
                )

        except Exception as error:
            st.error(
                "Could not analyze this file. "
                "Please check that it is a valid, readable PDF."
            )
            st.caption(f"Technical detail: {error}")
