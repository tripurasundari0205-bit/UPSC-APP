import streamlit as st
import pdfplumber
import io
import time
import random
from dataclasses import dataclass
from typing import List
from openai import OpenAI, error
import openai
from docx import Document
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet

# ---------------------
# Data classes
# ---------------------
@dataclass
class Question:
    qid: str
    text: str
    options: List[str]

@dataclass
class Solution:
    qid: str
    content_markdown: str

# ---------------------
# Helper functions
# ---------------------
def extract_text_from_pdf(file) -> str:
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def parse_questions(text: str) -> List[Question]:
    """
    Simplified parsing logic.
    Ideally, use regex or NLP to detect Q + options.
    """
    questions = []
    chunks = text.split("Q.")
    for idx, chunk in enumerate(chunks[1:], start=1):
        parts = chunk.strip().split("\n")
        q_text = parts[0]
        options = [p for p in parts[1:] if p.strip()]
        questions.append(Question(qid=f"Q{idx}", text=q_text, options=options))
    return questions

# ---------------------
# OpenAI API wrapper with retry
# ---------------------
def call_llm(system_prompt: str, user_prompt: str, max_retries: int = 5) -> str:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
            )
            return resp.choices[0].message.content

        except error.RateLimitError:
            wait_time = 2 ** attempt + random.random()
            st.warning(f"⚠️ Rate limit hit. Retrying in {wait_time:.1f}s...")
            time.sleep(wait_time)

        except error.APIError as e:
            wait_time = 2 ** attempt + random.random()
            st.warning(f"⚠️ API error: {e}. Retrying in {wait_time:.1f}s...")
            time.sleep(wait_time)

    st.error("❌ Failed after multiple retries due to API limits. Please try again later.")
    return "Error: API limit reached."

# ---------------------
# Solution generator
# ---------------------
def generate_solution(q: Question) -> Solution:
    system_prompt = (
        "You are an expert UPSC content creator. "
        "Generate solutions in the following format:\n"
        "1. Correct Answer – brief reason.\n"
        "2. Concept Behind the Question.\n"
        "3. Explanation of Each Option.\n"
        "4. Relevance/Current Affairs Linkage.\n"
        "5. Quick Summary for Revision."
    )

    user_prompt = f"Question: {q.text}\nOptions: {q.options}"
    answer = call_llm(system_prompt, user_prompt)
    return Solution(qid=q.qid, content_markdown=answer)

# ---------------------
# Export functions
# ---------------------
def export_docx(solutions: List[Solution]) -> bytes:
    doc = Document()
    doc.add_heading("UPSC Solution Book", 0)

    for sol in solutions:
        doc.add_heading(f"Question ID: {sol.qid}", level=1)
        doc.add_paragraph(sol.content_markdown)
        doc.add_page_break()

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()

def export_pdf(solutions: List[Solution]) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    flowables = []

    flowables.append(Paragraph("<b>UPSC Solution Book</b>", styles["Title"]))
    flowables.append(Spacer(1, 12))

    for sol in solutions:
        flowables.append(Paragraph(f"<b>Question ID: {sol.qid}</b>", styles["Heading2"]))
        flowables.append(Paragraph(sol.content_markdown.replace("\n", "<br/>"), styles["Normal"]))
        flowables.append(Spacer(1, 24))

    doc.build(flowables)
    return buf.getvalue()

# ---------------------
# Streamlit UI
# ---------------------
st.title("📘 UPSC Solution Book Generator")
st.write("Upload UPSC Previous Year Question PDFs and get structured solutions.")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    st.success("✅ PDF extracted")
    parsed = parse_questions(text)
    st.info(f"Found {len(parsed)} questions")

    if st.button("⚙️ Generate Solutions"):
        st.warning("Running LLM… this may take a few minutes")

        results: List[Solution] = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        total_qs = len(parsed)
        for idx, q in enumerate(parsed, start=1):
            status_text.text(f"Processing {q.qid} of {total_qs}...")
            try:
                sol = generate_solution(q)
                results.append(sol)
                with st.expander(f"✅ Solution for {q.qid}: {q.text[:60]}..."):
                    st.markdown(sol.content_markdown)
            except Exception as e:
                st.error(f"Failed for {q.qid}: {e}")

            progress_bar.progress(idx / total_qs)

        status_text.text("✅ All questions processed!")

        if results:
            docx_bytes = export_docx(results)
            pdf_bytes = export_pdf(results)

            st.download_button(
                label="⬇️ Download as DOCX",
                data=docx_bytes,
                file_name="UPSC_Solution_Book.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

            st.download_button(
                label="⬇️ Download as PDF",
                data=pdf_bytes,
                file_name="UPSC_Solution_Book.pdf",
                mime="application/pdf"
            )
