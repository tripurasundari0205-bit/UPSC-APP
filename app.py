import streamlit as st
from openai import OpenAI
from groq import Groq
import pdfplumber
import io
from typing import List
from docx import Document
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet

# -------------------
# Page Config
# -------------------
st.set_page_config(
    page_title="UPSC Solution Book Generator",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------
# Sidebar
# -------------------
st.sidebar.title("📘 UPSC Solution Generator")
# ---------------------------
# Sidebar Controls
# ---------------------------
st.sidebar.header("Settings")
year = st.sidebar.text_input("Year (optional)")
paper = st.sidebar.selectbox("Paper", ["Prelims", "Mains", "Other"], index=0)
process_limit = st.sidebar.number_input("Max questions to process", 1, 200, 20)
show_validator_notes = st.sidebar.checkbox("Show validator notes in output", value=True)
st.sidebar.info(
    """
    **Steps to use:**
    1. Upload a PDF with UPSC Previous Year Questions.  
    2. Click **Generate Solutions**.  
    3. View answers topic-wise in expandable sections.  
    4. Download full solution book as **PDF/DOCX**.  
    
    ⚡ AI will automatically switch to Groq if OpenAI fails.
    """
)
st.sidebar.markdown("---")
st.sidebar.markdown("👨‍💻 Built with ❤️ for UPSC aspirants")

# -------------------
# Utility Functions (unchanged logic)
# -------------------
def call_llm(system_prompt: str, user_prompt: str) -> str:
    try:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content

    except Exception as e:
        st.warning(f"⚠️ OpenAI failed ({e}), switching to Groq fallback...")
        try:
            groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
            resp = groq_client.chat.completions.create(
                model="hog2-wizard-15b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
            )
            return resp.choices[0].message.content
        except Exception as e2:
            st.error(f"❌ Both OpenAI and Groq failed: {e2}")
            return "Error: Unable to generate response at this time."

# -------------------
# Export functions (unchanged)
# -------------------
def export_docx(solutions: List) -> bytes:
    doc = Document()
    doc.add_heading("UPSC Solution Book", 0)
    for sol in solutions:
        doc.add_heading(f"Question ID: {sol.qid}", level=1)
        doc.add_paragraph(sol.content_markdown)
        doc.add_page_break()
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()

def export_pdf(solutions: List) -> bytes:
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

# -------------------
# Main UI
# -------------------
st.markdown("<h1 style='text-align: center;'>📘 UPSC Solution Book Generator</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Generate structured, exam-oriented solutions from UPSC PYQs</p>", unsafe_allow_html=True)
st.markdown("---")

uploaded_file = st.file_uploader("📂 Upload your UPSC PYQ PDF", type=["pdf"])

if uploaded_file:
      with st.spinner("Extracting text…"):
        raw_text = extract_text_from_pdf(uploaded)
    blocks = split_questions(raw_text)
    st.success("✅ File uploaded successfully! Ready to process.")

    if not blocks:
        st.error("Could not detect numbered questions in this PDF. Check formatting or try another file.")
    else:
        st.success(f"Detected {len(blocks)} question blocks.")
        parsed: List[Question] = []
        for i, b in enumerate(blocks[:process_limit]):
            q = parse_mcq_block(b)
            q.year, q.paper, q.page = year, paper, i + 1
            q.topic_tags = tag_topics(q.text)
            parsed.append(q)

        st.subheader("Preview detected questions")
        for q in parsed:
            with st.expander(f"Q: {q.text[:90]}..."):
                st.markdown("**Detected Type:** " + q.qtype)
                if q.options:
                    st.markdown("**Options:**\n\n" + "\n".join(q.options))
                st.markdown("**Tags:** " + ", ".join(q.topic_tags or []))

    if st.button("⚙️ Generate Solutions", type="primary", use_container_width=True):
        st.info("⏳ Running AI models… please wait, this may take a few minutes.")
        
        # Dummy placeholder since your parsing/generation is not pasted here
        # Replace with your actual parsing and solution generation
        results = []  # Assume your function fills this with solutions

        if results:
            st.success("✅ Solutions generated successfully!")
            for sol in results:
                with st.expander(f"🔎 Solution for Q: {sol.qid}"):
                    st.markdown(sol.content_markdown, unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                docx_bytes = export_docx(results)
                st.download_button(
                    label="⬇️ Download as DOCX",
                    data=docx_bytes,
                    file_name="UPSC_Solution_Book.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True
                )
            with col2:
                pdf_bytes = export_pdf(results)
                st.download_button(
                    label="⬇️ Download as PDF",
                    data=pdf_bytes,
                    file_name="UPSC_Solution_Book.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
