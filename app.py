# app.py
import re, io, json, time, uuid
import pdfplumber
import streamlit as st
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict

# ---------------------------
# Config
# ---------------------------
ALLOWED_DOMAINS = [
    ".gov.in", "pib.gov.in", "prsindia.org", "indiacode.nic.in", "rbi.org.in",
    "sebi.gov.in", "moef.gov.in", "niti.gov.in", "censusindia.gov.in",
    "un.org", "worldbank.org", "imf.org", "ipcc.ch", "undp.org", "unesco.org",
    "who.int", "fao.org", "oecd.org", "ncert.nic.in", "cag.gov.in"
]

SYSTEM_PROMPT = """You are an expert UPSC faculty member. Produce solutions STRICTLY in this structure:

Question: <verbatim question with options>

1. Correct Answer – State the correct option (A/B/C/D or ‘Not applicable’) and a one-sentence reason.

2. Concept Behind the Question – Explain the background, core concept, related theory, and why it matters for UPSC.

3. Explanation of Each Option – For every option, say “Correct” or “Incorrect” with 2–4 lines of reasoning and the concept behind it.

4. Relevance/Current Affairs Linkage – Link to present-day relevance, schemes, reports, or applications. If none, say “No direct current linkage.”

5. Quick Summary for Revision – 4–6 crisp bullet points.

Rules:
- Be accurate and concise. Avoid speculation.
- Cite only reliable sources. List sources at the end (bulleted).
- If unsure about correctness, say so and suggest verification.
- Keep tone exam-oriented; avoid fluff.
"""

USER_TEMPLATE = """PDF_Source_Metadata: {meta}

Question_Block:
{question_block}

Detected_Type: {qtype}

Topic_Tags: {tags}

Retrieved_Context:
{retrieved}
"""

# ---------------------------
# Data models
# ---------------------------
@dataclass
class Question:
    qid: str
    text: str
    options: List[str]
    qtype: str   # MCQ | Mains | Assertion-Reason | etc.
    year: Optional[str] = None
    paper: Optional[str] = None
    topic_tags: Optional[List[str]] = None
    page: Optional[int] = None

@dataclass
class Solution:
    qid: str
    content_markdown: str
    citations: List[str]

# ---------------------------
# PDF parsing
# ---------------------------
Q_START = re.compile(r"^\s*(\d{1,3})\s*[).:-]\s+", re.MULTILINE)
OPT_LINE = re.compile(r"^\s*[A-D]\s*[\)\].-]\s+", re.MULTILINE)

def extract_text_from_pdf(file: io.BytesIO) -> str:
    text_parts = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            # remove common headers/footers heuristically
            lines = [ln for ln in t.splitlines() if len(ln.strip()) > 0]
            text_parts.append("\n".join(lines))
    return "\n\n".join(text_parts)

def split_questions(raw: str) -> List[str]:
    # Split at question numbers while keeping the headers
    indices = [m.start() for m in Q_START.finditer(raw)]
    blocks = []
    if not indices:
        return blocks
    indices.append(len(raw))
    for i in range(len(indices)-1):
        blocks.append(raw[indices[i]:indices[i+1]].strip())
    return blocks

def parse_mcq_block(block: str) -> Question:
    # Extract question number (optional), stem, and options
    # Remove leading number "12) " etc.
    block_clean = re.sub(r"^\s*\d{1,3}\s*[).:-]\s*", "", block).strip()
    # Find options
    opt_matches = list(OPT_LINE.finditer(block_clean))
    options = []
    if opt_matches:
        # Split by option markers
        parts = OPT_LINE.split(block_clean)
        # parts[0] is the stem; subsequent parts align with options A-D
        stem = parts[0].strip()
        for m, p in zip(opt_matches, parts[1:]):
            # Recreate label
            label = block_clean[m.start():m.end()].strip()
            label_letter = re.findall(r"[A-D]", label)[0]
            option_text = p.strip()
            options.append(f"{label_letter}. {option_text}")
    else:
        stem = block_clean
    return Question(
        qid=str(uuid.uuid4())[:8],
        text=stem,
        options=options,
        qtype="MCQ" if options else "Mains"
    )

# ---------------------------
# Topic tagging (very light heuristic; you can replace with LLM)
# ---------------------------
KEYWORDS_TO_TAGS = {
    "parliament": "Polity", "governor": "Polity", "fundamental right": "Polity",
    "inflation": "Economy", "gdp": "Economy", "rbi": "Economy",
    "mangrove": "Environment", "biosphere": "Environment", "UNFCCC": "Environment",
    "dna": "Science & Tech", "quantum": "Science & Tech",
    "harappan": "History", "buddh": "History",
    "map": "Geography", "monsoon": "Geography",
}

def tag_topics(text: str) -> List[str]:
    lower = text.lower()
    tags = set()
    for k, v in KEYWORDS_TO_TAGS.items():
        if k in lower:
            tags.add(v)
    return list(tags) or ["General Studies"]

# ---------------------------
# Retrieval stub (optional)
# ---------------------------
def retrieve_snippets(question: Question) -> List[str]:
    """
    Placeholder for retrieval. Options:
      - Local vector store over NCERT/standard notes
      - Web search API restricted to ALLOWED_DOMAINS
    Return bullets like: "- Fact ... [PIB, 2023]"
    """
    return []  # start empty; upgrade later

# ---------------------------
# LLM call (provider-agnostic)
# ---------------------------
def call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Plug in your provider here.
    Example (OpenAI):
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":system_prompt},
                      {"role":"user","content":user_prompt}],
            temperature=0.2,
        )
        return resp.choices[0].message.content
    """
    raise NotImplementedError("Wire this to your preferred LLM provider.")

# ---------------------------
# Validation
# ---------------------------
def validate_markdown(md: str, q: Question) -> List[str]:
    errs = []
    # Structure checks
    for head in ["Question:", "1. Correct Answer", "2. Concept Behind the Question",
                 "3. Explanation of Each Option", "4. Relevance/Current Affairs Linkage",
                 "5. Quick Summary for Revision"]:
        if head.split(":")[0] not in md:
            errs.append(f"Missing section: {head}")
    # MCQ: ensure one correct option stated (A-D)
    if q.qtype == "MCQ":
        m = re.search(r"Correct Answer\s*–.*\b([ABCD])\b", md)
        if not m:
            errs.append("Correct Answer section missing A/B/C/D label.")
        # ensure each option labeled
        for opt_label in ["A", "B", "C", "D"]:
            if any(o.startswith(f"{opt_label}.") for o in q.options):
                if not re.search(rf"\bOption\s*{opt_label}\b|\b{opt_label}\)", md, re.IGNORECASE):
                    # soft warning; model may not use "Option X" literal
                    pass
    # Citations
    if "Sources" not in md and "References" not in md:
        errs.append("Missing sources/citations section.")
    return errs

# ---------------------------
# Orchestration
# ---------------------------
def generate_solution(q: Question) -> Solution:
    retrieved = retrieve_snippets(q)
    user_prompt = USER_TEMPLATE.format(
        meta=json.dumps({"year": q.year, "paper": q.paper, "page": q.page}),
        question_block=(q.text + ("\n\n" + "\n".join(q.options) if q.options else "")),
        qtype=q.qtype,
        tags=", ".join(q.topic_tags or []),
        retrieved="\n".join("- " + s for s in retrieved) if retrieved else "None"
    )
    md = call_llm(SYSTEM_PROMPT, user_prompt)
    errs = validate_markdown(md, q)
    if errs:
        md += "\n\n---\nValidator notes (for editor):\n" + "\n".join(f"- {e}" for e in errs)
    # naive citation scrape
    cites = re.findall(r"\((https?://[^\s)]+)\)", md)
    return Solution(qid=q.qid, content_markdown(md), citations=cites)

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="UPSC Solution Generator", page_icon="📘", layout="wide")
st.title("📘 UPSC PYQ → Structured Solution Generator")

st.sidebar.header("Settings")
year = st.sidebar.text_input("Year (optional)")
paper = st.sidebar.selectbox("Paper", ["Prelims", "Mains", "Other"], index=0)
process_limit = st.sidebar.number_input("Max questions to process", 1, 200, 20)

uploaded = st.file_uploader("Upload a PDF with previous year questions", type=["pdf"])

if uploaded:
    raw_text = extract_text_from_pdf(uploaded)
    blocks = split_questions(raw_text)
    st.success(f"Detected {len(blocks)} question blocks.")
    parsed: List[Question] = []
    for i, b in enumerate(blocks[:process_limit]):
        q = parse_mcq_block(b)
        q.year, q.paper, q.page = year, paper, i+1
        q.topic_tags = tag_topics(q.text)
        parsed.append(q)

    st.subheader("Preview detected questions")
    for q in parsed:
        with st.expander(f"Q: {q.text[:80]}..."):
            st.markdown("**Detected Type:** " + q.qtype)
            if q.options:
                st.markdown("**Options:**\n\n" + "\n".join(q.options))
            st.markdown("**Tags:** " + ", ".join(q.topic_tags or []))

    if st.button("⚙️ Generate Solutions"):
        st.warning("Please wire `call_llm` to your provider before running.")
        results: Dict[str, Dict] = {}
        for q in parsed:
            # md = generate_solution(q)  # will raise NotImplementedError until wired
            pass
        st.info("Export will appear here after LLM is wired.")

    st.download_button(
        "⬇️ Download raw extracted text (for debugging)",
        data=raw_text,
        file_name="extracted_text.txt",
        mime="text/plain"
    )
