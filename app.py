import streamlit as st
from openai import OpenAI
import requests
import time
import re
import fitz  # PyMuPDF for PDF reading

# ==============================
# Load API Keys from Secrets
# ==============================
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY")
GROQ_KEY = st.secrets.get("GROQ_API_KEY")

# Initialize OpenAI client if key exists
openai_client = None
if OPENAI_KEY:
    openai_client = OpenAI(api_key=OPENAI_KEY)

# Groq client (fallback)
# Groq client (fallback)
class GroqClient:
    def __init__(self, api_key):
        self.api_key = api_key

    def chat_completion(self, messages, model="llama3-70b-8192"):  # ✅ fixed model name
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"model": model, "messages": messages}
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"Groq Error: {response.text}"

groq_client = GroqClient(GROQ_KEY) if GROQ_KEY else None

# ==============================
# Utility: Extract Questions
# ==============================
def extract_questions_from_pdf(uploaded_file):
    """Extracts questions from UPSC PDF using regex."""
    pdf_doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in pdf_doc:
        text += page.get_text("text")

    # Split into questions (UPSC Qs usually have options A-D)
    raw_questions = re.split(r"Q\.\s*\d+", text, flags=re.IGNORECASE)
    cleaned_questions = [q.strip() for q in raw_questions if q.strip()]

    return cleaned_questions

# ==============================
# Utility: Get AI Answer
# ==============================
def get_solution(question):
    """Gets structured UPSC solution using OpenAI or Groq fallback."""
    messages = [
        {"role": "system", "content": "You are an expert UPSC mentor. Answer in the structured format: 1. Correct Answer, 2. Concept Behind the Question, 3. Explanation of Each Option, 4. Relevance/Current Affairs Linkage, 5. Quick Summary for Revision."},
        {"role": "user", "content": f"Prepare a complete solution for: {question}"}
    ]

    response_text = None

    # Try OpenAI
    if openai_client:
        try:
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )
            response_text = resp.choices[0].message.content
        except Exception as e:
            if "rate_limit" in str(e).lower() and groq_client:
                st.info("⚠️ OpenAI rate limited. Switching to Groq...")
                response_text = groq_client.chat_completion(messages)
            else:
                st.warning(f"⚠️ OpenAI Error: {e}")
            # Add delay to respect free-tier limits
            time.sleep(20)

    # If OpenAI failed → Try Groq
    if not response_text and groq_client:
        response_text = groq_client.chat_completion(messages)

    return response_text

# ==============================
# Streamlit UI
# ==============================
st.title("📘 UPSC PYQ Solution Generator (AI-Powered)")
st.write("Upload a UPSC PYQ PDF and get structured solutions instantly.")

uploaded_file = st.file_uploader("Upload UPSC PYQ PDF", type=["pdf"])

if uploaded_file:
    questions = extract_questions_from_pdf(uploaded_file)
    st.success(f"✅ Extracted {len(questions)} questions from PDF")

    for i, q in enumerate(questions, start=1):
        with st.expander(f"Question {i}"):
            solution = get_solution(q)
            if solution:
                st.markdown(solution)
            else:
                st.error("❌ No valid response (check API keys or try again).")
