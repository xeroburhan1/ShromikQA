import os
import sys
import streamlit as st
from dotenv import load_dotenv

# Path setup
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from rag.orchestrator import RAGOrchestrator

# Streamlit Page Configuration
st.set_page_config(
    page_title="Shromic AI — Bangladesh Labour Act RAG",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Payload CMS Dark Tech Aesthetic Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global Reset & Base Styling */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #09090B !important;
        color: #FAFAFA !important;
    }

    /* Completely Hide Streamlit Sidebar & Header Chrome */
    section[data-testid="stSidebar"] {
        display: none !important;
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    .main .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 960px !important;
    }

    /* Payload CMS Navbar */
    .payload-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 20px;
        background: #121215;
        border: 1px solid #27272A;
        border-radius: 12px;
        margin-bottom: 2.5rem;
    }
    .payload-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        color: #FFFFFF;
        letter-spacing: -0.02em;
    }
    .payload-logo {
        width: 26px;
        height: 26px;
        background: #22C55E;
        color: #000000;
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 0.9rem;
    }
    .payload-status {
        display: flex;
        align-items: center;
        gap: 8px;
        background: rgba(34, 197, 94, 0.08);
        border: 1px solid rgba(34, 197, 94, 0.2);
        color: #4ADE80;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 20px;
        font-family: 'JetBrains Mono', monospace;
    }
    .pulse-dot {
        width: 6px;
        height: 6px;
        background-color: #22C55E;
        border-radius: 50%;
        box-shadow: 0 0 8px #22C55E;
    }

    /* Hero Section */
    .hero-box {
        text-align: center;
        padding: 2rem 0 2rem 0;
    }
    .hero-tag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #18181B;
        border: 1px solid #27272A;
        color: #A1A1AA;
        font-size: 0.8rem;
        font-weight: 500;
        padding: 4px 14px;
        border-radius: 20px;
        margin-bottom: 1.2rem;
    }
    .hero-h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        line-height: 1.15;
        letter-spacing: -0.03em;
        color: #FFFFFF;
        margin-bottom: 1rem;
    }
    .hero-h1 span {
        background: linear-gradient(135deg, #22C55E 0%, #A7F3D0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-p {
        font-size: 1.05rem;
        color: #A1A1AA;
        max-width: 640px;
        margin: 0 auto 2.5rem auto;
        line-height: 1.6;
    }

    /* Streamlit Buttons Custom Styling for Prompt Grid */
    div.stButton > button {
        background-color: #141417 !important;
        color: #E4E4E7 !important;
        border: 1px solid #27272A !important;
        border-radius: 12px !important;
        padding: 14px 16px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        text-align: left !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
        box-shadow: none !important;
    }
    div.stButton > button:hover {
        background-color: #1F1F23 !important;
        border-color: #3F3F46 !important;
        color: #FFFFFF !important;
        transform: translateY(-2px);
    }

    /* Chat Messages High-Contrast Dark Styling */
    .stChatMessage {
        background-color: #141417 !important;
        border: 1px solid #27272A !important;
        border-radius: 14px !important;
        padding: 1.25rem !important;
        margin-bottom: 1.2rem !important;
        color: #FAFAFA !important;
    }
    .stChatMessage [data-testid="stMarkdownContainer"] p,
    .stChatMessage [data-testid="stMarkdownContainer"] li,
    .stChatMessage [data-testid="stMarkdownContainer"] div {
        color: #F4F4F5 !important;
        font-size: 0.96rem !important;
        line-height: 1.6 !important;
    }
    .stChatMessage [data-testid="stMarkdownContainer"] strong {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    /* Statutory Section Badges */
    .sec-tag {
        display: inline-flex;
        align-items: center;
        background: rgba(34, 197, 94, 0.12);
        color: #4ADE80;
        border: 1px solid rgba(34, 197, 94, 0.25);
        font-size: 0.78rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        margin-right: 6px;
    }
    
    /* Expanders & TextAreas */
    .stExpander {
        background-color: #0F0F12 !important;
        border: 1px solid #27272A !important;
        border-radius: 10px !important;
        margin-top: 1rem !important;
    }
    .stExpander [data-testid="stMarkdownContainer"] p {
        color: #A1A1AA !important;
    }
    textarea {
        background-color: #09090B !important;
        color: #D4D4D8 !important;
        border: 1px solid #27272A !important;
        border-radius: 8px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem !important;
    }

    /* Chat Input Bar Styling */
    .stChatInputContainer textarea {
        background-color: #141417 !important;
        color: #FFFFFF !important;
        border: 1px solid #27272A !important;
        border-radius: 12px !important;
    }
    .stChatInputContainer textarea:focus {
        border-color: #22C55E !important;
        box-shadow: 0 0 0 1px #22C55E !important;
    }
</style>
""", unsafe_allow_html=True)

# Payload Top Nav
st.markdown('''
    <div class="payload-nav">
        <div class="payload-brand">
            <div class="payload-logo">⚡</div>
            <span>Shromic AI</span>
        </div>
        <div class="payload-status">
            <div class="pulse-dot"></div>
            <span>FAISS Index Active (374 Sections)</span>
        </div>
    </div>
''', unsafe_allow_html=True)

@st.cache_resource
def get_orchestrator():
    load_dotenv()
    index_path = os.path.join(base_dir, "data/processed/faiss_index.bin")
    meta_path = os.path.join(base_dir, "data/processed/faiss_metadata.json")
    return RAGOrchestrator(index_path=index_path, meta_path=meta_path)

orchestrator = get_orchestrator()

# Chat History Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hero Screen (when chat is empty)
if len(st.session_state.messages) == 0:
    st.markdown('''
        <div class="hero-box">
            <div class="hero-tag">⚡ Bangladesh Labour Act 2006 RAG Agent</div>
            <div class="hero-h1">Legal Intelligence Grounded in <span>Statutory Text</span></div>
            <div class="hero-p">Ask any question regarding working hours, termination notice, leave entitlement, or dispute procedures. Answers strictly cite statutory section numbers.</div>
        </div>
    ''', unsafe_allow_html=True)

    # 4 Quick Prompt Cards Grid
    c1, c2 = st.columns(2)
    c3, c4 = st.columns(2)
    
    sample_selected = None
    
    with c1:
        if st.button("🚪 Section 27 — Termination Notice Period\nWhat notice is required for permanent workers?", key="btn1"):
            sample_selected = "What is the notice period for terminating a permanent worker under Section 27?"
            
    with c2:
        if st.button("⏱️ Section 100 — Daily Working Hours\nWhat are the maximum daily hours for adult workers?", key="btn2"):
            sample_selected = "What are maximum daily working hours for an adult worker under Section 100?"

    with c3:
        if st.button("🏖️ Section 117 — Annual Leave Calculation\nHow is annual leave with wages calculated in a factory?", key="btn3"):
            sample_selected = "How is annual leave with wages calculated for adult workers in a factory under Section 117?"

    with c4:
        if st.button("💼 Section 33 — Grievance Complaint Procedure\nWhat is the procedure & timeline for submitting a complaint?", key="btn4"):
            sample_selected = "What is the procedure for an aggrieved worker to submit a written complaint under Section 33?"

    st.markdown("<br>", unsafe_allow_html=True)
else:
    sample_selected = None

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "chunks" in msg and msg["chunks"]:
            with st.expander("🔎 View Grounding Statutory Text (FAISS + Section Override)"):
                for idx, chunk in enumerate(msg["chunks"]):
                    st.markdown(f"<span class='sec-tag'>Section {chunk.get('section_number')}</span> <strong style='color:#FAFAFA;'>{chunk.get('section_title')}</strong>", unsafe_allow_html=True)
                    st.caption(f"Source: {chunk.get('source_doc')} | Page: {chunk.get('page_number')} | Method: {chunk.get('retrieval_source')}")
                    st.text_area(f"Section {chunk.get('section_number')} Statutory Content", value=chunk.get("text"), height=110, key=f"hist_{msg['id']}_{idx}")

# Chat Input Bar
user_input = st.chat_input("Ask a question about the Bangladesh Labour Act...") or sample_selected

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input, "id": len(st.session_state.messages)})
    with st.chat_message("user"):
        st.markdown(user_input)
        
    with st.chat_message("assistant"):
        with st.spinner("Searching FAISS index & generating grounded response..."):
            res = orchestrator.answer_question(user_input, top_k=5)
            answer_text = res["answer"]
            chunks = res["retrieved_chunks"]
            
        st.markdown(answer_text)
        
        if chunks:
            with st.expander("🔎 View Grounding Statutory Text (FAISS + Section Override)"):
                for idx, chunk in enumerate(chunks):
                    st.markdown(f"<span class='sec-tag'>Section {chunk.get('section_number')}</span> <strong style='color:#FAFAFA;'>{chunk.get('section_title')}</strong>", unsafe_allow_html=True)
                    st.caption(f"Source: {chunk.get('source_doc')} | Page: {chunk.get('page_number')} | Method: {chunk.get('retrieval_source')}")
                    st.text_area(f"Section {chunk.get('section_number')} Statutory Content", value=chunk.get("text"), height=110, key=f"curr_{len(st.session_state.messages)}_{idx}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer_text,
        "chunks": chunks,
        "id": len(st.session_state.messages)
    })
