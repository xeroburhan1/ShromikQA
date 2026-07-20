import os
import re
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# API configuration
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = os.getenv("API_PORT", "8000")
API_URL = f"http://{API_HOST}:{API_PORT}/ask"

st.set_page_config(
    page_title="Bangladesh Labour Law LLM Assistant",
    page_icon="⚖️",
    layout="wide"
)

# Custom premium styling for UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
    }
    
    .main-title {
        background: linear-gradient(135deg, #1e3a8a 0%, #0d9488 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 10px;
    }
    
    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 1.2rem;
        margin-bottom: 40px;
    }
    
    .card-container {
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05), 0 2px 4px -2px rgb(0 0 0 / 0.05);
        border: 1px solid rgba(226, 232, 240, 0.8);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .card-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.08), 0 4px 6px -4px rgb(0 0 0 / 0.08);
    }
    
    .direct-answer-card {
        background: rgba(16, 185, 129, 0.03);
        border-left: 5px solid #10b981;
    }
    
    .sections-card {
        background: rgba(245, 158, 11, 0.03);
        border-left: 5px solid #f59e0b;
    }
    
    .source-card {
        background: rgba(59, 130, 246, 0.03);
        border-left: 5px solid #3b82f6;
    }
    
    .explanation-card {
        background: #ffffff;
        border-left: 5px solid #64748b;
    }
    
    .disclaimer-card {
        background: rgba(244, 63, 94, 0.03);
        border-left: 5px solid #f43f5e;
        color: #475569;
        font-size: 0.95rem;
    }
    
    .card-title {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 8px;
        color: #0f172a;
    }
    
    .source-badge {
        display: inline-block;
        background: #e2e8f0;
        color: #334155;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
        margin-right: 5px;
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

def parse_response(response_text: str) -> dict:
    """Parses LLM structured response into separate components using regular expressions."""
    sections = {
        "direct_answer": "",
        "sections": "",
        "source": "",
        "explanation": "",
        "disclaimer": ""
    }
    
    da_match = re.search(r'\*\*Direct Answer:\*\*(.*?)(?=\*\*Relevant Section\(s\):\*\*|\*\*Source:\*\*|\*\*Explanation:\*\*|\*\*Disclaimer:\*\*|$)', response_text, re.DOTALL | re.IGNORECASE)
    sec_match = re.search(r'\*\*Relevant Section\(s\):\*\*(.*?)(?=\*\*Direct Answer:\*\*|\*\*Source:\*\*|\*\*Explanation:\*\*|\*\*Disclaimer:\*\*|$)', response_text, re.DOTALL | re.IGNORECASE)
    src_match = re.search(r'\*\*Source:\*\*(.*?)(?=\*\*Direct Answer:\*\*|\*\*Relevant Section\(s\):\*\*|\*\*Explanation:\*\*|\*\*Disclaimer:\*\*|$)', response_text, re.DOTALL | re.IGNORECASE)
    exp_match = re.search(r'\*\*Explanation:\*\*(.*?)(?=\*\*Direct Answer:\*\*|\*\*Relevant Section\(s\):\*\*|\*\*Source:\*\*|\*\*Disclaimer:\*\*|$)', response_text, re.DOTALL | re.IGNORECASE)
    disc_match = re.search(r'\*\*Disclaimer:\*\*(.*?)(?=\*\*Direct Answer:\*\*|\*\*Relevant Section\(s\):\*\*|\*\*Source:\*\*|\*\*Explanation:\*\*|$)', response_text, re.DOTALL | re.IGNORECASE)
    
    if da_match: sections["direct_answer"] = da_match.group(1).strip()
    if sec_match: sections["sections"] = sec_match.group(1).strip()
    if src_match: sections["source"] = src_match.group(1).strip()
    if exp_match: sections["explanation"] = exp_match.group(1).strip()
    if disc_match: sections["disclaimer"] = disc_match.group(1).strip()
    
    # Fallback if parser failed
    if not sections["direct_answer"] and not sections["explanation"]:
        sections["direct_answer"] = response_text
        
    return sections

def main():
    st.markdown('<div class="main-title">⚖️ Bangladesh Labour Law Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">A verified RAG system grounded in Bangladesh Labour Act 2006 & amendments</div>', unsafe_allow_html=True)
    
    # Layout splits into query column and sidebar metadata
    col_left, col_right = st.columns([3, 1])
    
    with col_right:
        st.markdown("### 📋 System Information")
        st.info("""
        - **Data Freshness**: Up to 2018 Amendments (2025 Ordinance limitations applied).
        - **Multilingual Support**: Handles English and Bengali queries.
        - **Retrieval Engine**: Vector search (Cosine) + BM25 keyword fusion.
        - **Disclaimer**: General guidance only, not official legal advice.
        """)
        
        st.markdown("### 💡 Try Asking:")
        examples = [
            "Can my employer terminate me without notice?",
            "How many days of maternity leave am I entitled to?",
            "What is the overtime pay rate?",
            "How is gratuity calculated under the Labour Act?",
            "Can I be dismissed for joining a trade union?"
        ]
        for ex in examples:
            if st.button(ex, use_container_width=True):
                st.session_state["query_input"] = ex
                st.rerun()

    with col_left:
        # Check and get session state input
        query_val = st.session_state.get("query_input", "")
        
        user_query = st.text_input(
            "Ask a question about Bangladesh Labour Law:",
            value=query_val,
            placeholder="e.g. What is the limit on daily working hours?",
            key="user_query_input"
        )
        
        # Reset st.session_state.query_input after using it
        if "query_input" in st.session_state:
            del st.session_state["query_input"]
            
        if st.button("Submit Question", type="primary") or (user_query and user_query != query_val):
            if not user_query:
                st.warning("Please enter a question.")
                return
                
            st.write("---")
            with st.spinner("Searching files and generating answer..."):
                try:
                    response = requests.post(API_URL, json={"question": user_query})
                    if response.status_code == 200:
                        data = response.json()
                        answer_text = data.get("answer", "")
                        retrieved_chunks = data.get("retrieved_chunks", [])
                        
                        # Parse
                        parsed = parse_response(answer_text)
                        
                        # Render Direct Answer Card
                        st.markdown(f"""
                        <div class="card-container direct-answer-card">
                            <div class="card-title">🟢 Direct Answer</div>
                            <div style="font-size: 1.1rem; line-height: 1.5;">{parsed['direct_answer']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Render Sections Card
                        if parsed['sections']:
                            st.markdown(f"""
                            <div class="card-container sections-card">
                                <div class="card-title">📜 Relevant Section(s)</div>
                                <div style="font-size: 1rem; font-weight: 500;">{parsed['sections']}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        # Render Source Card
                        if parsed['source']:
                            st.markdown(f"""
                            <div class="card-container source-card">
                                <div class="card-title">📁 Source Citation</div>
                                <div style="font-size: 0.95rem;">{parsed['source']}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        # Render Explanation Card
                        if parsed['explanation']:
                            st.markdown(f"""
                            <div class="card-container explanation-card">
                                <div class="card-title">📖 Plain-Language Explanation</div>
                                <div style="line-height: 1.6; color: #334155;">{parsed['explanation']}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        # Render Disclaimer Card
                        disclaimer_text = parsed['disclaimer'] if parsed['disclaimer'] else "This is general information, not legal advice. Consult a qualified labour lawyer for advice on your specific situation."
                        st.markdown(f"""
                        <div class="card-container disclaimer-card">
                            <div class="card-title">⚠️ Legal Disclaimer</div>
                            <div>{disclaimer_text}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Render Retrieved Chunks expandable section
                        with st.expander("🔍 Show Retrieved Sources (for verification)"):
                            if not retrieved_chunks:
                                st.write("No source chunks retrieved.")
                            else:
                                for idx, chunk in enumerate(retrieved_chunks):
                                    meta = chunk.get("metadata", {})
                                    st.markdown(f"**Source {idx + 1}: {meta.get('source_doc', 'Unknown')}** - Chapter {meta.get('chapter', 'Unknown')} (Section {meta.get('section_number', 'Unknown')}) - Page {meta.get('page_number', 'Unknown')}")
                                    st.text_area(f"Text content of Source {idx + 1}", value=chunk['text'], height=150, key=f"src_txt_{idx}")
                    else:
                        st.error(f"Error calling API (Status {response.status_code}): {response.text}")
                except Exception as e:
                    st.error(f"Failed to connect to API server: {e}. Make sure the FastAPI backend is running.")

if __name__ == "__main__":
    main()
