import os
import sys
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a highly precise, authoritative legal AI assistant specializing in the Bangladesh Labour Act 2006 (and its amendments up to 2018 & Labour Rules 2015).

STRICT COMPLIANCE RULES:
1. Grounding: You MUST answer the user's question using ONLY the legal statutory text provided in the retrieved Context below. Do NOT use outside knowledge, case law, or speculation.
2. Section Citation: You MUST explicitly cite the relevant Section Number (e.g. "Section 27", "Section 100") for EVERY factual claim or rule you state.
3. Fallback: If the provided Context does NOT contain enough information to answer the question, state clearly: "I don't have enough information in my statutory corpus to answer this question."
4. Structure: Format your output with clear, structured Markdown headers:
   - **Direct Answer:** A concise 1-2 sentence response.
   - **Relevant Section(s):** Explicit list of cited legal sections.
   - **Explanation:** Detailed breakdown citing specific sub-sections and requirements.
   - **Disclaimer:** "This is general informational guidance based strictly on the Bangladesh Labour Act 2006 statutory text, not official legal advice."
"""

USER_PROMPT_TEMPLATE = """CONTEXT STATUTORY SECTIONS:
{context_text}

USER QUESTION:
{user_question}

Provide a grounded response adhering strictly to the system instructions.
"""

def format_context(chunks: list) -> str:
    formatted_parts = []
    for idx, chunk in enumerate(chunks):
        formatted_parts.append(
            f"--- Statutory Section Chunk {idx + 1} ---\n"
            f"Document: {chunk.get('source_doc', 'Labour Act 2006')}\n"
            f"Chapter: {chunk.get('chapter', 'Unknown')}\n"
            f"Section: {chunk.get('section_number', 'Unknown')}. {chunk.get('section_title', 'Unknown')}\n"
            f"Content:\n{chunk.get('text', '')}\n"
        )
    return "\n".join(formatted_parts)

def generate_answer(retrieved_chunks: list, user_question: str) -> str:
    """
    Generates an answer using the configured LLM API provider (Anthropic, Gemini, or OpenAI).
    Defaults to Anthropic API as requested by prompt.
    """
    load_dotenv(override=True)
    
    if not retrieved_chunks:
        return (
            "**Direct Answer:** I don't have enough information in my statutory corpus to answer this.\n\n"
            "**Relevant Section(s):** None\n\n"
            "**Explanation:** No context was retrieved from the statutory database to verify this query.\n\n"
            "**Disclaimer:** This is general informational guidance based strictly on the Bangladesh Labour Act 2006 statutory text, not official legal advice."
        )
        
    context_text = format_context(retrieved_chunks)
    user_prompt = USER_PROMPT_TEMPLATE.format(context_text=context_text, user_question=user_question)
    
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    
    # Auto-detect provider if default is requested but key missing
    if provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            provider = "gemini"
        elif os.getenv("OPENAI_API_KEY"):
            provider = "openai"

    logger.info(f"Generating LLM response via provider: {provider}")
    
    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return "Error: ANTHROPIC_API_KEY missing in .env file."
        try:
            from anthropic import Anthropic
            client = Anthropic(api_key=api_key)
            model_name = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")
            response = client.messages.create(
                model=model_name,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0.0
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API call failed: {e}")
            return f"Error during Anthropic API generation: {e}"

    elif provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return "Error: GEMINI_API_KEY / GOOGLE_API_KEY missing in .env file."
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            model = genai.GenerativeModel(model_name=model_name, system_instruction=SYSTEM_PROMPT)
            response = model.generate_content(user_prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            return f"Error during Gemini API generation: {e}"

    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "Error: OPENAI_API_KEY missing in .env file."
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            return f"Error during OpenAI API generation: {e}"

    elif provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return "Error: GROQ_API_KEY missing in .env file."
        try:
            import httpx
            from groq import Groq
            client = Groq(api_key=api_key, http_client=httpx.Client())
            model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq API call failed: {e}")
            return f"Error during Groq API generation: {e}"


    else:
        return f"Error: Unsupported LLM_PROVIDER '{provider}'."


if __name__ == "__main__":
    dummy_chunks = [{
        "source_doc": "labour_act_2006_en_mcci.pdf",
        "chapter": "CHAPTER IV",
        "section_number": "27",
        "section_title": "Termination of employment by worker",
        "text": "(1) A permanent worker may terminate his employment by giving 60 days notice."
    }]
    print(generate_answer(dummy_chunks, "What notice is required for permanent workers under Section 27?"))
