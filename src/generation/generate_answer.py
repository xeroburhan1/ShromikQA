import os
import logging
from dotenv import load_dotenv
from src.generation.prompt_templates import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def format_context(chunks: list) -> str:
    """Formats retrieved chunks with metadata headers for the LLM prompt context."""
    formatted_parts = []
    for idx, chunk in enumerate(chunks):
        meta = chunk.get("metadata", {})
        formatted_parts.append(
            f"--- Chunk {idx + 1} ---\n"
            f"Document: {meta.get('source_doc', 'Unknown')}\n"
            f"Chapter: {meta.get('chapter', 'Unknown')}\n"
            f"Section: {meta.get('section_number', 'Unknown')}. {meta.get('section_title', 'Unknown')}\n"
            f"Page: {meta.get('page_number', 'Unknown')}\n"
            f"Content:\n{chunk['text']}\n"
        )
    return "\n".join(formatted_parts)

def generate_answer(retrieved_chunks: list, user_question: str) -> str:
    """Generates an answer using the configured LLM provider and API keys."""
    load_dotenv(override=True)
    
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    
    # Format chunks context
    if not retrieved_chunks:
        # Strict rule: return a default answer if there are no chunks
        return (
            "**Direct Answer:** I don't have enough information in my corpus to answer this.\n"
            "**Relevant Section(s):** None\n"
            "**Source:** None\n"
            "**Explanation:** No context was retrieved from the database to verify this query.\n"
            "**Disclaimer:** This is general information, not legal advice. Consult a qualified labour lawyer for advice on your specific situation."
        )
        
    context_text = format_context(retrieved_chunks)
    user_prompt = USER_PROMPT_TEMPLATE.format(context_text=context_text, user_question=user_question)
    
    logger.info(f"Generating answer using provider: {provider}")
    
    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            # Fallback check if user has standard API key set in system env
            api_key = os.getenv("GOOGLE_API_KEY")
        
        if not api_key:
            logger.error("Gemini API key not found in environment.")
            return "Error: GEMINI_API_KEY or GOOGLE_API_KEY environment variable is missing. Please check your .env file."
            
        import time
        max_retries = 5
        retry_delay = 12.0
        for attempt in range(max_retries):
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
                model = genai.GenerativeModel(
                    model_name=model_name,
                    system_instruction=SYSTEM_PROMPT
                )
                response = model.generate_content(user_prompt)
                return response.text
            except Exception as e:
                err_msg = str(e)
                if "quota" in err_msg.lower():
                    logger.error(f"Gemini API daily quota limit reached: {e}")
                    return f"Error: Gemini API quota exceeded. {e}"
                if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                    logger.warning(f"Gemini API rate limit hit (429). Retrying in {retry_delay}s (Attempt {attempt+1}/{max_retries})...")
                    time.sleep(retry_delay)
                    retry_delay *= 1.5
                else:
                    logger.error(f"Gemini API call failed: {e}")
                    return f"Error occurred during Gemini API generation: {e}"
        return "Error: Gemini API rate limit exceeded after multiple retries."
            
    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OpenAI API key not found in environment.")
            return "Error: OPENAI_API_KEY is missing. Please check your .env file."
            
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
            return f"Error occurred during OpenAI API generation: {e}"
            
    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.error("Anthropic API key not found in environment.")
            return "Error: ANTHROPIC_API_KEY is missing. Please check your .env file."
            
        try:
            from anthropic import Anthropic
            client = Anthropic(api_key=api_key)
            model_name = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")
            response = client.messages.create(
                model=model_name,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API call failed: {e}")
            return f"Error occurred during Anthropic API generation: {e}"
            
    elif provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.error("Groq API key not found in environment.")
            return "Error: GROQ_API_KEY is missing. Please check your .env file."
            
        try:
            from groq import Groq
            client = Groq(api_key=api_key)
            model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-specdec")
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
            return f"Error occurred during Groq API generation: {e}"
            
    else:
        logger.error(f"Unsupported LLM provider: {provider}")
        return f"Error: Unsupported LLM provider '{provider}'. Must be 'gemini', 'openai', 'anthropic', or 'groq'."
