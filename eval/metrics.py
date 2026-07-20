import re
import logging

logger = logging.getLogger(__name__)

def calculate_recall_at_k(retrieved_chunks: list, expected_section: str, k: int) -> float:
    """
    Returns 1.0 if expected_section number is present within top-k retrieved chunks, else 0.0.
    """
    top_k_chunks = retrieved_chunks[:k]
    for chunk in top_k_chunks:
        sec_num = str(chunk.get("section_number", "")).strip()
        if sec_num == str(expected_section).strip():
            return 1.0
    return 0.0

def calculate_reciprocal_rank(retrieved_chunks: list, expected_section: str) -> float:
    """
    Returns 1 / rank (1-based) of the first occurrence of expected_section in retrieved_chunks, else 0.0.
    """
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        sec_num = str(chunk.get("section_number", "")).strip()
        if sec_num == str(expected_section).strip():
            return 1.0 / rank
    return 0.0

def verify_citation_hallucination(answer: str, expected_section: str, retrieved_chunks: list) -> dict:
    """
    Direct Hallucination Check:
    1. Checks if the generated answer cites the expected section number.
    2. Verifies that the cited section text actually exists in the retrieved context.
    """
    cited_sections = set(re.findall(r'(?:section|sec\.?|ধারা)\s*#?\s*(\d{1,3}[a-z]?)', answer, re.IGNORECASE))
    
    cites_expected = str(expected_section).strip() in cited_sections
    
    # Check if the retrieved chunk for expected section contains content
    matching_text = ""
    for chunk in retrieved_chunks:
        if str(chunk.get("section_number", "")).strip() == str(expected_section).strip():
            matching_text = chunk.get("text", "")
            break
            
    has_grounding_context = len(matching_text) > 0
    
    # Hallucination occurs if answer cites a section not present in retrieved context or wrong section
    is_faithful = cites_expected and has_grounding_context
    
    return {
        "cites_expected_section": cites_expected,
        "has_grounding_context": has_grounding_context,
        "is_faithful_unhallucinated": is_faithful,
        "cited_sections_found": list(cited_sections)
    }
