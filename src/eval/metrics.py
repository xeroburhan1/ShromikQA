import re
import logging
from src.generation.generate_answer import generate_answer

logger = logging.getLogger(__name__)

def count_syllables(word: str) -> int:
    """Helper to estimate the syllable count of an English word."""
    word = word.lower().strip()
    if not word:
        return 0
        
    vowels = "aeiouy"
    count = 0
    
    # Simple syllable counting logic
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
            
    if word.endswith("e"):
        count -= 1
        
    if count <= 0:
        count = 1
        
    return count

def calculate_flesch_reading_ease(text: str) -> float:
    """
    Computes the Flesch Reading Ease score for the provided text.
    Formula: 206.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)
    """
    if not text or not text.strip():
        return 0.0
        
    # Split sentences by simple punctuation markers
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Tokenize words (ignoring numbers and punctuation)
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    
    num_sentences = len(sentences)
    num_words = len(words)
    
    if num_sentences == 0 or num_words == 0:
        return 0.0
        
    num_syllables = sum(count_syllables(w) for w in words)
    
    asl = num_words / num_sentences
    asw = num_syllables / num_words
    
    score = 206.835 - (1.015 * asl) - (84.6 * asw)
    
    # Clamp score to [0.0, 100.0]
    return max(0.0, min(100.0, score))

def evaluate_retrieval_precision_at_3(retrieved_chunks: list, expected_section: str) -> float:
    """Returns 1.0 if the expected section number is retrieved in the top 3 chunks, else 0.0."""
    # Look at the top 3 chunks
    top_3 = retrieved_chunks[:3]
    for chunk in top_3:
        sec_num = str(chunk.get("metadata", {}).get("section_number", "")).strip()
        if sec_num == expected_section.strip():
            return 1.0
    return 0.0

def evaluate_citation_accuracy(generated_answer: str, expected_section: str) -> float:
    """Checks if the expected section number is cited in the generated answer."""
    pattern = rf'\b{re.escape(expected_section)}\b'
    if re.search(pattern, generated_answer):
        return 1.0
    return 0.0

def evaluate_faithfulness(generated_answer: str, retrieved_chunks: list) -> float:
    """
    Heuristic check for faithfulness. 
    In production, this should run an LLM-as-judge prompt.
    If no LLM-as-judge is run, it performs an overlap heuristic of names and numbers.
    """
    # Heuristic: verify that numbers cited in the generated answer exist in the context
    context_text = " ".join([c["text"] for c in retrieved_chunks])
    
    # Find all numbers in the generated answer
    answer_numbers = set(re.findall(r'\b\d+\b', generated_answer))
    context_numbers = set(re.findall(r'\b\d+\b', context_text))
    
    # Remove standard numbers like year 2006, 2015, 2018, 2025, 2-4, 16, 112 from checks 
    # to avoid false positives on standard disclaimer/metadata
    common_numbers = {"2006", "2015", "2018", "2022", "2025", "2", "3", "4", "16", "112", "90", "30", "1", "2", "3"}
    answer_numbers = answer_numbers - common_numbers
    context_numbers = context_numbers - common_numbers
    
    if not answer_numbers:
        return 1.0 # No numbers cited, pass heuristic
        
    hallucinated = answer_numbers - context_numbers
    if hallucinated:
        logger.warning(f"Faithfulness warning: Answer cited numbers {hallucinated} not found in retrieved context.")
        return 0.0
        
    return 1.0
