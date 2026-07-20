import re
import logging

logger = logging.getLogger(__name__)

def clean_extracted_text(text: str) -> str:
    """
    Cleans raw text by removing repetitive page headers, footers, 
    gazette publication notices, and redundant whitespace.
    """
    if not text:
        return ""
        
    # List of common header/footer/gazette patterns to remove
    patterns_to_remove = [
        # Gazette header patterns
        r"REGISTERED No\.\s+DA-1",
        r"The\s+Bangladesh\s+Gazette\s+Extraordinary\s+Published\s+by\s+Authority",
        r"THURSDAY,\s+OCTOBER\s+11,\s+2006",
        r"THE\s+BANGLADESH\s+GAZETTE,\s+EXTRA,\s+OCTOBER\s+11,\s+2006",
        r"GOVERNMENT\s+OF\s+THE\s+PEOPLE'S\s+REPUBLIC\s+OF\s+BANGLADESH",
        r"MINISTRY\s+OF\s+LAW,\s+JUSTICE\s+AND\s+PARLIAMENTARY\s+AFFAIRS",
        # Book/Publishing running headers/footers
        r"Bangladesh\s+Labour\s+Act,\s+2006",
        r"Bangladesh\s+Labor\s+Act,\s+2006",
        r"MCCI\s+Labour\s+Act\s+2006",
        r"Doulah\s+&\s+Doulah",
        r"www\.doulah\.net",
        # Section reference headers in some publications
        r"\[Part\s+I",
        r"PART\s+I\]"
    ]
    
    cleaned = text
    
    # Apply removals case-insensitively
    for pattern in patterns_to_remove:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        
    # Remove stand-alone page numbers (e.g., a number alone on a line)
    cleaned = re.sub(r'^\s*\d+\s*$', '', cleaned, flags=re.MULTILINE)
    
    # Remove lines consisting only of hyphens, underscores, stars
    cleaned = re.sub(r'^\s*[-_*=]+\s*$', '', cleaned, flags=re.MULTILINE)
    
    # Normalize multiple consecutive empty lines to a single empty line
    cleaned = re.sub(r'\n\s*\n+', '\n\n', cleaned)
    
    # Normalize multiple whitespaces
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    
    return cleaned.strip()

if __name__ == "__main__":
    sample_text = """
    REGISTERED No. DA-1
    THE BANGLADESH GAZETTE, EXTRA, OCTOBER 11, 2006
    Bangladesh Labour Act, 2006
    -------------------------
    27. Termination of employment by employer otherwise than by dismissal.- 
    (1) An employer may terminate the employment of a permanent worker...
    
    --- PAGE BREAK ---
    Bangladesh Labour Act, 2006
    18
    """
    print("Original length:", len(sample_text))
    cleaned = clean_extracted_text(sample_text)
    print("Cleaned text:")
    print(cleaned)
