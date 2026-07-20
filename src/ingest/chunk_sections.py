import re
import json
import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Pattern to match Chapter header, e.g., "CHAPTER I", "CHAPTER II", "CHAPTER X"
CHAPTER_PATTERN = re.compile(r'^\s*CHAPTER\s+([IVXLCDM\d]+)', re.IGNORECASE)

# Pattern to match Section header, e.g., "27. Termination of employment by employer otherwise than by dismissal.-"
# Matches: Group 1 = Section number, Group 2 = Section title
SECTION_PATTERN = re.compile(r'^\s*(\d{1,3}[A-Z]?)\.\s+([^.\n\-]{4,})(?:\.|\-|\.\-)', re.IGNORECASE)

# Pattern to match sub-sections, e.g., "(1)", "(2)", "(3)" at the start of a line or paragraph
SUBSECTION_PATTERN = re.compile(r'^\s*\((\d+)\)\s+')

# Pattern to match Schedule headers (e.g., THE FIRST SCHEDULE)
SCHEDULE_PATTERN = re.compile(r'^\s*(THE\s+\w+\s+)?SCHEDULE', re.IGNORECASE)

def split_large_section(text: str, section_num: str, section_title: str, max_tokens: int = 800) -> list:
    """
    If a section's text exceeds max_tokens (approx 3200 characters), 
    split it at sub-section boundaries (1), (2), etc.
    Prepends the section title to each sub-chunk to preserve context.
    """
    # Quick character-length heuristic: 1 token is ~4 characters. 800 tokens is ~3200 characters.
    if len(text) < max_tokens * 4:
        return [text]
        
    logger.info(f"Section {section_num} is long ({len(text)} chars). Splitting by sub-sections...")
    
    # Split text into lines/paragraphs
    paragraphs = text.split("\n")
    sub_chunks = []
    current_chunk_parts = []
    current_length = 0
    
    for para in paragraphs:
        # Check if this paragraph starts a new sub-section like (1), (2), etc.
        if SUBSECTION_PATTERN.match(para) and current_length > 0:
            # Combine current chunk parts
            sub_chunks.append("\n".join(current_chunk_parts))
            current_chunk_parts = [para]
            current_length = len(para)
        else:
            current_chunk_parts.append(para)
            current_length += len(para)
            
    if current_chunk_parts:
        sub_chunks.append("\n".join(current_chunk_parts))
        
    # Prepend section context to all split chunks except possibly the first one (which already contains it)
    final_chunks = []
    context_prefix = f"Section {section_num}. {section_title} (Continued) -\n"
    
    for idx, chunk in enumerate(sub_chunks):
        if idx == 0:
            final_chunks.append(chunk)
        else:
            final_chunks.append(context_prefix + chunk.strip())
            
    return final_chunks

def parse_and_chunk(raw_text: str, doc_name: str, law_version_date: str) -> list:
    """
    Parses the raw text, identifies Chapters and Sections, 
    and returns a list of chunk dictionaries.
    """
    chunks = []
    
    # Split raw text by pages using our page break marker
    pages = raw_text.split("--- PAGE BREAK ---")
    
    current_chapter = "Unknown"
    current_section_num = None
    current_section_title = None
    current_section_lines = []
    current_section_start_page = 1
    in_schedule = False
    
    def save_current_section(page_num):
        nonlocal current_section_num, current_section_title, current_section_lines, current_chapter, current_section_start_page
        if current_section_num is not None:
            section_text = "\n".join(current_section_lines).strip()
            if section_text:
                # Check and split if too large
                sub_texts = split_large_section(section_text, current_section_num, current_section_title)
                for idx, sub_txt in enumerate(sub_texts):
                    chunk_id = f"{doc_name.split('.')[0]}_sec{current_section_num}"
                    if len(sub_texts) > 1:
                        chunk_id += f"_p{idx+1}"
                        
                    chunks.append({
                        "chunk_id": chunk_id,
                        "chapter": current_chapter,
                        "section_number": current_section_num,
                        "section_title": current_section_title,
                        "text": sub_txt,
                        "source_doc": doc_name,
                        "page_number": current_section_start_page,
                        "law_version_date": law_version_date
                    })
            current_section_lines = []
            current_section_num = None
            current_section_title = None
    
    for page_idx, page_content in enumerate(pages):
        page_num = page_idx + 1
        lines = page_content.split("\n")
        
        for line in lines:
            # Check for Schedule start to stop section matching
            if SCHEDULE_PATTERN.match(line):
                save_current_section(page_num)
                in_schedule = True
                continue
                
            if in_schedule:
                continue
                
            # Check for Chapter change
            chap_match = CHAPTER_PATTERN.match(line)
            if chap_match:
                # Save previous section before changing chapter
                save_current_section(page_num)
                # Keep the chapter line for context
                current_chapter = line.strip()
                continue
                
            # Check for Section change
            sec_match = SECTION_PATTERN.match(line)
            if sec_match:
                # Save previous section
                save_current_section(page_num)
                
                # Start new section
                current_section_num = sec_match.group(1).strip()
                current_section_title = sec_match.group(2).strip()
                current_section_start_page = page_num
                current_section_lines = [line.strip()]
            else:
                if current_section_num is not None:
                    current_section_lines.append(line)
                    
    # Save the final section
    save_current_section(len(pages))
    
    logger.info(f"Parsing complete. Generated {len(chunks)} chunks.")
    return chunks

def save_chunks_to_jsonl(chunks: list, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(chunks)} chunks to {output_path}")

if __name__ == "__main__":
    # Test with dummy text
    dummy_text = """
    CHAPTER I
    PRELIMINARY
    1. Short title, commencement and application.- (1) This Act may be called the Bangladesh Labour Act, 2006.
    (2) It shall come into force at once.
    (3) It applies to the whole of Bangladesh.
    
    CHAPTER V
    V. Termination of employment by employer otherwise than by dismissal.-
    27. Termination of employment by employer otherwise than by dismissal.-
    (1) An employer may terminate the employment of a permanent worker by giving in writing to the worker ninety days notice.
    (2) An employer may terminate the employment of a temporary worker by giving in writing to the worker thirty days notice.
    
    --- PAGE BREAK ---
    28. Retrenchment.- (1) A worker may be retrenched from service on the ground of redundancy.
    """
    
    chunks = parse_and_chunk(dummy_text, "test_doc.pdf", "2006-10-11")
    print(json.dumps(chunks, indent=2))
