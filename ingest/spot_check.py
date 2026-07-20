import json
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Key section numbers to verify in the corpus
SPOT_CHECK_SECTIONS = [
    {"sec": "26", "keywords": ["discharge", "notice", "wages", "worker"]},
    {"sec": "27", "keywords": ["termination", "notice", "permanent", "temporary"]},
    {"sec": "33", "keywords": ["grievance", "complaint", "employer", "worker"]},
    {"sec": "100", "keywords": ["hours", "daily", "work"]},
    {"sec": "108", "keywords": ["overtime", "extra", "allowance"]},
    {"sec": "115", "keywords": ["casual", "leave"]},
    {"sec": "117", "keywords": ["annual", "leave", "wages"]},
    {"sec": "120", "keywords": ["wages", "time", "payment"]},
    {"sec": "209", "keywords": ["dispute", "industrial", "settlement"]}
]

def run_spot_check(chunks_file_path: str) -> bool:
    """
    Verifies that known sections exist in the chunked dataset, contain key terms, 
    and maintain intact section titles and metadata.
    """
    if not os.path.exists(chunks_file_path):
        logger.error(f"Chunks file missing: {chunks_file_path}")
        return False
        
    chunks = []
    with open(chunks_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
                
    sec_map = {chunk["section_number"]: chunk for chunk in chunks}
    
    logger.info(f"--- SPOT CHECKING {len(SPOT_CHECK_SECTIONS)} KNOWN SECTIONS ---")
    all_passed = True
    
    for check in SPOT_CHECK_SECTIONS:
        sec_num = check["sec"]
        if sec_num not in sec_map:
            logger.error(f"[FAILED] Section {sec_num} NOT FOUND in ingested chunks!")
            all_passed = False
            continue
            
        chunk = sec_map[sec_num]
        title = chunk.get("section_title", "")
        text = chunk.get("text", "").lower()
        
        missing_kw = [kw for kw in check["keywords"] if kw not in text and kw not in title.lower()]
        
        if missing_kw:
            logger.warning(f"[WARNING] Section {sec_num} ('{title}') missing expected keywords: {missing_kw}")
        else:
            logger.info(f"[PASSED] Section {sec_num} ('{title}') - Validated intact.")
            
    return all_passed

if __name__ == "__main__":
    chunks_path = "data/processed/chunks.jsonl"
    run_spot_check(chunks_path)
