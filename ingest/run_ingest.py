import os
import sys
import logging

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from ingest.clean_text import clean_extracted_text
from ingest.chunk_sections import parse_and_chunk, save_chunks_to_jsonl
from ingest.spot_check import run_spot_check
from src.ingest.extract_pdf import extract_text_from_pdf


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_ingestion_pipeline():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_pdf_path = os.path.join(base_dir, "data/raw/labour_act_2006_en_mcci.pdf")
    output_chunks_path = os.path.join(base_dir, "data/processed/chunks.jsonl")
    
    if not os.path.exists(raw_pdf_path):
        logger.error(f"Source PDF not found at {raw_pdf_path}")
        sys.exit(1)
        
    logger.info(f"Step 1: Extracting text from PDF: {raw_pdf_path}")
    raw_text = extract_text_from_pdf(raw_pdf_path)
    logger.info(f"Extracted {len(raw_text)} raw characters.")
    
    logger.info("Step 2: Cleaning headers, footers, and whitespace...")
    cleaned_text = clean_extracted_text(raw_text)
    logger.info(f"Cleaned text length: {len(cleaned_text)} characters.")
    
    logger.info("Step 3: Chunking text along statutory section boundaries...")
    chunks = parse_and_chunk(cleaned_text, doc_name="labour_act_2006_en_mcci.pdf", law_version_date="2018-10-24")
    
    logger.info(f"Step 4: Saving {len(chunks)} chunks to {output_chunks_path}")
    save_chunks_to_jsonl(chunks, output_chunks_path)
    
    logger.info("Step 5: Running spot-check validation on key sections...")
    passed = run_spot_check(output_chunks_path)
    
    if passed:
        logger.info("INGESTION PIPELINE COMPLETED SUCCESSFULLY!")
    else:
        logger.warning("Ingestion pipeline completed with spot-check warnings.")

if __name__ == "__main__":
    run_ingestion_pipeline()
