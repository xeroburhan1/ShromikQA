import os
import sys
import pdfplumber
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def ocr_page(pdf_path: str, page_idx: int) -> str:
    """Runs OCR on a single page of a PDF by converting it to an image first."""
    try:
        from pdf2image import convert_from_path
        import pytesseract
        
        logger.info(f"Attempting OCR on page {page_idx + 1} of {pdf_path}")
        # Convert only the specific page (1-indexed in convert_from_path)
        images = convert_from_path(pdf_path, first_page=page_idx + 1, last_page=page_idx + 1)
        if images:
            text = pytesseract.image_to_string(images[0])
            return text
    except ImportError:
        logger.warning("pdf2image or pytesseract is not installed. OCR fallback is unavailable.")
    except Exception as e:
        logger.error(f"OCR failed for page {page_idx + 1}: {e}")
    return ""

def ocr_full_document(pdf_path: str) -> str:
    """Runs OCR on all pages of a PDF by converting them to images."""
    try:
        from pdf2image import convert_from_path
        import pytesseract
        
        logger.info(f"Converting entire PDF to images for OCR: {pdf_path}")
        images = convert_from_path(pdf_path)
        logger.info(f"Converted {len(images)} pages. Starting OCR...")
        
        text_content = []
        for idx, img in enumerate(images):
            logger.info(f"Running OCR on page {idx + 1}/{len(images)}...")
            page_text = pytesseract.image_to_string(img)
            text_content.append(page_text)
            
        return "\n".join(text_content)
    except ImportError:
        logger.error("pdf2image or pytesseract is not installed. OCR fallback is unavailable.")
        return ""
    except Exception as e:
        logger.error(f"Full document OCR failed: {e}")
        return ""

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text from PDF using pdfplumber, falling back to OCR if pages are blank/scanned."""
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file does not exist: {pdf_path}")
        return ""
        
    logger.info(f"Extracting text from: {pdf_path}")
    text_content = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                # Try extracting text natively
                page_text = page.extract_text()
                
                # Check if page has sufficient text content
                if page_text and len(page_text.strip()) > 20:
                    text_content.append(page_text)
                else:
                    logger.warning(f"Page {page_idx + 1} has insufficient native text. Attempting OCR...")
                    page_text_ocr = ocr_page(pdf_path, page_idx)
                    if page_text_ocr and len(page_text_ocr.strip()) > 0:
                        text_content.append(page_text_ocr)
                    else:
                        text_content.append("") # Keep page indices aligned
    except Exception as e:
        logger.error(f"Error using pdfplumber: {e}. Falling back to full document OCR.")
        return ocr_full_document(pdf_path)
        
    full_text = "\n\n--- PAGE BREAK ---\n\n".join(text_content)
    logger.info(f"Finished extracting. Total length: {len(full_text)} characters.")
    return full_text

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_pdf.py <path_to_pdf>")
        sys.exit(1)
        
    pdf_path = sys.argv[1]
    text = extract_text_from_pdf(pdf_path)
    print(f"Extracted {len(text)} characters.")
