import os
import pdfplumber
import logging
from typing import List

logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts all text from a PDF file using pdfplumber.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")
    
    full_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
        return full_text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise e

def chunk_text(text: str, chunk_size: int = 10000) -> List[str]:
    """
    Splits text into chunks of specified size to stay within LLM context limits.
    In a real-world scenario, we might use recursive character splitter.
    """
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
