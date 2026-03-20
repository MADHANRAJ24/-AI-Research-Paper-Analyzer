import logging
import sys
import os
from dotenv import load_dotenv

def setup_logger():
    """
    Configures basic logging for the project.
    """
    load_dotenv()
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("AIResearchPaperAnalyzer")
