# AI Research Paper Analyzer

A production-ready multi-agent system built with Python and LangGraph to analyze academic PDF papers.

## Features
- **PDF Extraction**: Uses `pdfplumber` to extract text.
- **Multi-Agent Workflow**: Optimized for quality with specialized agents:
  - **Analyzer Agent**: Extracts problem, methodology, experiments, and findings.
  - **Summary Agent**: Generates a 150-200 word overview.
  - **Citation Agent**: Extracts references.
  - **Insights Agent**: Provides practical takeaways.
  - **Review Agent**: Evaluates outputs and triggers retries if quality is low (score < 7).
- **LangGraph Integration**: State-managed workflow with conditional retry logic (max 2 retries).

## Project Structure
```text
project/
├── main.py            # Orchestrator & Graph Definition
├── agents/            # Agent implementations
│   ├── base.py
│   ├── analyzer.py
│   ├── summary.py
│   ├── citations.py
│   └── reviewer.py
├── utils/             # Utility functions
│   ├── pdf_processor.py
│   └── logger.py
├── .env               # API Configuration (create from .env.example)
├── requirements.txt   # Dependencies
└── README.md
```

## Setup Instructions

1. **Clone the repository** (or copy files).
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure Environment**:
   - Rename `.env.example` to `.env`.
   - Add your `OPENAI_API_KEY`.
4. **Run the Analyzer**:
   - **CLI Mode**:
     ```bash
     python main.py --pdf path/to/your/paper.pdf
     ```
   - **Web UI Mode (Streamlit)**:
     ```bash
     streamlit run app.py
     ```

## Requirements
- Python 3.9+
- OpenAI API Key (GPT-4o-mini access)
