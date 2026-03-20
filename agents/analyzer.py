from agents.base import BaseAgent

class AnalyzerAgent(BaseAgent):
    def __init__(self):
        system_prompt = """
        You are a Research Paper Analyzer. Your task is to extract the following from the provided research paper text:
        1. Problem Statement
        2. Methodology
        3. Experiments
        4. Key Findings
        
        Provide a structured analysis.
        Return JSON in the format:
        {
            "content": {
                "problem_statement": "...",
                "methodology": "...",
                "experiments": "...",
                "key_findings": "..."
            },
            "score": 8,
            "status": "completed"
        }
        """
        super().__init__("AnalyzerAgent", system_prompt)
