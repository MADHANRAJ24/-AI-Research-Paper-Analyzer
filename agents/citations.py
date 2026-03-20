from agents.base import BaseAgent

class CitationAgent(BaseAgent):
    def __init__(self):
        system_prompt = """
        You are a Citation Extractor. Your task is to extract the main references and citations from the provided research paper text.
        List them in a structured way.
        
        Return JSON in the format:
        {
            "content": ["Citation 1", "Citation 2", ...],
            "score": 8,
            "status": "completed"
        }
        """
        super().__init__("CitationAgent", system_prompt)

class InsightAgent(BaseAgent):
    def __init__(self):
        system_prompt = """
        You are a Key Insights Agent. Your task is to generate practical takeaways and insights from the research paper analysis.
        What are the real-world applications?
        
        Return JSON in the format:
        {
            "content": ["Insight 1", "Insight 2", ...],
            "score": 8,
            "status": "completed"
        }
        """
        super().__init__("InsightAgent", system_prompt)
