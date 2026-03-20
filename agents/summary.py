from agents.base import BaseAgent

class SummaryAgent(BaseAgent):
    def __init__(self):
        system_prompt = """
        You are a Summary Generator. Your task is to generate a concise summary (150–200 words) of the research paper analysis provided.
        Ensure the summary captures the essence of the research.
        
        Return JSON in the format:
        {
            "content": "A 150-200 word summary here...",
            "score": 8,
            "status": "completed"
        }
        """
        super().__init__("SummaryAgent", system_prompt)
