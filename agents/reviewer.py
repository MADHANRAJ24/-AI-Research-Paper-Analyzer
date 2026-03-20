from agents.base import BaseAgent
from typing import Dict, Any

class ReviewAgent(BaseAgent):
    def __init__(self):
        system_prompt = """
        You are a Quality Review Agent. Your task is to evaluate the research paper analysis, summary, citations, and insights provided.
        Assign a score (1-10) based on clarity, completeness, and accuracy.
        
        If the quality is poor, provide a score below 7.
        If the quality is good, provide a score 7 or above.
        
        Return JSON in the format:
        {
            "content": "Brief feedback feedback on the overall quality.",
            "score": 8,
            "status": "approved"
        }
        """
        super().__init__("ReviewAgent", system_prompt)

    def evaluate(self, state: Dict[str, Any], agent_name: str) -> Dict[str, Any]:
        """
        Evaluates a specific agent's output.
        """
        agent_output = state.get(agent_name, {})
        # Create a specific prompt for evaluation
        evaluation_input = f"Evaluate the following {agent_name} output:\n{agent_output}"
        return self.invoke(evaluation_input)
