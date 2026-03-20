import os
from typing import Dict, Any, Type
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class AgentOutput(BaseModel):
    """
    Standard output format for all agents.
    """
    content: Any = Field(description="The actual content produced by the agent")
    score: int = Field(description="The score (1-10) assigned by the agent (used mostly by reviewer nodes or self-assessment)")
    status: str = Field(description="The status of the task (e.g., 'completed', 'approved')")

class BaseAgent:
    def __init__(self, name: str, system_prompt: str):
        self.name = name
        self.system_prompt = system_prompt
        self._llm = None  # Lazy initialization
        self.parser = JsonOutputParser(pydantic_object=AgentOutput)
        
    @property
    def llm(self):
        if self._llm is None:
            self._llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        return self._llm
        
    def _create_prompt(self):
        return ChatPromptTemplate.from_messages([
            ("system", self.system_prompt + "\n\nCRITICAL: Return ONLY a valid JSON object matching the requested schema."),
            ("human", "{input_text}")
        ])

    def invoke(self, input_text: str) -> Dict[str, Any]:
        """
        Generic invoke method for agents.
        """
        prompt = self._create_prompt()
        chain = prompt | self.llm | self.parser
        return chain.invoke({"input_text": input_text})
