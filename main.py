import os
import json
import logging
from typing import Dict, Any, List, TypedDict, Annotated, Literal
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

# Import agents
from agents.analyzer import AnalyzerAgent
from agents.summary import SummaryAgent
from agents.citations import CitationAgent, InsightAgent
from agents.reviewer import ReviewAgent

# Import utilities
from utils.pdf_processor import extract_text_from_pdf, chunk_text
from utils.logger import setup_logger

# Load environment variables
load_dotenv()

# Setup logging
logger = setup_logger()

# --- 1. Define State Class ---
class State(TypedDict):
    paper_text: str
    analysis: Dict[str, Any]
    summary: Dict[str, Any]
    citations: Dict[str, Any]
    insights: Dict[str, Any]
    scores: Dict[str, int]
    retry_counts: Dict[str, int]
    current_agent: str
    final_output: Dict[str, Any]

# --- 2. Initialize Agents ---
analyzer_agent = AnalyzerAgent()
summary_agent = SummaryAgent()
citation_agent = CitationAgent()
insight_agent = InsightAgent()
review_agent = ReviewAgent()

# --- 3. Define Nodes ---

def analyzer_node(state: State) -> State:
    logger.info("Node: Analyzer")
    result = analyzer_agent.invoke(state["paper_text"][:10000]) # Chunking if too long
    state["analysis"] = result
    state["current_agent"] = "analysis"
    return state

def summary_node(state: State) -> State:
    logger.info("Node: Summary Generator")
    # Feed the analysis to the summary generator
    input_text = json.dumps(state["analysis"]["content"])
    result = summary_agent.invoke(input_text)
    state["summary"] = result
    state["current_agent"] = "summary"
    return state

def citation_node(state: State) -> State:
    logger.info("Node: Citation Extractor")
    result = citation_agent.invoke(state["paper_text"][:10000])
    state["citations"] = result
    state["current_agent"] = "citations"
    return state

def insights_node(state: State) -> State:
    logger.info("Node: Key Insights Generator")
    input_text = json.dumps(state["analysis"]["content"])
    result = insight_agent.invoke(input_text)
    state["insights"] = result
    state["current_agent"] = "insights"
    return state

def review_node(state: State) -> State:
    logger.info(f"Node: Reviewer (checking {state['current_agent']})")
    current_agent = state["current_agent"]
    agent_output = state[current_agent]
    
    # Use the reviewer agent to evaluate
    review_result = review_agent.evaluate(state, current_agent)
    
    score = review_result.get("score", 0)
    state["scores"][current_agent] = score
    logger.info(f"Review for {current_agent}: Score {score}")
    
    return state

def final_node(state: State) -> State:
    logger.info("Node: Final Node - Aggregating results")
    state["final_output"] = {
        "analysis": state["analysis"]["content"],
        "summary": state["summary"]["content"],
        "citations": state["citations"]["content"],
        "insights": state["insights"]["content"],
        "overall_scores": state["scores"]
    }
    return state

# --- 4. Define Conditional Logic for Retries ---

def should_retry(state: State) -> Literal["retry", "next", "fail"]:
    current_agent = state["current_agent"]
    score = state["scores"].get(current_agent, 0)
    retry_count = state["retry_counts"].get(current_agent, 0)
    
    if score >= 7:
        logger.info(f"{current_agent} passed with score {score}")
        # Determine next node based on current agent
        if current_agent == "analysis": return "to_summary"
        if current_agent == "summary": return "to_citations"
        if current_agent == "citations": return "to_insights"
        if current_agent == "insights": return "to_final"
        return "next"
    
    if retry_count < 2:
        logger.warning(f"{current_agent} failed (score {score}). Retrying ({retry_count + 1}/2)...")
        state["retry_counts"][current_agent] = retry_count + 1
        return "retry"
    
    logger.error(f"{current_agent} failed max retries.")
    return "next" # Proceed anyway or fail? Requirement says "Limit retry to 2"

# Specific router functions for LangGraph
def route_after_analyzer(state: State):
    res = should_retry(state)
    if res == "retry": return "analyzer"
    return "summary"

def route_after_summary(state: State):
    res = should_retry(state)
    if res == "retry": return "summary"
    return "citations"

def route_after_citations(state: State):
    res = should_retry(state)
    if res == "retry": return "citations"
    return "insights"

def route_after_insights(state: State):
    res = should_retry(state)
    if res == "retry": return "insights"
    return "final"

# --- 5. Build Graph ---

workflow = StateGraph(State)

# Add Nodes
workflow.add_node("analyzer", analyzer_node)
workflow.add_node("summary", summary_node)
workflow.add_node("citations", citation_node)
workflow.add_node("insights", insights_node)
workflow.add_node("reviewer", review_node)
workflow.add_node("final", final_node)

# Add Edges
workflow.set_entry_point("analyzer")

# Logic Flow: Node -> Reviewer -> Conditional Edge (Retry or Next)
workflow.add_edge("analyzer", "reviewer")
workflow.add_conditional_edges(
    "reviewer",
    route_after_analyzer,
    {
        "analyzer": "analyzer",
        "summary": "summary"
    }
)

workflow.add_edge("summary", "reviewer")
workflow.add_conditional_edges(
    "reviewer",
    route_after_summary,
    {
        "summary": "summary",
        "citations": "citations"
    }
)

workflow.add_edge("citations", "reviewer")
workflow.add_conditional_edges(
    "reviewer",
    route_after_citations,
    {
        "citations": "citations",
        "insights": "insights"
    }
)

workflow.add_edge("insights", "reviewer")
workflow.add_conditional_edges(
    "reviewer",
    route_after_insights,
    {
        "insights": "insights",
        "final": "final"
    }
)

workflow.add_edge("final", END)

# Compile
app = workflow.compile()

# --- 6. Execution Loop ---

def run_analyzer(pdf_path: str):
    print(f"\n--- Starting Analysis for: {pdf_path} ---")
    
    try:
        text = extract_text_from_pdf(pdf_path)
        initial_state = {
            "paper_text": text,
            "analysis": {},
            "summary": {},
            "citations": {},
            "insights": {},
            "scores": {},
            "retry_counts": {"analysis": 0, "summary": 0, "citations": 0, "insights": 0},
            "current_agent": "",
            "final_output": {}
        }
        
        result = app.invoke(initial_state)
        
        print("\n--- Final Results ---")
        print(json.dumps(result["final_output"], indent=2))
        return result["final_output"]
        
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AI Research Paper Analyzer")
    parser.add_argument("--pdf", type=str, help="Path to the PDF file", required=True)
    args = parser.parse_args()
    
    run_analyzer(args.pdf)
