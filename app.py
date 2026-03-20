import streamlit as st
import os
import json
import tempfile
from utils.logger import setup_logger

# Setup logger
logger = setup_logger()

# Page configuration
st.set_page_config(
    page_title="AI Research Paper Analyzer",
    page_icon="📄",
    layout="wide"
)

# Sidebar for settings
with st.sidebar:
    st.title("Settings")
    api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    st.info("Upload a research paper in PDF format to get a comprehensive analysis using multi-agent AI.")

# Main UI
st.title("📄 AI Research Paper Analyzer")
st.markdown("Analyze research papers using a multi-agent system powered by LangGraph and GPT-4o-mini.")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("Please provide an OpenAI API Key in the sidebar to proceed.")
    else:
        if st.button("Analyze Paper"):
            with st.spinner("Analyzing paper... This may take a minute as multiple agents work on it."):
                from main import run_analyzer
                # Save uploaded file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    # Run the analyzer
                    result = run_analyzer(tmp_path)
                    
                    if result:
                        st.success("Analysis Complete!")
                        
                        # Display results in tabs
                        tab1, tab2, tab3, tab4, tab5 = st.tabs([
                            "Summary", "Core Analysis", "Citations", "Key Insights", "System Scores"
                        ])
                        
                        with tab1:
                            st.header("Summary")
                            st.write(result.get("summary", "No summary generated."))
                            
                        with tab2:
                            st.header("Core Analysis")
                            analysis = result.get("analysis", {})
                            st.subheader("Problem Statement")
                            st.write(analysis.get("problem_statement", "N/A"))
                            st.subheader("Methodology")
                            st.write(analysis.get("methodology", "N/A"))
                            st.subheader("Experiments")
                            st.write(analysis.get("experiments", "N/A"))
                            st.subheader("Key Findings")
                            st.write(analysis.get("key_findings", "N/A"))
                            
                        with tab3:
                            st.header("Citations & References")
                            citations = result.get("citations", [])
                            if citations:
                                for i, cite in enumerate(citations):
                                    st.markdown(f"{i+1}. {cite}")
                            else:
                                st.write("No citations extracted.")
                                
                        with tab4:
                            st.header("Key Insights & Takeaways")
                            insights = result.get("insights", [])
                            if insights:
                                for i, insight in enumerate(insights):
                                    st.markdown(f"- {insight}")
                            else:
                                st.write("No insights generated.")
                                
                        with tab5:
                            st.header("Agent Quality Scores")
                            scores = result.get("overall_scores", {})
                            for agent, score in scores.items():
                                st.metric(label=f"{agent.capitalize()} Agent", value=f"{score}/10")
                                st.progress(score / 10.0)
                        
                        # Raw JSON download
                        st.divider()
                        st.subheader("Raw Data")
                        st.download_button(
                            label="Download Analysis JSON",
                            data=json.dumps(result, indent=2),
                            file_name="analysis_results.json",
                            mime="application/json"
                        )
                    else:
                        st.error("Analysis failed. Please check the logs.")
                        
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    logger.error(f"Streamlit Error: {e}")
                finally:
                    # Clean up temp file
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

st.divider()
st.caption("Built with LangGraph, OpenAI, and Streamlit.")
