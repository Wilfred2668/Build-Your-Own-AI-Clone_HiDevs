import streamlit as st
import time
from datetime import datetime
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import wikipediaapi
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq

# Import the services from main app
from streamlit_app_simple import get_services

# Initialize services
services = get_services()

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'knowledge_base' not in st.session_state:
    st.session_state.knowledge_base = []
if 'evaluation_metrics' not in st.session_state:
    st.session_state.evaluation_metrics = []

def main():
    st.title("💬 AI Chat Interface")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key configuration
        st.subheader("🔑 Groq API Configuration")
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            help="Enter your Groq API key for enhanced AI responses"
        )
        
        if api_key:
            # Test API key (only show once per session)
            if 'api_key_tested' not in st.session_state or st.session_state.get('last_api_key') != api_key:
                try:
                    from streamlit_app_simple import AIService
                    test_ai = AIService(api_key)
                    if test_ai.llm:
                        st.success("✅ Groq API Key is valid and working!")
                    else:
                        st.error("❌ Groq API Key is invalid or not working")
                except Exception as e:
                    st.error("❌ Groq API Key is invalid or not working")
                st.session_state.api_key_tested = True
                st.session_state.last_api_key = api_key
        else:
            st.warning("⚠️ No API key provided - using fallback mode")
        
        # Knowledge base stats
        st.header("📊 Knowledge Base")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Articles", len(st.session_state.knowledge_base))
        with col2:
            st.metric("Chunks", len(services['knowledge_base'].documents))
        
        # Evaluation metrics
        if st.session_state.evaluation_metrics:
            st.header("📈 Evaluation")
            summary = services['evaluation'].get_metrics_summary()
            if summary:
                st.metric("Avg Relevance", f"{summary['avg_context_relevance']:.2f}")
                st.metric("Avg Coherence", f"{summary['avg_coherence']:.2f}")
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("📚 Data Loading")
        
        # Wikipedia search
        search_query = st.text_input("Search Wikipedia articles")
        if st.button("🔍 Search"):
            if search_query:
                with st.spinner("Searching Wikipedia..."):
                    articles = services['wikipedia'].search_articles(search_query, 5)
                    st.session_state.search_results = articles
                    st.success(f"Found {len(articles)} articles")
        
        # Display search results
        if 'search_results' in st.session_state and st.session_state.search_results:
            st.subheader("Search Results")
            for article in st.session_state.search_results:
                with st.expander(article['title']):
                    st.write(article['summary'][:200] + "...")
                    if st.button(f"📥 Load {article['title']}", key=article['title']):
                        with st.spinner(f"Loading {article['title']}..."):
                            full_article = services['wikipedia'].get_article_content(article['title'])
                            if full_article:
                                services['knowledge_base'].add_documents([full_article])
                                st.session_state.knowledge_base.append(full_article)
                                st.success(f"✅ {article['title']} loaded")
        
        # Clear knowledge base
        if st.session_state.knowledge_base:
            if st.button("🗑️ Clear Knowledge Base"):
                st.session_state.knowledge_base = []
                services['knowledge_base'].clear()
                st.success("Knowledge base cleared")
    
    with col2:
        st.header("💬 AI Chat")
        
        # Chat container with scroll
        chat_container = st.container()
        
        with chat_container:
            # Chat interface
            for i, message in enumerate(st.session_state.messages):
                if message["role"] == "user":
                    # User message with better styling
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, #667eea, #764ba2);
                        color: white;
                        padding: 1rem;
                        border-radius: 15px;
                        margin: 1rem 0;
                        margin-left: 20%;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    ">
                        <div style="font-weight: bold; margin-bottom: 0.5rem;">👤 You</div>
                        <div>{message["content"]}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # AI message with enhanced styling and metadata
                    confidence_color = "green" if message.get("confidence", 0) > 0.7 else "orange" if message.get("confidence", 0) > 0.4 else "red"

                    st.markdown(f"""
                    <div style="
                        background: rgba(255, 255, 255, 0.1);
                        border: 1px solid rgba(255, 255, 255, 0.2);
                        padding: 1.5rem;
                        border-radius: 15px;
                        margin: 1rem 0;
                        margin-right: 20%;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    ">
                        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                            <div style="font-weight: bold; margin-right: 1rem;">🤖 Assistant</div>
                            <div style="
                                background: {confidence_color};
                                color: white;
                                padding: 0.2rem 0.5rem;
                                border-radius: 10px;
                                font-size: 0.8rem;
                                font-weight: bold;
                            ">
                                Confidence: {message.get("confidence", 0):.1%}
                            </div>
                        </div>
                        <div style="line-height: 1.6; margin-bottom: 1rem;">{message["content"]}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Chat input
        if prompt := st.chat_input("Ask me anything..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.spinner("🤔 Thinking..."):
                # Retrieve relevant context
                context = services['knowledge_base'].search(prompt, 5)
                
                # Silent context retrieval - no messages
                if not context:
                    st.warning("⚠️ No relevant context found. Please load some articles first.")
                
                # Generate response
                from streamlit_app_simple import AIService
                ai_service = AIService(api_key)
                response = ai_service.generate_response(prompt, context)
                
                # Evaluate response
                evaluation = services['evaluation'].evaluate_response(
                    prompt, response['response'], context, response['processing_time']
                )
                st.session_state.evaluation_metrics.append(evaluation)
                
                # Add response to chat
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response['response'],
                    "sources": response['sources'],
                    "confidence": response['confidence']
                })
            
            st.rerun()
    
    # Evaluation dashboard
    if st.session_state.evaluation_metrics:
        st.header("📊 Evaluation Dashboard")
        
        summary = services['evaluation'].get_metrics_summary()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Queries", summary['total_queries'])
        with col2:
            st.metric("Avg Response Length", f"{summary['avg_response_length']:.0f}")
        with col3:
            st.metric("Avg Context Relevance", f"{summary['avg_context_relevance']:.2f}")
        with col4:
            st.metric("Avg Coherence", f"{summary['avg_coherence']:.2f}")
        
        # Metrics over time
        if len(st.session_state.evaluation_metrics) > 1:
            df = pd.DataFrame(st.session_state.evaluation_metrics)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            st.subheader("Performance Over Time")
            chart_data = df.set_index('timestamp')[['context_relevance', 'response_coherence']]
            st.line_chart(chart_data)

if __name__ == "__main__":
    main() 