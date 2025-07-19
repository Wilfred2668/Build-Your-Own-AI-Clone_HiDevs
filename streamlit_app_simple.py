import streamlit as st
import os
import json
import time
import requests
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import wikipediaapi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq

# Page configuration
st.set_page_config(
    page_title="GenAI RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .feature-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(45deg, #667eea, #764ba2);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        margin-left: 20%;
    }
    .assistant-message {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-right: 20%;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #5a6fd8, #6a4190);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'knowledge_base' not in st.session_state:
    st.session_state.knowledge_base = []
if 'evaluation_metrics' not in st.session_state:
    st.session_state.evaluation_metrics = []

class WikipediaService:
    def __init__(self):
        self.wiki = wikipediaapi.Wikipedia(
            user_agent='GenAI-RAG-Chatbot/1.0 (https://github.com/your-repo; your-email@example.com)',
            language='en'
        )
    
    def search_articles(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search Wikipedia articles"""
        try:
            # Use the correct search method
            search_results = self.wiki.search(query, results=limit)
            articles = []
            
            for title in search_results:
                page = self.wiki.page(title)
                if page.exists():
                    articles.append({
                        'title': page.title,
                        'summary': page.summary,
                        'url': page.fullurl,
                        'content': page.text[:5000]  # Limit content for processing
                    })
            
            return articles
        except AttributeError:
            # Fallback method if search doesn't exist
            try:
                # Try using requests to search Wikipedia API directly
                import requests
                
                search_url = "https://en.wikipedia.org/w/api.php"
                params = {
                    'action': 'query',
                    'format': 'json',
                    'list': 'search',
                    'srsearch': query,
                    'srlimit': limit,
                    'origin': '*'
                }
                
                response = requests.get(search_url, params=params)
                data = response.json()
                
                articles = []
                if 'query' in data and 'search' in data['query']:
                    for result in data['query']['search']:
                        # Get full page content
                        page = self.wiki.page(result['title'])
                        if page.exists():
                            articles.append({
                                'title': page.title,
                                'summary': page.summary,
                                'url': page.fullurl,
                                'content': page.text[:5000]
                            })
                
                return articles
            except Exception as e:
                st.error(f"Error searching Wikipedia: {str(e)}")
                return []
        except Exception as e:
            st.error(f"Error searching Wikipedia: {str(e)}")
            return []
    
    def get_article_content(self, title: str) -> Optional[Dict[str, Any]]:
        """Get full article content"""
        try:
            page = self.wiki.page(title)
            if page.exists() and page.text:
                content = page.text.strip()
                if len(content) > 100:  # Ensure we have meaningful content
                    return {
                        'title': page.title,
                        'content': content,
                        'url': page.fullurl,
                        'summary': page.summary
                    }
                else:
                    st.warning(f"⚠️ Article '{title}' has insufficient content")
                    return None
            else:
                st.warning(f"⚠️ Article '{title}' not found or empty")
                return None
        except Exception as e:
            st.error(f"❌ Error fetching article '{title}': {str(e)}")
            return None

class SimpleKnowledgeBase:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
        )
        self.documents = []
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to knowledge base"""
        try:
            for doc in documents:
                if not doc.get('content'):
                    st.warning(f"⚠️ No content found for {doc.get('title', 'Unknown')}")
                    continue
                
                chunks = self.text_splitter.split_text(doc['content'])
                for i, chunk in enumerate(chunks):
                    if chunk.strip():  # Only add non-empty chunks
                        self.documents.append({
                            'content': chunk,
                            'metadata': {
                                'source': doc.get('title', 'Unknown'),
                                'url': doc.get('url', ''),
                                'chunk_id': i
                            }
                        })
            
            st.success(f"✅ Added {len(chunks)} chunks to knowledge base")
        except Exception as e:
            st.error(f"❌ Error adding documents: {str(e)}")
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Enhanced keyword-based search with better relevance scoring"""
        if not self.documents:
            st.warning("⚠️ No documents in knowledge base. Please load some articles first.")
            return []
        
        query_words = set(query.lower().split())
        results = []
        
        for doc in self.documents:
            content_lower = doc['content'].lower()
            content_words = set(content_lower.split())
            
            # Calculate word overlap
            overlap = len(query_words.intersection(content_words))
            
            # Enhanced scoring: consider word frequency and position
            if overlap > 0:
                # Base score from word overlap
                base_score = overlap / len(query_words) if query_words else 0
                
                # Bonus for exact phrase matches
                phrase_bonus = 0
                for word in query_words:
                    if word in content_lower:
                        phrase_bonus += 0.1
                
                # Bonus for important words (longer words tend to be more specific)
                importance_bonus = sum(len(word) * 0.01 for word in query_words if word in content_lower)
                
                final_score = base_score + phrase_bonus + importance_bonus
                
                results.append({
                    'content': doc['content'],
                    'metadata': doc['metadata'],
                    'score': min(final_score, 1.0)  # Cap at 1.0
                })
        
        # Sort by score and return top k
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:k]
    
    def clear(self):
        """Clear all documents"""
        self.documents = []

class AIService:
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.llm = None
        if api_key and api_key.strip():
            try:
                self.llm = ChatGroq(
                    groq_api_key=api_key,
                    model_name="llama3-8b-8192"
                )
                # Test the connection with a simple prompt
                test_response = self.llm.invoke("Hello")
                if not test_response:
                    raise Exception("No response from Groq API")
            except Exception as e:
                self.llm = None
                raise e
    
    def generate_response(self, query: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate AI response with RAG using structured knowledge base"""
        start_time = time.time()
        
        # Prepare context for processing
        if not context:
            return {
                'response': "I don't have enough information to answer your question. Please load some Wikipedia articles first.",
                'sources': [],
                'confidence': 0.1,
                'processing_time': time.time() - start_time
            }
        
        # Build structured prompt with better context organization
        context_chunks = []
        for i, item in enumerate(context, 1):
            content = item['content'][:1000]  # Increased content length
            relevance = item.get('score', 0.0)
            context_chunks.append(f"""CHUNK {i} (Source: {item['metadata']['source']}, Relevance: {relevance:.2f}):
{content}""")
        
        context_text = "\n\n" + "="*50 + "\n\n".join(context_chunks)
        
        structured_prompt = f"""You are an AI assistant with access to Wikipedia knowledge. Analyze ALL provided chunks and create a comprehensive, structured answer.

KNOWLEDGE BASE CHUNKS:
{context_text}

USER QUESTION: {query}

ANALYSIS INSTRUCTIONS:
1. CAREFULLY ANALYZE ALL chunks above to understand the complete context
2. Look for the EXACT topic or keywords mentioned in the user's question
3. If the user's question contains words/concepts that are NOT present in ANY chunk, respond ONLY with: "I don't have information about that in my knowledge base."
4. If the topic/keywords ARE found in the chunks, proceed to create a structured answer

RESPONSE STRUCTURE (if information is found):
1. Start with a clear, direct answer to the user's question
2. Provide detailed explanations using information from ALL relevant chunks
3. Organize information logically with proper paragraphs
4. Include specific details, examples, and context from the chunks
5. End with: "Sources: [list unique source titles only, no duplicates]"

STRICT RULES:
- ONLY use information from the provided chunks
- DO NOT add any external knowledge or assumptions
- DO NOT suggest searching other sources
- If the user asks about something not in the chunks, say "I don't have information about that in my knowledge base."
- Be comprehensive but stay within the provided information

Please analyze the chunks and provide your response:"""

        try:
            if self.llm:
                # Use Groq LLM with structured prompt
                try:
                    # Add system message to enforce strict behavior
                    system_message = """You are a knowledge base assistant with STRICT rules:

1. ANALYZE ALL chunks carefully before responding
2. Look for EXACT keywords/topics from the user's question in the chunks
3. If the user's question contains words that are NOT found in ANY chunk, respond ONLY with: "I don't have information about that in my knowledge base."
4. DO NOT provide information about topics not explicitly mentioned in the chunks
5. DO NOT use any external knowledge or make assumptions
6. When providing answers, be comprehensive but stay within the provided information
7. Always end responses with proper source citations

You can ONLY provide information that is explicitly present in the provided knowledge base."""
                    
                    full_prompt = f"{system_message}\n\n{structured_prompt}"
                    llm_response = self.llm.invoke(full_prompt)
                    response = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
                    confidence = 0.9
                except Exception as groq_error:
                    # Fallback to rule-based response
                    response = self._rule_based_response(query, "\n".join([item['content'] for item in context]))
                    confidence = 0.6
            else:
                # Fallback to rule-based response
                response = self._rule_based_response(query, "\n".join([item['content'] for item in context]))
                confidence = 0.6
            
            # Get unique sources and limit to avoid repetition
            sources = list(dict.fromkeys([item['metadata']['source'] for item in context]))
            
            # Clean up response to remove duplicate sources in the text
            if "Sources:" in response:
                # Extract the sources part and clean it
                parts = response.split("Sources:")
                if len(parts) > 1:
                    main_content = parts[0].strip()
                    sources_text = parts[1].strip()
                    # Clean up sources text to remove duplicates
                    sources_list = [s.strip() for s in sources_text.split(',') if s.strip()]
                    unique_sources = list(dict.fromkeys(sources_list))
                    clean_sources_text = ", ".join(unique_sources)
                    response = f"{main_content}\n\nSources: {clean_sources_text}"
            
            return {
                'response': response,
                'sources': sources,
                'confidence': confidence,
                'processing_time': time.time() - start_time
            }
        
        except Exception as e:
            st.error(f"❌ Error generating response: {str(e)}")
            st.error(f"Error type: {type(e).__name__}")
            st.error(f"Error details: {e}")
            return {
                'response': "I encountered an error while processing your question. Please try again.",
                'sources': [],
                'confidence': 0.0,
                'processing_time': time.time() - start_time
            }
    
    def _rule_based_response(self, query: str, context: str) -> str:
        """Fallback rule-based response"""
        query_words = set(query.lower().split())
        sentences = context.split('.')
        
        # Find sentences that actually answer the question
        relevant_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            # Check if sentence contains words from the question
            if any(word in sentence_lower for word in query_words):
                relevant_sentences.append(sentence.strip())
        
        if relevant_sentences:
            # Provide more detailed response
            response = " ".join(relevant_sentences[:4])  # Increased from 2 to 4 sentences
            # Add some context if available
            if len(relevant_sentences) > 4:
                response += f" Additionally, {relevant_sentences[4]}"
            return response
        else:
            return "I don't have information about that in my knowledge base."

class EvaluationService:
    def __init__(self):
        self.metrics = []
    
    def evaluate_response(self, query: str, response: str, context: List[Dict[str, Any]], processing_time: float = 0) -> Dict[str, Any]:
        """Evaluate response quality"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response_length': len(response),
            'context_relevance': self._calculate_context_relevance(query, context),
            'response_coherence': self._calculate_coherence(response),
            'source_diversity': len(set([item['metadata']['source'] for item in context])),
            'processing_time': processing_time
        }
        
        self.metrics.append(metrics)
        return metrics
    
    def _calculate_context_relevance(self, query: str, context: List[Dict[str, Any]]) -> float:
        """Calculate relevance of retrieved context"""
        if not context:
            return 0.0
        
        query_words = set(query.lower().split())
        total_relevance = 0
        
        for item in context:
            content_words = set(item['content'].lower().split())
            overlap = len(query_words.intersection(content_words))
            relevance = overlap / len(query_words) if query_words else 0
            total_relevance += relevance
        
        return total_relevance / len(context)
    
    def _calculate_coherence(self, response: str) -> float:
        """Calculate response coherence (simplified)"""
        sentences = response.split('.')
        if len(sentences) <= 1:
            return 1.0
        
        # Simple coherence metric based on sentence length consistency
        lengths = [len(s.strip()) for s in sentences if s.strip()]
        if not lengths:
            return 0.0
        
        mean_length = np.mean(lengths)
        std_length = np.std(lengths)
        
        # Higher coherence for consistent sentence lengths
        coherence = 1.0 / (1.0 + std_length / mean_length) if mean_length > 0 else 0.0
        return min(coherence, 1.0)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of evaluation metrics"""
        if not self.metrics:
            return {}
        
        df = pd.DataFrame(self.metrics)
        
        return {
            'total_queries': len(self.metrics),
            'avg_response_length': df['response_length'].mean(),
            'avg_context_relevance': df['context_relevance'].mean(),
            'avg_coherence': df['response_coherence'].mean(),
            'avg_processing_time': df['processing_time'].mean(),
            'avg_source_diversity': df['source_diversity'].mean()
        }

# Initialize services
@st.cache_resource
def get_services():
    return {
        'wikipedia': WikipediaService(),
        'knowledge_base': SimpleKnowledgeBase(),
        'evaluation': EvaluationService()
    }

services = get_services()

# Main app
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 GenAI RAG Chatbot</h1>
        <p>A comprehensive AI-powered knowledge system with RAG, prompt engineering, and evaluation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Welcome section
    st.markdown("""
    ## 🎯 Welcome to the GenAI RAG Chatbot!
    
    A powerful AI-powered knowledge system with Wikipedia integration and intelligent search.
    """)
    
    # Instructions
    with st.expander("📖 How to Use", expanded=True):
        st.markdown("""
        ### 🚀 Quick Start
        
        **Step 1**: Click "Chat" in sidebar
        **Step 2**: Search & load Wikipedia articles
        **Step 3**: Ask questions and get AI responses
        **Step 4**: Monitor evaluation metrics
        
        ### 💡 Tips
        - Load 2-3 related articles for better coverage
        - Ask specific questions about loaded topics
        - Check evaluation dashboard for response quality
        """)
    
    # Feature showcase
    st.markdown("### ✨ Key Features")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**🔍 Smart Search**\n- Wikipedia integration\n- Keyword-based retrieval")
    with col2:
        st.markdown("**🤖 AI Chat**\n- Groq API integration\n- Context-aware responses")
    with col3:
        st.markdown("**📊 Evaluation**\n- Response quality metrics\n- Performance tracking")
    with col4:
        st.markdown("**🎯 RAG Pipeline**\n- Retrieval-Augmented Generation\n- Prompt engineering")
    
    st.divider()
    
    # Technology stack
    st.markdown("### 🛠️ Technology Stack")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Frontend**\n- Streamlit\n- Responsive design")
    with col2:
        st.markdown("**AI & Processing**\n- Groq API\n- LangChain\n- Wikipedia API")
    with col3:
        st.markdown("**Evaluation**\n- Custom metrics\n- Performance tracking")
    
    # Get started section
    st.divider()
    st.markdown("### 🚀 Ready to Start?")
    st.markdown("Click **\"Chat\"** in the sidebar to begin!")
    
    # Quick demo
    with st.expander("🎬 Quick Demo"):
        st.markdown("""
        **Example:**
        1. Search "machine learning"
        2. Load article
        3. Ask: "What is machine learning?"
        4. Get structured response with sources
        """)

if __name__ == "__main__":
    main() 