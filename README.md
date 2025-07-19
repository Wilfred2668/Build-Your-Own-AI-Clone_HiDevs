# 🤖 GenAI RAG Chatbot

A comprehensive AI-powered knowledge system with Wikipedia integration, intelligent search, and advanced AI responses using Retrieval-Augmented Generation (RAG) pipeline.

## 🚀 Features

- **🔍 Smart Search**: Wikipedia integration with keyword-based retrieval
- **🤖 AI Chat**: Groq API integration with context-aware responses
- **📊 Evaluation**: Real-time performance metrics and analytics
- **🎯 RAG Pipeline**: Advanced prompt engineering and knowledge management
- **🔒 Security**: No environment variables, API keys handled through UI only

## 📋 Prerequisites

- Python 3.8+
- Groq API key (optional, for enhanced AI responses)
- Internet connection

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Application

```bash
streamlit run streamlit_app_simple.py
```

### 3. Access App

Open browser: `http://localhost:8501`

## 📖 How to Use

### Step 1: Navigate to Chat

- Click "Chat" in sidebar
- Configure Groq API key (optional)

### Step 2: Load Knowledge Base

- Search Wikipedia articles (e.g., "machine learning")
- Click "Load Article" to add content

### Step 3: Chat with AI

- Ask questions about loaded content
- Get structured responses with sources
- Monitor evaluation metrics

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **AI**: Groq API (Llama models)
- **Search**: Custom keyword-based algorithm
- **Data**: Wikipedia API
- **Evaluation**: Custom metrics

## 🎯 Problem Statement Compliance

✅ **RAG Implementation**: Proper retrieval and generation pipeline  
✅ **Prompt Engineering**: Advanced prompts for structured responses  
✅ **Output Quality**: Context-aware, well-structured answers  
✅ **Performance**: Lightweight, fast implementation with Groq API  
✅ **Evaluation**: Comprehensive metrics and real-time analytics

## 📁 Project Structure

```
fluent-genai-buddy/
├── streamlit_app_simple.py    # Main app (Home page)
├── pages/1_Chat.py           # Chat interface
├── requirements.txt          # Dependencies
└── README.md                # This file
```

## 🔒 Security & Privacy

- **No Environment Variables**: API keys handled through UI only
- **No Data Storage**: All data in session state
- **Privacy-First**: No user data collection

## 🚀 Deployment

### Local Development

```bash
streamlit run streamlit_app_simple.py
```

### Streamlit Cloud

1. Push to GitHub
2. Connect at [share.streamlit.io](https://share.streamlit.io)
3. Deploy with main file: `streamlit_app_simple.py`


## 📊 Evaluation Metrics

- **Context Relevance**: Query-chunk matching
- **Response Coherence**: Answer quality
- **Processing Time**: Performance tracking
- **Source Diversity**: Information variety

---

**Built with by Wilfred for intelligent knowledge exploration**
