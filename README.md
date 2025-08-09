# 🤖 RAG ChatBot with Ollama

A local RAG chatbot using Ollama's Llama 3.2 3B model and ChromaDB for document question-answering.



## ✨ Features

- 🦙 **Local Ollama LLM** - Llama 3.2 3B (privacy-first)
- 📄 **Document Upload** - PDF, TXT, MD support
- 💬 **Conversational AI** - Context-aware responses
- 🔍 **Smart Retrieval** - ChromaDB vector search
- 📊 **Real-time Status** - Connection monitoring

## 🚀 Quick Setup

### 1. Install Ollama
```bash
# Windows/Mac: Download from https://ollama.ai
# Linux:
curl -fsSL https://ollama.ai/install.sh | sh
```

### 2. Start Ollama & Pull Model
```bash
ollama serve
ollama pull llama3.2:3b
```

### 3. Install & Run
```bash
pip install -r requirements.txt
python run_app.py
```

Open: `http://localhost:8501`

## 💡 Usage

1. **Process Documents**: Upload files → Click "🚀 Process Documents"
2. **Chat**: Ask questions about your documents
3. **View Sources**: Expand citations to see relevant excerpts

## 📋 Requirements

```txt
streamlit>=1.28.0
langchain>=0.0.350
chromadb>=0.4.0
sentence-transformers>=2.2.2
pypdf>=3.8.0
requests>=2.28.0
```

## 🔧 Troubleshooting

**Ollama not connected?**
```bash
ollama serve
ollama list  # Verify llama3.2:3b is available
```

**Model not found?**
```bash
ollama pull llama3.2:3b
```

## 📁 Files

- `app.py` - Main Streamlit application
- `run_app.py` - Launch script with checks
- `requirements.txt` - Dependencies

---

**Local AI • Privacy First • No API Keys Required**
