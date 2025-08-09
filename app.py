import streamlit as st
import os
import tempfile
import shutil
from pathlib import Path
import torch
from typing import List, Dict
import time
import json
import requests

# RAG components
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader, PyPDFLoader

import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🤖 RAG ChatBot with Ollama",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .user-message {
        border-left: 4px solid #2196f3;
        /* No background color - uses default */
    }
    
    .bot-message {
        border-left: 4px solid #9c27b0;
        /* No background color - uses default */
    }
    
    .source-doc {
        background-color: #fff3e0;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        border-left: 3px solid #ff9800;
    }
    
    .status-success {
        background-color: #e8f5e8;
        color: #2e7d32;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 4px solid #4caf50;
    }
    
    .status-error {
        background-color: #ffebee;
        color: #c62828;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 4px solid #f44336;
    }
    
    .status-warning {
        background-color: #fff8e1;
        color: #f57f17;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 4px solid #ffeb3b;
    }
    
    .ollama-status {
        background-color: #e8f4fd;
        color: #1565c0;
        padding: 0.75rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitRAGChatBot:
    def __init__(self):
        self.persist_directory = "./chroma_db"
        
    @st.cache_resource
    def load_embeddings(_self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """Load embedding model (cached)"""
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    @st.cache_resource
    def load_llm(_self, model_name="llama3.2:3b"):
        """Load Ollama LLM model (cached)"""
        try:
            # Check if Ollama is running
            try:
                response = requests.get("http://localhost:11434/api/tags", timeout=5)
                if response.status_code == 200:
                    available_models = [model['name'] for model in response.json().get('models', [])]
                    if model_name not in available_models:
                        st.warning(f"Model {model_name} not found. Available models: {available_models}")
                        return None
                else:
                    st.error("Ollama server is not responding. Make sure Ollama is running on localhost:11434")
                    return None
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to Ollama. Please ensure Ollama is installed and running.")
                st.info("💡 Start Ollama with: `ollama serve`")
                return None
            
            # Create Ollama LLM instance
            llm = Ollama(
                model=model_name,
                temperature=0.7,
                num_predict=512,
                top_k=40,
                top_p=0.9,
                repeat_penalty=1.1,
                base_url="http://localhost:11434"
            )
            
            # Test the connection with a simple query
            try:
                test_response = llm("Test")
                st.success(f"✅ Successfully connected to Ollama with {model_name}")
                return llm
            except Exception as e:
                st.error(f"❌ Failed to test Ollama connection: {str(e)}")
                return None
            
        except Exception as e:
            st.error(f"❌ Error loading Ollama model: {str(e)}")
            return None
    
    def process_uploaded_files(self, uploaded_files):
        """Process uploaded files and extract text"""
        documents = []
        
        with st.spinner("Processing uploaded files..."):
            for uploaded_file in uploaded_files:
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_file_path = tmp_file.name
                    
                    # Load document based on file type
                    if uploaded_file.name.endswith('.pdf'):
                        loader = PyPDFLoader(tmp_file_path)
                    else:
                        loader = TextLoader(tmp_file_path, encoding='utf-8')
                    
                    file_documents = loader.load()
                    
                    # Add filename to metadata
                    for doc in file_documents:
                        doc.metadata['source'] = uploaded_file.name
                        doc.metadata['upload_time'] = time.strftime("%Y-%m-%d %H:%M:%S")
                    
                    documents.extend(file_documents)
                    
                    # Clean up temporary file
                    os.unlink(tmp_file_path)
                    
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        
        return documents
    
    def create_sample_documents(self):
        """Create sample documents for demo"""
        sample_texts = [
            """Artificial Intelligence (AI) is intelligence demonstrated by machines, in contrast to natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals. Machine learning is a subset of AI that involves training algorithms to make predictions or decisions based on data.""",
            
            """Machine Learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention. There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.""",
            
            """Natural Language Processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data. Common NLP tasks include sentiment analysis, named entity recognition, machine translation, and text summarization.""",
            
            """Deep Learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Deep learning architectures such as deep neural networks, deep belief networks, recurrent neural networks and convolutional neural networks have been applied to fields including computer vision, speech recognition, natural language processing, and bioinformatics.""",
            
            """Computer Vision is an interdisciplinary scientific field that deals with how computers can gain high-level understanding from digital images or videos. From an engineering perspective, it seeks to understand and automate tasks that the human visual system can do. Computer vision tasks include image classification, object detection, image segmentation, and facial recognition.""",
            
            """Ollama is a tool for running large language models locally. It provides a simple API for creating, running, and managing models, as well as a library of pre-built models that can be easily used in a variety of applications. Ollama supports various model formats and provides efficient inference capabilities for local deployment."""
        ]
        
        documents = []
        for i, text in enumerate(sample_texts):
            doc = Document(
                page_content=text,
                metadata={"source": f"sample_document_{i+1}.txt", "type": "sample"}
            )
            documents.append(doc)
        
        return documents
    
    def chunk_documents(self, documents):
        """Split documents into chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=st.session_state.get('chunk_size', 1000),
            chunk_overlap=st.session_state.get('chunk_overlap', 200),
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        return chunks
    
    def create_vectorstore(self, documents, embeddings):
        """Create or update vector store"""
        chunks = self.chunk_documents(documents)
        
        # Check if vector store exists
        if os.path.exists(self.persist_directory):
            # Load existing vectorstore and add new documents
            vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=embeddings
            )
            vectorstore.add_documents(chunks)
        else:
            # Create new vectorstore
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=self.persist_directory
            )
        
        vectorstore.persist()
        return vectorstore, len(chunks)
    
    def setup_qa_chain(self, vectorstore, llm, chain_type="conversational"):
        """Setup QA chain"""
        if chain_type == "conversational":
            # Initialize conversation memory if not exists
            if 'conversation_memory' not in st.session_state:
                st.session_state.conversation_memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True,
                    output_key="answer"
                )
            
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": st.session_state.get('retrieval_k', 3)}
                ),
                memory=st.session_state.conversation_memory,
                return_source_documents=True,
                verbose=False
            )
        else:
            # Standard RAG chain
            prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Helpful Answer:"""
            
            PROMPT = PromptTemplate(
                template=prompt_template, 
                input_variables=["context", "question"]
            )
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": st.session_state.get('retrieval_k', 3)}
                ),
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )
        
        return qa_chain

def check_ollama_connection():
    """Check Ollama connection and return status"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code == 200:
            models = response.json().get('models', [])
            return True, models
        else:
            return False, []
    except:
        return False, []

def main():
    # Initialize session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = StreamlitRAGChatBot()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        
    if 'vectorstore_ready' not in st.session_state:
        st.session_state.vectorstore_ready = False
        
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = None
        
    if 'llm' not in st.session_state:
        st.session_state.llm = None
        
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
    
    # Main header
    st.markdown('<h1 class="main-header">🤖 RAG ChatBot with Ollama + ChromaDB</h1>', unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        st.subheader("🔧 Model Settings")
        
        embedding_model = st.selectbox(
            "Embedding Model",
            [
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/all-mpnet-base-v2",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            ],
            help="Choose the embedding model for document processing"
        )
        
        llm_model = st.selectbox(
            "Ollama Model", 
            [
                "llama3.2:3b",
                "llama3.1:8b",
                "llama2:7b",
                "mistral:7b",
                "codellama:7b",
                "gemma:7b"
            ],
            help="Choose the Ollama model for generating responses (make sure it's installed)"
        )
        
        # Advanced settings
        with st.expander("🔬 Advanced Settings"):
            chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 100)
            chunk_overlap = st.slider("Chunk Overlap", 50, 500, 200, 50)
            retrieval_k = st.slider("Retrieved Documents", 1, 10, 3, 1)
            chain_type = st.selectbox("Chain Type", ["conversational", "standard"])
            
            st.session_state.chunk_size = chunk_size
            st.session_state.chunk_overlap = chunk_overlap
            st.session_state.retrieval_k = retrieval_k
        
        # System status
        st.subheader("📊 System Status")
        device = "GPU" if torch.cuda.is_available() else "CPU"
        st.info(f"🖥️ Running on: {device}")
        
        # Ollama status check
        ollama_connected, available_models = check_ollama_connection()
        if ollama_connected:
            st.markdown('<div class="status-success">✅ Ollama Connected</div>', unsafe_allow_html=True)
            if available_models:
                st.write(f"📦 Available models: {len(available_models)}")
                model_names = [model['name'] for model in available_models]
                if llm_model in model_names:
                    st.success(f"✅ {llm_model} available")
                else:
                    st.warning(f"⚠️ {llm_model} not found")
                    st.info(f"💡 Run: ollama pull {llm_model}")
        else:
            st.markdown('<div class="status-error">❌ Ollama Not Connected</div>', unsafe_allow_html=True)
            st.info("💡 Start Ollama: `ollama serve`")
        
        if st.session_state.vectorstore_ready:
            st.markdown('<div class="status-success">✅ Vector Store Ready</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-warning">⚠️ Load documents first</div>', unsafe_allow_html=True)
        
        # Clear conversation
        if st.button("🗑️ Clear Conversation"):
            st.session_state.chat_history = []
            if 'conversation_memory' in st.session_state:
                st.session_state.conversation_memory.clear()
            st.rerun()
    
    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📄 Document Management", "📈 Analytics"])
    
    with tab2:
        st.header("📄 Document Management")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📁 Upload Documents")
            uploaded_files = st.file_uploader(
                "Choose files",
                accept_multiple_files=True,
                type=['txt', 'pdf', 'md'],
                help="Upload PDF, TXT, or MD files to add to the knowledge base"
            )
            
            use_sample = st.checkbox("Use sample documents for demo", value=True)
            
            if st.button("🚀 Process Documents", type="primary"):
                if not ollama_connected:
                    st.error("❌ Ollama is not connected. Please start Ollama first.")
                    return
                
                with st.spinner("Loading models and processing documents..."):
                    try:
                        # Load models
                        if st.session_state.embeddings is None:
                            st.session_state.embeddings = st.session_state.chatbot.load_embeddings(embedding_model)
                        
                        if st.session_state.llm is None:
                            st.session_state.llm = st.session_state.chatbot.load_llm(llm_model)
                        
                        if st.session_state.llm is None:
                            st.error("❌ Failed to load Ollama model. Please check the model name and try again.")
                            return
                        
                        # Process documents
                        documents = []
                        
                        if uploaded_files:
                            documents.extend(st.session_state.chatbot.process_uploaded_files(uploaded_files))
                        
                        if use_sample or not uploaded_files:
                            documents.extend(st.session_state.chatbot.create_sample_documents())
                        
                        if documents:
                            # Create vector store
                            vectorstore, chunk_count = st.session_state.chatbot.create_vectorstore(
                                documents, st.session_state.embeddings
                            )
                            
                            # Setup QA chain
                            st.session_state.qa_chain = st.session_state.chatbot.setup_qa_chain(
                                vectorstore, st.session_state.llm, chain_type
                            )
                            
                            st.session_state.vectorstore_ready = True
                            st.session_state.document_count = len(documents)
                            st.session_state.chunk_count = chunk_count
                            
                            st.success(f"✅ Successfully processed {len(documents)} documents into {chunk_count} chunks!")
                        else:
                            st.error("No documents to process!")
                            
                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")
        
        with col2:
            st.subheader("📊 Knowledge Base Info")
            
            if st.session_state.vectorstore_ready:
                st.metric("Documents", st.session_state.get('document_count', 0))
                st.metric("Text Chunks", st.session_state.get('chunk_count', 0))
                
                # Show document sources
                if hasattr(st.session_state, 'qa_chain') and st.session_state.qa_chain:
                    st.write("📚 **Document Sources:**")
                    st.info("Documents are loaded and ready for querying!")
            else:
                st.info("Process documents to see knowledge base statistics")
    
    with tab3:
        st.header("📈 System Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💬 Chat Statistics")
            st.metric("Total Messages", len(st.session_state.chat_history))
            st.metric("User Queries", len([msg for msg in st.session_state.chat_history if msg['role'] == 'user']))
        
        with col2:
            st.subheader("🔧 System Info")
            st.write(f"**Embedding Model:** {embedding_model}")
            st.write(f"**LLM Model:** {llm_model}")
            st.write(f"**Chunk Size:** {st.session_state.get('chunk_size', 1000)}")
            st.write(f"**Retrieval K:** {st.session_state.get('retrieval_k', 3)}")
        
        # Ollama system info
        if ollama_connected and available_models:
            st.subheader("🦙 Ollama Models")
            for model in available_models:
                st.write(f"• {model['name']} ({model.get('size', 'Unknown size')})")
        
        # Recent queries
        if st.session_state.chat_history:
            st.subheader("🕒 Recent Queries")
            recent_queries = [msg['content'] for msg in st.session_state.chat_history if msg['role'] == 'user'][-5:]
            for i, query in enumerate(reversed(recent_queries), 1):
                st.write(f"{i}. {query}")
    
    with tab1:
        st.header("💬 Chat with your Documents")
        
        # Check if system is ready
        if not ollama_connected:
            st.error("❌ Ollama is not connected. Please start Ollama and refresh the page.")
            st.info("💡 Run `ollama serve` in your terminal")
            return
        
        if not st.session_state.vectorstore_ready:
            st.warning("⚠️ Please process some documents first in the 'Document Management' tab!")
            return
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                if message['role'] == 'user':
                    st.markdown(f'<div class="chat-message user-message"><strong>👤 You:</strong><br>{message["content"]}</div>', 
                              unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-message bot-message"><strong>🤖 Bot:</strong><br>{message["content"]}</div>', 
                              unsafe_allow_html=True)
                    
                    # Show sources if available
                    if 'sources' in message and message['sources']:
                        with st.expander("📚 View Sources"):
                            for i, source in enumerate(message['sources'], 1):
                                st.markdown(f'<div class="source-doc"><strong>Source {i}:</strong><br>{source[:300]}...</div>', 
                                          unsafe_allow_html=True)
        
        # Chat input
        user_question = st.text_input(
            "Ask a question about your documents:",
            placeholder="What would you like to know?",
            key="user_input"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            send_button = st.button("Send 📤", type="primary")
        
        # Process user input
        if send_button and user_question and st.session_state.qa_chain:
            # Add user message to chat history
            st.session_state.chat_history.append({
                'role': 'user',
                'content': user_question,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            })
            
            # Get bot response
            with st.spinner("🤔 Thinking..."):
                try:
                    if chain_type == "conversational":
                        result = st.session_state.qa_chain({"question": user_question})
                        answer = result['answer']
                        sources = result.get('source_documents', [])
                    else:
                        result = st.session_state.qa_chain({"query": user_question})
                        answer = result['result']
                        sources = result.get('source_documents', [])
                    
                    # Add bot response to chat history
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': answer,
                        'sources': [doc.page_content for doc in sources[:2]] if sources else [],
                        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': f"Sorry, I encountered an error: {str(e)}",
                        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                    })
            
            # Rerun to update the chat display
            st.rerun()

if __name__ == "__main__":
    main()