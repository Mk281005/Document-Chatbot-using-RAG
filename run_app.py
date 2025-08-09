# run_app.py - Launch script for Streamlit RAG App with Ollama

import subprocess
import sys
import os
import requests
from pathlib import Path

def setup_streamlit_config():
    """Create Streamlit configuration directory and files"""
    
    # Create .streamlit directory
    streamlit_dir = Path(".streamlit")
    streamlit_dir.mkdir(exist_ok=True)
    
    # Create config.toml
    config_content = """
[global]
developmentMode = false

[server]
runOnSave = true
port = 8501
enableCORS = false
enableXsrfProtection = false
maxUploadSize = 200
maxMessageSize = 200

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
"""
    
    config_file = streamlit_dir / "config.toml"
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print("✅ Streamlit configuration created")

def check_ollama_status():
    """Check if Ollama is running and models are available"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print("✅ Ollama is running")
            print(f"📦 Available models: {len(models)}")
            
            # Check for llama3.2:3b specifically
            model_names = [model['name'] for model in models]
            if 'llama3.2:3b' in model_names:
                print("✅ llama3.2:3b model is available")
            else:
                print("⚠️  llama3.2:3b model not found")
                print("💡 Run: ollama pull llama3.2:3b")
            
            return True, model_names
        else:
            print("❌ Ollama server responded with error")
            return False, []
    except requests.exceptions.ConnectionError:
        print("❌ Ollama is not running")
        print("💡 Start Ollama with: ollama serve")
        return False, []
    except Exception as e:
        print(f"❌ Error checking Ollama: {e}")
        return False, []

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit', 'langchain', 'chromadb', 
        'sentence_transformers', 'pypdf', 'requests'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def create_directories():
    """Create necessary directories"""
    directories = ['uploads', 'chroma_db']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created directory: {directory}")

def launch_app():
    """Launch the Streamlit app"""
    try:
        print("🚀 Launching RAG ChatBot with Ollama...")
        print("📱 Open your browser at: http://localhost:8501")
        print("🛑 Press Ctrl+C to stop the app")
        print("=" * 50)
        
        # Launch streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "app.py", 
            "--server.headless", "true",
            "--server.enableWebsocketCompression", "false"
        ])
        
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error launching app: {e}")

def main():
    """Main setup and launch function"""
    print("🔧 Setting up RAG ChatBot with Ollama...")
    print("=" * 50)
    
    # Setup configuration
    setup_streamlit_config()
    
    # Check Ollama status
    ollama_running, models = check_ollama_status()
    
    # Check Python packages
    packages_ok = check_requirements()
    
    if not packages_ok:
        print("\n❌ Please install required packages first")
        return
    
    if not ollama_running:
        print("\n❌ Please start Ollama first")
        print("💡 In another terminal, run: ollama serve")
        return
    
    # Create directories
    create_directories()
    
    # Launch app
    print("\n🎯 All checks passed! Launching app...")
    launch_app()

if __name__ == "__main__":
    main()