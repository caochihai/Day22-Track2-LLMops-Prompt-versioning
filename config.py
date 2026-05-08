import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# LangChain / LangSmith configuration
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "day22-lab")

# OpenAI / Provider configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1") # Or your custom endpoint
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

def check_config():
    """Check if essential configuration is present."""
    missing = []
    if not LANGCHAIN_API_KEY:
        missing.append("LANGCHAIN_API_KEY")
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    
    if missing:
        print(f"⚠️ Warning: Missing configuration for: {', '.join(missing)}")
        print("Please update your .env file.")
    else:
        print("✅ Configuration loaded successfully.")
