import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# ── 1. Environment setup ────────────────────────────────────────────────────
# Load the .env file using python-dotenv
load_dotenv()

# Set LangSmith environment variables BEFORE importing LangChain to ensure tracing is enabled
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "day22-lab")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

# ── 2. LangChain + LangSmith imports ────────────────────────────────────────
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

# ── 3. LLM and Embeddings ───────────────────────────────────────────────────
# Create a ChatOpenAI instance. Using gpt-4o-mini as a cost-effective default.
llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)

# Create an OpenAIEmbeddings instance for vectorizing text
embeddings = OpenAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)


# ── 4. Build FAISS vector store ─────────────────────────────────────────────
def build_vectorstore():
    """
    Load the knowledge base, split into chunks, embed and index with FAISS.
    """
    # Read the dataset file (assumed to be in data/knowledge_base.txt as per README)
    data_path = Path("data/knowledge_base.txt")
    if not data_path.exists():
        # Fallback to creating a dummy directory and file if it doesn't exist
        data_path.parent.mkdir(exist_ok=True)
        data_path.write_text("Artificial Intelligence is transforming the world. RAG is a key technique.")
        
    text = data_path.read_text(encoding="utf-8")

    # Split text with standard RAG parameters
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    print(f"📊 Split into {len(chunks)} chunks")

    # Build and return the FAISS vectorstore
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore


# ── 5. RAG prompt template ──────────────────────────────────────────────────
# Define a template that instructs the LLM to use only provided context
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the context below to answer accurately.\n\nContext:\n{context}"),
    ("human",  "{question}"),
])


# ── 6. Build the RAG chain ──────────────────────────────────────────────────
def build_rag_chain(vectorstore):
    """
    Build a LangChain RAG chain using LCEL (pipe operator).
    """
    # Create a retriever from the vectorstore (top 3 relevant chunks)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Helper to join retrieved docs into a single string for the prompt
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Build the chain: context retrieval -> prompt formatting -> LLM -> output parsing
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain, retriever


# ── 7. Traced query function ────────────────────────────────────────────────
# Decorate with @traceable so every call is captured in LangSmith
@traceable(name="rag-query", tags=["rag", "step1"])
def ask(chain, question: str) -> str:
    """
    Run the RAG chain on a single question.
    """
    return chain.invoke(question)


# ── 8. Import sample questions from central utils ────────────────────────────
# Add root path to sys.path to find the 'utils' package
root_path = str(Path(__file__).parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

from utils.qa_pairs import SAMPLE_QUESTIONS


# ── 9. Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Step 1: LangSmith RAG Pipeline")
    print("=" * 60)

    # Verify environment
    if not os.getenv("LANGCHAIN_API_KEY") or not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Warning: Missing API keys in .env file.")

    # Build the vectorstore
    vectorstore = build_vectorstore()

    # Build the RAG chain
    chain, retriever = build_rag_chain(vectorstore)

    # Loop through all questions, call ask(), print results
    print(f"🚀 Processing {len(SAMPLE_QUESTIONS)} questions...")
    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        try:
            answer = ask(chain, question)
            # Đảm bảo answer là chuỗi để tránh lỗi slice/replace
            answer_str = str(answer) if answer else "No answer returned."
            
            print(f"[{i:02d}/{len(SAMPLE_QUESTIONS)}] Q: {question[:60]}")
            print(f"       A: {answer_str[:100].replace('\n', ' ')}...\n")
        except Exception as e:
            print(f"[{i:02d}/{len(SAMPLE_QUESTIONS)}] ERROR: {e}")

    # Print confirmation
    print(f"DONE: {len(SAMPLE_QUESTIONS)} traces sent to LangSmith project '{os.getenv('LANGCHAIN_PROJECT')}'")
    print("   Open https://smith.langchain.com to view traces.")


if __name__ == "__main__":
    main()
