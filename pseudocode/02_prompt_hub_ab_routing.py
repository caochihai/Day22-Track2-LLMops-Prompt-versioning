import os
import sys
import hashlib
from pathlib import Path
from dotenv import load_dotenv

# ── 1. Environment / imports ────────────────────────────────────────────────
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "day22-lab")

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import Client, traceable

# ── 2. Define two prompt templates ──────────────────────────────────────────
# PROMPT_V1 — concise, 2-4 sentence answers
SYSTEM_V1 = (
    "You are a helpful AI assistant. "
    "Answer the user's question using ONLY the provided context. "
    "Keep your answer concise (2-4 sentences). "
    "If the context does not contain the answer, say: 'I don't have enough information.'\n\n"
    "Context:\n{context}"
)
PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

# PROMPT_V2 — structured, expert 3-5 sentence answers
SYSTEM_V2 = (
    "You are an expert AI tutor. Provide a structured, accurate answer.\n\n"
    "Instructions:\n"
    "1. Read the context carefully.\n"
    "2. Identify the key facts relevant to the question.\n"
    "3. Write a clear, well-organized answer (3-5 sentences).\n"
    "4. State explicitly if the context lacks sufficient information.\n\n"
    "Context:\n{context}"
)
PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])

# Prompt Hub names (change these to your own unique names)
# Using a common prefix for consistency
PREFIX = "antigravity"
PROMPT_V1_NAME = f"{PREFIX}-rag-prompt-v1"
PROMPT_V2_NAME = f"{PREFIX}-rag-prompt-v2"


# ── 3. Push prompts to LangSmith Prompt Hub ──────────────────────────────────
def push_prompts_to_hub(client):
    """
    Upload both prompt versions to LangSmith Prompt Hub.
    """
    print(f"📤 Pushing prompts to Hub...")
    # Push PROMPT_V1
    try:
        url = client.push_prompt(PROMPT_V1_NAME, object=PROMPT_V1, description="V1 – concise answers")
        print(f"✅ Pushed V1 → {url}")
    except Exception as e:
        print(f"WARNING: V1 Push skipped or failed: {e}")

    # Push PROMPT_V2
    try:
        url = client.push_prompt(PROMPT_V2_NAME, object=PROMPT_V2, description="V2 – structured answers")
        print(f"✅ Pushed V2 → {url}")
    except Exception as e:
        print(f"⚠️  V2 Push Failed: {e}")


# ── 4. Pull prompts from Prompt Hub ─────────────────────────────────────────
def pull_prompts_from_hub(client):
    """
    Download both prompt versions from LangSmith Prompt Hub.
    """
    prompts = {}
    print(f"📥 Pulling prompts from Hub...")

    # Pull PROMPT_V1_NAME, fall back to local PROMPT_V1 on error
    try:
        prompts[PROMPT_V1_NAME] = client.pull_prompt(PROMPT_V1_NAME)
        print(f"  ↓ Pulled '{PROMPT_V1_NAME}' from Hub")
    except Exception:
        prompts[PROMPT_V1_NAME] = PROMPT_V1
        print(f"  ℹ️  Using local fallback for '{PROMPT_V1_NAME}'")

    # Pull PROMPT_V2_NAME, fall back to local PROMPT_V2 on error
    try:
        prompts[PROMPT_V2_NAME] = client.pull_prompt(PROMPT_V2_NAME)
        print(f"  ↓ Pulled '{PROMPT_V2_NAME}' from Hub")
    except Exception:
        prompts[PROMPT_V2_NAME] = PROMPT_V2
        print(f"  ℹ️  Using local fallback for '{PROMPT_V2_NAME}'")

    return prompts


# ── 5. A/B routing — deterministic hash ─────────────────────────────────────
def get_prompt_version(request_id: str) -> str:
    """
    Route a request to prompt V1 or V2 based on the MD5 hash of request_id.
    """
    # Compute MD5 hash of request_id, convert to integer
    hash_int = int(hashlib.md5(request_id.encode()).hexdigest(), 16)

    # Return V1 name if even, V2 name if odd
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME


# ── 6. Build vectorstore (reuse from step 1) ────────────────────────────────
def build_vectorstore():
    embeddings = OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    data_path = Path("data/knowledge_base.txt")
    if not data_path.exists():
        data_path.parent.mkdir(exist_ok=True)
        data_path.write_text("Artificial Intelligence is transforming the world. RAG is a key technique.")
        
    text = data_path.read_text(encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    return FAISS.from_texts(chunks, embeddings)


# ── 7. Traced A/B query function ────────────────────────────────────────────
@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version: str) -> dict:
    """
    Run the RAG chain using the given prompt version.
    """
    # Retrieve docs
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    # Run the chain
    answer = (prompt | llm | StrOutputParser()).invoke({"context": context, "question": question})

    # Return result
    return {"question": question, "answer": answer, "version": version}


# ── 8. Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Step 2: Prompt Hub A/B Routing")
    print("=" * 60)

    # Create LangSmith client
    client = Client(api_key=os.environ["LANGCHAIN_API_KEY"])

    # Push both prompts to Hub
    push_prompts_to_hub(client)

    # Pull both prompts from Hub
    prompts = pull_prompts_from_hub(client)

    # Build vectorstore, retriever, and LLM
    vectorstore = build_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )

    # Add root path to sys.path to find the 'utils' package
    root_path = str(Path(__file__).parent.parent)
    if root_path not in sys.path:
        sys.path.append(root_path)
    from utils.qa_pairs import SAMPLE_QUESTIONS
    
    v1_count = 0
    v2_count = 0
    
    print(f"\n🚀 Running {len(SAMPLE_QUESTIONS)} questions with A/B routing...")
    for i, question in enumerate(SAMPLE_QUESTIONS):
        try:
            request_id = f"req-{i:04d}"
            version_key = get_prompt_version(request_id)
            version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
            prompt = prompts[version_key]

            if version_tag == "v1": v1_count += 1
            else: v2_count += 1

            result = ask_ab(retriever, llm, prompt, question, version_tag)
            ans_str = str(result.get("answer", ""))
            print(f"[{i+1:02d}] [prompt-{version_tag}] {question[:55]}...")
            print(f"      A: {ans_str[:80].replace('\n', ' ')}...")
        except Exception as e:
            print(f"[{i+1:02d}] ERROR: {e}")

    # Print routing summary
    print(f"\n📊 Routing Summary: V1={v1_count}, V2={v2_count}")
    print(f"✅ Total {len(SAMPLE_QUESTIONS)} traces sent to LangSmith.")


if __name__ == "__main__":
    main()
