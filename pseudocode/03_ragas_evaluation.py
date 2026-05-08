import os
import sys
import json
import warnings
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# Suppress RAGAS and other deprecation warnings for cleaner output
warnings.filterwarnings("ignore")
load_dotenv()

# ── 1. Imports ───────────────────────────────────────────────────────────────
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

# ── 2. Import QA pairs from central utils ────────────────────────────────────
# Add root path to sys.path to find the 'utils' package
root_path = str(Path(__file__).parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

from utils.qa_pairs import QA_PAIRS


# ── 3. Prompt templates (same as step 2) ────────────────────────────────────
SYSTEM_V1 = (
    "You are a helpful AI assistant. "
    "Answer the user's question using ONLY the provided context. "
    "Keep your answer concise (2-4 sentences). "
    "If the context does not contain the answer, say: 'I don't have enough information.'\n\n"
    "Context:\n{context}"
)
PROMPT_V1 = ChatPromptTemplate.from_messages([("system", SYSTEM_V1), ("human", "{question}")])

SYSTEM_V2 = (
    "You are an expert AI tutor. Provide a structured, accurate answer.\n\n"
    "Instructions:\n"
    "1. Read the context carefully.\n"
    "2. Identify the key facts relevant to the question.\n"
    "3. Write a clear, well-organized answer (3-5 sentences).\n"
    "4. State explicitly if the context lacks sufficient information.\n\n"
    "Context:\n{context}"
)
PROMPT_V2 = ChatPromptTemplate.from_messages([("system", SYSTEM_V2), ("human", "{question}")])

PROMPTS = {
    "v1": PROMPT_V1,
    "v2": PROMPT_V2,
}


# ── 4. Build vectorstore (reuse logic from step 1) ───────────────────────────
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


# ── 5. Run RAG and capture outputs + contexts ────────────────────────────────
@traceable(name="rag-eval-run", tags=["eval", "step3"])
def run_rag(retriever, llm, prompt, question: str) -> dict:
    """
    Run the RAG chain for one question.
    """
    # Retrieve documents
    docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]   # ← list of strings!
    ctx_str = "\n\n".join(contexts)

    # Run the chain
    answer = (prompt | llm | StrOutputParser()).invoke({"context": ctx_str, "question": question})

    # Return both answer and contexts list
    return {"answer": answer, "contexts": contexts}


def collect_rag_outputs(vectorstore, prompt_version: str) -> list:
    """
    Run all 50 QA pairs through the given prompt version.
    Returns a list of dicts with keys: question, reference, answer, contexts.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    prompt = PROMPTS[prompt_version]

    results = []
    print(f"\nRunning 50 questions with prompt {prompt_version} ...")

    for i, qa in enumerate(QA_PAIRS, 1):
        out = run_rag(retriever, llm, prompt, qa["question"])
        results.append({
            "question":  qa["question"],
            "reference": qa["reference"],
            "answer":    out["answer"],
            "contexts":  out["contexts"],   # must be list[str]
        })
        if i % 10 == 0:
            print(f"  [{i:02d}/50] questions processed...")

    return results


# ── 6. Build RAGAS EvaluationDataset ────────────────────────────────────────
def build_ragas_dataset(rag_results: list):
    """
    Convert a list of RAG result dicts into a RAGAS EvaluationDataset.
    """
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]
    return EvaluationDataset(samples=samples)


# ── 7. Run RAGAS evaluation ──────────────────────────────────────────────────
def run_ragas_eval(rag_results: list, version: str) -> dict:
    """
    Evaluate RAG outputs with 4 RAGAS metrics.
    Returns a dict: {metric_name: mean_score}
    """
    print(f"\n📐 Running RAGAS evaluation for prompt {version} ...")

    dataset = build_ragas_dataset(rag_results)

    # Use specified evaluation models (Judge uses higher-tier models)
    llm_eval = ChatOpenAI(
        model=os.getenv("EVAL_MODEL_NAME", "gpt-4o"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    emb_eval = OpenAIEmbeddings(
        model=os.getenv("EVAL_EMBEDDING_MODEL", "text-embedding-3-large"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )

    # Run evaluate() — this makes many LLM calls!
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm_eval,
        embeddings=emb_eval,
    )

    # Extract mean scores
    scores = {}
    for key in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        raw = result[key]           # list of floats
        clean_values = [v for v in raw if v is not None]
        scores[key] = float(np.mean(clean_values)) if clean_values else 0.0

    # Print and return scores
    for k, v in scores.items():
        star = " ⭐" if k == "faithfulness" and v >= 0.8 else ""
        print(f"  {k:30s}: {v:.4f}{star}")
    return scores


# ── 8. Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Step 3: RAGAS Evaluation")
    print("=" * 60)

    # Build vectorstore
    vectorstore = build_vectorstore()

    # Collect outputs for V1 and V2
    v1_results = collect_rag_outputs(vectorstore, "v1")
    v2_results = collect_rag_outputs(vectorstore, "v2")

    # Run RAGAS evaluation on both
    v1_scores = run_ragas_eval(v1_results, "v1")
    v2_scores = run_ragas_eval(v2_results, "v2")

    # Print comparison table
    print("\n" + "-"*60)
    print(f"{'Metric':30s} | {'V1 Score':10s} | {'V2 Score':10s} | {'Winner'}")
    print("-"*60)
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        s1, s2 = v1_scores[metric], v2_scores[metric]
        winner = "← V1" if s1 > s2 else ("← V2" if s2 > s1 else "Tie")
        print(f"{metric:30s} | {s1:10.4f} | {s2:10.4f} | {winner}")

    # Check faithfulness target
    best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    if best_faith >= 0.8:
        print(f"\n✅ Target met: faithfulness = {best_faith:.4f}")
    else:
        print(f"\n⚠️  Below target ({best_faith:.4f}). Try adjusting chunking or prompts.")

    # Save JSON report to data/ragas_report.json
    report = {
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "target_met": best_faith >= 0.8,
    }
    report_path = Path("data/ragas_report.json")
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"💾 Saved {report_path}")


if __name__ == "__main__":
    main()
