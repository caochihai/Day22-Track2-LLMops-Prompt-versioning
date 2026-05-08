# 📊 Lab Evidence & Analysis — Day 22

## 1. Project Overview
This project implements a complete LLMOps pipeline including RAG tracing, Prompt Versioning (A/B Routing), RAGAS Evaluation, and Output Validation using Guardrails AI.

- **Target Project:** LLMopsPromptVersioning
- **LangSmith Project:** `track2_day7`
- **Student:** Cao Chi Hai
- **Student ID:** 2A202600011

## 2. Advanced Evaluation Setup (Strong Judge Pattern)
To ensure objective and high-quality assessment, this project implements a "Strong Judge" evaluation architecture:
- **Generation (Student):** `gpt-4o-mini` - Selected for production cost-efficiency and speed.
- **Evaluation (Judge):** `gpt-4o` - A superior model used to rigorously evaluate the student's outputs.
- **Evaluation Embedding:** `text-embedding-3-large` - Used for precise semantic similarity and relevancy scoring.

## 3. Evidence List
| File | Description |
|------|-------------|
| `01_langsmith_traces.png` | Trace history showing 50+ successful RAG queries. |
| `02_prompt_hub.png` | LangSmith Prompt Hub showing `v1` and `v2` versions. |
| `02_ab_routing_log.txt` | Console log demonstrating deterministic A/B routing. |
| `03_ragas_scores.png` | Comparison table of RAGAS metrics between V1 and V2. |
| `03_ragas_report.json` | Detailed JSON output of the evaluation run. |
| `04_pii_demo_log.txt` | Proof of successful PII redaction (Email, Phone, etc.). |
| `04_json_demo_log.txt` | Proof of automatic JSON repair for malformed outputs. |

## 4. RAGAS Metrics Analysis (Bonus Task)

### Performance Comparison
| Metric | V1 (Concise) | V2 (Expert Tutor) | Winner |
| :--- | :--- | :--- | :--- |
| **Faithfulness** | **0.7113** | 0.5211 | **V1** |
| **Answer Relevancy** | 0.7641 | **0.8169** | **V2** |
| **Context Recall** | 0.4900 | 0.5100 | V2 |
| **Context Precision** | 0.6600 | 0.6500 | V1 |

### Insights & Rationale
1. **Faithfulness:** **Prompt V1** significantly outperformed V2 in faithfulness. 
   - *Reason:* V1 uses a strict system prompt ("Answer using ONLY provided context", "Keep it concise"). This limits the LLM's tendency to hallucinate. V2 encourages the AI to act as an "Expert Tutor," which inadvertently causes it to bring in outside knowledge not present in our `knowledge_base.txt`.
2. **Answer Relevancy:** **Prompt V2** scored higher here. 
   - *Reason:* By acting as a tutor and providing structured answers, V2 addresses the user's intent more comprehensively, even if it sometimes strays from the provided context.
3. **Low Context Recall:** Both versions scored around 0.5.
   - *Reason:* This indicates that our current `knowledge_base.txt` is missing information for about 50% of the questions in the dataset, or the chunking strategy needs optimization to retrieve more relevant snippets.

## 5. Guardrails Validation
The implementation of **Guardrails AI** ensures that even if the LLM attempts to output sensitive PII or malformed JSON, the system automatically repairs it:
- **PIIDetector:** Uses regex-based fail-to-fix logic to redact emails and phone numbers.
- **JSONFormatter:** Successfully strips Markdown fences and fixes quote types/trailing commas.

---
*Created by Cao Chi Hai (2A202600011) as part of VinUni LLMOps Track 2.*