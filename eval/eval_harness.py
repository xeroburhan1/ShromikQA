import os
import sys
import json
import logging
import pandas as pd
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from rag.orchestrator import RAGOrchestrator
from eval.metrics import calculate_recall_at_k, calculate_reciprocal_rank, verify_citation_hallucination

def run_evaluation():
    load_dotenv()
    
    test_set_path = os.path.join(base_dir, "eval/test_set.json")
    results_csv_path = os.path.join(base_dir, "data/eval/results.csv")
    report_md_path = os.path.join(base_dir, "data/eval/eval_report.md")
    
    if not os.path.exists(test_set_path):
        logger.error(f"Test set missing: {test_set_path}")
        sys.exit(1)
        
    with open(test_set_path, 'r', encoding='utf-8') as f:
        test_set = json.load(f)
        
    logger.info(f"Loaded {len(test_set)} test cases. Initializing RAG Orchestrator...")
    orchestrator = RAGOrchestrator(
        index_path=os.path.join(base_dir, "data/processed/faiss_index.bin"),
        meta_path=os.path.join(base_dir, "data/processed/faiss_metadata.json")
    )
    
    eval_results = []
    
    for item in test_set:
        qid = item["id"]
        topic = item["topic"]
        question = item["question"]
        expected_sec = str(item["expected_section"])
        expected_summary = item["expected_answer_summary"]
        
        logger.info(f"Evaluating QID {qid} [{topic}]: '{question}'...")
        
        # Execute RAG Orchestrator
        res = orchestrator.answer_question(question, top_k=5)
        retrieved_chunks = res["retrieved_chunks"]
        answer = res["answer"]
        
        # Calculate retrieval metrics
        rec_1 = calculate_recall_at_k(retrieved_chunks, expected_sec, k=1)
        rec_3 = calculate_recall_at_k(retrieved_chunks, expected_sec, k=3)
        rec_5 = calculate_recall_at_k(retrieved_chunks, expected_sec, k=5)
        mrr = calculate_reciprocal_rank(retrieved_chunks, expected_sec)
        
        # Citation & Hallucination verification
        halluc_check = verify_citation_hallucination(answer, expected_sec, retrieved_chunks)
        
        eval_results.append({
            "id": qid,
            "topic": topic,
            "question": question,
            "expected_section": expected_sec,
            "recall_1": rec_1,
            "recall_3": rec_3,
            "recall_5": rec_5,
            "mrr": mrr,
            "cites_expected_section": halluc_check["cites_expected_section"],
            "has_grounding_context": halluc_check["has_grounding_context"],
            "unhallucinated_score": 1.0 if halluc_check["is_faithful_unhallucinated"] else 0.0,
            "generated_answer": answer
        })
        
    # Build summary DataFrame
    df = pd.DataFrame(eval_results)
    
    os.makedirs(os.path.dirname(results_csv_path), exist_ok=True)
    df.to_csv(results_csv_path, index=False, encoding='utf-8')
    logger.info(f"Saved detailed results to {results_csv_path}")
    
    # Compute aggregate metrics
    avg_rec1 = df["recall_1"].mean()
    avg_rec3 = df["recall_3"].mean()
    avg_rec5 = df["recall_5"].mean()
    avg_mrr = df["mrr"].mean()
    avg_unhallucinated = df["unhallucinated_score"].mean()
    
    # Generate Markdown Evaluation Report
    report_md = f"""# Evaluation Report - Bangladesh Labour Act RAG Chatbot

This report evaluates the RAG pipeline against a hand-labelled benchmark test set covering {len(test_set)} statutory questions across core legal topics (termination notice, working hours, leave entitlement, wages, dispute procedure).

## Pipeline Performance Summary

| Metric | Target | Achieved | Description |
| :--- | :---: | :---: | :--- |
| **Recall@1** | > 80% | **{avg_rec1:.1%}** | Ground-truth section retrieved at rank 1 |
| **Recall@3** | > 90% | **{avg_rec3:.1%}** | Ground-truth section included in top-3 candidates |
| **Recall@5** | > 95% | **{avg_rec5:.1%}** | Ground-truth section included in top-5 candidates |
| **MRR (Mean Reciprocal Rank)** | > 0.85 | **{avg_mrr:.4f}** | Average reciprocal rank of expected section |
| **Citation Hallucination Check** | 100% | **{avg_unhallucinated:.1%}** | Direct verification that cited section exists in retrieved text & matches ground truth |

## Topic Breakdown

"""
    for topic_name, group in df.groupby("topic"):
        report_md += f"### {topic_name}\n"
        report_md += f"- **Questions**: {len(group)}\n"
        report_md += f"- **Recall@1**: {group['recall_1'].mean():.1%}\n"
        report_md += f"- **MRR**: {group['mrr'].mean():.4f}\n"
        report_md += f"- **Citation Grounding**: {group['unhallucinated_score'].mean():.1%}\n\n"
        
    report_md += f"\nDetailed CSV results saved in [`results.csv`](file:///{results_csv_path}).\n"
    
    os.makedirs(os.path.dirname(report_md_path), exist_ok=True)
    with open(report_md_path, 'w', encoding='utf-8') as f:
        f.write(report_md)
    logger.info(f"Saved evaluation report to {report_md_path}")
    
    print("\n" + "="*50)
    print("EVALUATION COMPLETE")
    print(f"Recall@1:                   {avg_rec1:.1%}")
    print(f"Recall@3:                   {avg_rec3:.1%}")
    print(f"Recall@5:                   {avg_rec5:.1%}")
    print(f"MRR (Mean Reciprocal Rank): {avg_mrr:.4f}")
    print(f"Citation Hallucination Check:{avg_unhallucinated:.1%}")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_evaluation()
