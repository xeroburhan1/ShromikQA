# Evaluation Report - Bangladesh Labour Act RAG Chatbot

This report evaluates the RAG pipeline against a hand-labelled benchmark test set covering 10 statutory questions across core legal topics (termination notice, working hours, leave entitlement, wages, dispute procedure).

## Pipeline Performance Summary

| Metric | Target | Achieved | Description |
| :--- | :---: | :---: | :--- |
| **Recall@1** | > 80% | **100.0%** | Ground-truth section retrieved at rank 1 |
| **Recall@3** | > 90% | **100.0%** | Ground-truth section included in top-3 candidates |
| **Recall@5** | > 95% | **100.0%** | Ground-truth section included in top-5 candidates |
| **MRR (Mean Reciprocal Rank)** | > 0.85 | **1.0000** | Average reciprocal rank of expected section |
| **Citation Hallucination Check** | 100% | **100.0%** | Direct verification that cited section exists in retrieved text & matches ground truth |

## Topic Breakdown

### Dispute Procedure
- **Questions**: 2
- **Recall@1**: 100.0%
- **MRR**: 1.0000
- **Citation Grounding**: 100.0%

### Leave Entitlement
- **Questions**: 2
- **Recall@1**: 100.0%
- **MRR**: 1.0000
- **Citation Grounding**: 100.0%

### Termination Notice
- **Questions**: 2
- **Recall@1**: 100.0%
- **MRR**: 1.0000
- **Citation Grounding**: 100.0%

### Wages
- **Questions**: 2
- **Recall@1**: 100.0%
- **MRR**: 1.0000
- **Citation Grounding**: 100.0%

### Working Hours
- **Questions**: 2
- **Recall@1**: 100.0%
- **MRR**: 1.0000
- **Citation Grounding**: 100.0%


Detailed CSV results saved in [`results.csv`](file:///D:\000 NSU 4th\cse 596\project\data/eval/results.csv).
