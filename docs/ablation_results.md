# Retrieval Ablation Results

Full results for the retrieval ablation on the resume-screening pipeline. All runs are retrieval-only (no LLM in the loop).

## Setup

| Mode | Pipeline |
|---|---|
| Dense | Dense retrieval only (`text-embedding-3-small`, cosine similarity) |
| Dense + rerank | Dense → cross-encoder rerank (no BM25) |
| Fusion | BM25 + dense → Reciprocal Rank Fusion (k = 60, BM25 weight 2.0, dense 1.0) |
| Full | BM25 + dense → RRF → cross-encoder rerank (`cross-encoder/ms-marco-MiniLM-L-6-v2`) |

- **Index:** 258 section-level chunks from 50 resumes
- **Queries:** 28 recruiter-style queries (10 semantic, 9 numeric, 9 mixed)
- **Labels:** 343 graded query–resume judgments (200 full matches, grade 2; 143 partial, grade 1)
- **Metrics:** computed per resume (chunk hits collapsed to unique resumes, first occurrence wins). Recall and MRR count any grade ≥ 1 as relevant; nDCG@10 uses the grades as gains.
- **Candidate pools:** each retriever returns its top 200 chunks; the cross-encoder scores the top 180 candidates in both rerank modes.
- **Coverage caveat:** 6 labeled resumes are not in the index, which caps attainable recall on 16 of 28 queries (by one or two resumes on most). The cap is identical across modes.

## Aggregate results

Mean over queries in each group. Best value per row in **bold**.

| Query type | Metric | Dense | Dense + rerank | Fusion | Full |
|---|---|---:|---:|---:|---:|
| Overall (n=28) | Recall@5 | 0.306 | 0.386 | 0.323 | **0.387** |
| Overall (n=28) | Recall@10 | 0.492 | **0.542** | 0.501 | **0.542** |
| Overall (n=28) | MRR | 0.857 | **0.866** | 0.858 | **0.866** |
| Overall (n=28) | nDCG@10 | 0.616 | 0.689 | 0.646 | **0.690** |
| Semantic (n=10) | Recall@5 | 0.428 | **0.437** | 0.410 | **0.437** |
| Semantic (n=10) | Recall@10 | 0.581 | 0.612 | **0.655** | 0.612 |
| Semantic (n=10) | MRR | 0.950 | **1.000** | **1.000** | **1.000** |
| Semantic (n=10) | nDCG@10 | 0.798 | 0.839 | **0.862** | 0.839 |
| Numeric (n=9) | Recall@5 | 0.192 | 0.281 | 0.204 | **0.286** |
| Numeric (n=9) | Recall@10 | 0.394 | **0.429** | 0.364 | **0.429** |
| Numeric (n=9) | MRR | **0.852** | 0.714 | 0.698 | 0.714 |
| Numeric (n=9) | nDCG@10 | 0.481 | 0.546 | 0.462 | **0.548** |
| Mixed (n=9) | Recall@5 | 0.283 | **0.434** | 0.345 | **0.434** |
| Mixed (n=9) | Recall@10 | 0.491 | **0.576** | 0.467 | **0.576** |
| Mixed (n=9) | MRR | 0.759 | **0.870** | 0.861 | **0.870** |
| Mixed (n=9) | nDCG@10 | 0.548 | **0.666** | 0.591 | **0.666** |

## Per-query nDCG@10

`Relevant` is the number of labeled resumes (grade ≥ 1) for the query.

| ID | Type | Query | Relevant | Dense | Dense + rerank | Fusion | Full |
|---|---|---|---:|---:|---:|---:|---:|
| Q01 | semantic | Strong machine learning background with hands-on deep learning experience | 46 | 0.96 | 0.76 | 0.77 | 0.76 |
| Q02 | semantic | Frontend developer experienced with React | 12 | 0.84 | 0.80 | 0.87 | 0.80 |
| Q03 | semantic | Experience building RAG or LLM-based chatbots | 15 | 0.67 | 0.78 | 0.75 | 0.78 |
| Q04 | semantic | Computer vision experience with YOLO object detection | 18 | 0.76 | 0.89 | 0.89 | 0.89 |
| Q05 | semantic | NLP experience with transformer models like BERT | 18 | 0.92 | 0.84 | 0.94 | 0.84 |
| Q06 | semantic | Robotics or embedded systems hands-on experience | 10 | 0.67 | 0.73 | 0.82 | 0.73 |
| Q07 | semantic | Data analyst skilled in Power BI or Tableau dashboards | 14 | 0.70 | 0.91 | 0.86 | 0.91 |
| Q08 | semantic | Mobile app development experience with Flutter | 5 | 0.87 | 0.91 | 0.77 | 0.91 |
| Q09 | semantic | Candidate with research publications or conference papers | 5 | 0.58 | 0.78 | 0.95 | 0.78 |
| Q10 | semantic | Internship experience at ISRO or a space research organization | 3 | 1.00 | 1.00 | 1.00 | 1.00 |
| Q11 | numeric | CGPA above 8.5 on a 10-point scale | 17 | 0.62 | 0.49 | 0.65 | 0.49 |
| Q12 | numeric | CGPA above 9 on a 10-point scale | 7 | 0.30 | 0.00 | 0.20 | 0.00 |
| Q13 | numeric | At least 2 years of professional work experience | 5 | 0.26 | 0.56 | 0.27 | 0.56 |
| Q14 | numeric | Graduating in 2025 | 26 | 0.51 | 0.93 | 0.96 | 0.93 |
| Q15 | numeric | Already graduated (degree completed by end of 2024) | 12 | 0.36 | 0.55 | 0.13 | 0.55 |
| Q16 | numeric | Completed 3 or more internships | 21 | 0.59 | 0.49 | 0.46 | 0.51 |
| Q17 | numeric | Scored above 90% in class 12th board exams | 9 | 0.55 | 0.48 | 0.41 | 0.48 |
| Q18 | numeric | Achieved model accuracy above 95% in a project | 10 | 0.42 | 0.67 | 0.47 | 0.67 |
| Q19 | numeric | At least 2 research publications | 5 | 0.71 | 0.74 | 0.62 | 0.74 |
| Q20 | mixed | Python developer with AWS cloud experience | 13 | 0.40 | 0.65 | 0.77 | 0.65 |
| Q21 | mixed | Machine learning internship experience plus NLP skills | 16 | 0.54 | 0.45 | 0.18 | 0.45 |
| Q22 | mixed | IIT student with CGPA above 8 | 4 | 0.41 | 0.63 | 0.43 | 0.63 |
| Q23 | mixed | Frontend developer with 2+ years of professional experience | 8 | 0.84 | 0.78 | 0.71 | 0.78 |
| Q24 | mixed | Data scientist with a master's degree | 8 | 0.34 | 0.29 | 0.40 | 0.29 |
| Q25 | mixed | Full stack developer with Node.js and MongoDB | 8 | 0.73 | 0.82 | 0.71 | 0.82 |
| Q26 | mixed | Docker experience and Python skills | 7 | 0.52 | 0.89 | 0.83 | 0.89 |
| Q27 | mixed | LLM fine-tuning experience during an internship | 7 | 0.60 | 0.98 | 0.83 | 0.98 |
| Q28 | mixed | Computer vision internship and CGPA above 8 | 14 | 0.56 | 0.51 | 0.46 | 0.51 |

## Per-query Recall@5

`Relevant` is the number of labeled resumes (grade ≥ 1) for the query.

| ID | Type | Query | Relevant | Dense | Dense + rerank | Fusion | Full |
|---|---|---|---:|---:|---:|---:|---:|
| Q01 | semantic | Strong machine learning background with hands-on deep learning experience | 46 | 0.11 | 0.11 | 0.11 | 0.11 |
| Q02 | semantic | Frontend developer experienced with React | 12 | 0.42 | 0.42 | 0.42 | 0.42 |
| Q03 | semantic | Experience building RAG or LLM-based chatbots | 15 | 0.27 | 0.33 | 0.27 | 0.33 |
| Q04 | semantic | Computer vision experience with YOLO object detection | 18 | 0.22 | 0.28 | 0.28 | 0.28 |
| Q05 | semantic | NLP experience with transformer models like BERT | 18 | 0.28 | 0.28 | 0.28 | 0.28 |
| Q06 | semantic | Robotics or embedded systems hands-on experience | 10 | 0.50 | 0.40 | 0.40 | 0.40 |
| Q07 | semantic | Data analyst skilled in Power BI or Tableau dashboards | 14 | 0.29 | 0.36 | 0.36 | 0.36 |
| Q08 | semantic | Mobile app development experience with Flutter | 5 | 0.60 | 0.80 | 0.40 | 0.80 |
| Q09 | semantic | Candidate with research publications or conference papers | 5 | 0.60 | 0.40 | 0.60 | 0.40 |
| Q10 | semantic | Internship experience at ISRO or a space research organization | 3 | 1.00 | 1.00 | 1.00 | 1.00 |
| Q11 | numeric | CGPA above 8.5 on a 10-point scale | 17 | 0.18 | 0.18 | 0.24 | 0.18 |
| Q12 | numeric | CGPA above 9 on a 10-point scale | 7 | 0.14 | 0.00 | 0.14 | 0.00 |
| Q13 | numeric | At least 2 years of professional work experience | 5 | 0.00 | 0.40 | 0.20 | 0.40 |
| Q14 | numeric | Graduating in 2025 | 26 | 0.12 | 0.19 | 0.19 | 0.19 |
| Q15 | numeric | Already graduated (degree completed by end of 2024) | 12 | 0.08 | 0.33 | 0.00 | 0.33 |
| Q16 | numeric | Completed 3 or more internships | 21 | 0.19 | 0.19 | 0.14 | 0.24 |
| Q17 | numeric | Scored above 90% in class 12th board exams | 9 | 0.22 | 0.33 | 0.22 | 0.33 |
| Q18 | numeric | Achieved model accuracy above 95% in a project | 10 | 0.20 | 0.30 | 0.30 | 0.30 |
| Q19 | numeric | At least 2 research publications | 5 | 0.60 | 0.60 | 0.40 | 0.60 |
| Q20 | mixed | Python developer with AWS cloud experience | 13 | 0.08 | 0.38 | 0.31 | 0.38 |
| Q21 | mixed | Machine learning internship experience plus NLP skills | 16 | 0.19 | 0.12 | 0.06 | 0.12 |
| Q22 | mixed | IIT student with CGPA above 8 | 4 | 0.25 | 0.25 | 0.25 | 0.25 |
| Q23 | mixed | Frontend developer with 2+ years of professional experience | 8 | 0.50 | 0.62 | 0.38 | 0.62 |
| Q24 | mixed | Data scientist with a master's degree | 8 | 0.25 | 0.25 | 0.25 | 0.25 |
| Q25 | mixed | Full stack developer with Node.js and MongoDB | 8 | 0.50 | 0.62 | 0.50 | 0.62 |
| Q26 | mixed | Docker experience and Python skills | 7 | 0.29 | 0.71 | 0.57 | 0.71 |
| Q27 | mixed | LLM fine-tuning experience during an internship | 7 | 0.29 | 0.71 | 0.57 | 0.71 |
| Q28 | mixed | Computer vision internship and CGPA above 8 | 14 | 0.21 | 0.21 | 0.21 | 0.21 |

## Notes

- Dense + rerank and Full return the same top 10 resumes on 27 of 28 queries: with only 258 chunks, the cross-encoder scores 180 of them in either mode, so BM25 fusion has almost no effect once reranking runs.
- With 9–10 queries per type, per-type differences are directional. The cross-encoder's overall gain over dense is clearer (paired sign-flip test: Recall@5 p = 0.01, nDCG@10 p = 0.04).
