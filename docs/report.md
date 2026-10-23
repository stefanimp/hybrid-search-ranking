# Hybrid Search Ranking

### stefanimp · Technical report · 23 October 2026

A hybrid passage-ranking system that combines semantic retrieval, lexical evidence,
cross-encoder scores and supervised learning to rank.

## Outcome

| Measure | Result | Source |
| --- | --- | --- |
| Tournament position | First place | Final private leaderboard capture |
| Private MAP@10 | 0.63461 | Final private leaderboard capture |
| Local corpus | 100,000 passages | Archived corpus CSV |
| Query sets | 783 train / 224 test | Archived query CSVs |
| Maintenance parity | Identical prediction CSV bytes | Original and maintained pipeline comparison |

This individual University of Jaén project ranks relevant passages for natural-language queries.
It returns document identifiers rather than generated answers. The main difficulty is separating
passages that answer a question from passages that merely discuss the same topic.

Semantic retrieval provides broad coverage. Lexical matches help with names, dates and exact
phrases. A LambdaRank model learns interactions between those signals, and reciprocal rank
fusion combines its shortlist with complementary rankings. The resulting system uses the
strengths of each stage without requiring their raw scores to share the same scale.

## Reading the score

The leaderboard capture identifies 0.63461 as the final private result and states that the
private subset covers approximately 46% of test. The historical public score of 0.725 concerns
a different subset. Neither value is an average of the public and private results.

The capture remains offline because it includes personal details.

This maintenance work preserves the prediction behavior. It is not a new Kaggle evaluation
and does not claim a higher competitive score.

---

## Architecture

Cached candidate lists are merged by document ID. Unknown documents are filtered out, and each
retained query-document pair receives source scores, rank features and content features.

| Candidate source | Feature-construction depth | Contribution |
| --- | --- | --- |
| Basic / BM25 | 80 | Lexical precision |
| 100k cross-encoder | 60 | Dense and joint relevance scores |
| Full cross-encoder | 120 | Joint query-passage comparison |
| Dense | 260 | Semantic coverage |
| TF-IDF | 220 | Word and character evidence |

The observed training matrix has 73 columns. Content features include title and passage term
coverage, exact phrases, numeric overlap, first occurrence, log lengths, paragraph position
and question type. Feature order and historical transformations are retained.

## Learning and fusion

The LightGBM ranker uses LambdaRank, 520 estimators, learning rate 0.025, 31 leaves and seed 42.
Five folds split by query. Fixed score gaps of 0.18, 0.10 and 0.08 select a short list.

RRF contributes weight / (60 + rank). Its sources are the ranker shortlist (2.00, depth 10),
dense retrieval (1.50, depth 10), full cross-encoder (1.00, depth 40), 100k cross-encoder
(0.65, depth 40), and deeper dense retrieval (0.40, depth 100). The two dense contributions
are intentional. The final output keeps the shortlist length chosen by the ranker.

The maintained pipeline exports one final CSV by default. A flag enables the three historical
fusion variants. Their shared logic is implemented once rather than copied across exporters.
Unused helpers and the misleading threshold-optimization wrapper have been removed.

---

## Reproduction and quality checks

The project is an installable Python package with explicit input and output paths. The lockfile
records dependencies; the CLI validates CSV contracts before loading trusted joblib caches.
Each run needs a new or empty output directory so reference artifacts are not overwritten.

```text
uv sync --python 3.12 --locked --extra dev
uv run hybrid-search demo
uv run pytest -q
```

The demo is synthetic. Full replay needs the original four CSVs and five caches, which remain
outside Git. No original scripts for regenerating the neural caches were available. The caches
cover tournament queries and do not enable a live search interface for arbitrary new queries.

The original and maintained pipelines were run on identical inputs with PYTHONHASHSEED=0.
The final CSV and all three optional variants match byte for byte. Synthetic fixtures also
check features and candidate merging without distributing the dataset. Detailed input hashes,
output digests, dependency versions and current checks are recorded in verification.json.

## Limits that matter

- Internal validation inserts known positives into candidates before splitting folds. It is not an end-to-end retrieval evaluation.
- A historically named feature compares a document base with query text, not the query ID. Its behavior is retained for parity.
- The data page names corpus.csv; the delivered code uses corpus_100k.csv. Their equivalence cannot be established from the supplied captures.
- The original exporter uses Id,Predicted while the task references train_qrels.csv, whose local header is Id,Expected. A new upload needs template validation.

The author is stefanimp. Source evidence consists of the original academic files,
tournament captures and regression artifacts. The maintenance edition was prepared with AI
assistance. Code and documentation use MIT; third-party data and models retain their own terms.
