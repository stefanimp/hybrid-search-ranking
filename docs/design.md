# Design notes

## Candidate generation

Five cached sources contribute candidates in a fixed order: basic/BM25, cross-encoder 100k,
cross-encoder full, dense and TF-IDF. Their feature-construction depth limits are 80, 60,
120, 260 and 220. Candidates outside the local corpus are discarded. A document appearing
in several sources gets one row with signals from each source.

Source features include presence, reciprocal rank, inverse rank, raw scores and per-query
min-max normalization. Content features measure lexical coverage, longer query terms,
numeric overlap, exact phrases, first occurrence, lengths, question types and paragraph position.
The tournament training matrix has 73 feature columns.

## Learning and fusion

The ranker uses LambdaRank with 520 estimators, learning rate 0.025, 31 leaves, minimum
18 child samples and seed 42. Query groups remain intact during five-fold validation.
Fixed score gaps of 0.18, 0.10 and 0.08 select one to three passages; these are constants,
not the output of an optimization procedure.

| RRF source | Weight | Depth |
| --- | ---: | ---: |
| LambdaRank shortlist | 2.00 | 10 |
| Dense, shallow | 1.50 | 10 |
| Full cross-encoder blend | 1.00 | 40 |
| 100k cross-encoder blend | 0.65 | 40 |
| Dense, deep | 0.40 | 100 |

RRF adds `weight / (60 + rank)` for each occurrence. Its final list keeps the length chosen
by LambdaRank. The two dense contributions have different depths and are intentional.
Optional variants add TF-IDF at weights 0.20 or 0.60, or basic retrieval at weight 0.30.

## Validation limits

Known relevant documents are added to the candidate sets before internal folds are formed.
Their MAP@10 therefore does not estimate end-to-end recall. The private leaderboard score
of 0.63461 comes from the supplied final standings, not a local reevaluation of hidden labels.

The feature named `same_base_as_query_id` actually compares the passage base with query text.
Its historical behavior is retained because changing the feature would change the trained model.
Order-sensitive operations and tie handling are also preserved.

## Tournament contract

The supplied tournament task specifies MAP@10 and predictions for `test_queries.csv`.
The data page, reviewed offline,
names `corpus.csv`; the archived implementation uses `corpus_100k.csv`. Their equivalence and
the creation of the local corpus cannot be established from the available material.

The task describes the format of `train_qrels.csv`, whose header is `Id,Expected` locally.
The archived exporter writes `Id,Predicted`. This edition preserves that output for regression
comparison; a new competition submission would require checking the accepted template.

## Project scope

This is stefanimp's individual university project. The current maintenance edition was
prepared with AI assistance. Original academic files remain unchanged outside this repository;
historical source is retained in a separate local archive. Obsolete sweeps and instructions are omitted from
the current tree. Synthetic fixtures and output hashes preserve regression evidence without
redistributing the corpus. Original screenshots and academic handouts remain offline.
The public repository starts from this maintained edition and excludes the private archive.
