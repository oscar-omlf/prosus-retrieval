# Portuguese marketplace semantic search

Enter a Portuguese query, score the catalog, and return the best `k` items.
The implementation keeps search independent from offline comparison and evaluation.

Round 3 implementation is in progress. The approved design is in
[docs/final-plan.md](docs/final-plan.md); execution adjustments are recorded in
[docs/implementation-decisions.md](docs/implementation-decisions.md).

The local multilingual E5 backend needs no API key after setup. OpenAI embedding
and relevance-judging workflows require an environment key and explicit API opt-in.
No placeholder key is used for model calls. Results never imply human validation.

Place the supplied `5k_items_curated.csv` and `queries.csv` beside this README.
They are excluded from Git pending redistribution permission. Python 3.12+ is
required. Full setup, replay commands and measured results follow with the
completed implementation.
