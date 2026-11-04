# Semantic Search System for Food Dataset

This project matches Portuguese search queries to products in the supplied food
catalog and returns five ranked offers. I combine local E5 embeddings, BM25
keyword search, ingredient constraints and GPT-4.1-mini reranking.

**Contextual reranking achieved the highest nDCG@5 across 26 configurations on
all 100 supplied queries: 0.675, with a useful result in the top five for 86
queries and a useful first result for 73.** I evaluated 4,444 distinct
query-product pairs under one shared policy. The implementation also provides a
fast local version for applications that prioritize latency and operating cost.

## Search pipeline

```mermaid
flowchart LR
    A[Interpret the query] --> B[Retrieve and filter candidates]
    B --> C[Rerank up to 24 products]
    C --> D[Return the top five]
```

I prepare the catalog once: validate records, remove exact duplicates, compute
product embeddings, build keyword indexes and extract ingredient claims. Each
search then follows four steps.

### 1. Interpret the query

I separate the requested product from required attributes, excluded ingredients
and preferences. For **`hambúrguer sem queijo`**, the product is a burger and the
excluded ingredient is cheese. Occasion terms such as `para compartilhar` guide
ranking without specifying an exact number of servings.

A local parser handles supported simple requests. In contextual mode,
GPT-4.1-mini interprets complex uncached queries. The interpreter attaches each
requirement to its original query span. I retain the original query during
retrieval and reranking to preserve its wording and context.

### 2. Retrieve candidates and enforce ingredient exclusions

I tested semantic retrieval with E5, keyword retrieval with BM25, and combinations
of both. E5 matches related meanings across different wording. BM25 matches exact
product names, brands and ingredients. The selected pipeline uses both and builds
product text from names, types, categories, descriptions and declared portions.

I check ingredient claims in product names and descriptions before limiting
the candidate list. For `sem queijo`, I reject products that explicitly include
cheese, including varieties such as cheddar and cheese in included side dishes.
I retain products that do not specify whether they contain cheese. I also
distinguish optional additions from default ingredients and keep negation scoped
to the named ingredient: `sem cheddar` does not establish absence of other cheeses.
Each extracted claim retains its source field and text span.

Filtering before truncation lets me replace excluded products with compatible
results farther down the rankings. Supported ingredient absence can improve the
rank of otherwise comparable products. Historical customizations describe past
orders; I exclude them from claims about current options and default ingredients.

I assemble up to 24 distinct candidates by taking turns across three rankings:

| Ranking | Implementation |
| --- | --- |
| Keyword | Portuguese-normalized, stemmed BM25 over name, category and description, weighted 3/1/1. |
| Simple hybrid | Raw-query E5 and keyword results combined through reciprocal-rank fusion. |
| Constraint-aware | Original and positive-query retrieval, lexical matching, and bounded product-identity and attribute adjustments. |

Each ranking supplies up to 100 candidates. Reciprocal-rank fusion sums
`1 / (60 + rank)` to combine rankings without calibrating BM25 and cosine scores.
The positive query omits excluded ingredient terms, so `sem queijo` does not give
positive keyword credit to cheese. I fixed these settings for the experiments;
a branch-removal experiment would measure how far I can simplify the merger.

### 3. Rerank the candidates

GPT-4.1-mini reads the original query alongside candidate names, categories,
taxonomy and description excerpts of up to 600 characters. It returns five
ordered candidate IDs. I shuffle input order deterministically and omit retrieval
scores and evaluation labels to reduce anchoring on the initial rankings.

The reranker evaluates the complete request. For example, it can distinguish a
prepared lunch from a disposable plate even when both listings match the word
`prato`. The fast `constraint-hybrid` mode returns the local ranking directly;
`contextual` adds this reranking step.

### 4. Validate and return results

I validate that the selected IDs belong to the shortlist, contain no duplicates
and pass the ingredient checks again. Scores 5 through 1 represent the final
ordering. If fewer than five compatible offers exist, contextual mode returns
an explicit error; the fast mode supports a shorter result list.

## Design decisions

### Retrieve first, then rerank

I score all 4,997 products locally with inexpensive vector and keyword
operations. I apply the language model only to the 24-product shortlist.
This concentrates model inference on plausible matches and bounds the amount of
product text sent per request.

Ranking every listing with a language model would require processing the whole
catalog for each query. As the catalog grows, candidate retrieval keeps the
expensive reranking stage bounded. I can later replace exact vector search with
an approximate index while preserving the reranker interface. Candidate coverage
remains essential: the reranker can select only products that retrieval supplies.

### Multilingual E5-small for embeddings

I use the pinned `intfloat/multilingual-e5-small` encoder on CPU through ONNX
Runtime. It produces 384-dimensional vectors using `query:` and `passage:`
prefixes, masked mean pooling and L2 normalization. I rank vectors by cosine
similarity, calculated as a dot product after normalization.

E5-small provides Portuguese semantic retrieval, a compact local index and zero
embedding API cost per query. On the common 100-query evaluation, the submitted
semantic baseline scored 0.429 nDCG, E5-base scored 0.424, E5-large INT8 scored
0.444, and full-text MiniLM scored 0.257. I retained E5-small and invested in
contextual reranking, which raised the final pipeline to 0.675. The larger encoder
was tested as a retriever, not as a replacement inside the final pipeline; that
combination remains an experiment. Hosted embedding experiments encountered
access failures, so they produced no equivalent comparison.

### Separate relevance ranking from explicit constraints

A cheese burger can have high semantic similarity to a request for a burger
without cheese. I therefore enforce explicit exclusions through ingredient
checks rather than expecting embedding similarity to encode negation. This also
lets me distinguish an explicit contradiction from missing product information.

### GPT-4.1-mini for reranking

GPT-4.1-mini contextual reranking achieved the highest nDCG and Success@5 in the
100-query comparison.
I selected it for the submitted contextual pipeline and retained the local
constraint-aware ranking as the low-latency alternative. Limiting model input to
24 products and caching validated responses controls recurring API cost.

### Exact local indexes and SQLite

An exact vector matrix and in-process BM25 indexes fit this catalog. SQLite stores
caches and cumulative API accounting. This keeps deployment to one application
without a separate vector database. I deferred images, behavioral ranking and
fine-tuning to focus the budget on text retrieval and measured ranking changes.

## Experiments and results

I tested 26 recoverable configurations on **all 100 assignment queries**. I froze
rankings and parameters before judging, pooled their top-five outputs, and graded
all **4,444 distinct query-product pairs**. This gives every method the same
relevance policy and judged pool across **13,000 returned positions**. I did not
tune methods against these new scores.

### What the experiments taught me

| Question | Evidence from the 100-query comparison | Design decision |
| --- | --- | --- |
| Does product representation matter? | The E5-small full-text baseline scored 0.429 nDCG, versus 0.307 without taxonomy and 0.197 with name/description only. | Retain useful product metadata. Preserve ingredient evidence separately for constraints. |
| Should I use a larger encoder? | The submitted E5-small baseline scored 0.429; E5-base scored 0.424; E5-large INT8 scored 0.444; MiniLM full text scored 0.257. | Retain the compact local encoder. Contextual reranking delivers the larger observed improvement. |
| Do keywords complement semantic retrieval? | BM25 scored 0.424; the original hybrid scored 0.461; lean hybrid scored 0.457. | Keep both signals for candidate generation. These full-system comparisons motivate the combination; a branch ablation would isolate each candidate source's contribution. |
| Does structured interpretation help? | Grounded retrieval scored 0.517. The two facet variants scored 0.449 and 0.437. | Retain query interpretation; additional facet rules alone did not improve the grounded method. |
| Can a local cross-encoder replace contextual reasoning? | The two tested cross-encoder configurations scored 0.404 and 0.385. Contextual reranking scored 0.675. | Select contextual reranking. Keep local ranking as the faster alternative. |
| Do explicit constraints still need a separate mechanism? | The supplied 100 queries contain no explicit ingredient exclusions. Separate diagnostics cover negation, ingredient hierarchy, optional additions and unspecified ingredients. | Enforce supported contradictions directly and retain unknown ingredient status as eligible. Evaluate exclusion behavior separately from this relevance benchmark. |

### Complete comparison

The table is sorted by nDCG@5. Method names identify the frozen configurations;
variants change representation, encoder, lexical preprocessing, fusion or ranking.
The submitted `semantic-local`, `lexical`, `hybrid-local`, `constraint-hybrid` and
`contextual` modes are directly accessible through the application.

| Configuration | nDCG@5 | Success@5 | Precision@5 | MRR@5 | Mandatory conflicts |
| --- | ---: | ---: | ---: | ---: | ---: |
| **contextual** | **0.675** | **86/100** | **55.0%** | **0.780** | **99/500** |
| grounded-local | 0.517 | 79/100 | 40.8% | 0.629 | 135/500 |
| hybrid-local | 0.461 | 75/100 | 36.6% | 0.565 | 171/500 |
| lean-hybrid | 0.457 | 80/100 | 37.2% | 0.599 | 175/500 |
| rrf-pt-0.25 | 0.449 | 77/100 | 38.4% | 0.561 | 164/500 |
| grounded-facets | 0.449 | 76/100 | 36.0% | 0.603 | 167/500 |
| constraint-hybrid | 0.447 | 77/100 | 37.0% | 0.575 | 168/500 |
| small-fields-rrf-0.25 | 0.446 | 77/100 | 36.6% | 0.573 | 171/500 |
| e5-large-int8-full | 0.444 | 77/100 | 38.0% | 0.633 | 175/500 |
| base-pt-rrf-0.25 | 0.439 | 74/100 | 37.6% | 0.570 | 179/500 |
| grounded-facets-v1 | 0.437 | 77/100 | 35.4% | 0.604 | 170/500 |
| rrf-pt-1 | 0.434 | 81/100 | 35.6% | 0.556 | 182/500 |
| semantic-local | 0.429 | 76/100 | 35.4% | 0.522 | 172/500 |
| lexical | 0.424 | 73/100 | 32.0% | 0.569 | 207/500 |
| e5-base-full | 0.424 | 69/100 | 35.4% | 0.556 | 184/500 |
| query-prefix-rrf-small | 0.422 | 73/100 | 35.8% | 0.515 | 176/500 |
| reranker-union20 | 0.404 | 76/100 | 32.4% | 0.542 | 205/500 |
| field-lexical | 0.401 | 73/100 | 30.4% | 0.556 | 214/500 |
| reranker-e5-top40 | 0.385 | 68/100 | 30.4% | 0.490 | 202/500 |
| lexical-name-description | 0.369 | 64/100 | 28.4% | 0.471 | 229/500 |
| lexical-fields-3-1 | 0.349 | 61/100 | 25.0% | 0.487 | 250/500 |
| lexical-pt | 0.338 | 66/100 | 26.4% | 0.464 | 245/500 |
| e5-no-taxonomy | 0.307 | 66/100 | 25.2% | 0.457 | 231/500 |
| minilm-full | 0.257 | 52/100 | 19.8% | 0.367 | 270/500 |
| e5-name-description | 0.197 | 47/100 | 15.4% | 0.273 | 300/500 |
| minilm-name-description | 0.194 | 43/100 | 15.4% | 0.277 | 297/500 |

**I selected contextual reranking:** it leads all four ranking-quality measures,
returns a useful first result for **73/100 queries**, and records the fewest judged
mandatory conflicts. Its nDCG gain over grounded-local is **+0.158**, with a 95%
intent-group bootstrap interval of **[+0.098, +0.213]**. The gain over the submitted
semantic baseline is **+0.246 [+0.176, +0.317]**. I used 2,000 bootstrap replicates
across 64 intent groups. These intervals measure query variation conditional on
the automated labels and are unadjusted for multiple comparisons.

The 26 configurations include completed historical variants as well as the
submitted modes. I excluded an incomplete gateway reranker, unfinished hosted
embedding comparisons and a discarded facet-v3 implementation that I could not
faithfully reproduce. Historical grounded variants reuse their original cached
interpretations for the 30 development queries and existing production
interpretations for the remaining 70. These compare configurations as recovered,
rather than isolating a single changed component.

### Development and evaluation scope

There is no model training in this project. Development still changes prompts,
representations, candidate selection and ranking rules, so I reserved queries to
check those choices. The full 100-query result summarizes the supplied assignment
set; it includes 30 queries previously used for development. I also report the
original remaining 70 separately under the same new pool and policy.

| Method | Development 30 nDCG / Success | Remaining 70 nDCG / Success |
| --- | ---: | ---: |
| Contextual reranking | **0.734 / 24/30** | **0.650 / 62/70** |
| Grounded retrieval | 0.604 / 22/30 | 0.481 / 57/70 |
| Constraint-aware ranking | 0.498 / 22/30 | 0.426 / 55/70 |
| Semantic baseline | 0.421 / 20/30 | 0.432 / 56/70 |

I did not tune later methods against the original 70-query final evaluation. Those
queries had appeared in earlier evaluations, so this re-evaluation is not a newly
collected holdout.

### Earlier independently authored query study

Before the full comparison, I tested four architectures on 30 development queries
and 12 independently model-authored queries. The author received no catalog,
previous queries or rankings. The pools contained 390 and 152 pairs respectively.
I preserve this study separately because its pools and judging policy differ
from the 100-query benchmark; scores cannot be compared across those studies.

| Method | Development nDCG@5 | Held-out nDCG@5 | Held-out Success@5 |
| --- | ---: | ---: | ---: |
| Keyword ranking | 0.434 | 0.646 | 9/12 |
| Simple hybrid | 0.530 | 0.706 | 9/12 |
| Constraint-aware ranking | 0.654 | 0.774 | 10/12 |
| Contextual reranking | **0.787** | **0.799** | **11/12** |

An earlier 30-query synthetic constraint study scored constraint-hybrid at
**0.778 nDCG and 20/30 Success**, versus grounded retrieval at **0.702 and 17/30**
and the semantic baseline at **0.669 and 17/30**. That study supported retaining
explicit constraints. Its separate query set, pool and policy also remain
associated with those results.

## Evaluation methodology

### Define relevance without supplied labels

I use a 0-3 rubric: **0** means irrelevant or contradictory, **1** partial,
**2** useful, and **3** complete. I first freeze query-only intent cards that
separate identity, required attributes, exclusions, preferences and ambiguity.
The card author sees no products. The product judge receives those cards and
actual listing fields, with opaque IDs and no method names, ranking positions,
retrieval scores or behavioral profiles. It cites source fields and text spans.

I assess identity, positive requirements, exclusions and preferences separately.
Deterministic checks only lower the model's raw grade: mandatory contradictions
become 0; unknown identity or positive requirements cap the grade at 1; unknown
exclusions, incomplete preferences or ambiguity cap it at 2. An unspecified
excluded ingredient remains eligible. Historical customizations do not establish
current availability, and false dietary flags do not prove ingredient inclusion.

### Judge a common pool of actual outputs

GPT-4.1-mini graded all 4,444 unique pairs in the combined top-five pool under the
`all100-source-relevance-v1` protocol. I used new annotations throughout and
preserved historical labels and metrics. Accepted
annotations remained fixed when I corrected response-format and citation-schema
errors during execution; the relevance rules did not change.

| Metric | Interpretation |
| --- | --- |
| nDCG@5 | Graded ordering quality against the ideal ranking in the shared judged pool, using gains of `2^grade - 1` and logarithmic rank discounts. |
| Success@5 | Fraction of queries with at least one useful result, graded 2 or 3. |
| Precision@5 | Fraction of the five returned products graded useful. |
| MRR@5 | Reciprocal rank of the first useful result, averaged across queries; no useful result contributes zero. |
| Mandatory conflicts | Returned positions with a judged identity or required-attribute contradiction, counted once per position. |

One query has no positive grade anywhere in the pool. I exclude it from nDCG,
which therefore averages **99 queries**; Success, Precision and MRR include all
**100**. Pooling the top five supports fair comparison of returned rankings. It
does not measure recall against every relevant item in the 4,997-offer catalog.

The supplied queries contain **no explicit ingredient exclusions**. Mandatory
conflicts in this table therefore measure identity and positive requirements,
not dietary-exclusion error rates. I test exclusions separately with diagnostic
cases for ingredient hierarchy, negation, optional ingredients, included sides,
ambiguity, unspecified status and candidate replenishment. A passing diagnostic
suite and a relevance benchmark answer different questions.

### Check the automated judge

GPT-4.1 independently reviewed **100 pairs, one per query**. The first 40 pairs
were selected before primary judging. After observing disagreement, I froze a
plan to expand the audit to 100; the additional selection used query/pair hashes,
without grades, ranks or method identities. I kept the primary annotations fixed.

The two models agreed on usefulness for **79/100 pairs** and on the exact grade
for **56/100**. Replacing all 100 audited annotations with the independent reviews
still places contextual reranking first: **0.667 nDCG versus 0.515** for
grounded-local. All judgments are automated. The primary judge and contextual
reranker both use GPT-4.1-mini, so shared-model bias remains a material limitation;
the separate GPT-4.1 audit tests the ranking decision's sensitivity to disagreement.

In the earlier 12-query study, GPT-4.1 was the primary judge; smaller independent
checks agreed on usefulness for 10/12 development and 5/6 held-out pairs. I keep
those checks associated with their original study rather than combining them
with the new audit. Historical artifacts remain archived privately; this source
repository includes their summaries.

### Latency and cost

Earlier component measurements recorded **26.5 ms median** for local top-100
candidate generation with cached query plans and embeddings. Thirty first API
ranking requests added **1.6 seconds median** and approximately **$0.0016 per
ranking request**. These exclude a complete uncached interpretation path. The
100-query re-evaluation verified cached serving outputs; it did not remeasure
uncached end-to-end latency or operating cost for every configuration.

The full re-evaluation and independent audit added **$1.153763** in API charges.
Cumulative accounting is **$3.777124**, including prior uncertain reservations,
within the **$4.50 ceiling**. All 710 new operations are accounted for, with no new
pending or uncertain charges. This excludes local compute and assistant usage.

## Dataset and operating limits

The supplied 5,000 rows contain three exact duplicate occurrences, leaving
**4,997 offers from 1,806 merchants**. Only 1,707 carry a prepared-food category;
the catalog also contains groceries, drinks and household products. I search
all categories to preserve valid assignment queries.

There are 853 blank descriptions and 134 zero prices. I preserve missing product
facts as unknown and do not interpret zero prices as free offers. The ingredient
matcher recognizes supported expressions in names and descriptions; it cannot
certify absence when a listing omits an ingredient. Reranking uses 600-character
description excerpts and depends on retrieval supplying a relevant candidate.

I validate model outputs before returning results. The last uncached novel-query
probe failed query-provenance validation; the application stopped that request
without retrying or substituting another service. Deployment work includes live
merchant/location/stock eligibility, current option data, freshness and monitoring.

## Setup and usage

Use Python **3.12+** and place `5k_items_curated.csv` and `queries.csv` in the
project root. Git ignores data, credentials, downloaded models, caches and outputs.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python scripts/main.py setup
python scripts/main.py build
```

On macOS/Linux, activate with `source .venv/bin/activate`. Setup downloads the
pinned local assets, approximately 490 MB. `requirements-lock.txt` records the
tested Windows dependency environment. The installed `food-search` command calls
the same CLI as `scripts/main.py`.

Set `OPENAI_NEW_KEY` in ignored `.env` to run contextual search:

```dotenv
OPENAI_NEW_KEY=your_key_here
```

```powershell
python scripts/main.py search "hamburguer sem queijo" --mode contextual --allow-api --json
python scripts/main.py batch --mode contextual --allow-api --output outputs/results.csv
python scripts/main.py serve --mode contextual --allow-api --port 8766
python scripts/main.py budget
```

The server binds to localhost and exposes `/health` and
`/search?q=hamburguer%20sem%20queijo&k=5`. Batch output includes query/item IDs,
ranks, scores, mode and release ID. Use `--overwrite` to replace an existing file.

| Mode | Purpose |
| --- | --- |
| `contextual` | Selected retrieval-plus-reranking pipeline; supports k=1..5. |
| `constraint-hybrid` | Fast local version; supports simple or cached query interpretations offline. |
| `semantic-local` | Original offline baseline and CLI default; performs semantic retrieval without exclusion filtering. |
| `lexical`, `hybrid-local` | Original offline comparison methods. |

`--allow-api` applies to contextual mode. Complex uncached queries may require
both interpretation and ranking calls. I cache validated results by inputs and
policy. Calls use the official OpenAI endpoint and reserve against a persistent
**$4.50 cumulative ceiling**. Preserve the ledger when reusing local state.
Development, re-evaluation and validation accounted for **$3.777124**, including
prior uncertain reservations. This ledger tracks API use, excluding local compute
and assistant usage.

## Source organization and verification

```text
scripts/main.py                 # setup/build/search/batch/serve/budget
src/prosus_search/
  cli.py, runtime.py             # orchestration and configuration
  catalog.py, representation.py  # input validation and product text
  indexing.py, artifacts.py      # index builds and versioned releases
  local_embeddings.py            # E5 inference
  retrieval.py, grounded.py      # retrieval and local ranking
  constraints.py, facets.py      # ingredient extraction and matching
  understanding.py               # query interpretation
  quality.py, model_client.py    # contextual reranking and API calls
  store.py                       # caches and cumulative accounting
  batch.py, server.py            # CSV export and localhost HTTP
  eligibility.py, snapshots.py   # eligibility and immutable state
config.toml                     # paths, models, retrieval and budget
pyproject.toml                   # package and pinned direct dependencies
requirements-lock.txt            # tested environment
```

I verified the cleaned application with **135 core regressions** and **17
evaluation tests**. The checks cover grading arithmetic, complete shared-pool
coverage, source citations, grade caps and budget guards. I independently
recomputed all 26 metric rows, revalidated all 4,544 primary/audit annotations,
and reproduced identical report files on a second scoring run.

The index build passed. HTTP replay matched **all 500 contextual result IDs across
100 queries**, with API access disabled. Source and dataset hashes remained
unchanged during re-evaluation. These checks establish reproducible cached
serving. The next experiments would ablate candidate sources, measure shortlist
coverage and run a full uncached latency benchmark.
