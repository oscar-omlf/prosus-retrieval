# Semantic Search System for Food Dataset

This project matches Portuguese search queries to products in the supplied food
catalog and returns five ranked offers. I combine local E5 embeddings, BM25
keyword search, ingredient constraints and GPT-4.1-mini reranking.

**I selected contextual reranking because it achieved the highest nDCG@5 among
the four methods in my final comparison: 0.787 on 30 development queries and
0.799 on 12 held-out queries.** The implementation also provides a fast local
version for applications that prioritize latency and operating cost.

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
embedding API cost per query. It outperformed the tested MiniLM encoder.
E5-base increased development nDCG from 0.508 to 0.564, while both models achieved
Success@5 of 10/30; I retained E5-small for its smaller runtime footprint and the
paired comparison reported below. Hosted embedding experiments encountered access
failures, so they produced no equivalent comparison.

### Separate relevance ranking from explicit constraints

A cheese burger can have high semantic similarity to a request for a burger
without cheese. I therefore enforce explicit exclusions through ingredient
checks rather than expecting embedding similarity to encode negation. This also
lets me distinguish an explicit contradiction from missing product information.

### GPT-4.1-mini for reranking

GPT-4.1-mini achieved the highest nDCG and Success@5 in the final held-out sample.
I selected it for the submitted contextual pipeline and retained the local
constraint-aware ranking as the low-latency alternative. Limiting model input to
24 products and caching validated responses controls recurring API cost.

### Exact local indexes and SQLite

An exact vector matrix and in-process BM25 indexes fit this catalog. SQLite stores
caches and cumulative API accounting. This keeps deployment to one application
without a separate vector database. I deferred images, behavioral ranking and
fine-tuning to focus the budget on text retrieval and measured ranking changes.

## Experimental development

I tested changes in stages to identify whether errors came from product
representation, the encoder, query interpretation, constraints or final ordering.
Within each experiment, competing methods shared the same judged product pool.
The table records the motivation, result and decision for each major experiment.

Each row reports its own comparison. Expanding a judged pool changes the ideal
ranking used by nDCG; evaluation-policy versions also differ between stages.

| Experiment and motivation | Result | Decision |
| --- | --- | --- |
| Remove metadata to test whether it distracts from product identity. | Full E5 text: 0.518 nDCG; no taxonomy: 0.458; name/description only: 0.391. | Retain metadata in the original baseline. Build the later constraint-aware representation from explicit product claims. |
| Compare keyword search and a smaller multilingual encoder with E5. | Best tested BM25 variant: 0.499; MiniLM full text: 0.337; E5: 0.518. | Keep E5 and use BM25 in candidate generation. |
| Combine lexical and dense rankings to improve exact and semantic matching. | Dense-dominant fusion: 0.522 versus E5: 0.518; gain 95% interval [-0.070, +0.070]. | Use fusion as a candidate source; retain E5 as the encoder. |
| Increase encoder capacity with E5-base. | E5-base: 0.564 versus E5-small: 0.508; both succeed on 10/30 queries; gain interval [-0.041, +0.159]. | Retain the smaller encoder. E5-large INT8 completed inference but did not complete relevance evaluation. |
| Test a local cross-encoder for improved ordering. | Dense/union reranking: 0.504 / 0.490 versus E5: 0.500; reranker-only p95 latency: 2.2-2.5 seconds. | Reject this configuration on the measured quality/latency trade-off. |
| Add semantic query interpretation and product-attribute matching. | Grounded retrieval: 0.844 versus 0.592; Success: 9/30 versus 7/30. Hard-attribute contradictions increased. | Retain semantic interpretation and implement a separate exclusion filter. |
| Add structured product facets to the grounded method. | Selected facet version: 0.737 versus grounded control: 0.727; both succeed on 9/30 queries. | Continue with explicit constraints rather than expanding facet rules alone. |
| Test exclusion handling on 30 fresh synthetic queries. | Constraint-hybrid: 0.778; grounded: 0.702; E5: 0.669. Success: 20/30 versus 17/30 for each alternative. | Retain constraint-aware retrieval and use it as the control for contextual reranking. |

Field-separated fusion and query-prefix variants produced rankings with incomplete
judgment coverage. An earlier source-cited LLM reranker completed 5/30 queries
before an API rejection. I excluded these unfinished comparisons from method
selection.

### Final comparison and selected method

I compared four architectures on 30 development queries using a shared pool of
390 query-product pairs. I then froze the methods and evaluated 12 independently
model-authored queries using a shared 152-pair pool. The query author received no
catalog, previous queries or rankings.

| Method | Development nDCG@5 | Held-out nDCG@5 | Held-out Success@5 | Held-out MRR@5 | Held-out mandatory conflicts |
| --- | ---: | ---: | ---: | ---: | ---: |
| Keyword ranking | 0.434 | 0.646 | 9/12 | 0.646 | 9/60 |
| Simple hybrid | 0.530 | 0.706 | 9/12 | 0.750 | 10/60 |
| Constraint-aware ranking | 0.654 | 0.774 | 10/12 | **0.833** | **2/60** |
| Contextual reranking | **0.787** | **0.799** | **11/12** | 0.819 | 4/60 |

**I selected contextual reranking for its leading nDCG on both samples and its
11/12 held-out Success@5.** The local constraint-aware method offers a faster
operating point and recorded fewer mandatory conflicts on the held-out sample.

The paired nDCG gain over the local control was +0.133 on development data, with
95% interval [+0.029, +0.257], and +0.024 on held-out data, with interval
[-0.095, +0.165]. These intervals describe query variation conditional on the
model judgments. The 12-query sample provides an initial check of generalization.

Local control top-100 generation measured **26.5 ms median with cached query plans
and embeddings**. Thirty first API ranking requests added **1.6 seconds median**
and approximately **$0.0016 per ranking request**. These component timings exclude
a complete uncached interpretation path.

## Evaluation methodology

The assignment supplies no relevance labels. I pooled the top-five results from
competing methods and used blinded model judgments grounded in product fields.
The grader assigned a score from 0 to 3: irrelevant or contradictory, partial,
useful, or complete. An otherwise relevant product with an unspecified excluded
ingredient can receive grade 2; an explicit contradiction requires grade 0.

I report four measures:

| Metric | Interpretation |
| --- | --- |
| nDCG@5 | Quality of graded ordering against the best ranking in the judged pool. |
| Success@5 | Fraction of queries with at least one useful result, graded 2 or 3. |
| MRR@5 | Reciprocal rank of the first useful result, averaged across queries; queries with no useful top-five result contribute zero. |
| Mandatory conflicts | Returned positions with a judged identity or attribute contradiction. |

GPT-4.1 judged the held-out products independently of the GPT-4.1-mini reranker.
Preselected model checks agreed on usefulness for 10/12 development pairs and
5/6 held-out pairs. All relevance labels are model-generated. One held-out
exclusion judgment remained disputed in the source audit.

I calculate nDCG for the 29 development and 11 held-out queries whose pools have
positive ideal gain. Success and MRR include all queries. Product pools are
shallow and queries are synthetic, so the results measure performance on these
samples rather than full-catalog recall or live shopper behavior.

I versioned changes to the judging policy and preserved earlier labels.
Historical development judgments applied stricter rules to unknown requirements.
The latest held-out query-card policy also permits open discovery without forcing
a mandatory product identity; I amended it after a query-only pilot, before
product judging. I kept rankings fixed. I did not reuse the original 70-query
final evaluation for tuning later methods. Historical judgments and replay
artifacts remain archived privately; this repository includes their summaries.

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
Development and validation accounted for **$2.623361**, including uncertain
reservations. This ledger tracks API use, excluding local compute and assistant
usage.

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

I verified the cleaned application with 135 core regressions, a complete index
build and exact replay of all 500 previously exported result IDs. The next
experiments would expand the untouched evaluation sample and ablate the three
candidate rankings to measure their individual contribution.
