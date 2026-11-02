# Portuguese marketplace semantic search

Type a Portuguese query and choose how many results to return. The system embeds
the query, scores all **4,997 distinct catalog items**, and returns the top `k`.
Product vectors are prepared once. Search never calls a relevance judge.

**Working now:** offline multilingual E5, BM25 and their RRF combination;
100 queries × five results × three runs. **API spend: $0.** The OpenAI comparisons
and blinded LLM evaluator are implemented but have not been run against live APIs.
Relevance metrics remain unavailable; no winner or human validation is claimed.
Semantic-local is the provisional default and has known qualitative misses.

[Ranked results](results/selected-top5.csv) · [Measured results](docs/results.md) ·
[Five-slide presentation](docs/presentation.pptx) · [Slide text](docs/slides.md)

## Quick start

Requires Python 3.12+, network access for setup, and the supplied
`5k_items_curated.csv` and `queries.csv` in the repository root. Raw inputs, model
weights, caches and detailed outputs are excluded from Git because redistribution
permission is unconfirmed. Submitted result CSVs contain stable query/item IDs and
scores; join them to the supplied inputs for full text.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
python -m prosus_search setup --local-model
python -m prosus_search preflight
$build = python -m prosus_search build --backend local | ConvertFrom-Json
python -m prosus_search search "pizza de queijo" --k 5
```

On macOS/Linux activate with `source .venv/bin/activate`; use the same Python
commands and copy the returned `build_manifest` path where needed. The lock file
records the tested Windows/Python 3.12 environment. GitHub Actions also passes the
offline fixture tests on Ubuntu; full-catalog timing was measured on Windows.
Alternatively use `pip install -e '.[local,dev]'` and record resolved
versions. Setup downloads approximately 490 MB. Subsequent local operation needs
no internet or API key.

```powershell
python -m prosus_search search "prato para compartilhar" --k 10 --json
python -m prosus_search search "pizza de queijo" --mode hybrid-local --k 5
python -m prosus_search search "pizza de queijo" --mode lexical --k 5
$run = python -m prosus_search benchmark --build $build.build_manifest | ConvertFrom-Json
$benchmark = Join-Path $run.output_directory 'benchmark.json'
python -m prosus_search evaluate --benchmark $benchmark --dry-run
python -m prosus_search export --benchmark $benchmark --run semantic-local --output results/selected-top5.csv
python -m prosus_search budget
```

`k` defaults to five. Semantic/lexical accept `1..catalog size`; interactive hybrid
accepts up to 100 because its candidate depth is fixed. Scores are cosine, BM25 or
RRF ranking signals, **not probabilities**, and differ across score types. No
relevance cutoff is imposed: unsupported queries still receive ranked candidates.

For repeated requests, reuse the loaded service:

```python
from prosus_search.service import SearchService
search = SearchService(mode="semantic-local")
results = search.search("sopa para o inverno", k=5)
for item in results["results"]:
    print(item["rank"], item["name"], item["item_id"])
```

The convenience function `prosus_search.service.search(query, k=5)` also works;
creating a fresh service for every request repeats loading.

## Preparation and comparison

Strict CSV/JSON validation preserves provenance and collapses only identical
repeated item IDs. Separate merchant offers remain distinct. Shared text includes
name, category, readable taxonomy, description and declared portion tags. Missing
facts stay missing. Dietary flags, price, profiles and images do not affect
retrieval text. Dense text preserves accents; BM25 folds accents/case while
retaining negation and numbers. Oversized inputs fail rather than truncate.

The local model is the author's full-precision ONNX export of
[intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small),
MIT license, commit `614241f622f53c4eeff9890bdc4f31cfecc418b3`. Verified files,
required `query:`/`passage:` prefixes, attention-mask mean pooling and L2
normalization produce 384-dimensional vectors. Search never downloads files.

Exact NumPy scoring uses stable item-ID ties. Hybrid uses equal-weight RRF,
`k=60`, top 100 per branch, positive lexical matches only. Failures never silently
fall back to another retrieval method.

`config.toml` controls paths, model spaces, initial parameters and budgets; unknown
keys fail. Paths resolve beside the file. Fixed benchmark builds verify input
hashes and the grouped split. New data requires versioned hashes and an appropriate
evaluation protocol. Direct lexical search can ingest a changed catalog; semantic
search needs a rebuilt release. `--release` pins interactive search to a snapshot.

## Hosted experiments and evaluation

Configure `OPENAI_API_KEY` securely in the environment when available. Never put
keys in source, Git, saved terminal commands or chat. Placeholder keys are rejected.
Cached replay is the default; paid requests require `--allow-api`.

```powershell
$hosted = python -m prosus_search build --backend hosted --allow-api | ConvertFrom-Json
$comparison = python -m prosus_search benchmark --build $hosted.build_manifest | ConvertFrom-Json
python -m prosus_search evaluate --benchmark (Join-Path $comparison.output_directory 'benchmark.json') --allow-api
```

Five hosted configurations compare BM25, small/large embeddings and each hybrid
with common evidence. SDK mechanics use synthetic tests; actual account access,
live schema compatibility and hosted latency remain unverified. Fixtures never
become dataset measurements.

Evaluation pools top-five candidates, hides system/rank/score, creates query-only
intent cards and requests pointwise Portuguese source-backed grades. Validation
checks exact quotes/scalars, requirement IDs and grade caps. Missing evidence
differs from contradiction. Cached outputs and distinct replicate IDs support
bounded repeat, equivalent-prompt, grader and independent intent-card checks.
One retry total per logical assessment is budgeted; failures remain missing.

The frozen split has **30 development / 70 final queries**, grouped by intent.
Final judging requires an explicit end-of-iteration decision and frozen selection.
No final labels were generated. The CLI gate is procedural, not a security boundary.
Human calibration remains an optional future enhancement.

With valid labels, reports mark nDCG@5, Success@5, MRR@5 and constraint rates as
**LLM-judged**. Incomplete pools use the common complete-query subset and keep
full-split metrics null. No zero imputation, full-catalog recall claim or promotion
from incomplete evidence. Bootstrap intervals condition on automated labels and
do not measure judge accuracy. Frozen selection rules retain simplicity when
differences are inconclusive; see [the protocol](protocols/README.md).

The persistent ledger reserves before requests and includes pending/uncertain
charges. It stops at **$4.50**, leaving $0.50 untouched: $0.50 embeddings, $1.40
development, $0.85 final, $1 Round 4 and $0.75 recovery. SDK retries are disabled;
application retries reserve again. Never delete the ledger to bypass limits.
The ledger cannot observe unrelated account spending. Verify tariffs and remaining
credit before paid execution.

## Replay and tests

```powershell
python -m pytest -q
python scripts/measure_local.py
python scripts/verify_local.py --build $build.build_manifest --benchmark $benchmark
python scripts/replay_benchmark.py --build $build.build_manifest --reference $benchmark
```

Tests cover parsing, alignment, scoring/fusion, accounting, interrupted publishing,
evidence validation, metrics and end-to-end benchmark mechanics using synthetic
fixtures. They need no raw data, pretrained downloads or paid APIs.

`work/state.sqlite3` stores caches and the ledger. Immutable releases under
`work/artifacts/releases/<hash>/` contain evidence, ordered IDs, vectors and
checksums. Build manifests record input/config/code/package versions. Each
`outputs/<run>/` contains top-five CSV/JSONL, top-100 diagnostics, selected results
and measurements. Replay takes an explicit build path rather than a mutable
current pointer. Keep detailed outputs and judgment payloads private.

Changed text or portion tags re-embed affected items. Excluded-metadata updates
reuse vectors in a new evidence snapshot; deletions disappear from new releases.
Model/prefix/epoch changes create separate cache spaces. Saved artifacts replay
results; hosted aliases cannot guarantee identical fresh inference. An incomplete
build never replaces the current pointer. Missing assets require `setup`; corrupt
releases require rebuilding; hosted cache misses require explicit API access.

## Scope and delivery

Better discovery and cold-start merchant exposure are business hypotheses;
conversion and fairness are unmeasured. Production geography, inventory, eligibility
and trusted dietary facts require authoritative feeds. Consider ANN after measured
latency/memory pressure; consider reranking only when judged candidates are useful
but misordered. No server, vector database or UI is needed for this benchmark.

[The frozen plan](docs/final-plan.md) remains authoritative, supplemented by
[recorded execution decisions](docs/implementation-decisions.md). Round 4 has not
begun. Send the GitHub link and setup instructions at least **one day before the
interview**; the interview date was not provided.
