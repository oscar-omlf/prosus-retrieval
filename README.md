# Portuguese marketplace semantic search

Type a Portuguese query and choose how many results to return. The system embeds
the query, scores all **4,997 distinct catalog items**, and returns the top `k`.
Product vectors are prepared once. Search never calls a relevance judge.

An experimental grounded method is explicitly selectable with
`python -m prosus_search search "hambúrguer sem queijo para viagem" --mode grounded-local --allow-api --json`.
It retains E5-small, builds separate source-backed product text, retrieves dense
and lexical candidates, and applies a small compatibility scorer. Query-only
semantic interpretation uses the budgeted gateway and persists validated plans;
cached queries work offline without `--allow-api`. An uncached query fails clearly
when API calls are disabled, rather than silently using a different parser.
Historical customization records are kept separately and establish neither base
ingredients nor current availability. The first construction encodes the catalog;
later constructions validate and reuse that matrix. This mode does not change
the incumbent or current-release pointer. Detailed results and limitations are
recorded in [the completed semantic evaluation](review/GROUNDED-SEMANTIC.md).

`--mode grounded-facets` selects an additional experimental source-facet matcher
over the same E5-small candidate pool. It preserves multiword identity modifiers,
exact source evidence, component scope, unknowns and conflicts; compatibility
tiers precede retrieval similarity. It still uses the existing query interpreter:
cached plans work offline, while unseen queries require `--allow-api`. It does
not use the remote reranker. The first complete 30-query development comparison
was weaker than grounded-local (nDCG .674 versus .731 on the same fully judged
522-pair pool), so neither the incumbent nor the default was changed. See
[the initial measured experiment](review/evidence/facets-v1.json).

```powershell
python scripts/score_grounded.py --facets
python scripts/complete_grounded.py --folder work/grounded-facets
```

**Working now:** offline multilingual E5, BM25 and their RRF combination;
100 queries × five results × three local runs. The independent review implements
a hosted-small comparison option and runs blinded evaluation through the assignment gateway;
the full hosted embedding experiment was blocked by access denial.
See the [review report](review/REPORT.md) for measured quality, coverage, spending
and limitations. Semantic-local remains the incumbent until frozen development
selection justifies a change; passing tests is not evidence of search quality.

**Historical final review:** operational prototype, quality targets unmet. Frozen final
E5 estimates are nDCG@5 **0.695**, Success@5 **47.1%**, MRR@5 **0.381**, and mandatory
contradiction exposure **30.3%**. All 755 pooled pairs are judged after explicit
adjudication, but repeat checks remain incomplete and no human calibration was
available. No method earned promotion on development. The 180-test suite, fresh
inference and exact 1,500-row replay pass. Conservative API spending is
**$0.622962**, including uncertain calls; the existing $4.50 ceiling is unchanged.

**Current development cycle:** the frozen v5 interpretation evaluation is complete:
527 reading/item judgments plus40 independent repeats, with no retained ranking
improvement. Incumbent primary-reading nDCG is **.590**, Success **7/30**;
these changed judgments are not a change in rankings or comparable gains over
the historical protocol. [Current decision](review/INTENT-DECISION-v5.md) and
[every-query diagnosis](review/V5-QUERY-DIAGNOSIS.md) show limitations and alternate
readings. New field/fusion experiments remain separate until fully evaluated.
Cumulative project API accounting is **$0.691229**, including uncertainty.
The original final set is already used; new experiments do not treat it as untouched.

Loaded snapshots now isolate metadata, vectors, configuration and model identity.
The repaired service passes417 tests and the fresh-inference audit. Its latest
60-second, four-client genuine-miss test measures **62.76QPS**, **109.20ms p95**,
zero errors and0.834GiB loaded memory; these are local library measurements.
See [pipeline map](review/PIPELINE-AUDIT-v6.md), [measured runtime](review/PERFORMANCE-v6.md)
and [candidate/abstention limits](review/CANDIDATES-AND-ABSTENTION-v6.md).
[Acceptance criteria](review/ACCEPTANCE-v2.md), [experiment decisions](review/EXPERIMENTS-v2.md),
[current progress](review/PROGRESS.md), [production path](review/PRODUCTION-PATH-v2.md)
and [cost boundaries](review/COST-v2.md) distinguish demonstrated results from
remaining work. Ranking relevance targets remain unmet.

[Ranked results](review/evidence/selected-top5.csv) · [Independent review](review/REPORT.md) ·
[Historical five-slide presentation](review/presentation/independent-review.pptx) ·
[Evaluation protocol](review/EVALUATION.md)

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
records the tested Windows/Python 3.12 environment. The repository includes an
Ubuntu CI workflow; this review ran locally on Windows.
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
creating a fresh service for every request repeats loading. A service admits at
most four simultaneous calls by default and raises `SearchBusyError` when full.
Callers should use bounded backoff or an explicit overload response. One loaded
service completed 5,418 genuine query-cache misses, including SQLite persistence,
in 60.03 seconds: **90.26 QPS, 92.72 ms p95, 141.55 ms p99**, with zero errors.
Loaded RSS was 0.829 GiB and peak RSS 1.181 GiB. A separate mixed-load run offered
102 requests/second and rejected 1.60%; saturation and rejected work must be
reported with accepted-request latency. These host-local measurements use short
development queries plus unique suffixes and do not establish production API
capacity. See [performance protocol and evidence](review/PERFORMANCE-v2.md).

The library also accepts a validated, expiring eligibility snapshot and filters
before candidate selection. Empty or small eligible sets return fewer than `k`
results. A marketplace backend must supply authoritative stock/geography/merchant
eligibility and authenticate its context; these feeds are proposed production
work. See [the demonstrated eligibility contract](review/ELIGIBILITY-v2.md).

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
evaluation protocol. Lexical search uses the same immutable current snapshot;
before the first build it can ingest raw CSV. Changes require a rebuilt release.
`--release` pins interactive search to a snapshot, including lexical mode; pair it
with an explicit `--mode` when pinning a different model family from the current default.

## Hosted experiments and evaluation

Paste your API key after `OPENAI_API_KEY=` in the repository-root `.env` file
(copy `.env.example` if needed), or set `OPENAI_API_KEY` in the environment.
Configuration loading reads only that key from `.env` beside `config.toml`;
an existing environment variable takes precedence. `.env` is ignored by Git,
and its contents never enter saved configuration or manifests. Never put keys
in tracked files, logs, saved terminal commands or chat. Placeholder keys are
rejected. Cached replay is the default; paid requests require `--allow-api`.

The supplied key belongs to **`https://lcm-litellm.prosus.com/`**, configured in
`[api].base_url`. Gateway model names include `openai/`. Do not send that key to
the direct OpenAI endpoint. The review verified root embedding calls and current
judge inference; `/v1/models` lists models but can include deprecated entries.
The originally suggested nano model rejected inference as deprecated. The
versioned evaluator uses `openai/gpt-6-luna` and independent `openai/gpt-6-sol`
checks with the Responses API. See the protocol for conservative prices and
pilot corrections. Explicit endpoint/model/cache hashes prevent accidental reuse
across providers.

```powershell
$hosted = python -m prosus_search build --backend comparison --allow-api | ConvertFrom-Json
$comparison = python -m prosus_search benchmark --build $hosted.build_manifest | ConvertFrom-Json
$comparisonBenchmark = Join-Path $comparison.output_directory 'benchmark.json'
python -m prosus_search evaluate --benchmark $comparisonBenchmark --pilot-only --allow-api
python -m prosus_search evaluate --benchmark $comparisonBenchmark --judge-workers 4 --allow-api
```

The optional comparison includes BM25, E5, hosted-small and each model's hybrid,
using common evidence. **The review's hosted-small catalog build was blocked by
gateway access denial after successful small probes; its partial index was not
published or evaluated.** The judged review therefore compares the three complete
local runs. `--backend local` needs no API; `--backend hosted` retains the separate
small/large experiment. An incomplete build never changes the serving pointer.
Hosted batches use 32 inputs; live failures remain charged conservatively.
Fixtures never become dataset labels.

Evaluation pools top-five candidates, hides system/rank/score, creates query-only
intent cards and requests pointwise Portuguese source-backed grades. Validation
checks exact quotes/scalars, requirement IDs and grade caps. Missing evidence
differs from contradiction. Cached outputs and distinct replicate IDs support
bounded repeat, equivalent-prompt, grader and independent intent-card checks.
One retry per primary assessment is budgeted. Exhausted invalid primary outputs
receive a separately recorded independent-model adjudication, with at most two
attempts. Originals remain intact, primary validity is reported separately, and
unresolved failures remain missing. Diagnostic arms are never adjudicated.

The frozen split has **30 development / 70 final queries**, grouped by intent.
Final judging requires an explicit end-of-iteration decision and frozen selection.
The CLI gate protects the holdout; it is procedural, not a security boundary or
a renewed approval request. The review records whether final judging was released.
Human calibration has not been performed.

With valid labels, reports mark nDCG@5, Success@5, MRR@5 and constraint rates as
**LLM-judged**. Incomplete pools use the common complete-query subset and keep
full-split metrics null. Zero ideal gain gives undefined nDCG with its denominator;
those queries remain in Success/MRR. No zero imputation, full-catalog recall claim or promotion
from incomplete evidence. Bootstrap intervals condition on automated labels and
do not measure judge accuracy. Frozen selection rules retain simplicity when
differences are inconclusive; BM25 is eligible to win. See the
[review protocol](review/EVALUATION.md).

The persistent ledger reserves before requests and includes pending/uncertain
charges. It stops at **$4.50**, leaving $0.50 untouched: $0.50 embeddings, $1.40
development, $0.85 final, $1 Round 4 and $0.75 recovery. SDK retries are disabled;
application retries reserve again. Up to four independent judge calls can overlap;
per-key locks prevent duplicate submissions and SQLite reservations enforce the
same shared ceiling. Never delete the ledger to bypass limits.
The ledger cannot observe unrelated account spending. Verify tariffs and remaining
credit before paid execution. The user confirmed the key was unused with $5
available and standard rates before the review's paid calls; the provider's budget
route itself is blocked. Reported totals include failed/uncertain requests.

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

The independent review, protocol amendments, measured figures and durable
[progress record](review/PROGRESS.md) live outside `docs/`, as requested. The
latest user instruction authorizes implementation and iteration without the
earlier round gates. Nothing was published or pushed. The assignment asks for
the GitHub link at least **one day before the interview**; its date was not provided.
