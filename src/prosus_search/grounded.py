"""Grounded product search with cached, budgeted semantic query understanding.

Historical options never establish ingredients or current availability.
"""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from pathlib import Path
import os
import re
import time
import uuid

import numpy as np
import snowballstemmer

from .artifacts import file_hash, load_current, load_release
from .benchmark import local_client, open_store, read_json, write_json
from .catalog import load_catalog
from .eligibility import EligibilitySnapshot
from .evaluation import canonical_hash
from .representation import lexical_tokens, normalize_text, portion_claims
from .retrieval import LexicalIndex, normalize_vectors, stable_rank
from .snapshots import freeze_json

VERSION = 'grounded-semantic-v2'
STOP = frozenset('a o as os de da do das dos em na no nas nos e para por um uma uns umas ao aos que seu sua'.split())
GENERIC = frozenset('outro outros geral gerais alimento alimentos preparado preparados prato pratos categoria categoriais'.split())
PREPARATIONS = ('frito', 'assado', 'grelhado', 'empanado', 'crocante', 'cozido', 'defumado')
POLICY = {'candidate_depth': 100, 'rrf_constant': 60, 'type_match': 1.0,
          'type_mismatch_signal': -.75, 'requirement_match': .30,
          'explicit_contradiction': -.80, 'preference_match': .10, 'domain_match': .75}


@lru_cache(maxsize=50000)
def stem(word):
    # Separate mutable Snowball instances avoid cross-request state sharing.
    return snowballstemmer.stemmer('portuguese').stemWord(word)


def keys(text):
    return frozenset(stem(word) for word in lexical_tokens(text) if word not in STOP)


GENERIC_KEYS = keys(' '.join(GENERIC))


def product(record):
    m = record.metadata
    taxonomy = m.get('taxonomy', {})
    type_sources = [(f'/taxonomy/{level}', taxonomy.get(level, '').replace('_', ' ').lower())
                    for level in ('l2', 'l1')]
    type_sources.append(('/category_name', m.get('category_name', '')))
    selected = next(((path, value) for path, value in type_sources if keys(value) - GENERIC_KEYS), ('/name', m['name']))
    description = normalize_text(m.get('description', ''))
    text = f"Produto: {normalize_text(m['name'])}\nTipo: {normalize_text(selected[1])}"
    if m.get('category_name'):
        text += '\nCategoria: ' + normalize_text(m['category_name'])
    if description:
        text += '\nDescrição: ' + description
    portions = sorted({claim.people for claim in portion_claims(record)})
    if portions:
        text += '\nPorções declaradas separadamente: ' + '; '.join(f'serve {n} pessoas' for n in portions)
    evidence = [{'path': '/name', 'quote': m['name']},
                {'path': selected[0], 'quote': m['name'] if selected[0] == '/name' else
                 (taxonomy[selected[0].split('/')[-1]] if selected[0].startswith('/taxonomy/') else m['category_name'])}]
    if description:
        evidence.append({'path': '/description', 'quote': m['description']})
    evidence.extend({'path': '/' + re.sub(r'\[(\d+)\]', r'/\1', claim.field_path).replace('.', '/'), 'quote': claim.raw_value}
                    for claim in portion_claims(record))
    return freeze_json({'item_id': record.item_id, 'text': text, 'identity': m['name'] + ' / ' + selected[1],
        'domain': ('prepared_food' if taxonomy.get('l0') == 'ALIMENTOS_PREPARADOS' else
                   'grocery' if taxonomy.get('l0') in {'BEBIDAS','MERCEARIA','FLV','BOMBONIERE','FRIOS_LATICINIOS','PROTEINAS'} else
                   'unknown' if taxonomy.get('l0') in {None,'','OUTROS'} else 'retail'),
        'type_keys': sorted(set().union(*(keys(value) for _, value in type_sources)) - GENERIC_KEYS),
        'name_keys': sorted(keys(m['name'])), 'base_keys': sorted(keys(m['name'] + ' ' + description)),
        'explicit_text': m['name'] + '\n' + m.get('description', ''),
        'portion_claims': portions, 'portion_conflict': len(portions) > 1, 'evidence': evidence,
        'unknowns': ['ingredient list completeness', 'current option availability', 'dietary certification'],
        'historical_options': [{'source_path': f'/itemProfile/metrics/customizations/{i}', 'record': dict(option)}
                               for i, option in enumerate(record.historical_options)],
        'historical_options_are_base_ingredients': False})


def interpret(query, vocabulary):
    original = normalize_text(query)
    if not original:
        raise ValueError('Query must not be blank')
    head = re.split(r'\s+(?:com|sem|para|estilo)\s+', original, maxsplit=1, flags=re.IGNORECASE)[0]
    # A category can contain modifiers such as "rápido" or "crocante".
    # Do not promote every matching word to product identity. Anchor the first
    # content word; an unknown container phrase can expose its "de Y" content.
    words = [w for w in lexical_tokens(head) if w not in STOP and stem(w) not in GENERIC_KEYS]
    anchor = words[0] if words else ''
    identity_keys = keys(anchor)
    if anchor.endswith('s') and len(anchor) > 3 and not identity_keys & vocabulary:
        singular = keys(anchor[:-1])
        if singular & vocabulary:
            identity_keys = singular
    if not identity_keys & vocabulary:
        content = re.search(r'\bde\s+(\w+)', head, re.IGNORECASE)
        if content and keys(content.group(1)) & vocabulary:
            identity_keys = keys(content.group(1))
    identity_keys &= vocabulary
    requirements, preferences = [], []
    for match in re.finditer(r'\b(com|sem|estilo)\s+(.+?)(?=\s+(?:com|sem|para|estilo)\s+|[,;.]|$)', original, re.IGNORECASE):
        relation, value = match.group(1).lower(), match.group(2).strip()
        requirements.append({'relation': relation, 'value': value, 'source_span': match.group(0)})
    for word in PREPARATIONS:
        matching = [w for w in original.split() if lexical_tokens(w) and stem(lexical_tokens(w)[0]) == stem(word)]
        if matching:
            requirements.append({'relation': 'attribute', 'value': matching[0], 'source_span': matching[0]})
    portion = re.search(r'\b(\d+)\s+pessoas?\b', original, re.IGNORECASE)
    if portion:
        requirements.append({'relation': 'portion', 'value': portion.group(1), 'source_span': portion.group(0)})
    occasion = re.search(r'\bpara\s+(.+)$', original, re.IGNORECASE)
    if occasion and not portion:
        preferences.append({'value': occasion.group(1), 'source_span': occasion.group(0)})
    canonical = head + ''.join(' ' + req['source_span'] for req in requirements
                              if req['source_span'].casefold() not in head.casefold())
    return {'original': original, 'identity_phrase': head, 'identity_keys': sorted(identity_keys),
            'requirements': requirements, 'preferences': preferences, 'canonical': canonical,
            'ambiguity': ['No catalog type identified'] if not identity_keys else [],
            'interpreter': 'catalog-derived lexical parser; no LLM sense resolution'}


def compatibility(plan, facts):
    wanted = set(plan['identity_keys'])
    identity = set(facts['type_keys']) | set(facts['name_keys'])
    type_match = len(wanted & identity) / len(wanted) if wanted else None
    statuses = []
    text = normalize_text(facts['explicit_text']).casefold()
    for req in plan['requirements']:
        value, relation = req['value'], req['relation']
        if relation == 'portion':
            claims = facts['portion_claims']
            status = ('unknown' if not claims or len(claims) > 1 else
                      'supported' if int(value) in claims else 'contradicted')
        else:
            terms = keys(value)
            present = bool(terms) and terms <= set(facts['base_keys'])
            # Only literal explicit exclusions, not taxonomy or dietary flags,
            # establish a negative relation. Complex scope remains a limitation.
            excluded = 'sem ' + normalize_text(value).casefold() in text
            status = ('supported' if excluded else 'contradicted' if present else 'unknown') if relation == 'sem' else (
                'contradicted' if excluded else 'supported' if present else 'unknown')
        statuses.append({**req, 'status': status})
    supported = sum(r['status'] == 'supported' for r in statuses)
    contradictions = sum(r['status'] == 'contradicted' for r in statuses)
    preference = sum(bool(keys(p['value'])) and keys(p['value']) <= set(facts['base_keys']) for p in plan['preferences'])
    adjustment = (POLICY['type_match'] * type_match if type_match is not None else 0.)
    if wanted and type_match == 0 and facts['type_keys']:
        adjustment += POLICY['type_mismatch_signal']
    adjustment += POLICY['requirement_match'] * supported + POLICY['explicit_contradiction'] * contradictions
    adjustment += POLICY['preference_match'] * preference
    domain = plan.get('domain','unknown')
    domain_match = None if 'unknown' in (domain,facts['domain']) else domain == facts['domain']
    if domain_match is not None:
        adjustment += POLICY['domain_match'] * (1 if domain_match else -1)
    return adjustment, {'type_lexical_coverage': type_match, 'requirements': statuses,
                        'domain_match':domain_match,
                        'unknown_requirements': sum(r['status'] == 'unknown' for r in statuses),
                        'historical_options_used_as_evidence': False}


class GroundedSearch:
    def __init__(self, config, baseline_release_id=None, *, allow_api=False, rerank=False):
        self.config = config
        baseline_release_id = baseline_release_id or load_current(config.resolve('artifacts'))['models']['semantic-local']
        anchor = load_release(config.resolve('artifacts'), baseline_release_id,
                              config['local']['model'], config['local']['dimensions'], config['local']['revision'])
        catalog = load_catalog(config.resolve('catalog'))
        if catalog.source_hash != anchor.manifest['source_hash'] or tuple(r.item_id for r in catalog.records) != anchor.ids:
            raise ValueError('Grounded inputs differ from the pinned incumbent catalog')
        self.records = tuple(replace(r, metadata=freeze_json(r.metadata), historical_options=freeze_json(r.historical_options)) for r in catalog.records)
        self.ids = anchor.ids
        self.facts = tuple(product(record) for record in self.records)
        self.vocabulary = frozenset().union(*(set(f['type_keys']) | set(f['name_keys']) for f in self.facts))
        from .understanding import SemanticInterpreter
        self.interpreter = SemanticInterpreter(config,allow_api=allow_api)
        self.reranker = None
        if rerank:
            from .reranking import EvidenceReranker
            self.reranker = EvidenceReranker(config,allow_api=allow_api)
        self.mode = 'grounded-rerank' if rerank else 'grounded-local'
        self.client = local_client(config, open_store(config))
        texts = [f['text'] for f in self.facts]
        self.identity = {'version': VERSION, 'source_hash': catalog.source_hash, 'anchor': baseline_release_id,
                         'representation_hash': canonical_hash(texts), 'policy': POLICY,
                         'producer_sha256': file_hash(__file__), 'model_fingerprint': self.client.manifest['model_fingerprint']}
        self.identity['query_interpreter_fingerprint']=self.interpreter.fingerprint
        if self.reranker is not None:
            self.identity['reranker_fingerprint']=self.reranker.fingerprint
        self.release_id = canonical_hash(self.identity)
        matrix_identity = {'representation_hash': self.identity['representation_hash'],
                           'model_fingerprint': self.identity['model_fingerprint']}
        root = config.resolve('artifacts') / 'grounded-vectors' / canonical_hash(matrix_identity)
        root.mkdir(parents=True, exist_ok=True)
        matrix_path, receipt_path = root / 'vectors.npy', root / 'manifest.json'
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            if receipt['identity'] != matrix_identity or file_hash(matrix_path) != receipt['vector_sha256']:
                raise ValueError('Grounded representation cache differs')
            self.vectors = np.load(matrix_path, allow_pickle=False)
        else:
            self.vectors = self.client.embed(texts, representation_version=VERSION, kind='passage', bypass_cache=True)
            temporary = root / (uuid.uuid4().hex + '.npy')
            np.save(temporary, self.vectors, allow_pickle=False)
            os.replace(temporary, matrix_path)
            write_json(receipt_path, {'identity': matrix_identity, 'vector_sha256': file_hash(matrix_path)})
        if self.vectors.shape != (len(self.ids), 384) or not np.isfinite(self.vectors).all() or not np.allclose(np.linalg.norm(self.vectors, axis=1), 1., atol=1e-5):
            raise ValueError('Invalid grounded vector alignment/normalization')
        self.vectors.flags.writeable = False
        self.lexical = LexicalIndex([lexical_tokens(r.metadata['name'] + ' ' + f['identity'] + ' ' + r.metadata.get('description', '')) for r, f in zip(self.records, self.facts)])

    def search(self, query, k=5, *, bypass_query_cache=False, eligibility=None):
        start = time.perf_counter()
        if type(k) is not int or not 1 <= k <= min(100, len(self.ids)):
            raise ValueError('Grounded k must be between 1 and min(100, catalog size)')
        if self.reranker is not None and k>5:
            raise ValueError('Source-aware reranking supports up to five returned items')
        if eligibility is not None and not isinstance(eligibility, EligibilitySnapshot):
            raise ValueError('Provide an EligibilitySnapshot')
        allowed = eligibility.validate_for(self.release_id, self.ids) if eligibility is not None else None
        eligible = np.asarray([i for i, item_id in enumerate(self.ids) if allowed is None or item_id in allowed], dtype=np.int64)
        if not len(eligible):
            response = {'query': query, 'mode': self.mode, 'release_id': self.release_id, 'results': [], 'scores_are_probabilities': False}
            if eligibility is not None:
                eligibility.assert_fresh()
                response['eligibility'] = {'revision': eligibility.revision, 'context_id': eligibility.context_id,
                    'valid_until': eligibility.valid_until, 'eligible_count': 0}
            return response
        raw_plan = self.interpreter.interpret(query)
        identity = interpret(raw_plan['identity'], self.vocabulary)
        plan = {'original':query, 'identity_phrase':raw_plan['identity'], 'identity_keys':identity['identity_keys'],
                **{key:raw_plan[key] for key in ('canonical','domain','requirements','preferences','ambiguity','alternatives')},
                'interpreter':'query-only LLM; cached budgeted structured output; no evaluation inputs'}
        vectors = self.client.embed([plan['original'], plan['canonical']], representation_version=VERSION + '-query', kind='query', bypass_cache=bypass_query_cache)
        dense = self.vectors @ normalize_vectors(vectors).T
        lexical = self.lexical.score(plan['original'])
        retrieval = np.zeros(len(self.ids), dtype=np.float64)
        for values, lexical_branch in [(dense[:, 0], False), (dense[:, 1], False), (lexical, True)]:
            order = stable_rank([self.ids[i] for i in eligible], values[eligible], min(100, len(eligible)))
            rows = eligible[order]
            if lexical_branch:
                rows = rows[values[rows] > 0]
            retrieval[rows] += 60. / (60 + np.arange(1, len(rows) + 1))
        candidates = np.flatnonzero(retrieval > 0)
        values, explanations = [], {}
        for i in candidates:
            adjustment, explanations[i] = compatibility(plan, self.facts[i])
            values.append(retrieval[i] / 3. + adjustment)
        order = stable_rank([self.ids[i] for i in candidates], np.asarray(values), min(10 if self.reranker is not None else k, len(candidates)))
        results = []
        for rank, offset in enumerate(order, 1):
            i = candidates[offset]
            results.append({'item_id': self.ids[i], 'merchant_id': self.records[i].merchant_id, 'rank': rank,
                'score': float(values[offset]), 'score_kind': VERSION, 'name': self.records[i].metadata['name'],
                'description': self.records[i].metadata.get('description', ''), 'explanation': explanations[i]})
        reranking = None
        if self.reranker is not None:
            results,reranking=self.reranker.rank(query,plan,results,{r.item_id:r.metadata for r in self.records},k)
        response = {'query': query, 'mode': self.mode, 'release_id': self.release_id, 'results': results,
                    'interpretation': plan, 'scores_are_probabilities': False,
                    'timing_ms': {'total': (time.perf_counter() - start) * 1000}}
        if reranking is not None:
            response['reranking']=reranking
        if eligibility is not None:
            eligibility.assert_fresh()
            response['eligibility'] = {'revision': eligibility.revision, 'context_id': eligibility.context_id,
                'valid_until': eligibility.valid_until, 'eligible_count': len(eligible)}
        return response
