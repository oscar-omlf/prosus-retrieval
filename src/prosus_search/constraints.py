"""Source-backed ingredient constraints; unknown facts never disqualify offers.

This deliberately bounded vocabulary is not a recipe or allergy knowledge base.
Only ingredient presence/absence with applicable component scope can filter.
"""
from __future__ import annotations

import re
from functools import lru_cache

from .facets import fold, identity, lineage, PARENTS, product_facets, requirement
from .grounded import keys
from .snapshots import freeze_json

VERSION = 'constraint-hybrid-v1'
ALIASES = {
    'queijo': ('queijo', 'queijos'), 'cheddar': ('cheddar',),
    'mussarela': ('mussarela', 'mucarela', 'muçarela', 'mozzarella'),
    'parmesao': ('parmesao',), 'provolone': ('provolone',),
    'gorgonzola': ('gorgonzola',), 'requeijao': ('requeijao',),
    'cream_cheese': ('cream cheese',), 'catupiry': ('catupiry',),
    'bacon': ('bacon',), 'cebola': ('cebola', 'cebolas'),
    'tomate': ('tomate', 'tomates'), 'leite': ('leite',),
    'gluten': ('gluten',), 'lactose': ('lactose',),
    'ovo': ('ovo', 'ovos'), 'pimenta': ('pimenta', 'pimentas'),
}
PARENT = {v: 'queijo' for v in (
    'cheddar', 'mussarela', 'parmesao', 'provolone', 'gorgonzola',
    'requeijao', 'cream_cheese', 'catupiry')}
LOOKUP = {fold(alias): value for value, aliases in ALIASES.items() for alias in aliases}
LOOKUP.update({'queijo ' + alias: value for alias, value in tuple(LOOKUP.items())
               if value in PARENT and value != 'cream_cheese'})
INGREDIENT = re.compile(r'(?<!\w)(' + '|'.join(
    sorted(map(re.escape, LOOKUP), key=len, reverse=True)) + r')(?!\w)')
OPTION = re.compile(r'\b(?:opciona\w*|adiciona\w*|acrescimo\w*|pode adicionar|a sua escolha|escolha|escolher)\b')
SIDE = re.compile(r'\b(?:acompanh\w*|guarnic\w*|servid[oa] com)\b')
RELATION = re.compile(r'(?<!\w)(sem|nao contem|livre de|com|contem|zero|s/|c/)(?!\w)')
POLICY = {'identity_supported': .70, 'identity_mismatch': -.60,
          'attribute_support': .25, 'attribute_contradiction': -.40,
          'name_coverage': .15, 'domain_mismatch': -.45}


class ConstraintInterpreter:
    """Full-consumption local ingredient grammar; explicit semantic route otherwise."""
    def __init__(self, semantic):
        from .artifacts import file_hash
        from .evaluation import canonical_hash
        self.semantic = semantic
        self.fingerprint = canonical_hash({'code': file_hash(__file__), 'semantic': semantic.fingerprint})

    def interpret(self, query):
        from .understanding import validate
        clauses = list(re.finditer(r'\s+(com|sem)\s+(.+?)(?=\s+(?:com|sem)\s+|$)', query, re.I))
        head = query[:clauses[0].start()] if clauses else query
        # No guess about occasions, brands, measurements or complex syntax.
        simple_head = fold(head) in {'hamburguer', 'pizza', 'sanduiche', 'wrap', 'sopa', 'salada', 'burrito'}
        end = len(head)
        requirements = []
        for clause in clauses:
            if clause.start() != end or not wanted_ingredients(clause.group(2))[1]:
                break
            requirements.append({'relation': clause.group(1).lower(), 'value': clause.group(2),
                                 'source_span': clause.group().strip()})
            end = clause.end()
        if simple_head and end == len(query):
            value = {'identity': head, 'canonical': query, 'domain': 'unknown',
                'requirements': requirements, 'preferences': [], 'ambiguity': [], 'alternatives': []}
            if fold(head) == 'hamburguer':
                value['ambiguity'] = ['Sanduíche pronto ou hambúrguer para preparar: consulta não especifica.']
                value['alternatives'] = ['hambúrguer pronto', 'hambúrguer para preparar']
            return {**validate(value, query), '_interpreter': 'constraint local grammar; full consumption; no API'}
        return {**self.semantic.interpret(query), '_interpreter': 'cached/budgeted semantic interpretation; no silent fallback'}


@lru_cache(maxsize=16384)
def ingredient_claims(text, path):
    # fold retains the character offsets for ordinary Portuguese catalog text.
    # Exact original source slices are stored and verified at extraction time.
    normalized = fold(text)
    claims = []
    for sentence in re.finditer(r'[^.!?;\n]+', normalized):
        value = sentence.group()
        boundary = normalized.rfind('\n\n', 0, sentence.start())
        paragraph_start = boundary + 2 if boundary >= 0 else 0
        prefix = normalized[paragraph_start:sentence.start()]
        section_option = re.search(r'(?:^|\n)\s*(?:adicionais|opcionais|opcoes)[^.!?\n]{0,60}\n', prefix)
        section_side = re.search(r'(?:^|\n)\s*(?:acompanh\w*|guarnic\w*)\s*:\s*\n', prefix)
        for match in INGREDIENT.finditer(value):
            # Optionality belongs to a list/phrase, not everything after the
            # first option marker in the entire description.
            left = max(value.rfind(',', 0, match.start()), value.rfind('|', 0, match.start())) + 1
            right = min([p for p in (value.find(',', match.end()), value.find('|', match.end())) if p >= 0] or [len(value)])
            phrase = value[left:right]
            options = OPTION.search(phrase)
            heading = OPTION.search(value[:match.start()])
            optional_list = heading and ':' in value[heading.end():match.start()]
            side = SIDE.search(value)
            possible = re.search(r'\bpode conter\b', value[:match.start()])
            scope = ('optional' if options or optional_list or section_option else 'possible' if possible else
                     'included_side' if section_side or side and match.start() >= side.start() else 'main')
            relations = list(RELATION.finditer(value[:match.start()]))
            last = relations[-1].group() if relations else None
            # A contrast ends a preceding negation; comma-separated ingredient
            # lists and conjunctions preserve it until an explicit new relation.
            tail = value[relations[-1].end():match.start()] if relations else ''
            absent = last in {'sem', 'nao contem', 'livre de', 'zero', 's/'} and not re.search(r'\b(?:mas|porem)\b', tail)
            absent |= bool(re.match(r'\s+(?:zero|free)\b', value[match.end():]))
            start, end = sentence.start() + match.start(), sentence.start() + match.end()
            quote = text[start:end]
            if fold(quote) != match.group():
                continue  # normalization mismatch must not invent provenance
            claims.append(freeze_json({'value': LOOKUP[match.group()],
                'polarity': 'absent' if absent else 'present', 'scope': scope,
                'path': path, 'quote': quote, 'start': start, 'end': end}))
    return tuple(claims)


def product(record):
    m = record.metadata
    return freeze_json({'ingredients': ingredient_claims(m['name'], '/name') +
        ingredient_claims(m.get('description', ''), '/description'),
        'facets': product_facets(record), 'historical_options_used': False})


@lru_cache(maxsize=4096)
def wanted_ingredients(value):
    claims = ingredient_claims(value, '/query')
    covered = set().union(*(keys(c['quote']) for c in claims)) if claims else set()
    # A partial vocabulary match cannot silently consume an unknown qualifier.
    complete = bool(claims) and not (keys(value) - covered - keys('e de do da'))
    return tuple(dict.fromkeys(c['value'] for c in claims)), complete


def ingredient_status(req, facts):
    wanted, complete = wanted_ingredients(req['value'])
    states, evidence = [], []
    violates = False
    for value in wanted:
        applicable = [c for c in facts['ingredients'] if c['scope'] in {'main', 'included_side'} and
            (req['relation'] == 'sem' or c['scope'] == 'main')]
        # Present subtype -> parent; absent parent -> absent subtype. Never
        # infer absence of an entire family from absence of just one variety.
        positive = [c for c in applicable if c['polarity'] == 'present' and value in lineage(c['value'], PARENT)]
        negative = [c for c in applicable if c['polarity'] == 'absent' and c['value'] in lineage(value, PARENT)]
        contradiction = bool(positive) if req['relation'] == 'sem' else bool(negative)
        support = bool(negative) if req['relation'] == 'sem' else bool(positive)
        states.append('conflict' if positive and negative else 'contradicted' if contradiction else 'supported' if support else 'unknown')
        # Conflicting explicit default inclusion is not equivalent to unspecified.
        violates |= contradiction and complete
        evidence.extend(positive + negative)
    if not complete:
        states.append('unknown')
    status = next((s for s in ('contradicted', 'conflict', 'unknown') if s in states), 'supported')
    return {'status': status, 'evidence': evidence, 'violates': violates,
            'complete_ingredient_parse': complete}


def constraints(plan, facts):
    statuses = []
    for req in plan['requirements']:
        values, _ = wanted_ingredients(req['value'])
        if req['relation'] in {'sem', 'com'} and values:
            statuses.append({**req, **ingredient_status(req, facts)})
    return statuses


def positive_query(plan):
    # Excluded ingredient names must not receive ordinary positive BM25 credit.
    return ' '.join([plan['identity_phrase']] + [r['value'] for r in plan['requirements'] if r['relation'] != 'sem'] +
                    [p['value'] for p in plan['preferences']])


def compatibility(plan, facts, source, statuses):
    wanted = identity(plan)
    offered = set(facts['facets']['types'])
    supported = bool(wanted['types'] and any(q in lineage(p, PARENTS) for q in wanted['types'] for p in offered))
    identity_status = 'supported' if supported else 'mismatch_signal' if wanted['types'] and offered else 'unknown'
    score = POLICY['identity_supported'] if supported else POLICY['identity_mismatch'] if offered and wanted['types'] else 0.
    statuses = list(statuses)
    identity_attributes = [{'relation': 'attribute', 'value': c['quote'], 'source_span': c['quote']}
        for c in wanted['claims'] if c['facet'] not in {'type', 'ingredient'}]
    for req in list(plan['requirements']) + identity_attributes:
        if any(s['source_span'] == req['source_span'] and s['relation'] == req['relation'] for s in statuses):
            continue
        state, evidence = requirement(req, facts['facets'], source)
        # Cuisine labels may overlap; a different cuisine alone is not proof of
        # impossibility. These weak noningredient features never hard-filter.
        if state == 'contradicted' and req['relation'] == 'estilo':
            state = 'unknown'
        statuses.append({**req, 'status': state, 'evidence': evidence, 'violates': False})
    n = max(1, len(statuses))
    score += POLICY['attribute_support'] * sum(s['status'] == 'supported' for s in statuses) / n
    score += POLICY['attribute_contradiction'] * sum(s['status'] == 'contradicted' for s in statuses) / n
    query_keys = set(wanted['keys'])
    score += POLICY['name_coverage'] * len(query_keys & set(source['name_keys'])) / max(1, len(query_keys))
    domain_mismatch = (plan.get('domain') == 'prepared_food' and source['domain'] == 'retail') or (
        plan.get('domain') == 'retail' and source['domain'] in {'prepared_food', 'grocery'})
    if domain_mismatch:
        score += POLICY['domain_mismatch']
    return score, {'identity_status': identity_status, 'requirements': statuses,
        'unknown_requirements': sum(s['status'] in {'unknown', 'conflict'} for s in statuses),
        'domain_mismatch_signal': domain_mismatch, 'historical_options_used_as_evidence': False,
        'ingredient_list_completeness': 'unknown', 'scores_are_probabilities': False}
