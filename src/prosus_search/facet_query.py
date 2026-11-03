"""Exact local grammar for simple requests; explicit semantic route otherwise.

Only a fully consumed supported grammar uses the local path. Complex cuisine,
occasion, brand and ambiguous queries retain the versioned semantic interpreter.
"""
from __future__ import annotations

import re

from .artifacts import file_hash
from .evaluation import canonical_hash
from .understanding import validate
from .facets import fold

TYPE = r'(?:hamb[uú]rguer|pizza|sandu[ií]che|wrap|sopa|salada|burrito)'
CLAUSE = re.compile(r'\s+(com|sem)\s+([\wÀ-ÿ]+(?:\s+e\s+[\wÀ-ÿ]+)*)', re.I)
INGREDIENTS = frozenset(('queijo cheddar mussarela bacon cebola tomate leite ovo ovos '
                         'catupiry requeijao pimenta gluten lactose').split())


class FacetQueryInterpreter:
    def __init__(self, semantic):
        self.semantic=semantic
        self.fingerprint=canonical_hash({'local_grammar':file_hash(__file__),'semantic':semantic.fingerprint})

    def interpret(self, query):
        head=re.match(r'^('+TYPE+r')(?=\s|$)',query,re.I)
        requirements=[]
        end=head.end() if head else 0
        if head:
            for match in CLAUSE.finditer(query,end):
                if match.start()!=end or any(fold(w) not in INGREDIENTS for w in re.split(r'\s+e\s+',match.group(2),flags=re.I)):
                    break
                requirements.append({'relation':match.group(1).casefold(),'value':match.group(2),
                                     'source_span':match.group(0).strip()})
                end=match.end()
        if head and end==len(query):
            value={'identity':head.group(1),'canonical':query,'domain':'prepared_food',
                   'requirements':requirements,'preferences':[],'ambiguity':[],'alternatives':[]}
            return {**validate(value,query),'_interpreter':'local exact grammar; full query consumed; no API'}
        return {**self.semantic.interpret(query),'_interpreter':'query-only semantic model; cached budgeted structured output'}
