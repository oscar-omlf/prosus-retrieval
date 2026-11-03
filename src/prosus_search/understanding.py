"""Cached, query-only semantic interpretation through the existing budgeted client."""
from __future__ import annotations

import json
import re
from copy import deepcopy
from decimal import Decimal, InvalidOperation

from .runtime import open_store
from .config import token_counter
from .model_client import ModelClient, ModelConfig, ModelValidationError, _payload
from .runtime import canonical_hash
from .artifacts import file_hash

INSTRUCTIONS = '''Interprete uma consulta em português para busca em marketplace misto:
restaurantes, supermercado e varejo. Você recebe somente a consulta, nunca itens,
rankings, exemplos de avaliação ou notas. Identifique o produto solicitado pelo
sentido da frase inteira. Normalize sinônimos e traduções literais culinárias
quando justificável; não invente ingredientes, receita, quantidade ou fatos de
produtos. Preserve marcas, exclusões, preparo, estilo e quantidades explícitas.
Separe requisitos explícitos de preferências de ocasião/uso. Uma refeição para
família não implica quantidade numérica. Se recipiente versus alimento, ou outro
sentido, for ambíguo, escolha a leitura mais provável e registre alternativas.
identity é apenas o tipo de produto, canonical é uma consulta portuguesa natural
que conserva todos os requisitos. source_span deve ser trecho exato da consulta.
relation: com=ingrediente incluído, sem=exclusão, estilo=variante/cultura explícita,
attribute=outro atributo explícito, portion=número explícito de pessoas (value
deve conter somente os dígitos). Não acrescente atributos implícitos de receitas.
domain é prepared_food, grocery, retail ou unknown. Seja conservador com ambiguidade.'''


def obj(properties):
    return {'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}


STRING = {'type':'string'}
SCHEMA = obj({'identity':STRING,'canonical':STRING,
    'domain':{'type':'string','enum':['prepared_food','grocery','retail','unknown']},
    'requirements':{'type':'array','items':obj({
        'relation':{'type':'string','enum':['com','sem','estilo','attribute','portion']},
        'value':STRING,'source_span':STRING})},
    'preferences':{'type':'array','items':obj({'value':STRING,'source_span':STRING})},
    'ambiguity':{'type':'array','items':STRING},
    'alternatives':{'type':'array','items':STRING}})


def normalize_quantity_dimension(value):
    """Correct only explicit measured units misclassified as person portions."""
    fixed=deepcopy(value)
    for req in fixed.get('requirements',[]):
        if req.get('relation')!='portion':continue
        span=req.get('source_span','')
        measured=re.search(r'\b(\d+(?:[.,]\d+)?)\s*(?:ml|l|litros?|g|kg|gramas?|quilogramas?)\b',span,re.I)
        if not measured or re.search(r'\bpessoas?\b',span,re.I):continue
        try:
            same=Decimal(req['value'].replace(',','.'))==Decimal(measured.group(1).replace(',','.'))
        except (InvalidOperation,AttributeError):continue
        if same:
            req.update(relation='attribute',value=span)
    return fixed


def validate(value, query):
    if set(value) != set(SCHEMA['properties']) or value['domain'] not in SCHEMA['properties']['domain']['enum']:
        raise ModelValidationError('Invalid interpretation fields/domain')
    for name in ('identity','canonical'):
        if not isinstance(value[name],str) or not value[name].strip():
            raise ModelValidationError('Missing semantic identity/canonical query')
    for field in ('requirements','preferences'):
        if not isinstance(value[field],list):
            raise ModelValidationError('Interpretation clauses must be lists')
        for clause in value[field]:
            expected = {'value','source_span'} | ({'relation'} if field=='requirements' else set())
            if (set(clause)!=expected or not isinstance(clause['source_span'],str)
                    or not clause['source_span'] or clause['source_span'] not in query
                    or not isinstance(clause['value'],str) or not clause['value'].strip()):
                raise ModelValidationError('Interpretation clause lacks query provenance')
            if field=='requirements' and clause['relation'] not in SCHEMA['properties']['requirements']['items']['properties']['relation']['enum']:
                raise ModelValidationError('Unknown interpretation relation')
            if clause.get('relation')=='portion':
                words={'um':'1','uma':'1','dois':'2','duas':'2','tres':'3','três':'3',
                       'quatro':'4','cinco':'5','seis':'6','sete':'7','oito':'8',
                       'nove':'9','dez':'10','onze':'11','doze':'12'}
                written=re.search(r'\b('+ '|'.join(words) +r')\s+pessoas?\b',clause['source_span'],re.I)
                supported=clause['value'] in re.findall(r'\b\d+\b',clause['source_span']) or (
                    written is not None and words[written.group(1).lower()]==clause['value'])
                if not clause['value'].isdigit() or not supported:
                    raise ModelValidationError('Portion must preserve an explicit digit or written person count')
    for field in ('ambiguity','alternatives'):
        if not isinstance(value[field],list) or any(not isinstance(x,str) for x in value[field]):
            raise ModelValidationError('Invalid semantic ambiguity alternatives')
    return value


class SemanticInterpreter:
    def __init__(self,config,*,allow_api=False):
        factory = ModelConfig.gpt41
        self.client=ModelClient(open_store(config),config=factory(
            model=config['models']['query'],base_url='https://api.openai.com/v1/',
            timeout_seconds=30,protocol_version='grounded-query-semantic-v1'),
            allow_api=allow_api,token_counter=token_counter(config),max_attempts=1)
        self.fingerprint=canonical_hash({'instructions':INSTRUCTIONS,'schema':SCHEMA,
                                         'configuration':self.client.config.__dict__,
                                         'producer_sha256':file_hash(__file__)})

    def interpret(self,query):
        payload=_payload(self.client.config,INSTRUCTIONS,{'consulta':query},SCHEMA,'query_interpretation')
        record=self.client._execute(payload,'grounded-semantic-query-v1','development','query-understanding',
                                    lambda value:validate(value,query))
        if record['status']!='valid' and record.get('reason')=='invalid_structured_output':
            # Revalidate preserved raw output after the declared written-count
            # validator correction. Never edit the returned interpretation or
            # overwrite the rejected response; no extra API request is needed.
            for attempt in record.get('attempts',[]):
                try:
                    value=validate(normalize_quantity_dimension(json.loads(attempt.get('visible_output') or '')),query)
                except (ValueError,TypeError,KeyError,ModelValidationError):
                    continue
                recovered={**record,'status':'valid','value':value,
                    'validation_recovery':'explicit-quantity-dimension-v1',
                    'derived_from_rejected_cache_key':record['cache_key']}
                self.client.store.put('query-validation-recovery:'+canonical_hash([record['cache_key'],'quantity-dimension-v1']),recovered)
                record=recovered
                break
        if record['status']!='valid' and record.get('reason')=='invalid_structured_output' and self.client.config.model.startswith('openai/gpt-4.1'):
            # Declared query-only adjudication; no products, rankings or grades.
            reviewer=ModelClient(self.client.store,config=ModelConfig.gpt41('openai/gpt-4.1',
                base_url=self.client.config.base_url,input_token_cap=4096,output_token_cap=2048,
                timeout_seconds=45,protocol_version='query-schema-adjudication-v1'),
                allow_api=self.client.allow_api,token_counter=self.client.token_counter,max_attempts=1)
            recovery_payload=_payload(reviewer.config,INSTRUCTIONS,{'consulta':query},SCHEMA,'query_interpretation')
            record=reviewer._execute(recovery_payload,'query-schema-adjudication-v1','recovery','query-understanding',
                lambda value:validate(normalize_quantity_dimension(value),query))
        if record['status']!='valid':
            attempts=record.get('attempts',[])
            status=attempts[-1].get('http_status') if attempts else None
            reason=record.get('reason','missing')
            raise RuntimeError(f'Semantic interpretation unavailable: {reason}; HTTP {status}. '
                               'Uncached queries require explicit --allow-api; no lexical fallback was used.')
        return record['value']
