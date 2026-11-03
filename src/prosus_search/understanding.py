"""Cached, query-only semantic interpretation through the existing budgeted client."""
from __future__ import annotations

from .benchmark import open_store
from .config import token_counter
from .judge import JudgeClient, JudgeConfig, JudgmentValidationError, _payload
from .evaluation import canonical_hash

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


def validate(value, query):
    if set(value) != set(SCHEMA['properties']) or value['domain'] not in SCHEMA['properties']['domain']['enum']:
        raise JudgmentValidationError('Invalid interpretation fields/domain')
    for name in ('identity','canonical'):
        if not isinstance(value[name],str) or not value[name].strip():
            raise JudgmentValidationError('Missing semantic identity/canonical query')
    for field in ('requirements','preferences'):
        if not isinstance(value[field],list):
            raise JudgmentValidationError('Interpretation clauses must be lists')
        for clause in value[field]:
            expected = {'value','source_span'} | ({'relation'} if field=='requirements' else set())
            if (set(clause)!=expected or not isinstance(clause['source_span'],str)
                    or not clause['source_span'] or clause['source_span'] not in query
                    or not isinstance(clause['value'],str) or not clause['value'].strip()):
                raise JudgmentValidationError('Interpretation clause lacks query provenance')
            if field=='requirements' and clause['relation'] not in SCHEMA['properties']['requirements']['items']['properties']['relation']['enum']:
                raise JudgmentValidationError('Unknown interpretation relation')
            if clause.get('relation')=='portion' and (not clause['value'].isdigit() or clause['value'] not in clause['source_span']):
                raise JudgmentValidationError('Portion must preserve explicit query digits')
    for field in ('ambiguity','alternatives'):
        if not isinstance(value[field],list) or any(not isinstance(x,str) for x in value[field]):
            raise JudgmentValidationError('Invalid semantic ambiguity alternatives')
    return value


class SemanticInterpreter:
    def __init__(self,config,*,allow_api=False):
        factory = JudgeConfig.gpt41 if config['evaluation']['primary_model'].removeprefix('openai/').startswith('gpt-4.1') else JudgeConfig
        self.client=JudgeClient(open_store(config),config=factory(
            model=config['evaluation']['primary_model'],base_url=config['api']['base_url'],
            timeout_seconds=30,protocol_version='grounded-query-semantic-v1'),
            allow_api=allow_api,token_counter=token_counter(config),max_attempts=1)
        self.fingerprint=canonical_hash({'instructions':INSTRUCTIONS,'schema':SCHEMA,
                                         'configuration':self.client.config.__dict__})

    def interpret(self,query):
        payload=_payload(self.client.config,INSTRUCTIONS,{'consulta':query},SCHEMA,'query_interpretation')
        record=self.client._execute(payload,'grounded-semantic-query-v1','development','query-understanding',
                                    lambda value:validate(value,query))
        if record['status']!='valid':
            attempts=record.get('attempts',[])
            status=attempts[-1].get('http_status') if attempts else None
            reason=record.get('reason','missing')
            raise RuntimeError(f'Semantic interpretation unavailable: {reason}; HTTP {status}. '
                               'Uncached queries require explicit --allow-api; no lexical fallback was used.')
        return record['value']
