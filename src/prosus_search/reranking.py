"""Optional source-aware shortlist ranking; no evaluation cards or labels."""
from __future__ import annotations

from dataclasses import replace

from .benchmark import open_store
from .config import token_counter
from .evaluation import canonical_hash
from .judge import (JudgeClient, JudgeConfig, JudgmentValidationError, JudgePreflightError,
                    EVIDENCE_SCHEMA, evidence_view, resolve_pointer, _payload)
from .understanding import obj, STRING

INSTRUCTIONS = '''Ordene produtos de um marketplace para a consulta original.
Você recebe uma interpretação auxiliar, que pode estar errada, e um conjunto de
itens com fatos originais. Não recebe notas de avaliação nem posições de busca.
Entenda o produto solicitado pela frase inteira, não por palavras isoladas.
Priorize identidade correta e requisitos explícitos antes de estilo/ocasião.
Ausência de evidência não é contradição. Não invente receitas, ingredientes,
estilo, temperatura, embalagem, origem ou quantidade. Flags dietéticas isoladas
não certificam segurança. Uma alternativa pode ser útil, mas não substitua
silenciosamente requisito explícito ou restrição alimentar. Observe partes,
prato completo versus ingrediente, quantidade e comida versus recipiente.
Retorne exatamente numero_resultados itens, em ordem.
grade: 0 irrelevante ou contradição explícita; 1 correspondência parcial ou
requisito sem evidência; 2 útil, identidade e requisitos comprovados; 3 também
preferências atendidas. conflict=true exige grade=0; hard_supported=false limita
grade a 1. Cite ao menos uma evidência exata do respectivo item para cada escolha.
Cada path é um JSON Pointer relativo à raiz de catalogo do próprio item:
use /name ou /description ou /taxonomy/l2, nunca catalogo.name. Para texto,
quote deve ser um trecho literal não vazio e scalar=null. Para booleano/número,
quote=null e scalar deve ser o valor original.
As notas orientam apenas esta ordenação e não são probabilidades ou certificações.'''

SCHEMA = obj({'results':{'type':'array','minItems':1,'maxItems':5,'items':obj({
    'item_id':STRING,'grade':{'type':'integer','enum':[0,1,2,3]},
    'hard_supported':{'type':'boolean'},'conflict':{'type':'boolean'},
    'evidence':{'type':'array','minItems':1,'items':EVIDENCE_SCHEMA}})}})


class RerankingError(RuntimeError):
    """Safe execution metadata; never exposes a provider response body."""
    def __init__(self, record):
        self.reason = record.get('reason', 'missing')
        attempts = record.get('attempts', [])
        self.receipt = {'http_status': attempts[-1].get('http_status') if attempts else None,
                        'error_type': attempts[-1].get('error_type') if attempts else None,
                        'preflight': record.get('preflight')}
        super().__init__('Source-aware reranking failed: '+self.reason+'. No silent ranking fallback.')


def validate(value, evidence, count):
    if set(value)!={'results'} or not isinstance(value['results'],list) or len(value['results'])!=count:
        raise JudgmentValidationError('Reranker must return the requested number of items')
    ids=[r.get('item_id') for r in value['results']]
    if len(set(ids))!=count or not set(ids)<=set(evidence):
        raise JudgmentValidationError('Reranker returned invented or duplicate IDs')
    for row in value['results']:
        if set(row)!={'item_id','grade','hard_supported','conflict','evidence'} or type(row['grade']) is not int or row['grade'] not in range(4):
            raise JudgmentValidationError('Invalid reranking fields or grade')
        if type(row['conflict']) is not bool or type(row['hard_supported']) is not bool:
            raise JudgmentValidationError('Invalid reranking status')
        if (row['conflict'] and row['grade']!=0) or (not row['hard_supported'] and row['grade']>1):
            raise JudgmentValidationError('Reranking grade contradicts its evidence status')
        if not isinstance(row['evidence'],list) or not row['evidence']:
            raise JudgmentValidationError('Reranking requires source evidence')
        for ref in row['evidence']:
            if set(ref)!={'path','quote','scalar'}:
                raise JudgmentValidationError('Invalid reranking source reference')
            original=resolve_pointer(evidence[row['item_id']],ref['path'])
            if isinstance(original,str):
                if ref['scalar'] is not None or not isinstance(ref['quote'],str) or not ref['quote'] or ref['quote'] not in original:
                    raise JudgmentValidationError('Reranking quote is not in the original source')
            elif type(original) in (bool,int,float):
                if ref['quote'] is not None or type(ref['scalar']) is not type(original) or ref['scalar']!=original:
                    raise JudgmentValidationError('Reranking scalar differs from source')
            else:
                raise JudgmentValidationError('Reranking must quote a scalar source field')
    return value


class EvidenceReranker:
    def __init__(self,config,*,allow_api=False):
        self.client=JudgeClient(open_store(config),config=replace(JudgeConfig.secondary(),
            model=config['evaluation']['secondary_model'],base_url=config['api']['base_url'],
            input_token_cap=4096,output_token_cap=2048,timeout_seconds=45,
            protocol_version='grounded-source-reranking-v1'),allow_api=allow_api,
            token_counter=token_counter(config),max_attempts=2)
        self.fingerprint=canonical_hash({'instructions':INSTRUCTIONS,'schema':SCHEMA,
            'config':self.client.config.__dict__,'max_candidates':10,'full_evidence_overflow':'reduce-candidate-count'})

    def rank(self,query,plan,candidates,metadata,k):
        selected=list(candidates[:10])
        while True:
            evidence={r['item_id']:evidence_view(metadata[r['item_id']]) for r in selected}
            data={'consulta':query,'interpretacao_auxiliar':plan,'numero_resultados':min(k,len(selected)),
                  'itens':[{'item_id':item_id,'catalogo':evidence[item_id]} for item_id in sorted(evidence)]}
            payload=_payload(self.client.config,INSTRUCTIONS,data,SCHEMA,'product_ranking')
            try:
                self.client.preflight(payload)
                break
            except JudgePreflightError:
                if len(selected)<=min(k,len(candidates)):
                    raise
                selected.pop()  # Never truncate or synthesize a product's evidence.
        record=self.client._execute(payload,'grounded-source-reranking-v1','round4','product-reranking',
            lambda value:validate(value,evidence,min(k,len(selected))))
        if record.get('status')!='valid':
            raise RerankingError(record)
        indexed={r['item_id']:r for r in selected}
        ordered=sorted(enumerate(record['value']['results']),key=lambda x:(-x[1]['grade'],x[0]))
        result=[]
        for rank,(_,row) in enumerate(ordered,1):
            original=indexed[row['item_id']]
            result.append({**original,'rank':rank,'score':float(row['grade'])+(len(ordered)-rank)/1000,
                'score_kind':'source-aware-llm-reranking-v1',
                'reranking':{'grade':row['grade'],'hard_supported':row['hard_supported'],
                             'conflict':row['conflict'],'evidence':row['evidence']}})
        return result,{'model':self.client.config.model,'input_candidates':len(selected),
                       'cache_hit':record.get('cache_hit',False),'fingerprint':self.fingerprint}
