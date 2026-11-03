"""Small source-grounded facet matcher; no benchmark data or remote inference.

The vocabulary describes ordinary catalog concepts, not query rewrites. Missing
values remain unknown; mutually exclusive values only conflict in the same facet.
"""
from __future__ import annotations

import re
import unicodedata

from .grounded import keys
from .snapshots import freeze_json

VERSION = 'source-facets-v1'

# Canonical values and ordinary Portuguese/catalog aliases. These are not
# inferred recipes: e.g. a falafel label never establishes its ingredients.
VOCABULARY = {
    'type': {
        'pizza':'pizza pizzas', 'sandwich':'sanduiche sanduiches sanduba lanche natural',
        'burger':'hamburguer hamburgueres burger smashburger', 'wrap':'wrap wraps',
        'burrito':'burrito burritos', 'bread':'pao paes pita baguete ciabatta focaccia',
        'soup':'sopa sopas caldo caldos', 'rice':'arroz risoto', 'pasta':'macarrao massa massas lasanha nhoque rondelli',
        'salad':'salada saladas', 'bento':'bento bentos', 'meal':'marmita marmitex executivo refeicao',
        'chicken':'frango galeto', 'beef':'carne bovina alcatra picanha contrafile',
        'pork':'suino suina porco pernil lombo linguica salsicha salsichao',
        'fish':'peixe tilapia salmao merluza bacalhau', 'seafood':'camarao lula polvo',
        'potato':'batata batatas fritas', 'pastry':'pastel pasteis', 'snack':'coxinha esfiha empada quibe croquete',
        'falafel':'falafel', 'kebab':'kebab shawarma', 'dessert':'trufa brownie brigadeiro bombom pave pudim',
        'cake':'bolo torta', 'icecream':'sorvete sorvetes', 'acai':'acai', 'popcorn':'pipoca',
        'sushi':'sushi sashimi temaki', 'combo':'combo combinados',
        'disinfectant':'desinfetante desinfetantes', 'wipes':'lenco umedecido',
        'sponge':'esponja esponjas', 'detergent':'detergente', 'soap':'sabonete sabao',
        'shampoo':'shampoo xampu', 'diaper':'fralda fraldas', 'container':'lancheira marmiteira recipiente',
        'soda':'refrigerante refrigerantes', 'juice':'suco sucos', 'water':'agua', 'beer':'cerveja cervejas',
    },
    'cut': {'wing':'asa asas asinha', 'thigh':'coxa sobrecoxa', 'breast':'peito',
            'whole':'inteiro inteira', 'fillet':'file filet'},
    'preparation': {'fried':'frito frita fritos fritas fritura', 'grilled':'grelhado grelhada',
                    'roasted':'assado assada', 'boiled':'cozido cozida',
                    'breaded':'empanado empanada milanesa', 'smoked':'defumado defumada'},
    'texture': {'crisp':'crocante crocantes', 'creamy':'cremoso cremosa', 'soft':'macio macia fofinho fofinha'},
    'cuisine': {'korean':'coreano coreana coreanos coreanas', 'japanese':'japones japonesa',
                'chinese':'chines chinesa', 'thai':'tailandes tailandesa', 'indian':'indiano indiana',
                'lebanese':'libanes libanesa', 'egyptian':'egipcio egipcia',
                'greek':'grego grega', 'mexican':'mexicano mexicana', 'italian':'italiano italiana',
                'german':'alemao alema', 'brazilian':'brasileiro brasileira',
                'mediterranean':'mediterraneo mediterranea', 'asian':'asiatico asiatica'},
    'scent': {'lavender':'lavanda', 'citrus':'citrus citrico citrica', 'floral':'floral flores',
              'eucalyptus':'eucalipto', 'pine':'pinho'},
    'ingredient': {x:x for x in ('queijo','cheddar','mussarela','bacon','cebola','tomate','leite',
                                  'gluten','lactose','ovo','catupiry','requeijao','pimenta')},
}
PARENTS = {'burger':'sandwich', 'wrap':'sandwich', 'burrito':'sandwich', 'kebab':'sandwich',
           'bento':'meal', 'chicken':'meal', 'beef':'meal', 'pork':'meal', 'fish':'meal', 'seafood':'meal'}
PROTEINS = {'chicken','beef','pork','fish','seafood'}
CUISINE_PARENTS = {'korean':'asian','japanese':'asian','chinese':'asian','thai':'asian',
                  'greek':'mediterranean','lebanese':'mediterranean','egyptian':'mediterranean'}
GENERIC = keys('produto produtos comida comidas prato pratos jantar almoco alimento alimentos refeicao refeicoes porcao porcoes')
SIDE = re.compile(r'\b(?:acompanh\w*|guarnic\w*|servid[oa] com|opciona\w*|escolha|escolher|adiciona\w*)\b')


def fold(text):
    return ''.join(c for c in unicodedata.normalize('NFKD', text.casefold()) if not unicodedata.combining(c))


def phrase(text, value):
    return bool(re.search(r'(?<!\w)'+re.escape(fold(value))+r'(?!\w)', fold(text)))


# Match complete word stems, preserving the exact original source substring.
LOOKUP = {facet:{key:value for value,aliases in values.items() for key in keys(aliases)}
          for facet,values in VOCABULARY.items()}


def extract(text, path, *, scope='main'):
    result=[]
    side=SIDE.search(fold(text)) if path=='/description' else None
    for word in re.finditer(r'\b\w+\b', text):
        stems=keys(word.group())
        if not stems:
            continue
        for facet,lookup in LOOKUP.items():
            value=next((lookup[s] for s in stems if s in lookup),None)
            if value is None:
                continue
            # Ambiguous standalone words do not establish a retail identity.
            if facet=='type' and value in {'wipes','sandwich'}:
                if value=='wipes' and not phrase(text,'umedecido'):
                    continue
                if value=='sandwich' and fold(word.group()) in {'natural','naturais'}:
                    continue
            before=fold(text[max(0,word.start()-30):word.start()])
            negative=bool(re.search(r'\b(?:sem|nao contem|livre de)\s+(?:\w+\s+){0,1}$',before))
            result.append({'facet':facet,'value':value,'polarity':'absent' if negative else 'present',
                'scope':'option_or_side' if side and word.start()>=side.start() else scope,
                'path':path,'quote':word.group()})
    return result


def product_facets(record):
    m=record.metadata
    claims=extract(m['name'],'/name')+extract(m.get('description',''),'/description')
    # A restaurant category/taxonomy establishes type, not ingredients or scent.
    for path,text in [('/category_name',m.get('category_name',''))]+[
            ('/taxonomy/'+level,m.get('taxonomy',{}).get(level,'').replace('_',' ')) for level in ('l1','l2')]:
        for claim in extract(text,path):
            if claim['facet']!='type':
                continue
            # Taxonomy quotes must occur in the original scalar, including case.
            original=m['taxonomy'].get(path.rsplit('/',1)[-1],'') if path.startswith('/taxonomy/') else text
            word=next((w for w in re.findall(r'[^_\W]+',original) if keys(w)==keys(claim['quote'])),None)
            if word:
                claims.append({**claim,'quote':word})
    ordered=[c['value'] for c in claims if c['facet']=='type' and c['path']=='/name' and c['polarity']=='present']
    forms=[v for v in ordered if v not in PROTEINS]
    name_types=set(forms if 'combo' in forms else forms[:1] or ordered[:1])
    catalog_types={c['value'] for c in claims if c['facet']=='type' and c['path']!='/description' and c['polarity']=='present'}
    taxonomy_forms={c['value'] for c in claims if c['facet']=='type' and c['path']!='/name' and c['path']!='/description' and c['value'] not in PROTEINS}
    selected=taxonomy_forms if name_types and name_types<=PROTEINS and taxonomy_forms else name_types or catalog_types
    return freeze_json({'claims':claims,'types':sorted(selected),
                        'type_sources_conflict':bool(name_types and catalog_types-name_types),
                        'historical_options_used':False})


def identity(plan):
    text=plan['identity_phrase']
    claims=extract(text,'/query/identity')
    ordered=[c for c in claims if c['facet']=='type']
    primary=ordered[0]['value'] if ordered else None
    for c in ordered[1:]:
        if primary in lineage(c['value'],PARENTS):primary=c['value']
    return {'types':{primary} if primary else set(),'keys':keys(text)-GENERIC,'claims':claims}


def lineage(value, parents):
    values={value}
    while value in parents:
        value=parents[value];values.add(value)
    return values


def requirement(req, facts, source):
    """Conservative matching: absence isn't a negation, nor is another component."""
    relation,value=req['relation'],req['value']
    if relation=='portion':
        claims=source['portion_claims']
        return ('unknown' if not claims else 'conflict' if len(claims)>1 else
                'supported' if int(value) in claims else 'contradicted'),[]
    wanted=[c for c in extract(value,'/query/requirement') if c['facet']!='type']
    evidence=[];states=[]
    for w in wanted:
        candidates=[c for c in facts['claims'] if c['facet']==w['facet'] and c['scope']=='main']
        matching=[c for c in candidates if c['value']==w['value'] or
                  (w['facet']=='cuisine' and w['value'] in lineage(c['value'],CUISINE_PARENTS))]
        positive=[c for c in matching if c['polarity']=='present']
        negative=[c for c in matching if c['polarity']=='absent']
        if positive and negative:
            states.append('conflict');evidence.extend(matching);continue
        if positive or negative:
            supported=bool(negative) if relation=='sem' else bool(positive)
            states.append('supported' if supported else 'contradicted');evidence.extend(matching);continue
        # Different ingredients/preparations may coexist. Scent, declared
        # cuisine and physical cut provide a mismatch only when unambiguous.
        exclusive=w['facet'] in {'scent','cut','cuisine'}
        other=[c for c in candidates if c['polarity']=='present']
        other_values={c['value'] for c in other}
        related=w['facet']=='cuisine' and any(c['value'] in lineage(w['value'],CUISINE_PARENTS) for c in other)
        if exclusive and len(other_values)==1 and not related and relation!='sem':
            states.append('contradicted');evidence.extend(other)
        else:
            states.append('unknown')
    # A compound request must preserve its complete meaning. Facet tokens
    # alone cannot establish e.g. "forno a lenha" or "estilo vovo".
    text=source['explicit_text']
    exact=phrase(text,value)
    terms=keys(value)
    matched_terms=set().union(*(keys(c['quote']) for c in wanted)) if wanted else set()
    remainder=terms-matched_terms-GENERIC
    # Generic culinary facet headings do not add a new product fact.
    remainder-=keys('estilo sabor fragrancia preparo preparado')
    if remainder or not wanted:
        if exact:
            # An exact mention confined to optional/side text remains unknown.
            main=text.split('\n',1)[0]+' '+SIDE.split(fold(text.split('\n',1)[-1]),maxsplit=1)[0]
            explicit_negative=phrase(main,'sem '+value)
            state=('supported' if explicit_negative else 'contradicted') if relation=='sem' else (
                   'contradicted' if explicit_negative else 'supported')
            if not phrase(main,value):state='unknown'
            states.append(state)
            if state!='unknown':
                evidence.extend({'path':e['path'],'quote':e['quote']} for e in source['evidence'] if phrase(e['quote'],value))
        else:
            states.append('unknown')
    state=next((s for s in ('contradicted','conflict','unknown') if s in states),'supported')
    return state,evidence


def compatibility(plan, facts, source):
    wanted=identity(plan);offered=set(facts['types'])
    compatible=bool(wanted['types'] and any(q in lineage(p,PARENTS) for q in wanted['types'] for p in offered))
    # A prepared ingredient or soup is not a sandwich just because it contains
    # bread in the description. Unknown types do not count as contradictions.
    identity_status='supported' if compatible else 'contradicted' if wanted['types'] and offered else 'unknown'
    first_word=next(iter(re.findall(r'\b\w+\b',plan['identity_phrase'])), '')
    if not wanted['types'] and keys(first_word)&GENERIC and plan.get('domain')=='prepared_food' and source['domain']=='prepared_food':
        identity_status='supported'
    if not wanted['types'] and wanted['keys'] and wanted['keys']<=set(source['name_keys']):
        identity_status='supported'
    identity_evidence=[c for c in facts['claims'] if c['facet']=='type' and c['path']!='/description']
    statuses=[]
    # Cuts/protein modifiers in a multiword identity remain mandatory.
    extras=[{'relation':'attribute','value':c['quote'],'source_span':c['quote']} for c in wanted['claims'] if c['facet']=='cut']
    extras.extend({'relation':'com','value':c['quote'],'source_span':c['quote']} for c in wanted['claims']
        if c['facet']=='type' and c['value'] not in wanted['types'] and
        not any(c['value'] in lineage(v,PARENTS) for v in wanted['types']))
    for req in list(plan['requirements'])+extras:
        state,evidence=requirement(req,facts,source)
        statuses.append({**req,'status':state,'evidence':evidence})
    contradicted=identity_status=='contradicted' or any(s['status']=='contradicted' for s in statuses)
    domain_conflict=(plan.get('domain')=='prepared_food' and source['domain']=='retail') or (
                     plan.get('domain')=='retail' and source['domain'] in {'prepared_food','grocery'})
    contradicted=contradicted or domain_conflict
    uncertain=any(s['status'] in {'unknown','conflict'} for s in statuses)
    tier=3 if contradicted else 2 if identity_status=='unknown' else 1 if uncertain else 0
    coverage=len(wanted['keys']&set(source['name_keys']))/len(wanted['keys']) if wanted['keys'] else 0.
    matches=sum(s['status']=='supported' for s in statuses)/max(1,len(statuses))
    # Retrieval contributes <1 and cannot cross the ten-point compatibility
    # tiers. Name coverage and support are bounded tie breakers.
    score=10.*(3-tier)+.5*coverage+.25*matches
    return score,{'identity_status':identity_status,'identity_types':sorted(wanted['types']),
        'product_types':sorted(offered),'identity_evidence':identity_evidence,'tier':tier,'domain_conflict':domain_conflict,
        'requirements':statuses,'unknown_requirements':sum(s['status'] in {'unknown','conflict'} for s in statuses),
        'type_sources_conflict':facts['type_sources_conflict'],'historical_options_used_as_evidence':False}
