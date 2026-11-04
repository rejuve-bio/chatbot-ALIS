"""Source-bound comparative evidence, with no LLM-authored truth values."""
from __future__ import annotations
import hashlib
import json
import re

SOURCE_POLICY='source-comparative-support-v1'
MEASURES=('HR','RR','OR','MD','SMD','RD')
POPULATIONS=('human','animal','cellular','unclear')
DESIGNS=('randomized','comparative_nonrandomized','noncomparative','review','protocol','unclear')
COMPARISONS=('target_vs_control','dose_comparison','within_group','none','unclear')


def text(limit):return {'type':'string','minLength':1,'maxLength':limit}


EFFECT_SCHEMA={'type':'object','additionalProperties':False,
    'required':['measure','estimate','ci_lower','ci_upper','ci_level','direction','quote'],
    'properties':{'measure':{'type':'string','enum':list(MEASURES)},
                  **{k:dict(text(35),pattern=r'^[+−-]?(?:\d+(?:[.·]\d+)?|\.\d+)%?$') for k in ('estimate','ci_lower','ci_upper','ci_level')},
                  'direction':{'type':'string','enum':['higher','lower','unclear']},'quote':text(1800)}}
SOURCE_SCHEMA={'type':'object','additionalProperties':False,
    'required':['source_id','study_id','population_type','design','comparison','condition_match','intervention_match',
                'outcome_role','population','comparator','outcome','source_quotes','effect','exclusion_reason'],
    'properties':{'source_id':text(70),'study_id':{'anyOf':[text(100),{'type':'null'}]},
       'population_type':{'type':'string','enum':list(POPULATIONS)},'design':{'type':'string','enum':list(DESIGNS)},
       'comparison':{'type':'string','enum':list(COMPARISONS)},
       'condition_match':{'type':'string','enum':['direct','related','unrelated','unclear']},
       'intervention_match':{'type':'string','enum':['administered_target','mentioned_only','unclear']},
       'outcome_role':{'type':'string','enum':['primary','secondary','unclear']},
       'population':text(500),'comparator':text(500),'outcome':text(500),
       'source_quotes':{'type':'array','maxItems':3,'items':text(1200)},
       'effect':{'anyOf':[EFFECT_SCHEMA,{'type':'null'}]},
       'exclusion_reason':{'type':'string','maxLength':700}}}


def validate_source_assessments(rows,context):
    if not isinstance(rows,list) or len(rows)>8:raise ValueError('Invalid source-assessment list.')
    sources={s['id']:s for s in context['sources'] if not s['id'].startswith('handoff:')}
    seen=set()
    for row in rows:
        if not isinstance(row,dict) or set(row)!=set(SOURCE_SCHEMA['required']):
            raise ValueError('Missing or unexpected source-assessment fields.')
        if row['source_id'] not in sources:raise ValueError('An upstream summary is not a primary study source.')
        source=sources[row['source_id']]
        for key in ('population_type','design','comparison','condition_match','intervention_match','outcome_role'):
            if row[key] not in SOURCE_SCHEMA['properties'][key]['enum']:raise ValueError('Invalid source classification: '+key)
        for key in ('population','comparator','outcome'):
            if not isinstance(row[key],str) or not row[key].strip() or len(row[key])>500:raise ValueError('Invalid source context.')
        if not isinstance(row['exclusion_reason'],str) or len(row['exclusion_reason'])>700:raise ValueError('Invalid exclusion reason.')
        if not isinstance(row['source_quotes'],list) or len(row['source_quotes'])>3:raise ValueError('Invalid source quotes.')
        for quote in row['source_quotes']:
            if not isinstance(quote,str) or not quote or len(quote)>1200 or quote not in source['text']:
                raise ValueError('A source-classification quote is not in the inspected source.')
        study=row['study_id']
        if study is not None and (not isinstance(study,str) or not re.fullmatch(r'(?:NCT\d{8}|ISRCTN\d+)',study)
                                   or study not in source['text']):
            raise ValueError('A study identifier must occur in the inspected source; do not invent trial identities.')
        effect=row['effect']
        if effect is not None:
            if not isinstance(effect,dict) or set(effect)!=set(EFFECT_SCHEMA['required']):raise ValueError('Invalid comparative effect fields.')
            if effect['measure'] not in MEASURES or effect['direction'] not in ('higher','lower','unclear'):
                raise ValueError('Invalid measure or direction.')
            for key in ('estimate','ci_lower','ci_upper','ci_level'):
                if not isinstance(effect[key],str) or not re.fullmatch(r'[+−-]?(?:\d+(?:[.·]\d+)?|\.\d+)',effect[key]):
                    raise ValueError('Effect '+key+' must be a reported numerical literal; received '+repr(effect[key])[:50]+'.')
            quote=effect['quote']
            if not isinstance(quote,str) or not quote or len(quote)>1800 or quote not in source['text']:
                raise ValueError('The comparative effect quote is not in the inspected primary source.')
        identity=row['source_id']
        if identity in seen:raise ValueError('Only one unambiguous primary effect per source is supported; do not select or overwrite multiple effects.')
        seen.add(identity)
    # A source with unavailable information remains an explicit gap, never a
    # fabricated confidence or an automatic negative result.
    if set(sources)-{r['source_id'] for r in rows}:
        raise ValueError('Every inspected primary source needs an assessment or an explicit unscored record.')
    return rows


def source_state_key(row):
    """Reasons/endorsement counts never create a new numerical observation."""
    value={k:row[k] for k in ('source_id','study_id','population_type','design','comparison','condition_match',
                             'intervention_match','outcome_role','population','comparator','outcome','effect','exclusion_reason')}
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
