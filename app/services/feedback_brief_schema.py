"""A cited, read-only explanation contract; no verdict or numerical score fields."""
from __future__ import annotations

import copy
import hashlib
import json

POLICY='omega-readonly-review-brief-v1'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def text(limit):return {'type':'string','minLength':1,'maxLength':limit}


CITATION={'type':'object','additionalProperties':False,'required':['id','quote'],
          'properties':{'id':text(80),'quote':text(300)}}
ITEM={'type':'object','additionalProperties':False,'required':['explanation','citations'],
      'properties':{'explanation':text(600),'citations':{'type':'array','minItems':1,'maxItems':2,'items':CITATION}}}
SCHEMA={'type':'object','additionalProperties':False,'required':['study_findings','review_discussion','next_questions'],
        'properties':{'study_findings':{'type':'array','maxItems':2,'items':ITEM},
                      'review_discussion':{'type':'array','maxItems':2,'items':ITEM},
                      'next_questions':{'type':'array','maxItems':2,'items':{
                          'type':'object','additionalProperties':False,'required':['question','reference_ids'],
                          'properties':{'question':text(400),'reference_ids':{'type':'array','minItems':1,'maxItems':3,'items':text(80)}}}}}}


def references(context):
    return {x['id']:x['text'] for x in context['sources']+context['review_notes']}


def schema_for(context):
    schema=copy.deepcopy(SCHEMA)
    for field,key in (('study_findings','sources'),('review_discussion','review_notes')):
        schema['properties'][field]['items']=copy.deepcopy(ITEM)
        ids=[x['id'] for x in context[key]]
        if not ids:schema['properties'][field]['maxItems']=0
        else:schema['properties'][field]['items']['properties']['citations']['items']['properties']['id']['enum']=ids
    ids=list(references(context))
    if ids:schema['properties']['next_questions']['items']['properties']['reference_ids']['items']['enum']=ids
    else:schema['properties']['next_questions']['maxItems']=0
    return schema


def validate(value,context):
    def keys(row,expected):
        if not isinstance(row,dict) or set(row)!=set(expected):raise ValueError('Unexpected review-brief fields.')
    def string(row,limit):
        if not isinstance(row,str) or not row.strip() or len(row)>limit:raise ValueError('Invalid review-brief text.')
    keys(value,SCHEMA['required']);refs=references(context);word_count={}
    for field,kind in (('study_findings','sources'),('review_discussion','review_notes')):
        rows=value[field]
        if not isinstance(rows,list) or len(rows)>2:raise ValueError('The review brief exceeded its item limit.')
        allowed={x['id'] for x in context[kind]}
        for row in rows:
            keys(row,('explanation','citations'));string(row['explanation'],600)
            citations=row['citations']
            if not isinstance(citations,list) or not 1<=len(citations)<=2:raise ValueError('Each explanation needs a checked citation.')
            for cited in citations:
                keys(cited,('id','quote'));string(cited['quote'],300)
                if cited['id'] not in allowed or cited['quote'] not in refs[cited['id']]:
                    raise ValueError('Omega quoted text or a review that was not in the inspected packet.')
                if kind=='sources':word_count[cited['id']]=word_count.get(cited['id'],0)+len(cited['quote'].split())
    if any(n>25 for n in word_count.values()):raise ValueError('The brief quoted too much from a publication.')
    questions=value['next_questions']
    if not isinstance(questions,list) or len(questions)>2:raise ValueError('Too many review questions.')
    for row in questions:
        keys(row,('question','reference_ids'));string(row['question'],400)
        if not isinstance(row['reference_ids'],list) or not 1<=len(row['reference_ids'])<=3 or any(x not in refs for x in row['reference_ids']):
            raise ValueError('A review question references uninspected evidence.')
    return value
