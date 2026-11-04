"""Source-effect likelihood inputs and native PLN; no clinician-vote inputs.

The (log) normal approximation uses the reported estimate and CI. Directional
maximized likelihoods are normalized to strength, and w=z²/2 nats is mapped by
native Truth__w2c. This is an explicit experimental research convention, not a
posterior probability, GRADE certainty or a probability of patient benefit.
"""
from __future__ import annotations
from decimal import Decimal
import hashlib
import json
import math
import re
from statistics import NormalDist

from app.services.feedback_pln import NativePLNError, execute_native

ADAPTER='source-comparative-normal-support-v1'
SEMANTICS='Experimental directional statistical-support index from source-reported comparative estimates and uncertainty. No reviewer votes, inferred study-quality defaults or patient-effect probability.'
NUM=r'[+−-]?(?:\d+(?:[.·]\d+)?|\.\d+)'
INTERVAL=re.compile(r'(?P<point>'+NUM+r')\s*(?:%|[A-Za-zµμ]+(?:/[A-Za-zµμ]+)?)?\s*[,;\[(]*\s*'
    r'(?P<level>\d+(?:[.·]\d+)?)\s*%\s*(?:confidence interval|CI)\s*[:=,\[(]*\s*'
    r'(?P<lo>'+NUM+r')\s*%?\s*(?:to|,|[–—-])\s*(?P<hi>'+NUM+r')',re.I)


def number(value):return float(Decimal(value.replace('−','-').replace('·','.')))
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()


def prepared(row,source):
    reasons=[]
    if source['id']!=row['source_id'] or hashlib.sha256(source['text'].encode()).hexdigest()!=source['text_sha256']:
        raise NativePLNError('Source identity/hash mismatch; no numerical result accepted.')
    if not source.get('abstract_available',False):reasons.append('No primary-source abstract/results text was retrieved.')
    if row['population_type']!='human':reasons.append('No human-population effect established by this source.')
    if row['design'] not in ('randomized','comparative_nonrandomized'):reasons.append('Not a primary comparative human study with reported results.')
    if row['comparison']!='target_vs_control':reasons.append('The contrast does not isolate this target intervention versus a control; dose comparisons are separate claims.')
    if row['condition_match']!='direct':reasons.append('The reported outcome does not directly establish the stated condition claim.')
    if row['intervention_match']!='administered_target':reasons.append('The intervention was not established as the tested target in this contrast.')
    if row['outcome_role']!='primary':reasons.append('No source-defined primary outcome is established; do not select the best-looking secondary result.')
    if row['exclusion_reason']:reasons.append(row['exclusion_reason'])
    effect=row['effect']
    if effect is None:reasons.append('No reported comparative estimate with an explicit confidence interval.')
    if reasons:return None,reasons
    if effect['quote'] not in source['text']:return None,['The reported effect quote does not occur in this source snapshot.']
    if not row['source_quotes']:return None,['The study context has no source quote.']
    if effect['direction']=='unclear':return None,['The benefit direction requires clarification.']
    observed=tuple(number(effect[k]) for k in ('estimate','ci_level','ci_lower','ci_upper'))
    matches=[tuple(number(m[k]) for k in ('point','level','lo','hi')) for m in INTERVAL.finditer(effect['quote'])]
    if observed not in matches:return None,['The numerical tuple was not found together as one reported effect/CI; no guessed numbers enter PLN.']
    e,level,lo,hi=observed
    if not all(math.isfinite(v) for v in observed) or not 0<level<100 or not lo<hi or not lo<=e<=hi:
        return None,['Invalid reported estimate or confidence interval.']
    ratio=effect['measure'] in ('HR','RR','OR')
    if ratio and lo<=0:return None,['Ratio estimates and interval limits must be positive.']
    quote=effect['quote']
    patterns={'HR':r'\bHR\b|hazard ratio','RR':r'\bRR\b|relative risk|risk ratio','OR':r'\bOR\b|odds ratio',
              'MD':r'mean difference|between.group difference|difference|treatment difference',
              'SMD':r'\bSMD\b|standardi[sz]ed mean difference','RD':r'\bRD\b|risk difference|absolute risk reduction'}
    if not re.search(patterns[effect['measure']],quote,re.I):return None,['The quoted effect type is not established.']
    values=[e,lo,hi]
    transformed=[math.log(v) for v in values] if ratio else values
    centre=(transformed[1]+transformed[2])/2
    rounding=0.0
    for factor,key,value in ((.5,'estimate',e),(.25,'ci_lower',lo),(.25,'ci_upper',hi)):
        resolution=10.0**Decimal(effect[key].replace('−','-').replace('·','.')).as_tuple().exponent
        rounding+=factor*resolution/(abs(value) if ratio else 1)
    if abs(transformed[0]-centre)>rounding+1e-10*max(1,abs(centre)):
        return None,['The interval is asymmetric beyond reported rounding; the supported normal approximation does not apply.']
    q=float(format(NormalDist().inv_cdf((1+level/100)/2),'.14g'))
    se=(transformed[2]-transformed[1])/(2*q)
    signed=transformed[0]*(1 if effect['direction']=='higher' else -1)
    z=signed/se
    if not all(math.isfinite(v) for v in (q,se,z,z*z)) or se<=0:return None,['Invalid normal approximation.']
    return {'source_id':row['source_id'],'source_sha256':source['text_sha256'], 'study_id':row['study_id'],
        'measure':effect['measure'],'raw_effect':effect,'transformed':transformed,'normal_quantile':q,
        'direction_sign':1 if effect['direction']=='higher' else -1,'approximate_se':se,'z':z,
        'population':row['population'],'outcome':row['outcome'],'comparator':row['comparator'],'design':row['design']},[]


def score_sources(records):
    usable=[];unscored=[];conflicts=[]
    for record in records:
        if record.get('status')=='contested':
            conflicts.append(record['source']['id']);continue
        value,reasons=prepared(record['facts'],record['source'])
        if value:usable.append(value)
        else:unscored.append({'source_id':record['source']['id'],'reasons':reasons})
    # Repeated reports of one registered study and exactly repeated estimates
    # are one input. Different endpoints/contrasts from the same trial are not
    # independent evidence and cannot be silently fused.
    unique={}
    for item in usable:
        key=(item['study_id'] or item['source_id'],item['measure'],tuple(item['transformed']),item['direction_sign'],
             item['population'].casefold(),item['outcome'].casefold(),item['comparator'].casefold())
        unique.setdefault(key,item)
    inputs=list(unique.values())
    result={'adapter':ADAPTER,'semantics':SEMANTICS,'strength':None,'confidence':None,'score_100':None,
            'status':'not_scorable','inputs':inputs,'unscored':unscored,'conflicts':conflicts,'native':None,
            'source_count':len(records),'numerical_input_count':len(inputs)}
    if conflicts:
        result['reason']='A source-specific correction is contested; adjudication is required.';return result
    if not inputs:
        result['reason']='No admissible source-reported comparative effect with uncertainty.';return result
    first=inputs[0]
    compatibility=lambda x:(x['measure'],x['population'].casefold(),x['outcome'].casefold(),x['comparator'].casefold(),x['design'],x['direction_sign'])
    independence={r['source']['id']:bool(r.get('independence_confirmed',False)) for r in records}
    if len(inputs)>1 and (any(compatibility(x)!=compatibility(first) for x in inputs)
                          or any(not x['study_id'] for x in inputs) or len({x['study_id'] for x in inputs})!=len(inputs)):
        result.update(status='needs_target',reason='Multiple estimates have incompatible or unverified study contexts. Select a precise comparison; they are not pooled or ranked as one drug score.')
        return result
    if len(inputs)>1 and not all(independence.get(x['source_id'],False) for x in inputs):
        result.update(status='needs_target',reason='Participant independence has not been explicitly verified. Different trial IDs alone do not establish it; source effects are not pooled or ranked as one drug score.')
        return result
    lines=['!(import! &self lib_pln)']
    expected=[]
    for index,item in enumerate(inputs):
        point,lo,hi=map(repr,item['transformed']);q=repr(item['normal_quantile']);direction=repr(float(item['direction_sign']))
        body=f'(let* (($se (/ (- {hi} {lo}) (* 2.0 {q}))) ($z (/ (* {point} {direction}) $se)) ($w (/ (* $z $z) 2.0)) ($r (exp (- 0.0 $w))) ($s (if (>= $z 0.0) (/ 1.0 (+ 1.0 $r)) (/ $r (+ 1.0 $r)))) ($t (stv $s (Truth__w2c $w))))'
        lines.append(f'(= (source_{index}) {body} $t))')
        lines.append(f'!(let $t (source_{index}) (source_value_{index} $t))')
        if index==0:lines.append('(= (combined_0) (source_0))')
        else:lines.append(f'(= (combined_{index}) (Truth__Revision (combined_{index-1}) (source_{index})))')
        w=item['z']**2/2;r=math.exp(-w)
        expected.append((1/(1+r) if item['z']>=0 else r/(1+r),w/(w+1),w))
    last=len(inputs)-1
    lines.append(f'!(let $t (combined_{last}) (source_evidence_result $t))')
    lines.append(f'!(let (stv $s $c) (combined_{last}) (source_evidence_index (* 100.0 (* $s $c))))')
    program='\n'.join(lines)+'\n'
    native=execute_native(program)
    values=re.findall(r'\(source_value_(\d+)\s+\(stv\s+([-+\deE.]+)\s+([-+\deE.]+)\)\)',native['stdout'])
    if len(values)!=len(inputs) or {int(x[0]) for x in values}!=set(range(len(inputs))):raise NativePLNError('Missing or duplicate native source results.')
    for index,s,c in values:
        exp_s,exp_c,_=expected[int(index)]
        if not math.isclose(float(s),exp_s,abs_tol=1e-9) or not math.isclose(float(c),exp_c,abs_tol=1e-9):raise NativePLNError('Native source result differs from the reported-estimate adapter.')
    pair=re.findall(r'\(source_evidence_result\s+\(stv\s+([-+\deE.]+)\s+([-+\deE.]+)\)\)',native['stdout'])
    index=re.findall(r'\(source_evidence_index\s+([-+\deE.]+)\)',native['stdout'])
    if len(pair)!=1 or len(index)!=1:raise NativePLNError('No unique native evidence assessment.')
    s,c=map(float,pair[0]);score=float(index[0])
    if not all(math.isfinite(v) for v in (s,c,score)) or not 0<=s<=1 or not 0<=c<=1 or not 0<=score<=100:
        raise NativePLNError('Native evidence assessment is outside its domain.')
    if not math.isclose(score,100*s*c,abs_tol=1e-8):raise NativePLNError('Native evidence-index mismatch.')
    result.update(status='calculated',strength=s,confidence=c,score_100=score,native=native)
    result['conflicting_directions']=any(x['z']>0 for x in inputs) and any(x['z']<0 for x in inputs)
    return result
