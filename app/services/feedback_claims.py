"""Explicit, source-quoted claim clarification using the existing approval memory.

Only publication identifiers leave the process when a linked abstract is fetched.
No language model is called and no research claim is generated automatically.
Quote matching verifies the excerpt, not the scientific correctness of a claim.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

SCHEMA = '''
CREATE TABLE IF NOT EXISTS claim_publications (
 text_sha256 TEXT NOT NULL, source_id TEXT NOT NULL,
 source_json TEXT NOT NULL, saved_epoch REAL NOT NULL,
 PRIMARY KEY(source_id,text_sha256)
);
CREATE INDEX IF NOT EXISTS claim_publication_id ON claim_publications(source_id,saved_epoch);
'''


def normalized(value):
    return ' '.join(str(value).split())


def source_digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def pmid_id(value):
    match=re.fullmatch(r'(?:PMID\s*:?\s*)?(\d+)',str(value).strip(),re.I)
    if not match:raise ValueError('Choose a PubMed ID supplied with this intervention (PMID:...).')
    return 'PMID:'+match[1]


def linked_pmids(data):
    values=[p.get('pmid') for p in data.get('publications',[]) if isinstance(p,dict)]
    values+=data.get('pmids',[]) or []
    result=[]
    for value in values:
        if value is None:continue
        try:identity=pmid_id(value)
        except ValueError:continue
        if identity not in result:result.append(identity)
    return result


def fetch_abstract(source_id):
    pmid=pmid_id(source_id).split(':')[1]
    query=urlencode({'query':'EXT_ID:'+pmid+' AND SRC:MED','format':'json','resultType':'core','pageSize':1})
    request=Request('https://www.ebi.ac.uk/europepmc/webservices/rest/search?'+query,
                    headers={'User-Agent':'ALIS-Feedback-Claim-Review/1.0','Accept':'application/json'})
    try:
        with urlopen(request,timeout=20) as response:
            raw=response.read(2_000_001)
        if len(raw)>2_000_000:raise ValueError('Publication response exceeded the retrieval limit.')
        records=json.loads(raw).get('resultList',{}).get('result',[])
        record=next((r for r in records if str(r.get('pmid',r.get('id','')))==pmid and r.get('source','MED')=='MED'),None)
        if record is None:raise ValueError('The supplied PMID was not found in Europe PMC.')
        abstract=normalized(html.unescape(re.sub(r'<[^>]+>',' ',record.get('abstractText',''))))
        if not abstract:raise ValueError('This linked publication has no available abstract. Choose another supplied source.')
        title=html.unescape(record.get('title',''))
        text='Title: '+title+'\nAbstract: '+abstract
        return {'id':'PMID:'+pmid,'title':title,'text':text,'abstract_available':True,
                'url':'https://pubmed.ncbi.nlm.nih.gov/'+pmid+'/',
                'retrieval':'Europe PMC MED core abstract','text_sha256':source_digest(text)}
    except ValueError:raise
    except Exception as error:
        raise ValueError('Could not retrieve the linked abstract ('+type(error).__name__+'). No claim or review was changed. Saved sources remain available.') from None


class ClaimReview:
    def claim_source_ids(self,candidate):
        return linked_pmids(self._source_data(candidate))

    def _save_claim_publication(self,candidate,source):
        if source['id'] not in self.claim_source_ids(candidate):
            raise ValueError('The source is not linked to this intervention in the supplied handoff.')
        if not source.get('abstract_available') or source_digest(source.get('text',''))!=source.get('text_sha256'):
            raise ValueError('The saved abstract failed its integrity check.')
        with self.transaction():
            self.db.execute('INSERT OR IGNORE INTO claim_publications VALUES (?,?,?,?)',
                            (source['text_sha256'],source['id'],json.dumps(source,sort_keys=True,ensure_ascii=False),time.time()))
        return source

    def claim_source(self,candidate,source_id=None):
        ids=self.claim_source_ids(candidate)
        if not ids:raise ValueError('No linked PMID is available for a quoted claim. This entry needs an upstream publication reference.')
        source_id=pmid_id(source_id) if source_id else ids[0]
        if source_id not in ids:raise ValueError('Choose a publication supplied with this intervention: '+', '.join(ids))
        for record in self.source_records(candidate):
            source=record['source']
            if source['id']==source_id and source.get('abstract_available'):
                return self._save_claim_publication(candidate,source),'saved source assessment'
        row=self.db.execute('SELECT * FROM claim_publications WHERE source_id=? ORDER BY saved_epoch DESC LIMIT 1',(source_id,)).fetchone()
        if row and time.time()-row['saved_epoch']<=86400:
            source=json.loads(row['source_json'])
            if source_digest(source['text'])!=row['text_sha256']:raise ValueError('Saved publication failed its integrity check.')
            return source,'saved public abstract'
        source=fetch_abstract(source_id)
        return self._save_claim_publication(candidate,source),'retrieved public abstract'

    def saved_claim_source(self,candidate,source_id):
        source_id=pmid_id(source_id)
        if source_id not in self.claim_source_ids(candidate):return None
        row=self.db.execute('SELECT * FROM claim_publications WHERE source_id=? ORDER BY saved_epoch DESC LIMIT 1',(source_id,)).fetchone()
        if row is None:return None
        source=json.loads(row['source_json'])
        if source_digest(source['text'])!=row['text_sha256']:raise ValueError('Saved publication failed its integrity check.')
        return source

    def claim_history(self,candidate):
        generic='The supplied evidence supports retaining '+candidate['name']+' as a research candidate for '+candidate['condition']+'.'
        return [dict(id=r['id'],origin=r['origin'],created_at=r['created_at'],
                     statement=json.loads(r['data_json']).get('review_claim') or generic)
                for r in self.db.execute('SELECT * FROM versions WHERE candidate_id=? ORDER BY rowid',(candidate['id'],))]

    def validate_claim_grounding(self,candidate,grounding):
        if not isinstance(grounding,dict):raise ValueError('Use claim N to attach a checked source quotation to the proposed claim.')
        source_id=grounding.get('source_id')
        if source_id not in self.claim_source_ids(candidate):raise ValueError('Claim source must be linked in the supplied handoff.')
        row=self.db.execute('SELECT * FROM claim_publications WHERE text_sha256=? AND source_id=?',
                            (grounding.get('source_text_sha256'),source_id)).fetchone()
        if row is None:raise ValueError('Read the linked abstract before proposing a claim.')
        source=json.loads(row['source_json'])
        if source_digest(source['text'])!=row['text_sha256']:raise ValueError('The supporting abstract failed its integrity check.')
        quote=normalized(grounding.get('quote',''))
        if len(quote)<20 or quote not in normalized(source['text']):
            raise ValueError('The supporting quote must match a passage in the saved abstract (at least 20 characters). No proposal was saved.')
        return {'source_id':source_id,'quote':quote,'source_text_sha256':row['text_sha256'],
                'url':source['url'],'source':source,'scope':'published-study',
                'check':'literal quote matched; claim meaning requires clinician approval'}

    def propose_claim(self,candidate,reviewer,statement,source_id,quote,reason='Clarify the review statement to match the published study.'):
        self.require_clinician(reviewer)
        source,_=self.claim_source(candidate,source_id)
        grounding=self.validate_claim_grounding(candidate,{'source_id':source['id'],'source_text_sha256':source['text_sha256'],'quote':quote})
        return self.propose_correction(candidate['id'],candidate['current_version'],reviewer,'review_claim',statement,
                                       reason,source['id'],reusable=True,grounding=grounding)

    def claim_proposals(self,candidate):
        return [dict(r) for r in self.db.execute('SELECT * FROM corrections WHERE candidate_id=? AND field=? ORDER BY id',
                                                (candidate['id'],'review_claim'))]
