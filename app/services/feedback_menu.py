"""Guided clinician review over the existing store, approvals and native PLN.

The ordinary path is select -> read -> judge -> optional note -> saved result.
No language model is used. Command mode remains available separately.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import sqlite3

from app.services.feedback_auth import AuthenticationError, interactive_login
from app.services.feedback_atlas import ReviewedAtlas, AtlasError
from app.services.feedback_brief import ReviewBrief
from app.services.feedback_omega import OmegaReviewError
from app.services.feedback_backend import ensure_alis_ready
from app.services.feedback_pln import NativePLNError
from app.services.feedback_source_display import excerpt, source_sections
from app.services.feedback_triage import CONCERNS, LABELS, review_status, priority_key
from app.services.feedback_terminal import Terminal, clip, confidence as compact_confidence, strength as compact_strength

ERRORS=(ValueError,OSError,sqlite3.Error,AuthenticationError,NativePLNError,AtlasError,OmegaReviewError)


def confidence_text(value):
    if value is None:return 'unknown'
    text=f'{value:.6f}'
    return '<1.000000' if value<1 and text=='1.000000' else text


def support_text(value):
    if not value or value['strength'] is None:return 'unreviewed' if not value or value['status']=='unreviewed' else 'strength and confidence unknown'
    return f"{100*value['strength']:.1f}% decisive support | count confidence {confidence_text(value['confidence'])}"


class GuidedFeedback:
    def __init__(self,session,say,handoff_path=None):
        self.s=session;self.store=session.store;self.ui=Terminal(emit=say);self.say=self.ui.text
        self.notice=None;self.offset=0;self.back_offsets=[];self.next_offset=0;self.previous_offset=0;self.atlas_error=None
        self.path=Path(handoff_path).expanduser().resolve() if handoff_path else None
        self.view="all";self.min_decisive=0
        self.atlas=ReviewedAtlas(session)

    def refresh_atlas(self,announce=False):
        self.s.require_access()
        try:
            result=self.atlas.refresh(self.min_decisive,progress=lambda message:self.ui.text('Updating reviewed atlas…',kind='muted'))
        except AtlasError as error:
            self.atlas_error=str(error)
            return None
        self.atlas_error=None
        if announce:
            if self.notice is None:self.notice={'message':'Atlas #'+str(result['snapshot_id'])+' is current.'}
            self.notice['atlas']=result
        return result

    def atlas_changes(self,result):
        direct=[c for c in result['changes'] if c['cause_events'] or not c['before'] or c['before']['version_id']!=c['after']['version_id']]
        shifted=[c for c in result['changes'] if c not in direct]
        ordered=direct if direct else shifted
        for change in ordered[:6]:
            before=change['before'];after=change['after']
            self.ui.section('Entry '+str(change['entry_number'])+' · '+change['name'])
            if before:
                self.ui.delta(before,after)
                self.say(before['status']+' → '+after['status']+'  ·  order '+str(before['research_order'])+' → '+str(after['research_order']))
            else:self.say('New entry · '+after['status'])
            for event in change['cause_events'][-3:]:
                self.say('Review #'+str(event['id'])+' · '+event['reviewer']+' · '+event['decision']+
                         (' · replaces #'+str(event['supersedes_id']) if event['supersedes_id'] else ''),kind='muted')
            if before and before==after:self.say('Note updated; strength, confidence and order unchanged.',kind='muted')
        if direct and shifted:self.say(str(len(shifted))+' other entries shifted position; their appraisals are unchanged.',kind='muted')
        if len(ordered)>6:self.say(str(len(ordered)-6)+' further changes are retained in the export.',kind='muted')

    def home(self):
        result=self.refresh_atlas()
        subtitle=self.s.reviewer.name+'  ·  '+(self.path.name if self.path else 'Opened review file')
        if result:subtitle+='  ·  Atlas #'+str(result['snapshot_id'])
        self.ui.begin('Interventions',subtitle)
        if self.notice:
            if self.notice.get('error'):self.ui.text(self.notice['error'],kind='danger')
            else:
                self.ui.text(self.notice.get('message','Atlas updated'),kind='success')
                if 'before' in self.notice:self.ui.delta(self.notice['before'],self.notice['after'])
                atlas=self.notice.get('atlas')
                if atlas:
                    caused=[c for c in atlas['changes'] if c['cause_events']]
                    for change in caused[:1]:
                        if change['before']:
                            self.say(change['before']['status']+' → '+change['after']['status']+'  ·  research order '+str(change['before']['research_order'])+' → '+str(change['after']['research_order']))
                        event=change['cause_events'][-1]
                        self.say('Review #'+str(event['id'])+' · '+event['reviewer']+' · '+event['decision'],kind='muted')
            self.ui.rule()
        if self.atlas_error:
            self.ui.text('Reviews are saved; atlas refresh is pending.',kind='warning')
            self.say(self.atlas_error,kind='muted')
        if result:
            all_rows=result['atlas']['entries']
            reviewed=sum(r['routing']['total']>0 for r in all_rows)
            needs=sum(r['routing']['status']=='needs_review' for r in all_rows)
            self.say(f'{len(all_rows)} entries  ·  {reviewed} reviewed  ·  {needs} need review',kind='muted')
            if self.view!='all' or self.min_decisive:
                self.say('Filter: '+('All' if self.view=='all' else LABELS[self.view])+f'  ·  minimum decisive reviews: {self.min_decisive}',kind='muted')
            rows=[r for r in all_rows if self.view=='all' or r['routing']['status']==self.view]
            size=max(4,min(12,self.ui.height-16-(6 if self.notice else 0)))
            if self.offset>=len(rows):self.offset=0
            footer_lines=3+sum(max(1,(len(line)+self.ui.width-1)//self.ui.width) for line in ('[number] Open  ·  [n/p] Next/previous  ·  [r] Atlas  ·  [v] Filter','[a] Account  ·  [f] File  ·  [?] Help  ·  [q] Quit'))
            budget=max(4,self.ui.height-self.ui.lines-footer_lines)
            displayed=[];used=0;last_group=None
            for row in rows[self.offset:self.offset+size]:
                group=(row['condition'],row['category'])
                label=group[0]+' · '+group[1].replace('_',' ')
                group_cost=(2 if self.ui.width>=70 else 1)+max(1,(len(label)+self.ui.width-1)//self.ui.width) if group!=last_group else 0
                cost=group_cost+(1 if self.ui.width>=70 else 3)
                if displayed and used+cost>budget:break
                displayed.append(row);used+=cost;last_group=group
            end=self.offset+len(displayed)
            self.next_offset=end if end<len(rows) else self.offset
            self.previous_offset=max(0,self.offset-size)
            last=None
            for row in displayed:
                group=(row['condition'],row['category'])
                if group!=last:
                    self.ui.section(group[0]+' · '+group[1].replace('_',' '));last=group
                    block=[]
                    for other in displayed[displayed.index(row):]:
                        if (other['condition'],other['category'])!=group:break
                        block.append(other)
                    self.ui.entries(block)
            if not rows:self.say('No entries match this view. Choose v to change the filter; all entries remain saved.')
            else:self.say(f'Showing {self.offset+1}–{end} of {len(rows)}  ·  Reviews = decisive / total',kind='muted')
        else:
            for c in self.store.candidates():self.say(f"{c['position']:>2}. {c['name']} | {c['condition']}")
        self.ui.rule()
        self.say('[number] Open  ·  [n/p] Next/previous  ·  [r] Atlas  ·  [v] Filter')
        self.say('[a] Account  ·  [f] File  ·  [?] Help  ·  [q] Quit')

    def reviewed_atlas(self):
        result=self.refresh_atlas()
        if not result:
            self.notice={'error':'Atlas refresh is pending. '+str(self.atlas_error or '')};return
        self.ui.begin('Reviewed atlas #'+str(result['snapshot_id']),result['created_at'])
        parent=result['previous_snapshot_id']
        if parent:self.say('Changes since atlas #'+str(parent),kind='muted')
        else:self.say('Starting atlas. Later versions retain the changes caused by reviews.',kind='muted')
        if result['changes']:self.atlas_changes(result)
        else:self.say('No changes to strength, status or ordering.')
        self.ui.section('Execution')
        self.say('Native Omega · local workflow · zero model calls',kind='muted')
        self.say('Read feedback → apply review policy → assemble atlas',kind='muted')
        self.ui.menu([('1','Change history'),('2','Export reviewed atlas JSON'),('3','Native PLN calculation'),('0','Back')])
        choice=input('Atlas > ').strip()
        if choice=='1':
            self.s.require_access()
            rows=self.store.db.execute('SELECT id,created_at,result_json FROM reviewed_atlases WHERE dataset_id=? AND scope_key=? ORDER BY id DESC LIMIT 10',
                                       (self.store.active_dataset(),self.atlas._scope(self.min_decisive))).fetchall()
            lines=[]
            for saved in reversed(rows):
                value=json.loads(saved['result_json']);lines+=['ATLAS #'+str(saved['id'])+' · '+saved['created_at']]
                for change in value['changes']:
                    if not change['cause_events']:continue
                    lines.append(change['name']+' · '+change['condition'])
                    if change['before']:
                        lines.append('Strength '+compact_strength(change['before']['strength'])+' → '+compact_strength(change['after']['strength'])+
                                     '; confidence '+compact_confidence(change['before']['confidence'])+' → '+compact_confidence(change['after']['confidence']))
                        lines.append(change['before']['status']+' → '+change['after']['status']+'; order '+str(change['before']['research_order'])+' → '+str(change['after']['research_order']))
                    for event in change['cause_events']:lines.append('Review #'+str(event['id'])+' · '+event['reviewer']+' · '+event['decision'])
                lines.append('')
            self.ui.document('Atlas change history','\n'.join(lines))
        elif choice=='2':
            default=self.store.path.parent/'atlases'/('reviewed-atlas-'+str(result['snapshot_id'])+'.json')
            answer=input('Export path (Enter for '+str(default)+'): ').strip()
            target=Path(answer).expanduser() if answer else default;target.parent.mkdir(parents=True,exist_ok=True)
            self.notice={'message':'Export saved: '+str(self.atlas.export(result,target))}
        elif choice=='3':
            value=input('Intervention entry number: ').strip();self.s.require_access()
            c=self.store.resolve(value);a=self.store.belief_assessment(c);native=(a or {}).get('native')
            if native:self.ui.document('Native PLN · '+c['name'],native['program']+'\n'+native['stdout'],native['engine'])
            else:self.ui.document('Native PLN','No decisive input for this version. Strength and confidence remain unknown.')

    def filters(self):
        self.s.require_access();self.ui.begin('Filter interventions')
        self.ui.menu([('1','All'),('2','More support'),('3','Needs review'),('4','More opposition'),('5','Not reviewed'),('6','Set minimum decisive reviews'),('0','Back')])
        choice=input('Filter > ').strip()
        views={'1':'all','2':'more_support','3':'needs_review','4':'lower_support','5':'unreviewed'}
        if choice in views:self.view=views[choice]
        elif choice=='6':
            value=input('Minimum decisive reviewers (0 = no minimum): ').strip()
            if not value.isdigit():raise ValueError('Enter a nonnegative whole number.')
            self.min_decisive=int(value)
            self.notice={'message':'Coverage filter updated. Appraisals and records are unchanged.'}
        elif choice!='0':raise ValueError('Choose a displayed filter.')
        self.offset=0;self.back_offsets=[]

    def run(self):
        while True:
            try:
                if self.s.authentication is None:
                    self.ui.begin('Sign in');self.say('[a] Sign in  ·  [q] Quit')
                elif not self.store.db.execute("SELECT 1 FROM settings WHERE key='dataset'").fetchone():
                    self.ui.begin('Open a review file',self.s.reviewer.name);self.say('[f] Open JSON  ·  [a] Account  ·  [q] Quit')
                else:self.home()
                choice=input('Select > ').strip().casefold()
                if choice in ('q','quit','exit'):return
                if choice=='a':self.notice=None;self.offset=0;self.back_offsets=[];self.switch_account();continue
                if choice=='f':self.notice=None;self.offset=0;self.back_offsets=[];self.open_file();continue
                if choice=='v':self.filters();continue
                if choice=='r':self.reviewed_atlas();continue
                if choice in ('n','p'):
                    if choice=='n':
                        if self.next_offset!=self.offset:self.back_offsets.append(self.offset)
                        self.offset=self.next_offset
                    else:self.offset=self.back_offsets.pop() if self.back_offsets else self.previous_offset
                    self.notice=None;continue
                if choice in ('help','?'):
                    self.ui.document('How to review',
                        'Choose an intervention ID, read its evidence, then select Support, Oppose or Uncertain. Add an optional note.\n\n'
                        'Strength: the share of decisive current judgments choosing Support.\n'
                        'Count confidence: N/(N+1), where N is Support + Oppose.\n'
                        'Uncertain: recorded separately, without positive or negative weight.\n'
                        'Reviews: decisive / total reviewers. These are research appraisals, not treatment certainty.\n\n'
                        'A new response replaces your current response for this evidence version. History is preserved.\n'
                        'Use r for atlas changes, export and the native PLN calculation.\n'
                        'Use n/p to change list pages. Entry IDs never change, and any ID can be opened from any page.');continue
                if choice.isdigit():
                    self.s.require_access();self.notice=None;self.intervention(self.store.resolve(choice)['id']);continue
                self.notice={'error':'Choose an intervention ID or a displayed menu key.'}
            except ERRORS as error:self.notice={'error':str(error)}

    def open_file(self):
        value=input('Review JSON file path (Enter to cancel): ').strip().strip('"\'')
        if not value:return
        path=Path(value);result=self.s.import_handoff(path);self.path=path.expanduser().resolve()
        self.say(f"Opened {result['total']} interventions."+(f" Reused {result['learned']} approved statement(s)." if result['learned'] else ''))

    def switch_account(self):
        if self.s.provider=='alis':ensure_alis_ready(self.s.api_url,self.s.auto_start)
        self.s.authentication=None
        self.s.authentication=interactive_login(self.store,self.s.provider,self.s.api_url)
        if self.path:
            result=self.s.import_handoff(self.path)
            self.say('Opened the selected file for '+self.s.reviewer.name+'.')
        else:self.open_file()

    def load_source(self,candidate,source_id=None):
        ids=self.store.claim_source_ids(candidate)
        if not ids:return None
        if source_id is None and candidate.get('claim_approved'):
            approved=(candidate['data'].get('review_claim_grounding') or {}).get('source')
            if approved and approved['id'] in ids:
                self.say('Reading the study snapshot attached to the approved wording…')
                return approved
        try:
            self.say('Reading the linked study (no model call)…')
            source,_=self.store.claim_source(candidate,source_id or ids[0])
            return source
        except (ValueError,OSError) as error:
            self.say('Study text unavailable: '+str(error));return None

    def current_score(self,c):
        a=self.store.belief_assessment(c);rows=self.store.current_reviews(c);t=review_status(a,rows,self.min_decisive)
        compact=self.ui.height<38
        if compact:self.ui.raw()
        else:self.ui.section('Research appraisal')
        self.ui.metrics(a,t,compact=compact)
        if t['status']=='needs_review' and not compact:self.say(t['reason'],kind='muted')
        for concern in t['concerns'][:2]:self.say(concern['reviewer']+' flagged: '+concern['text'],2,kind='warning')
        if len(t['concerns'])>2:self.say(f"{len(t['concerns'])-2} more concerns in review history.",kind='muted')

    def card(self,c,source):
        compact=self.ui.height<38
        self.ui.begin(c['name'],c['condition']+'  ·  Entry '+str(c.get('position') or next(r['position'] for r in self.store.candidates() if r['id']==c['id'])),compact=compact)
        if not compact:self.ui.text('REVIEW QUESTION',kind='muted')
        self.say(c['proposition'])
        if c.get('claim_approved'):
            fact=next(f for f in c['learned_knowledge'] if f['field']=='review_claim')
            self.say('Wording approved by '+fact['approver']+'.',kind='muted')
        self.ui.section('Evidence'+(' · '+source['id'] if source and compact else ''))
        if source:
            if not compact:
                if source.get('title'):self.say(clip(source['title'],self.ui.width),kind='bold')
                self.say(source['id']+'  '+source.get('url',''),kind='muted')
            methods=re.search(r'\bMethods\s*:\s*(.*?)(?=\bResults\s*:|\bConclusions?\s*:|$)',source.get('text',''),re.I|re.S)
            if methods:
                quote,cut=excerpt(methods.group(1),135 if compact else 230);self.say('Studied: '+quote+(' …' if cut else ''),2)
            sections=source_sections(source)
            if sections:
                part=sections[0];quote,cut=excerpt(part['quote'],140 if compact else 430)
                self.say(('Findings: ' if part['label'].startswith('Reported results') else 'Abstract: ')+quote+(' …' if cut or part['truncated'] else ''),2)
            self.say('Source excerpts · full abstract and Omega brief under [4]',kind='muted')
        else:
            quote,cut=excerpt(str(c['data'].get('why') or c['data'].get('note') or 'No explanation supplied.'),400)
            self.say(quote+(' …' if cut else ''),2);self.say('Supplied explanation · study text unavailable',kind='muted')
        self.current_score(c)
        reviews=self.store.current_reviews(c)
        self.ui.section('Reviewer notes'+(' · showing 2 of '+str(len(reviews)) if len(reviews)>2 else ''))
        if not reviews:self.say('No responses yet.',kind='muted')
        shown=reviews[-2:]
        mine=next((row for row in reviews if row['clinician_id']==self.s.reviewer.id),None)
        if mine and mine not in shown:shown=[mine,reviews[-1]]
        for row in shown:
            own=' (you)' if row['clinician_id']==self.s.reviewer.id else ''
            if compact:
                line=clip(row['reviewer']+own,22)+' · '+row['decision'].title()
                if row['rationale']:line+=' · '+row['rationale']
                self.say(clip(line,self.ui.width))
            else:
                self.say(row['reviewer']+own+' · '+row['decision'].title(),kind='bold')
                if row['rationale']:
                    note,cut=excerpt(row['rationale'],160);self.say(note+(' …' if cut else ''),2)
        if len(reviews)>2 and not compact:self.say(f'{len(reviews)-2} other response(s) in review history.',kind='muted')

    def intervention(self,identity):
        self.s.require_access();self.store.require_active_candidate(identity)
        c=self.store.candidate(identity);source=self.load_source(c);source_version=c['current_version'];redraw=True;displayed_reviews=None
        while True:
            try:
                self.s.require_access();self.store.require_active_candidate(identity);c=self.store.candidate(identity)
                if c['current_version']!=source_version:
                    source=self.load_source(c);source_version=c['current_version'];redraw=True;displayed_reviews=None
                elif source and source['id'] not in self.store.claim_source_ids(c):source=self.load_source(c);redraw=True
                current_reviews=tuple(r['id'] for r in self.store.current_reviews(c))
                if current_reviews!=displayed_reviews:redraw=True
                if redraw:self.card(c,source);displayed_reviews=current_reviews
                self.ui.rule();self.say('[1] Support   [2] Oppose   [3] Uncertain')
                self.say('[4] Evidence / history   [5] Correction / concern   [0] Back')
                choice=input('Review > ').strip()
                if choice=='0':return
                if choice in ('1','2','3'):
                    self.submit(c,{'1':'support','2':'oppose','3':'uncertain'}[choice],source);return
                if choice=='4':
                    old=source;source=self.more(c,source);redraw=(source is not old or self.notice is not None);continue
                if choice=='5':self.record_concern(c,source);redraw=True;continue
                self.ui.text('Choose 1, 2, 3, 4, 5 or 0.',kind='warning');redraw=False
            except ERRORS as error:self.ui.text(str(error),kind='danger');redraw=False

    def submit(self,c,decision,source):
        note=input('Note for other reviewers (Enter to skip): ').strip()
        self.s.require_access(c['patient_id']);citation=source['id'] if source else ''
        self.store.save_draft(c,self.s.reviewer,decision,note,citation,notes_only=True)
        result=self.store.review(c['id'],c['current_version'],self.s.reviewer,decision,note,citation,notes_only=True,submitted_decision=decision)
        self.notice={'message':('Saved '+decision.title()+' · '+c['name'] if result['changed'] else 'Already saved · no duplicate contribution'),
                     'before':result['belief_before'],'after':result['belief_after']}
        self.offset=0;self.back_offsets=[];self.refresh_atlas(announce=True)

    def more(self,c,source):
        self.ui.begin('Evidence & review history',c['name']+' · '+c['condition'])
        self.ui.menu([('1','Read full abstract'),('2','Choose another linked study'),('3','Reviewer responses & history'),('4','Withdraw my response'),('5','Omega review brief · optional'),('0','Back')])
        choice=input('Evidence > ').strip()
        if choice=='1':self.ui.document('Study abstract',source['text'] if source else 'No study text is available.',(source['id']+' · '+source.get('url','')) if source else c['name'])
        elif choice=='2':
            ids=self.store.claim_source_ids(c)
            for i,value in enumerate(ids,1):self.say(f'{i}. {value}')
            selected=input('Study number (Enter to cancel): ').strip()
            if selected.isdigit() and 1<=int(selected)<=len(ids):return self.load_source(c,ids[int(selected)-1])
        elif choice=='3':
            lines=['CURRENT RESPONSES']
            reviews=self.store.current_reviews(c)
            if not reviews:lines.append('No current responses.')
            for row in reviews:
                lines+=['',row['reviewer']+' · '+row['decision']+' · Review #'+str(row['id']),row['created_at']]
                if row['rationale']:lines.append(row['rationale'])
                if row['citation']:lines.append('Reference: '+row['citation'])
                if row.get('concern_kind'):lines.append('Concern: '+CONCERNS[row['concern_kind']]+': '+row['concern_text'])
            lines+=['','RECENT CHANGES']
            for event in self.store.history(c)['reviews'][-10:]:
                lines.append('#'+str(event['id'])+' · '+event['reviewer']+' · '+event['decision']+' · '+event['created_at'])
                if event.get('concern_kind'):lines.append(CONCERNS[event['concern_kind']]+': '+event['concern_text'])
                elif event.get('rationale'):lines.append(event['rationale'])
            lines+=['','STATEMENT HISTORY']
            for version in self.store.claim_history(c):lines.append(('Current: ' if version['id']==c['current_version'] else 'Earlier: ')+version['statement'])
            old=sum(r['version_id']!=c['current_version'] for r in self.store.history(c)['reviews'])
            if old:lines.append(f'{old} older review event(s) belong to earlier statement versions; they remain in history.')
            self.ui.document('Reviewer history','\n'.join(lines),c['name']+' · '+c['condition'])
        elif choice=='5':self.omega_brief(c,source)
        elif choice=='4':
            current=next((r for r in self.store.current_reviews(c) if r['clinician_id']==self.s.reviewer.id),None)
            if not current:self.say('You have no current response to withdraw.',kind='muted')
            elif input('Withdraw your current response? [y/N]: ').strip().casefold()=='y':
                self.s.require_access(c['patient_id'])
                result=self.store.review(c['id'],c['current_version'],self.s.reviewer,'withdrawn','Withdrawn by reviewer from the guided menu.',notes_only=True)
                self.notice={'message':'Response withdrawn · history preserved','before':result['belief_before'],'after':result['belief_after']}
                self.refresh_atlas(announce=True)
        return source

    def omega_brief(self,c,source):
        assistant=ReviewBrief(self.s)
        sources=[source] if source else []
        request=assistant.context(c,sources)
        if not request['sources'] and not request['review_notes']:
            self.say('No study text or substantive review notes are available to explain. No model call was made.');return
        key=assistant.key(c,request)
        stored=assistant.cached(c,request)
        if stored is None:
            rejected=self.store.cached_analysis('brief-rejected:'+key,ttl=10**12)
            if rejected:
                self.say('The previous request for this exact brief failed: '+rejected['result']['error'])
                if input('Make one new model request to retry? [y/N]: ').strip().casefold()!='y':return
            identity=assistant.client.identity()
            if not self.store.has_external_consent(self.s.reviewer,identity):
                self.say('The optional brief uses '+identity['model']+' at '+identity['destination']+'.')
                self.say('It receives the selected study, intervention/condition and a bounded sample of reviewer notes. Patient identifiers, account names/emails and credentials are excluded or redacted. It cannot change reviews, scores or ordering.')
                if input('Allow the authorized file text to be sent for this optional brief? [y/N]: ').strip().casefold()!='y':
                    self.say('No text was sent. Reviewing and scoring remain available.');return
                self.store.record_external_consent(self.s.reviewer,identity)
            self.say('Omega is preparing a read-only review brief. At most one model call; no automatic retries.')
            try:stored=assistant.generate(c,sources,request,progress=self.say)
            except OmegaReviewError as error:
                self.store.save_analysis('brief-rejected:'+key,{'error':str(error)})
                self.say('Optional brief unavailable: '+str(error))
                self.say('Your reviews, PLN appraisal and atlas remain available and unchanged.');return
        value=stored['result'];coverage=value['context']['coverage']
        lines=['Draft explanation for checking · source facts and scores are unchanged.','',
               str(coverage['sources_inspected'])+' linked abstract(s) · '+str(coverage['selected_notes'])+' selected notes / '+str(coverage['reviews_with_notes'])+' nonempty responses · '+str(coverage['current_reviews'])+' current reviewers.']
        if coverage['selected_notes']<coverage['reviews_with_notes']:lines.append(coverage['selection']+' Other notes remain in history.')
        brief=value['brief']
        for label,field in (('WHAT THE STUDY REPORTS','study_findings'),('REVIEWER DISCUSSION','review_discussion')):
            lines+=['',label]
            if not brief[field]:lines.append('No inspected finding is available.' if field=='study_findings' else 'No explanatory notes were supplied.' if not coverage['reviews_with_notes'] else 'No substantive explanation was established from the selected notes.')
            for item in brief[field]:
                lines.append(item['explanation'])
                for citation in item['citations']:lines.append(citation['id']+' — “'+citation['quote']+'”')
                lines.append('')
        lines+=['QUESTIONS TO RESOLVE']
        for item in brief['next_questions']:lines.append(item['question']+' ['+', '.join(item['reference_ids'])+']')
        if value.get('omissions'):lines+=['','An item containing an unreported quantity was omitted. Original model output is preserved in the explanation cache.']
        lines+=['','Native Omega · '+value['runtime']['model']+' · '+('0 new model calls · saved brief' if stored['cached'] else '1 model call')]
        self.ui.document('Omega review brief','\n'.join(lines),c['name']+' · '+('Saved · no new model call' if stored['cached'] else 'New explanation'))

    def record_concern(self,c,source):
        current=next((r for r in self.store.current_reviews(c) if r['clinician_id']==self.s.reviewer.id),None)
        self.ui.begin('Correction or concern',c['name']+' · '+c['condition'])
        self.say('To revise your judgment, choose Support/Oppose/Uncertain on the review screen.',kind='muted')
        for i,(_,label) in enumerate(CONCERNS.items(),1):self.say(f'{i}. '+label)
        if current and current.get('concern_kind'):self.say('6. Clear my recorded concern')
        self.say('0. Back')
        choice=input('Correction / concern category: ').strip()
        if choice=='0':return
        if choice=='6' and current and current.get('concern_kind'):
            self.s.require_access(c['patient_id'])
            self.store.review(c['id'],c['current_version'],self.s.reviewer,current['decision'],current['rationale'],current['citation'],
                              notes_only=True,submitted_decision=current['decision'],concern_kind='',concern_text='')
            self.say("Your concern was cleared. Its earlier record remains in history; other clinicians' concerns are unchanged.")
            self.refresh_atlas(announce=True);return
        if not choice.isdigit() or not 1<=int(choice)<=len(CONCERNS):raise ValueError('Choose a displayed concern category.')
        kind=list(CONCERNS)[int(choice)-1]
        text=input('What needs correcting or checking? (Enter to cancel): ').strip()
        if not text:return
        decision=current['decision'] if current else 'uncertain'
        if current is None:
            value=input('Your judgment: 1 Support  2 Oppose  3 Uncertain [3]: ').strip() or '3'
            if value not in ('1','2','3'):raise ValueError('Choose Support, Oppose or Uncertain.')
            decision={'1':'support','2':'oppose','3':'uncertain'}[value]
        self.s.require_access(c['patient_id'])
        self.store.review(c['id'],c['current_version'],self.s.reviewer,decision,current['rationale'] if current else '',
                          source['id'] if source else current['citation'] if current else '',notes_only=True,
                          submitted_decision=decision,concern_kind=kind,concern_text=text)
        self.say('Saved against your clinician identity. This flags the entry for review; it does not rewrite the publication or turn the concern into an extra PLN vote.')
        self.refresh_atlas(announce=True)

    def wording(self,c,source,pending):
        if pending:
            for i,row in enumerate(pending,1):self.say(f'{i}. Proposed by {row["proposer"]}: {row["replacement"]}')
            choice=input('Proposal number to inspect, n for a new proposal, or 0 Back: ').strip().casefold()
            if choice=='0':return
            if choice!='n':
                if not choice.isdigit() or not 1<=int(choice)<=len(pending):raise ValueError('Choose a displayed proposal number.')
                row=pending[int(choice)-1];grounding=json.loads(row['grounding_json'])
                self.say('Current statement: '+c['proposition'])
                self.say('Proposed statement: '+row['replacement'])
                self.say('Proposed by: '+row['proposer'])
                self.say('Supporting source: '+grounding['source_id']+' — '+grounding['url'])
                self.say('Supporting passage: '+grounding['quote'],2)
                self.s.claim_source_context(grounding['source'])
                if row['proposer_clinician_id']==self.s.reviewer.id:
                    self.say('Another clinician must approve your proposal. Switch reviewer from the intervention list.');return
                decision=input('1 Approve wording   2 Reject wording   0 Back: ').strip()
                if decision not in ('1','2'):return
                self.s.require_access(c['patient_id'])
                self.store.resolve_correction(row['id'],self.s.reviewer,decision=='1',
                    'Wording and supporting source checked by reviewer.' if decision=='1' else 'Proposed wording not approved by reviewer.')
                self.say('Wording approved. The revised statement starts its own reviews; earlier responses stay in history.' if decision=='1' else 'Proposal rejected. The current statement is unchanged.')
                return
        if not source:
            self.say('A linked study abstract is needed before proposing source-backed wording.');return
        statement=input('Corrected statement (Enter to cancel): ').strip()
        if not statement:return
        sections=source_sections(source)
        if not sections:raise ValueError('No supporting text is available.')
        quote=sections[0]['quote']
        while True:
            self.say('Proposed statement: '+statement)
            self.say('Supporting passage from '+source['id']+': '+quote,2)
            choice=input('1 Submit proposal   2 Paste a different supporting passage   0 Cancel: ').strip()
            if choice=='0':return
            if choice=='2':quote=input('Exact supporting passage: ').strip();continue
            if choice!='1':continue
            self.s.require_access(c['patient_id'])
            self.store.propose_claim(c,self.s.reviewer,statement,source['id'],quote)
            self.say('Proposal saved. The current statement and score are unchanged until another clinician approves it.');return
