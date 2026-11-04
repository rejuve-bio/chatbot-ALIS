"""Readable terminal output without changing entry identity or review semantics."""
import io
import json
import os
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch

from app.services.feedback_terminal import Terminal,ANSI,visible_width,clip,confidence
from app.services.feedback_auth import register_local,login_local
from app.services.feedback_store import FeedbackStore
from app.services.feedback_menu import GuidedFeedback
from app.services.feedback_atlas import ReviewedAtlas
from tests.test_feedback_atlas import PolicyOnlyClient

Session=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/feedback'))['Session']


class TerminalLayout(unittest.TestCase):
    def test_no_color_and_untrusted_terminal_escape_sequences(self):
        out=io.StringIO()
        with patch.dict(os.environ,{'NO_COLOR':'1','TERM':'xterm-256color'}):
            ui=Terminal(lambda value='':print(value,file=out),interactive=True)
            ui.can_clear=False;ui.text('\x1b[31mAn untrusted\x1b[0m note',kind='warning')
        self.assertNotIn('\x1b',out.getvalue());self.assertIn('An untrusted note',out.getvalue())

    def test_wide_text_and_tables_fit_the_terminal(self):
        out=[]
        with patch.dict(os.environ,{'COLUMNS':'80','LINES':'24'}):
            ui=Terminal(out.append,interactive=False,color=False)
            row=dict(entry_number=123,name='長い intervention name '+('a'*100),routing=dict(label='Needs review',status='needs_review',decisive=2,total=3),
                     appraisal=dict(strength=.5,confidence=2/3))
            ui.entries([row])
            self.assertTrue(all(visible_width(line)<=ui.width for line in out))
            self.assertIn('123',out[-1]);self.assertIn('0.667',out[-1]);self.assertIn('2/3',out[-1])
        self.assertLessEqual(visible_width(clip('長い名前',5)),5)

    def test_large_confidence_never_rounds_up_to_certainty(self):
        self.assertEqual(confidence(None),'—');self.assertEqual(confidence(10000/10001),'<1.000')

    def test_source_pager_keeps_all_text_available_in_piped_output(self):
        out=[];ui=Terminal(out.append,interactive=False,color=False)
        ui.document('Full source','First finding.\n\nOpposing finding.\nFinal limitation.')
        text='\n'.join(out)
        for phrase in ('First finding.','Opposing finding.','Final limitation.'):self.assertIn(phrase,text)
        self.assertNotIn('\x1b',text)

    def test_last_reader_page_labels_enter_as_back(self):
        out=[];ui=Terminal(out.append,interactive=True,color=False);ui.can_clear=False
        with patch.dict(os.environ,{'COLUMNS':'80','LINES':'24'}),patch('builtins.input',side_effect=['n','']):
            ui.document('Full source','\n'.join('Source line '+str(i) for i in range(25)))
        self.assertIn('[Enter/0] Back',out[-1])


class PageNavigation(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.store=FeedbackStore(Path(self.tmp.name)/'reviews.sqlite3')
        register_local(self.store,'Trial','trial@example.test','Synthetic-password-2026')
        self.auth=login_local(self.store,'trial@example.test','Synthetic-password-2026')
        self.path=Path(self.tmp.name)/'handoff.json'
        self.path.write_text(json.dumps({'patient_id':'synthetic','risk_areas':[{'disease_name':'Condition','investigational_evidence':[
            {'name':'Candidate '+str(i)} for i in range(1,21)]}]}))
        self.session=Session(self.store,self.auth,'local','http://127.0.0.1:5001');self.session.import_handoff(self.path)
        self.output=[];self.menu=GuidedFeedback(self.session,self.output.append,self.path)
        self.menu.ui.interactive=False;self.menu.ui.color=False;self.menu.ui.can_clear=False
        self.menu.atlas=ReviewedAtlas(self.session,PolicyOnlyClient())

    def tearDown(self):self.store.close();self.tmp.cleanup()

    def test_pagination_preserves_all_records_and_ids_can_open_from_any_page(self):
        with patch.dict(os.environ,{'COLUMNS':'80','LINES':'24'}),patch('builtins.input',side_effect=['n','20','1','Layout trial','q']):self.menu.run()
        c=self.store.resolve('20');reviews=self.store.current_reviews(c)
        self.assertEqual(len(self.store.candidates()),20)
        self.assertEqual(reviews[0]['decision'],'support')
        text='\n'.join(self.output)
        self.assertIn('Showing 1–8 of 20',text);self.assertIn('Showing 9–16 of 20',text)
        self.assertIn('Saved Support · Candidate 20',text)
        self.assertIn('Before:',text);self.assertIn('After:',text)

    def test_card_no_longer_repeats_after_opening_evidence_tools(self):
        with patch('builtins.input',side_effect=['1','4','0','0','q']):self.menu.run()
        text='\n'.join(self.output)
        self.assertEqual(text.count('The supplied evidence supports retaining Candidate 1 as a research candidate for'),1)

    def test_current_viewer_can_see_their_own_response_even_after_other_reviewers(self):
        c=self.store.resolve('1')
        self.store.review(c['id'],c['current_version'],self.auth.clinician,'support','My original reason.',notes_only=True)
        for name in ('other-a','other-b'):
            register_local(self.store,name,name+'@example.test','Synthetic-password-2026')
            other=login_local(self.store,name+'@example.test','Synthetic-password-2026')
            self.store.review(c['id'],c['current_version'],other.clinician,'uncertain','Another reason.',notes_only=True)
        self.menu.card(self.store.resolve('1'),None)
        text='\n'.join(self.output)
        self.assertIn('Trial (you)',text);self.assertIn('My original reason.',text)
        self.assertEqual(len(self.store.current_reviews(c)),3)


if __name__=='__main__':unittest.main()
