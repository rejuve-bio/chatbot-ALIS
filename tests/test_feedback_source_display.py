import unittest
from unittest.mock import patch

from app.services.feedback_source_display import source_sections


class SourceDisplayChecks(unittest.TestCase):
    def source(self, text):
        return dict(id='PMID:39940362',text=text,abstract_available=True)

    def test_results_are_literal_and_separate_from_author_interpretation(self):
        source=self.source('Title. Methods : Groups were compared. Results : Marker A rose in both groups. '
                           'Marker B rose only in one group. Conclusions : Further studies are needed.')
        rows=source_sections(source)
        self.assertEqual(len(rows),2)
        self.assertEqual(rows[0]['quote'],'Marker A rose in both groups. Marker B rose only in one group.')
        self.assertEqual(rows[1]['quote'],'Further studies are needed.')
        self.assertTrue(all(r['quote'] in source['text'] for r in rows))
        self.assertFalse(any(r['truncated'] for r in rows))

    def test_unstructured_abstract_is_not_assigned_an_invented_results_section(self):
        source=self.source('This study measured a marker. Both groups improved. More research is needed.')
        rows=source_sections(source)
        self.assertEqual(rows[0]['quote'],source['text'])
        self.assertIn('Retrieved abstract',rows[0]['label'])

    def test_metadata_only_is_not_presented_as_study_results(self):
        source=self.source('A title with no retrieved abstract.')
        source['abstract_available']=False
        self.assertEqual(source_sections(source),[])

    def test_conclusions_only_also_shows_the_available_source_context(self):
        rows=source_sections(self.source('Methods : A small study. Conclusions : An intervention may be promising.'))
        self.assertIn('Retrieved abstract',rows[0]['label'])
        self.assertIn("Authors' interpretation",rows[1]['label'])

    def test_long_source_excerpts_are_marked_and_remain_literal(self):
        source=self.source('Results : '+('A finding was reported. '*150)+'Conclusions : Check the source.')
        rows=source_sections(source)
        self.assertTrue(rows[0]['truncated'])
        self.assertLessEqual(len(rows[0]['quote']),1800)
        self.assertIn(rows[0]['quote'],source['text'])

    def test_no_numeric_or_model_service_is_used(self):
        with patch('app.services.feedback_pln.execute_native',side_effect=AssertionError('Unexpected calculation')):
            self.assertTrue(source_sections(self.source('Results : A change was reported.')))


if __name__=='__main__':unittest.main()
