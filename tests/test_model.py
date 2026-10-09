import tempfile,unittest
from pathlib import Path
from model import TriageModel,ROOT

class TriageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.model=TriageModel()
    def test_training_example(self):
        result=self.model.predict('My card was charged twice for the monthly subscription')
        self.assertEqual(result['label'],'billing')
        self.assertTrue(result['supporting_terms'])
    def test_unknown_words_abstain(self):
        result=self.model.predict('qzxv qzxv qzxv')
        self.assertTrue(result['needs_review']);self.assertIsNone(result['label'])
        self.assertEqual(result['reason'],'no_known_terms')
    def test_invalid_input(self):
        for text in ['',None,123,'a'*5001]:
            with self.assertRaises(ValueError):self.model.predict(text)
    def test_scores_sum_to_one(self):
        self.assertAlmostEqual(sum(self.model.predict('Please help with my subscription')['scores'].values()),1,places=4)
    def test_test_vocabulary_not_fitted(self):
        self.assertNotIn('qzxv',self.model.pipeline.named_steps['tfidf'].vocabulary_)
        self.model.predict('qzxv')
        self.assertNotIn('qzxv',self.model.pipeline.named_steps['tfidf'].vocabulary_)
    def test_evaluation_and_leakage_guard(self):
        report=self.model.evaluate();self.assertEqual(report['test_rows'],16)
        self.assertGreater(report['accuracy'],report['baseline_accuracy'])
        with self.assertRaisesRegex(ValueError,'overlap'):self.model.evaluate(ROOT/'data/train.csv')
    def test_duplicate_training_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'rows.csv';path.write_text('text,label\nsame text,a\nSame text,b\n')
            with self.assertRaisesRegex(ValueError,'Duplicate'):TriageModel(path)

if __name__=='__main__':unittest.main()
