"""Train-only TF-IDF fitting, class probabilities, abstention and local explanations."""
import csv
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.dummy import DummyClassifier

ROOT=Path(__file__).resolve().parent
def load_rows(path):
    with Path(path).open(encoding='utf-8',newline='') as handle:
        rows=list(csv.DictReader(handle))
    if not rows or any(not row.get('text','').strip() or not row.get('label','').strip() for row in rows):
        raise ValueError('CSV requires non-empty text and label columns')
    normalized=[row['text'].strip().casefold() for row in rows]
    if len(set(normalized))!=len(normalized):raise ValueError('Duplicate ticket texts are not allowed within a split')
    return rows

class TriageModel:
    def __init__(self,train_path=ROOT/'data/train.csv',threshold=0.45):
        if not 0<=threshold<=1:raise ValueError('threshold must be in [0, 1]')
        self.threshold=threshold
        self.rows=load_rows(train_path)
        if len(set(row['label'] for row in self.rows))<2:raise ValueError('At least two labels are required')
        self.pipeline=Pipeline([
            ('tfidf',TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,min_df=1)),
            ('classifier',LogisticRegression(C=4,max_iter=2000,random_state=42,class_weight='balanced'))])
        self.pipeline.fit([row['text'] for row in self.rows],[row['label'] for row in self.rows])

    def predict(self,text):
        if not isinstance(text,str) or not 3<=len(text.strip())<=5000:
            raise ValueError('text must contain 3–5000 characters')
        vectorizer=self.pipeline.named_steps['tfidf']; classifier=self.pipeline.named_steps['classifier']
        vector=vectorizer.transform([text]);probabilities=classifier.predict_proba(vector)[0]
        index=int(np.argmax(probabilities));label=str(classifier.classes_[index])
        confidence=float(probabilities[index]);known=vector.nnz>0
        needs_review=not known or confidence<self.threshold
        # Per-feature logit contributions, not causal explanations or calibrated confidence.
        weights=classifier.coef_[index] if len(classifier.classes_)>2 else classifier.coef_[0]*(1 if index==1 else -1)
        contributions=vector.toarray()[0]*weights
        names=vectorizer.get_feature_names_out()
        terms=[{'term':str(names[i]),'contribution':round(float(contributions[i]),5)}
               for i in np.argsort(contributions)[::-1] if contributions[i]>0][:5]
        return {'label':None if needs_review else label,'suggested_label':label if known else None,
                'needs_review':needs_review,'probability':round(confidence,5),
                'probability_note':'Uncalibrated model output, not a guarantee of correctness',
                'reason':'no_known_terms' if not known else ('low_probability' if needs_review else 'classified'),
                'scores':{str(c):round(float(p),5) for c,p in zip(classifier.classes_,probabilities)},
                'supporting_terms':terms}

    def evaluate(self,test_path=ROOT/'data/test.csv'):
        test=load_rows(test_path)
        train_text={r['text'].strip().casefold() for r in self.rows}
        if train_text & {r['text'].strip().casefold() for r in test}:
            raise ValueError('Train/test text overlap detected')
        texts=[row['text'] for row in test];truth=[row['label'] for row in test]
        predictions=self.pipeline.predict(texts)
        outputs=[self.predict(text) for text in texts]
        accepted=[i for i,out in enumerate(outputs) if not out['needs_review']]
        baseline=DummyClassifier(strategy='most_frequent').fit(np.zeros((len(self.rows),1)),[r['label'] for r in self.rows])
        return {'dataset':'80 hand-authored synthetic tickets; 64 train / 16 held-out',
                'warning':'Tiny illustrative benchmark, not evidence of real support-desk accuracy',
                'accuracy':float(accuracy_score(truth,predictions)),
                'classification_report':classification_report(truth,predictions,output_dict=True,zero_division=0),
                'labels':self.pipeline.classes_.tolist(),
                'confusion_matrix':confusion_matrix(truth,predictions,labels=self.pipeline.classes_).tolist(),
                'baseline_accuracy':float(accuracy_score(truth,baseline.predict(np.zeros((len(test),1))))),
                'review_threshold':self.threshold,'coverage':len(accepted)/len(test),
                'accepted_accuracy':float(accuracy_score([truth[i] for i in accepted],[predictions[i] for i in accepted])) if accepted else None,
                'train_rows':len(self.rows),'test_rows':len(test)}
