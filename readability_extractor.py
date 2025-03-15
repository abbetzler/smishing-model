from sklearn.base import BaseEstimator, TransformerMixin
import textstat
import pandas as pd


class ReadabilityFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        readability_features = X.apply(self._extract_readability_features)
        return pd.DataFrame(list(readability_features))

    def _extract_readability_features(self, text):
        return {
            'flesch_reading_ease': textstat.flesch_reading_ease(text),
            'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
            'gunning_fog': textstat.gunning_fog(text),
            'smog_index': textstat.smog_index(text),
            'dale_chall_score': textstat.dale_chall_readability_score(text),
            'ari': textstat.automated_readability_index(text),
            'coleman_liau_index': textstat.coleman_liau_index(text),
            'linsear_write': textstat.linsear_write_formula(text),
            'avg_word_length': sum(len(word) for word in text.split()) / len(text.split() if len(text) > 0 else "0"),
            'avg_sentence_length': len(text.split()) / len(text.split('.')),
            'pct_complex_words': sum(1 for word in text.split() if textstat.syllable_count(word) >= 3) / (len(text.split()) if len(text) > 0 else 1)
        }
