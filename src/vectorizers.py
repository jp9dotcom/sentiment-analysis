"""
src/vectorizers.py
Vectorizers including custom wrappers for Word2Vec and SBERT to make them sklearn-compatible.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Dynamic TF-IDF (Accepts Unigrams, Bigrams, or Both)
from sklearn.feature_extraction.text import CountVectorizer

class BM25Transformer(BaseEstimator, TransformerMixin):
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.vectorizer = CountVectorizer(stop_words="english")
        self.idf = None
        self.avgdl = None

    def fit(self, X, y=None):
        X_counts = self.vectorizer.fit_transform(X)

        n_docs = X_counts.shape[0]
        df = np.bincount(X_counts.indices, minlength=X_counts.shape[1])

        self.idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1)

        doc_lengths = np.asarray(X_counts.sum(axis=1)).ravel()
        self.avgdl = doc_lengths.mean()

        return self

    def transform(self, X):
        X_counts = self.vectorizer.transform(X).astype(np.float64)

        doc_lengths = np.asarray(X_counts.sum(axis=1)).ravel()

        rows, cols = X_counts.nonzero()

        for i, j in zip(rows, cols):
            tf = X_counts[i, j]
            denom = tf + self.k1 * (
                1 - self.b + self.b * doc_lengths[i] / self.avgdl
            )
            X_counts[i, j] = self.idf[j] * (tf * (self.k1 + 1)) / denom

        return X_counts

def get_tfidf(ngram_range=(1, 2)):
    """
    Returns a TF-IDF vectorizer.
    :param ngram_range: Tuple (min_n, max_n). 
                        (1,1) = Unigrams, (2,2) = Bigrams, (1,2) = Both.
    """
    return TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=20000, # Prevents OOM, keeps top 20k words
        min_df=2,           # Ignores extremely rare words
        stop_words='english'
    )

# 2. Word2Vec Wrapper
class Word2VecTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vector_size=300, window=5, min_count=1, workers=4):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None

    def fit(self, X, y=None):
        from gensim.models import Word2Vec
        # X is an array of strings. We need to tokenize them into lists of words.
        tokenized_docs = [doc.split() for doc in X]
        self.model = Word2Vec(
            sentences=tokenized_docs, 
            vector_size=self.vector_size, 
            window=self.window, 
            min_count=self.min_count, 
            workers=self.workers
        )
        return self

    def transform(self, X):
        # Average the word vectors for each document to get a single 300-d vector
        tokenized_docs = [doc.split() for doc in X]
        result = []
        for doc in tokenized_docs:
            valid_words = [w for w in doc if w in self.model.wv]
            if valid_words:
                # Average the vectors
                doc_vec = np.mean([self.model.wv[w] for w in valid_words], axis=0)
            else:
                # Fallback for unknown words
                doc_vec = np.zeros(self.vector_size)
            result.append(doc_vec)
        return np.array(result)

def get_word2vec():
    return Word2VecTransformer()

# 3. Sentence-BERT Wrapper
class SBertTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = None

    def fit(self, X, y=None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(self.model_name)
        return self

    def transform(self, X):
        # SBERT directly encodes lists of strings into 384-d dense vectors
        return self.model.encode(X, show_progress_bar=False)

def get_sbert():
    return SBertTransformer()

def get_bm25():
    return BM25Transformer()


def get_all_vectorizers():
    return {
        'tfidf': get_tfidf(),
        'bm25': get_bm25(),
        'word2vec': get_word2vec(),
        'sbert': get_sbert()
    }
