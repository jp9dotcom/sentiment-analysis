"""
src/vectorizers.py
Vectorizers including custom wrappers for Word2Vec and SBERT to make them sklearn-compatible.
"""
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# 1. Dynamic TF-IDF (Accepts Unigrams, Bigrams, or Both)

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

# 4. BM25 Transformer
class BM25Transformer(BaseEstimator, TransformerMixin):
    """
    BM25 Transformer compatible with scikit-learn pipelines.
    Uses CountVectorizer to compute term frequencies and applies BM25 weighting.
    
    Parameters
    ----------
    k1 : float, default=1.5
        Term frequency saturation parameter.
    b : float, default=0.75
        Length normalization parameter.
    ngram_range : tuple, default=(1, 1)
        N-gram range for CountVectorizer.
    max_features : int, default=20000
        Maximum number of features.
    min_df : int, default=2
        Minimum document frequency.
    stop_words : str or list, default='english'
        Stop words to remove.
    """
    def __init__(self, k1=1.5, b=0.75, ngram_range=(1, 1), max_features=20000, min_df=2, stop_words='english'):
        self.k1 = k1
        self.b = b
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.min_df = min_df
        self.stop_words = stop_words
        self.vectorizer = None
        self.idf_ = None
        self.avgdl_ = None

    def fit(self, X, y=None):
        self.vectorizer = CountVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            min_df=self.min_df,
            stop_words=self.stop_words
        )
        X_counts = self.vectorizer.fit_transform(X)
        
        # Compute document lengths
        doc_lengths = X_counts.sum(axis=1).A1
        self.avgdl_ = np.mean(doc_lengths)
        
        # Compute IDF values (BM25 uses a slightly different IDF formulation)
        n_samples = X_counts.shape[0]
        df = np.array((X_counts > 0).sum(axis=0)).flatten()
        # BM25 IDF: log((N - df + 0.5) / (df + 0.5))
        self.idf_ = np.log((n_samples - df + 0.5) / (df + 0.5) + 1)
        
        return self

    def transform(self, X):
        if self.vectorizer is None:
            raise ValueError("BM25Transformer not fitted yet. Call fit() first.")
        
        X_counts = self.vectorizer.transform(X)
        
        # Get document lengths
        doc_lengths = X_counts.sum(axis=1).A1
        
        # Apply BM25 formula
        # BM25 score = IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (dl / avgdl)))
        
        # Convert to CSR for efficient row operations
        X_csr = X_counts.tocsr()
        
        # Compute denominator for each non-zero element
        # For each document, we need: tf + k1 * (1 - b + b * dl / avgdl)
        # We can compute the length normalization factor per document
        length_norm = self.k1 * (1 - self.b + self.b * doc_lengths / self.avgdl_)
        
        # Apply BM25 transformation to non-zero elements
        data = X_csr.data
        indices = X_csr.indices
        indptr = X_csr.indptr
        
        new_data = np.zeros_like(data)
        for i in range(X_csr.shape[0]):
            start, end = indptr[i], indptr[i + 1]
            tf = data[start:end]
            term_indices = indices[start:end]
            
            # BM25 formula
            idf_vals = self.idf_[term_indices]
            denom = tf + length_norm[i]
            new_data[start:end] = idf_vals * (tf * (self.k1 + 1)) / denom
        
        X_bm25 = sparse.csr_matrix((new_data, indices, indptr), shape=X_csr.shape)
        return X_bm25

    def get_feature_names_out(self, input_features=None):
        if self.vectorizer is None:
            raise ValueError("BM25Transformer not fitted yet. Call fit() first.")
        return self.vectorizer.get_feature_names_out(input_features)



